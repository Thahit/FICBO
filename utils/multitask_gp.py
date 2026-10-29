import torch
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.likelihoods import Likelihood
from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood
from gpytorch.priors import LogNormalPrior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.kernels.positive_index import PositiveIndexKernel
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from typing import Any, Dict, Optional, Tuple

MTGP_KINDS = ("icm", "kronecker")

MIN_LENGTHSCALE = 2.5e-2
MIN_NOISE = 1e-4


def _enforce_constraint(module: torch.nn.Module, name: str, lower_bound: float) -> None:
    # botorch transform=None unenforced, adam walked lengthscale to -0.06, use softplus
    value = getattr(module, name).detach().clone().clamp_min(lower_bound * 2.0)
    module.register_constraint(f"raw_{name}", GreaterThan(lower_bound))
    setattr(module, name, value)


def _unit_cube(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # botorch kernel priors assume unit cube
    span = (hi - lo).clamp_min(1e-8)
    return ((x - lo) / span).clamp(0.0, 1.0)


def _standardize(y: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # y [B, N, 1], per env
    y_mean = y.mean(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)
    # n=1 -> std nan, the t=0 case
    fallback = y.abs().mean(dim=dim, keepdim=True).clamp(min=1.0)
    y_std = torch.where(torch.isfinite(y_std) & (y_std >= 1e-6), y_std, fallback)
    return (y - y_mean) / y_std, y_mean, y_std


def _subsample_batched(B: int, n: int, max_n: Optional[int], device: torch.device) -> torch.Tensor:
    # [B, min(n, max_n)], own subset per env
    if max_n is None or n <= max_n:
        return torch.arange(n, device=device).unsqueeze(0).expand(B, n)
    return torch.argsort(torch.rand(B, n, device=device), dim=1)[:, :max_n]


class _BatchedPositiveIndexKernel(PositiveIndexKernel):
    # botorch var * eye only broadcasts unbatched, diag_embed as in gpytorch IndexKernel

    def _eval_covar_matrix(self) -> torch.Tensor:
        cf = self.covar_factor
        covar = cf @ cf.transpose(-1, -2) + torch.diag_embed(self.var)
        if self.unit_scale_for_target:
            norm_factor = covar[..., self.target_task_index, self.target_task_index]
            covar = covar / norm_factor.unsqueeze(-1).unsqueeze(-1)
        return covar


class BatchedMultiTaskGP(ExactGP):

    def __init__(
        self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: Likelihood, rank: int,
        ) -> None:
        super().__init__(train_x, train_y, likelihood)
        batch_shape = torch.Size([train_x.shape[0]])
        dim_x = train_x.shape[-1] - 1
        # as botorch MultiTaskGP plus env batch. positive task kernel, sign barely identified
        self.mean_module = MultitaskMean(ConstantMean(batch_shape=batch_shape), num_tasks=2)
        base_kernel = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=dim_x, batch_shape=batch_shape,
            )
        _enforce_constraint(base_kernel, "lengthscale", MIN_LENGTHSCALE)
        # free outputscale, context std understates pool spread. B[0,0] pinned, still identifiable
        self.data_covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)
        self.task_covar_module = _BatchedPositiveIndexKernel(num_tasks=2, rank=rank, batch_shape=batch_shape)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        # last col = task, 0 objective, 1 feedback
        x_data, task_idx = x[..., :-1], x[..., -1:].long()
        mean_all_tasks = self.mean_module(x_data)
        mean_x = mean_all_tasks.gather(-1, task_idx).squeeze(-1)
        covar_x = self.data_covar_module(x_data)
        covar_i = self.task_covar_module(task_idx)
        return MultivariateNormal(mean_x, covar_x.mul(covar_i))


def _build_icm_gp(
    x_ctx: torch.Tensor, y_ctx: torch.Tensor, x_fb: torch.Tensor, y_fb: torch.Tensor,
    rank: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    # task 0 objective on ctx, task 1 feedback on full pool
    B, n_ctx = x_ctx.shape[0], x_ctx.shape[1]
    keep_n = x_fb.shape[1]
    y_ctx_std, y_mean, y_std = _standardize(y_ctx, dim=1)
    y_fb_std, _, _ = _standardize(y_fb, dim=1)

    task_0 = torch.zeros(B, n_ctx, 1, dtype=x_ctx.dtype, device=x_ctx.device)
    task_1 = torch.ones(B, keep_n, 1, dtype=x_fb.dtype, device=x_fb.device)
    train_x = torch.cat(
        [torch.cat([x_ctx, task_0], dim=-1), torch.cat([x_fb, task_1], dim=-1)], dim=1,
        )
    train_y = torch.cat([y_ctx_std, y_fb_std], dim=1).squeeze(-1)
    rank_eff = rank if rank is not None else 2
    return train_x, train_y, rank_eff, y_mean, y_std


def _build_kronecker_gp(
    x_ctx: torch.Tensor, y_ctx: torch.Tensor, fb_ctx: torch.Tensor, rank: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    # block-aligned, pool feedback unused
    return _build_icm_gp(x_ctx, y_ctx, x_ctx, fb_ctx, rank)


_JITTER_SCHEDULE = (1e-2, 1e-1, 1.0)


def _mll_loss(
    mll: ExactMarginalLogLikelihood, model: BatchedMultiTaskGP, train_x: torch.Tensor,
    train_y: torch.Tensor, diag: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
    # one bad env breaks the batched cholesky, escalate jitter
    for jitter in _JITTER_SCHEDULE:
        try:
            with gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
                # task noise reads task col from train_x
                loss = -mll(model(train_x), train_y, train_x).mean()
        except Exception as exc:
            diag["errors"].append(f"mll@jitter={jitter:g}: {type(exc).__name__}: {exc}")
            continue
        if torch.isfinite(loss):
            diag["max_jitter"] = max(diag["max_jitter"], jitter)
            return loss
        diag["errors"].append(f"mll@jitter={jitter:g}: non-finite loss")
    return None


def _load_matching_state(module: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    # N grew since last step, load only shape-matching keys, else whole load no-ops
    current = module.state_dict()
    matching = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    module.load_state_dict(matching, strict=False)


def _build_likelihood(B: int) -> HadamardGaussianLikelihood:
    # per-task noise, shared one would swamp the few objective points
    return HadamardGaussianLikelihood(
        num_tasks=2, batch_shape=torch.Size([B]),
        noise_prior=LogNormalPrior(loc=-4.0, scale=1.0),
        noise_constraint=GreaterThan(MIN_NOISE),
        task_feature_index=-1,
        )


def _fit_batch(
    train_x: torch.Tensor, train_y: torch.Tensor, rank: int, n_iter: int, lr: float,
    prev_state: Optional[Dict[str, Any]], diag: Dict[str, Any],
    ) -> Tuple[BatchedMultiTaskGP, Likelihood, Optional[Dict[str, Any]]]:
    # all B in one adam loop, so no fit_gpytorch_mll
    device = train_x.device
    B = train_x.shape[0]

    # match data dtype, double for cholesky
    likelihood = _build_likelihood(B).to(device=device, dtype=train_x.dtype)
    model = BatchedMultiTaskGP(train_x, train_y, likelihood, rank).to(device=device, dtype=train_x.dtype)

    if prev_state is not None:
        try:
            _load_matching_state(model, prev_state["model"])
            _load_matching_state(likelihood, prev_state["likelihood"])
        except Exception as exc:
            diag["errors"].append(f"warm_start: {type(exc).__name__}: {exc}")

    if any(torch.isnan(p).any() for p in model.parameters()):
        diag["errors"].append("warm_start produced NaN parameters; reinitialised")
        model = BatchedMultiTaskGP(train_x, train_y, likelihood, rank).to(
            device=device, dtype=train_x.dtype
            )

    trained_any = False
    if n_iter > 0:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(n_iter):
            optimizer.zero_grad()
            loss = _mll_loss(mll, model, train_x, train_y, diag)
            if loss is None:
                # skip just this step
                diag["steps_skipped"] += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            diag["steps_taken"] += 1
            trained_any = True

    model.eval()
    likelihood.eval()

    diag["trained"] = trained_any
    _record_hypers(model, likelihood, diag)

    state = None
    if trained_any and not any(torch.isnan(p).any() for p in model.parameters()):
        state = {"model": model.state_dict(), "likelihood": likelihood.state_dict()}
    return model, likelihood, state


def _record_hypers(
    model: BatchedMultiTaskGP, likelihood: Likelihood, diag: Dict[str, Any],
    ) -> None:
    # can go non-positive silently, record
    with torch.no_grad():
        lengthscale = model.data_covar_module.base_kernel.lengthscale.detach()
        noise = likelihood.noise.detach()
        task_covar = model.task_covar_module.covar_matrix.detach()
        corr = task_covar[..., 0, 1] / (task_covar[..., 0, 0] * task_covar[..., 1, 1]).sqrt()

    diag["lengthscale"] = lengthscale.mean(dim=0).flatten().tolist()
    # [obj, fb], fb >> obj -> kernel can't explain feedback
    diag["noise"] = noise.mean(dim=0).flatten().tolist()
    diag["task_corr"] = corr.mean().item()
    if (lengthscale < MIN_LENGTHSCALE).any() or (noise < MIN_NOISE).any():
        diag["errors"].append(
            f"constraint violated: min lengthscale={lengthscale.min().item():.4g} "
            f"(floor {MIN_LENGTHSCALE:g}), min noise={noise.min().item():.4g} (floor {MIN_NOISE:g})"
            )


def _new_diag() -> Dict[str, Any]:
    return {
        "trained": False, "steps_taken": 0, "steps_skipped": 0, "max_jitter": 0.0,
        "posterior_fallback": 0, "lengthscale": None, "noise": None, "task_corr": None,
        "errors": [],
        }


def fit_multitask_gps(
    context_x: torch.Tensor, context_y: torch.Tensor, context_feedback: torch.Tensor,
    pool_x: torch.Tensor, pool_feedback: torch.Tensor, *,
    kind: str = "icm", rank: Optional[int] = None, max_pool: Optional[int] = 64,
    n_iter: int = 50, prev_states: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    # one batched GP, botorch MultiTaskGP rejects batch dims
    # returns (fits, state, diag)
    if kind not in MTGP_KINDS:
        raise NotImplementedError(f"Unknown multi-task GP kind: {kind}")

    diag = _new_diag()
    device = context_x.device
    x_ctx = context_x.detach().double()
    y_ctx = context_y.detach().double()
    fb_ctx = context_feedback.detach().double()

    all_x = torch.cat([x_ctx, pool_x.detach().double()], dim=1)
    lo = all_x.min(dim=1, keepdim=True).values
    hi = all_x.max(dim=1, keepdim=True).values

    try:
        if kind == "icm":
            fb_all_y = torch.cat([fb_ctx, pool_feedback.detach().double()], dim=1)
            keep = _subsample_batched(all_x.shape[0], all_x.shape[1], max_pool, device)
            x_fb = torch.gather(all_x, 1, keep.unsqueeze(-1).expand(-1, -1, all_x.shape[-1]))
            y_fb = torch.gather(fb_all_y, 1, keep.unsqueeze(-1).expand(-1, -1, fb_all_y.shape[-1]))
            train_x, train_y, rank_eff, y_mean, y_std = _build_icm_gp(
                _unit_cube(x_ctx, lo, hi), y_ctx, _unit_cube(x_fb, lo, hi), y_fb, rank,
                )
        else:
            train_x, train_y, rank_eff, y_mean, y_std = _build_kronecker_gp(
                _unit_cube(x_ctx, lo, hi), y_ctx, fb_ctx, rank,
                )
    except Exception as exc:
        diag["errors"].append(f"build: {type(exc).__name__}: {exc}")
        return None, None, diag

    # 1 ctx point -> target 0, nothing to fit
    n_iter_eff = 0 if x_ctx.shape[1] < 2 else n_iter
    if n_iter_eff == 0 and n_iter > 0:
        diag["errors"].append(f"fit skipped: only {x_ctx.shape[1]} context point(s)")

    model, likelihood, state = _fit_batch(
        train_x, train_y, rank_eff, n_iter_eff, 0.1, prev_states, diag,
        )

    fits = {
        "model": model, "likelihood": likelihood, "kind": kind,
        "lo": lo, "hi": hi, "y_mean": y_mean, "y_std": y_std, "diag": diag,
        }
    return fits, state, diag


def multitask_posterior(
    fits: Optional[Dict[str, Any]], x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # x [B, N, dim_x] -> mean, std [B, N] of objective task, original y units
    B, N = x.shape[0], x.shape[1]
    mean = torch.zeros(B, N, dtype=x.dtype, device=x.device)
    std = torch.ones(B, N, dtype=x.dtype, device=x.device)
    if fits is None:
        return mean, std
    diag = fits.get("diag")

    x_norm = _unit_cube(x.detach().double(), fits["lo"], fits["hi"])
    task_0 = torch.zeros(B, N, 1, dtype=x_norm.dtype, device=x_norm.device)
    x_task0 = torch.cat([x_norm, task_0], dim=-1)

    # escalating jitter as _mll_loss. latent f, obs noise would flatten ucb
    m = v = None
    for jitter in _JITTER_SCHEDULE:
        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
                post = fits["model"](x_task0)
                m, v = post.mean, post.variance
            break
        except Exception as exc:
            if diag is not None:
                diag["errors"].append(f"posterior@jitter={jitter:g}: {type(exc).__name__}: {exc}")
            continue
    if m is None:
        # fallback mean 0 std 1, argmax -> slot 0, not random
        if diag is not None:
            diag["posterior_fallback"] += 1
        return mean, std

    y_mean = fits["y_mean"].squeeze(-1)
    y_std = fits["y_std"].squeeze(-1)
    mean = torch.nan_to_num((m * y_std + y_mean).to(x.dtype), nan=0.0)
    std = torch.nan_to_num((v.clamp_min(1e-8).sqrt() * y_std).to(x.dtype), nan=1.0)
    return mean, std

