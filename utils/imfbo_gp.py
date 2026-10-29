# iMFBO, Fan et al. UAI 2024. feedback = noisy obs of f, learned sigma_lf(x), one GP
# differs from mtgp (2nd task), gp_feat (input col), rgpe (base task)

import math
import torch
import torch.nn.functional as F
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import Likelihood
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import Noise
from gpytorch.priors import LogNormalPrior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.operators import DiagLinearOperator
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from typing import Any, Dict, List, Optional, Tuple

from utils.multitask_gp import (
    MIN_LENGTHSCALE, MIN_NOISE, _JITTER_SCHEDULE,
    _enforce_constraint, _unit_cube, _standardize, _subsample_batched, _load_matching_state,
    )

IMFBO_NOISE_MODELS = ("parametric", "gp")
IMFBO_ACQUISITIONS = ("nvucb", "nucb", "ucb")

MIN_SCALE = 5e-2
MAX_LF_NOISE = 1e4
RAW_CLAMP = 30.0
LOG_NOISE_CLAMP = 10.0
INIT_HF_NOISE = 1e-2
INIT_LF_NOISE = 1e-1
LOG_NOISE_OUTPUTSCALE = 0.5
LOG_NOISE_JITTER = 1e-4
MLP_PRIOR_SCALE = 1.0
HUGE_LF_NOISE = 1e6


def _lf_noise_from_raw(raw: torch.Tensor) -> torch.Tensor:
    # softplus not exp, adam has no line search, exp overflows
    return MIN_NOISE + F.softplus(raw.clamp(-RAW_CLAMP, RAW_CLAMP)).clamp(max=MAX_LF_NOISE)


class _LFNoiseMLP(torch.nn.Module):

    def __init__(self, B: int, dim_x: int, hidden: int, init_raw: float) -> None:
        # w2 ~0 -> homoscedastic start, exact 0 stalls w1
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(B, dim_x, hidden) / max(dim_x, 1) ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(B, 1, hidden))
        self.w2 = torch.nn.Parameter(torch.randn(B, hidden, 1) * 1e-2)
        self.b2 = torch.nn.Parameter(torch.full((B, 1, 1), init_raw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, dim_x] -> raw noise [B, N]
        h = torch.tanh(torch.baddbmm(self.b1, x, self.w1))
        return torch.baddbmm(self.b2, h, self.w2).squeeze(-1)

    def penalty(self) -> torch.Tensor:
        # MAP penalty [B], not register_prior, that broadcasts silently
        return sum(
            p.reshape(p.shape[0], -1).square().sum(-1) for p in (self.w1, self.w2)
            ) / (2.0 * MLP_PRIOR_SCALE ** 2)


class _LogNoiseGPPrior(torch.nn.Module):

    def __init__(self, B: int, dim_x: int) -> None:
        # eq. 12 MAP, GP prior on log sigma^2 stops rows collapsing to floor
        super().__init__()
        batch_shape = torch.Size([B])
        self.mean = torch.nn.Parameter(torch.full((B, 1), math.log(INIT_LF_NOISE)))
        base_kernel = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=dim_x, batch_shape=batch_shape,
            )
        _enforce_constraint(base_kernel, "lengthscale", MIN_LENGTHSCALE)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)
        self.covar_module.outputscale = torch.full((B,), LOG_NOISE_OUTPUTSCALE)

    def log_prob(self, x_lf: torch.Tensor, z_lf: torch.Tensor) -> torch.Tensor:
        # [B, n_lf, d], [B, n_lf] -> [B]. prior, fixed jitter ok
        n = z_lf.shape[-1]
        K = self.covar_module(x_lf).to_dense()
        eye = torch.eye(n, dtype=K.dtype, device=K.device)
        L = torch.linalg.cholesky(K + LOG_NOISE_JITTER * eye)
        r = (z_lf - self.mean).unsqueeze(-1)
        quad = (r * torch.cholesky_solve(r, L)).sum(dim=(-2, -1))
        logdet = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (quad + logdet + n * math.log(2.0 * math.pi))


class _SourceNoise(Noise):

    def __init__(
        self, B: int, n_train: int, n_hf: int, dim_x: int, noise_model: str, hidden: int,
        ) -> None:
        # train_x passed explicitly at fit, eval gets it from DefaultPredictionStrategy
        super().__init__()
        self.noise_model = noise_model
        self.n_hf = n_hf
        self.lf_noise_override: Optional[float] = None
        self.register_parameter("raw_noise", torch.nn.Parameter(torch.zeros(B, 1)))
        # enforced softplus, unlike botorch, see multitask_gp._enforce_constraint
        self.register_constraint("raw_noise", GreaterThan(MIN_NOISE))
        self.register_prior("hf_noise_prior", LogNormalPrior(loc=-4.0, scale=1.0), _hf_noise_param)
        self.noise = torch.full((B, 1), INIT_HF_NOISE)

        if noise_model == "parametric":
            self.lf_net = _LFNoiseMLP(B, dim_x, hidden, _inv_softplus(INIT_LF_NOISE))
            self.raw_lf = None
        else:
            # full training block, no gather, hf rows get zero grad
            self.lf_net = None
            self.register_parameter(
                "raw_lf", torch.nn.Parameter(torch.full((B, n_train), math.log(INIT_LF_NOISE))),
                )

    @property
    def noise(self) -> torch.Tensor:
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: torch.Tensor) -> None:
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def lf_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.lf_noise_override is not None:
            return torch.full_like(x[..., 0], self.lf_noise_override)
        if self.noise_model == "parametric":
            return _lf_noise_from_raw(self.lf_net(x))
        # exp ok, prior is on log sigma^2
        return MIN_NOISE + self.raw_lf.clamp(-LOG_NOISE_CLAMP, LOG_NOISE_CLAMP).exp()

    def forward(
        self, *params: Any, shape: Optional[torch.Size] = None, **kwargs: Any,
        ) -> DiagLinearOperator:
        # tensor at fit, train_inputs tuple at eval, as HadamardGaussianLikelihood
        x_aug = params[0] if torch.is_tensor(params[0]) else params[0][0]
        x, is_lf = x_aug[..., :-1], (x_aug[..., -1] > 0.5).to(x_aug.dtype)
        # blend, broadcasts [B, 1] hf noise
        lf = self.lf_noise(x)
        return DiagLinearOperator(self.noise + is_lf * (lf - self.noise))


class SourceGaussianLikelihood(_GaussianLikelihoodBase):

    def __init__(self, noise_covar: _SourceNoise) -> None:
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> torch.Tensor:
        return self.noise_covar.noise


class BatchedIMFBOGP(ExactGP):

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: Likelihood) -> None:
        # one GP over f, lf rows see a*f + b
        # a > 0, sign barely identified with few points
        super().__init__(train_x, train_y, likelihood)
        batch_shape = torch.Size([train_x.shape[0]])
        dim_x = train_x.shape[-1] - 1
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        base_kernel = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=dim_x, batch_shape=batch_shape,
            )
        _enforce_constraint(base_kernel, "lengthscale", MIN_LENGTHSCALE)
        # default Positive() already enforced
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)
        self.register_parameter("raw_scale", torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        self.register_constraint("raw_scale", GreaterThan(MIN_SCALE))
        self.scale = torch.ones(*batch_shape, 1)
        self.register_parameter("offset", torch.nn.Parameter(torch.zeros(*batch_shape, 1)))

    @property
    def scale(self) -> torch.Tensor:
        return self.raw_scale_constraint.transform(self.raw_scale)

    @scale.setter
    def scale(self, value: torch.Tensor) -> None:
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        # last col = source, 0 hf, 1 feedback. test points 0 -> latent f
        x_data, is_lf = x[..., :-1], (x[..., -1] > 0.5).to(x.dtype)
        s = 1.0 + is_lf * (self.scale - 1.0)
        mean_x = s * self.mean_module(x_data) + is_lf * self.offset
        # mul with tensor exact, with LinearOperator rank-100 lanczos approx
        covar_x = self.covar_module(x_data).mul(s.unsqueeze(-1) * s.unsqueeze(-2))
        return MultivariateNormal(mean_x, covar_x)


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _hf_noise_param(module: _SourceNoise) -> torch.Tensor:
    return module.noise


def _build_imfbo_data(
    x_ctx: torch.Tensor, y_ctx: torch.Tensor, fb_ctx: torch.Tensor,
    x_pool: torch.Tensor, fb_pool: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    # [hf ctx | lf ctx | lf pool], ctx pairs never subsampled, only evidence for a, b
    B, n_ctx = x_ctx.shape[0], x_ctx.shape[1]
    x_lf = torch.cat([x_ctx, x_pool], dim=1)
    y_lf = torch.cat([fb_ctx, fb_pool], dim=1)

    src_hf = torch.zeros(B, n_ctx, 1, dtype=x_ctx.dtype, device=x_ctx.device)
    src_lf = torch.ones(B, x_lf.shape[1], 1, dtype=x_lf.dtype, device=x_lf.device)
    train_x = torch.cat(
        [torch.cat([x_ctx, src_hf], dim=-1), torch.cat([x_lf, src_lf], dim=-1)], dim=1,
        )
    train_y = torch.cat([y_ctx, y_lf], dim=1).squeeze(-1)
    return train_x, train_y, n_ctx


def _map_penalty(
    model: BatchedIMFBOGP, noise: _SourceNoise, train_x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
    # MAP penalty [B] in neg log density, None on failure
    if noise.noise_model == "parametric":
        return noise.lf_net.penalty()
    x_lf = train_x[:, noise.n_hf:, :-1]
    z_lf = noise.raw_lf[:, noise.n_hf:]
    try:
        return -model.lf_noise_prior.log_prob(x_lf, z_lf)
    except Exception:
        return None


def _mll_loss(
    mll: ExactMarginalLogLikelihood, model: BatchedIMFBOGP, noise: _SourceNoise,
    train_x: torch.Tensor, train_y: torch.Tensor, diag: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
    # one bad env breaks the batched cholesky, escalate jitter
    num_data = train_x.shape[1]
    for jitter in _JITTER_SCHEDULE:
        try:
            with gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
                # _SourceNoise reads source col from train_x
                loss = -mll(model(train_x), train_y, train_x)
            penalty = _map_penalty(model, noise, train_x)
            if penalty is None:
                diag["errors"].append(f"map_penalty@jitter={jitter:g}: failed")
                continue
            # mll divides by num_data, so does penalty
            loss = (loss + penalty / num_data).mean()
        except Exception as exc:
            diag["errors"].append(f"mll@jitter={jitter:g}: {type(exc).__name__}: {exc}")
            continue
        if torch.isfinite(loss):
            diag["max_jitter"] = max(diag["max_jitter"], jitter)
            diag["last_penalty"] = float(penalty.mean().item())
            return loss
        diag["errors"].append(f"mll@jitter={jitter:g}: non-finite loss")
    return None


def _param_groups(
    model: BatchedIMFBOGP, noise: _SourceNoise, lr: float, noise_lr: float,
    ) -> Tuple[List[Dict[str, Any]], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    # clip noise and kernel grads separately, different scales
    noise_ids = set()
    noise_params: List[torch.nn.Parameter] = []
    for module in (noise.lf_net, getattr(model, "lf_noise_prior", None)):
        if module is not None:
            noise_params.extend(module.parameters())
    if noise.raw_lf is not None:
        noise_params.append(noise.raw_lf)
    for p in noise_params:
        noise_ids.add(id(p))
    gp_params = [p for p in model.parameters() if id(p) not in noise_ids]

    groups = [
        {"params": [p for p in gp_params if p.requires_grad], "lr": lr},
        {"params": [p for p in noise_params if p.requires_grad], "lr": noise_lr},
        ]
    return groups, gp_params, noise_params


def _fit_batch(
    train_x: torch.Tensor, train_y: torch.Tensor, n_hf: int, noise_model: str, n_iter: int,
    n_warmup: int, lr: float, noise_lr: float, hidden: int, fit_scale: bool,
    prev_state: Optional[Dict[str, Any]], diag: Dict[str, Any],
    ) -> Tuple[BatchedIMFBOGP, Likelihood, Optional[Dict[str, Any]]]:
    # all B in one adam loop, so no fit_gpytorch_mll
    device, dtype = train_x.device, train_x.dtype
    B, n_train = train_x.shape[0], train_x.shape[1]
    dim_x = train_x.shape[-1] - 1

    # match data dtype, double for cholesky
    source_noise = _SourceNoise(B, n_train, n_hf, dim_x, noise_model, hidden)
    likelihood = SourceGaussianLikelihood(source_noise).to(device=device, dtype=dtype)
    model = BatchedIMFBOGP(train_x, train_y, likelihood).to(device=device, dtype=dtype)
    if noise_model == "gp":
        model.lf_noise_prior = _LogNoiseGPPrior(B, dim_x).to(device=device, dtype=dtype)

    if prev_state is not None:
        try:
            _load_matching_state(model, prev_state["model"])
            _load_matching_state(likelihood, prev_state["likelihood"])
            # raw_lf shape changes per step, seed from prev mean
            if source_noise.raw_lf is not None and "lf_noise_mean" in prev_state:
                with torch.no_grad():
                    source_noise.raw_lf.copy_(
                        prev_state["lf_noise_mean"].to(device=device, dtype=dtype).expand(B, n_train)
                        )
        except Exception as exc:
            diag["errors"].append(f"warm_start: {type(exc).__name__}: {exc}")

    if any(torch.isnan(p).any() for p in model.parameters()):
        diag["errors"].append("warm_start produced NaN parameters; reinitialised")
        source_noise = _SourceNoise(B, n_train, n_hf, dim_x, noise_model, hidden)
        likelihood = SourceGaussianLikelihood(source_noise).to(device=device, dtype=dtype)
        model = BatchedIMFBOGP(train_x, train_y, likelihood).to(device=device, dtype=dtype)
        if noise_model == "gp":
            model.lf_noise_prior = _LogNoiseGPPrior(B, dim_x).to(device=device, dtype=dtype)

    # a, b, mean unidentified with few ctx, bad fit would warm-start every step
    model.raw_scale.requires_grad_(fit_scale)
    diag["scale_frozen"] = not fit_scale

    trained_any = False
    if n_iter > 0:
        model.train()
        likelihood.train()
        groups, gp_params, noise_params = _param_groups(model, source_noise, lr, noise_lr)
        optimizer = torch.optim.Adam(groups)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        for it in range(n_iter):
            # freeze z for warm-up, cold joint start unstable
            if source_noise.raw_lf is not None:
                source_noise.raw_lf.requires_grad_(it >= n_warmup)

            optimizer.zero_grad()
            loss = _mll_loss(mll, model, source_noise, train_x, train_y, diag)
            if loss is None:
                # skip just this step
                diag["steps_skipped"] += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gp_params, max_norm=1.0)
            if noise_params:
                torch.nn.utils.clip_grad_norm_(noise_params, max_norm=1.0)
            optimizer.step()
            diag["steps_taken"] += 1
            trained_any = True

    model.eval()
    likelihood.eval()

    diag["trained"] = trained_any
    _record_hypers(model, source_noise, train_x, diag)

    state = None
    if trained_any and not any(torch.isnan(p).any() for p in model.parameters()):
        state = {"model": model.state_dict(), "likelihood": likelihood.state_dict()}
        if source_noise.raw_lf is not None:
            state["lf_noise_mean"] = source_noise.raw_lf[:, n_hf:].mean(dim=-1, keepdim=True).detach()
    return model, likelihood, state


def _record_hypers(
    model: BatchedIMFBOGP, noise: _SourceNoise, train_x: torch.Tensor, diag: Dict[str, Any],
    ) -> None:
    # failures recover silently, record what collapses quietly
    with torch.no_grad():
        lengthscale = model.covar_module.base_kernel.lengthscale.detach()
        outputscale = model.covar_module.outputscale.detach()
        hf_noise = noise.noise.detach()
        scale = model.scale.detach()
        offset = model.offset.detach()
        lf = noise.lf_noise(train_x[..., :-1]).detach()[:, noise.n_hf:]

    diag["lengthscale"] = lengthscale.mean(dim=0).flatten().tolist()
    # lengthscale at floor = interpolating noise, read with lf_snr
    diag["lengthscale_floor_frac"] = (
        lengthscale <= MIN_LENGTHSCALE * 2.0
        ).to(lengthscale.dtype).mean().item()
    diag["outputscale"] = outputscale.mean().item()
    diag["hf_noise"] = hf_noise.mean().item()
    diag["scale"] = scale.mean().item()
    diag["offset"] = offset.mean().item()
    diag["lf_noise_mean"] = lf.mean().item()
    diag["lf_noise_p05"] = lf.quantile(0.05).item()
    diag["lf_noise_p95"] = lf.quantile(0.95).item()
    diag["lf_noise_floor_frac"] = (lf <= MIN_NOISE * 1.01).to(lf.dtype).mean().item()
    # >~1 informative, <<0.1 on 3-D benches, lf rows drop out
    snr = (scale.squeeze(-1).square() * outputscale) / lf.mean(dim=-1).clamp_min(1e-12)
    diag["lf_snr"] = snr.mean().item()

    if (lengthscale < MIN_LENGTHSCALE).any() or (hf_noise < MIN_NOISE).any() or (scale < MIN_SCALE).any():
        diag["errors"].append(
            f"constraint violated: min lengthscale={lengthscale.min().item():.4g} "
            f"(floor {MIN_LENGTHSCALE:g}), min hf_noise={hf_noise.min().item():.4g} "
            f"(floor {MIN_NOISE:g}), min scale={scale.min().item():.4g} (floor {MIN_SCALE:g})"
            )
    if not torch.isfinite(lf).all():
        diag["errors"].append("non-finite low-fidelity noise")


def _new_diag() -> Dict[str, Any]:
    return {
        "trained": False, "steps_taken": 0, "steps_skipped": 0, "max_jitter": 0.0,
        "posterior_fallback": 0, "n_hf": 0, "n_lf": 0, "scale_frozen": False,
        "lengthscale": None, "lengthscale_floor_frac": None, "outputscale": None,
        "hf_noise": None, "scale": None, "offset": None,
        "lf_noise_mean": None, "lf_noise_p05": None, "lf_noise_p95": None,
        "lf_noise_floor_frac": None, "lf_snr": None, "last_penalty": None,
        "gamma_mean": None, "lf_share": None,
        "errors": [],
        }


def fit_imfbo_gp(
    context_x: torch.Tensor, context_y: torch.Tensor, context_feedback: torch.Tensor,
    pool_x: torch.Tensor, pool_feedback: torch.Tensor, *,
    noise_model: str = "parametric", max_pool: Optional[int] = 192, n_iter: int = 50,
    n_warmup: int = 10, lr: float = 0.1, noise_lr: float = 0.1, noise_hidden: int = 16,
    min_ctx_for_scale: int = 3, prev_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    # all B in one fit, as multitask_gp.fit_multitask_gps
    # returns (fits, state, diag), fits has sigma_hf for nvucb gamma
    if noise_model not in IMFBO_NOISE_MODELS:
        raise NotImplementedError(f"Unknown iMFBO noise model: {noise_model}")

    diag = _new_diag()
    device = context_x.device
    x_ctx = context_x.detach().double()
    y_ctx = context_y.detach().double()
    fb_ctx = context_feedback.detach().double()
    x_pool_all = pool_x.detach().double()
    fb_pool_all = pool_feedback.detach().double()

    all_x = torch.cat([x_ctx, x_pool_all], dim=1)
    lo = all_x.min(dim=1, keepdim=True).values
    hi = all_x.max(dim=1, keepdim=True).values

    try:
        y_ctx_std, y_mean, y_std = _standardize(y_ctx, dim=1)
        # standardize over full pool before subsampling, stable stats, a=1 b=0 init
        fb_all = torch.cat([fb_ctx, fb_pool_all], dim=1)
        fb_all_std, fb_mean, fb_std = _standardize(fb_all, dim=1)
        fb_ctx_std = fb_all_std[:, :x_ctx.shape[1]]
        fb_pool_std = fb_all_std[:, x_ctx.shape[1]:]

        keep = _subsample_batched(all_x.shape[0], x_pool_all.shape[1], max_pool, device)
        x_pool = torch.gather(
            x_pool_all, 1, keep.unsqueeze(-1).expand(-1, -1, x_pool_all.shape[-1]),
            )
        fb_pool = torch.gather(
            fb_pool_std, 1, keep.unsqueeze(-1).expand(-1, -1, fb_pool_std.shape[-1]),
            )

        train_x, train_y, n_hf = _build_imfbo_data(
            _unit_cube(x_ctx, lo, hi), y_ctx_std, fb_ctx_std, _unit_cube(x_pool, lo, hi), fb_pool,
            )
    except Exception as exc:
        diag["errors"].append(f"build: {type(exc).__name__}: {exc}")
        return None, None, diag

    diag["n_hf"] = n_hf
    diag["n_lf"] = train_x.shape[1] - n_hf

    # 1 ctx point -> target 0, nothing to fit
    n_iter_eff = 0 if x_ctx.shape[1] < 2 else n_iter
    if n_iter_eff == 0 and n_iter > 0:
        diag["errors"].append(f"fit skipped: only {x_ctx.shape[1]} context point(s)")

    model, likelihood, state = _fit_batch(
        train_x, train_y, n_hf, noise_model, n_iter_eff, n_warmup, lr, noise_lr, noise_hidden,
        x_ctx.shape[1] >= min_ctx_for_scale, prev_state, diag,
        )

    with torch.no_grad():
        sigma_hf = (likelihood.noise_covar.noise.detach().sqrt() * y_std.squeeze(-1))

    fits = {
        "model": model, "likelihood": likelihood, "noise_model": noise_model,
        "lo": lo, "hi": hi, "y_mean": y_mean, "y_std": y_std, "fb_mean": fb_mean, "fb_std": fb_std,
        "sigma_hf": sigma_hf, "n_hf": n_hf, "n_lf": diag["n_lf"], "diag": diag,
        }
    return fits, state, diag


def imfbo_posterior(
    fits: Optional[Dict[str, Any]], x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # x [B, N, dim_x] -> mean, std [B, N] of latent f, original y units
    # no obs noise, nvucb/nucb add it
    B, N = x.shape[0], x.shape[1]
    mean = torch.zeros(B, N, dtype=x.dtype, device=x.device)
    std = torch.ones(B, N, dtype=x.dtype, device=x.device)
    if fits is None:
        return mean, std
    diag = fits.get("diag")

    x_norm = _unit_cube(x.detach().double(), fits["lo"], fits["hi"])
    src = torch.zeros(B, N, 1, dtype=x_norm.dtype, device=x_norm.device)
    x_hf = torch.cat([x_norm, src], dim=-1)

    # same escalating jitter as training
    m = v = None
    for jitter in _JITTER_SCHEDULE:
        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
                post = fits["model"](x_hf)
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


def imfbo_query_noise(
    fits: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
    # [B, 1] noise of the queried source, not feedback noise
    if fits is None:
        return torch.zeros(1, 1)
    return fits["sigma_hf"]


def imfbo_scores(
    acquisition: str, mean: torch.Tensor, std: torch.Tensor, *,
    sigma_hf: torch.Tensor, kappa: float = 2.0,
    ) -> torch.Tensor:
    # nvucb shrinks bonus, nucb inflates it, both -> ucb as sigma_hf -> 0
    var = std.square()
    hf2 = sigma_hf.to(std).square()
    if acquisition == "ucb":
        return mean + kappa * std
    if acquisition == "nucb":
        return mean + kappa * (var + hf2).clamp_min(1e-12).sqrt()
    if acquisition == "nvucb":
        gamma = std / (var + hf2).clamp_min(1e-12).sqrt()
        return mean + kappa * gamma * std
    raise NotImplementedError(f"Unknown iMFBO acquisition: {acquisition}")


def imfbo_gamma(
    std: torch.Tensor, sigma_hf: torch.Tensor,
    ) -> torch.Tensor:
    # [B, N], ~1 -> nvucb = nucb = ucb
    var = std.square()
    return std / (var + sigma_hf.to(std).square()).clamp_min(1e-12).sqrt()


def imfbo_lf_share(
    fits: Optional[Dict[str, Any]], x: torch.Tensor,
    ) -> torch.Tensor:
    # [B] in [0, 1], how much lf rows move the mean, rerun with huge lf noise
    if fits is None:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    model = fits["model"]
    noise = fits["likelihood"].noise_covar
    mean_full, _ = imfbo_posterior(fits, x)
    noise.lf_noise_override = HUGE_LF_NOISE
    try:
        # drop cached prediction strategy, holds old noise, diag would be 0
        model.prediction_strategy = None
        mean_hf_only, _ = imfbo_posterior(fits, x)
    finally:
        noise.lf_noise_override = None
        model.prediction_strategy = None

    delta = (mean_full - mean_hf_only).abs().mean(dim=-1)
    scale = mean_hf_only.abs().mean(dim=-1).clamp_min(1e-8)
    return (delta / scale).clamp(0.0, 1.0)
