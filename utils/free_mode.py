import math
import torch
import torch.nn.functional as F
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import (
    LogExpectedImprovement, ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement,
    _ei_helper, _log_ei_helper, _scaled_improvement,
    )
from botorch.utils.probability.utils import ndtr as Phi
from linear_operator.utils.errors import NotPSDError
from botorch.exceptions.errors import ModelFittingError
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from attrdictionary import AttrDict
from utils.misc import calculate_gmm_variance

# clamp-mode floor, for ei_clamp_frac
LOG_EI_CLAMP_FLOOR = math.log(1e-30)


def estimate_reference_optimum(
    task: Any, *, n_samples: int = 20000, num_restarts: int = 10, n_polish_steps: int = 100, lr: float = 0.05,
    ) -> float:
    # MC max too low on sharp benches (schwefel) -> negative regret, polish top samples
    device = getattr(task, "device", "cpu")
    total_dim = task.total_dim
    design_scale = task.design_scale
    benchmark_name = task.benchmark_name

    with torch.no_grad():
        probe = task.sample_data(1, n_samples, total_dim=total_dim).to(device)
        y_probe = task.forward(probe, benchmark_name).squeeze(0).squeeze(-1)  # [n_samples]
        k = min(num_restarts, n_samples)
        top_idx = torch.topk(y_probe, k=k).indices
        starts = probe[0, top_idx].clone()  # [k, total_dim]

    x = starts.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(n_polish_steps):
        optimizer.zero_grad()
        y = task.forward(x.unsqueeze(0), benchmark_name).squeeze(0).squeeze(-1)  # [k]
        loss = -y.sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(-design_scale, design_scale)

    with torch.no_grad():
        y_polished = task.forward(x.unsqueeze(0), benchmark_name).squeeze(0).squeeze(-1)
        best = torch.max(y_probe.max(), y_polished.max())
    return best.item()


def weighted_mean_std(
    mixture_means: torch.Tensor, mixture_stds: torch.Tensor, mixture_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = (mixture_means * mixture_weights).sum(dim=-1)
    std = torch.sqrt((mixture_weights * mixture_stds ** 2).sum(dim=-1) + 1e-8)
    return mean, std


def expected_improvement(mean: torch.Tensor, std: torch.Tensor, best_f: torch.Tensor) -> torch.Tensor:
    # botorch _ei_helper, same as log_expected_improvement
    sigma = std + 1e-8
    u = _scaled_improvement(mean, sigma, best_f, maximize=True)
    return sigma * _ei_helper(u)


def log_expected_improvement(
    mean: torch.Tensor, std: torch.Tensor, best_f: torch.Tensor, mode: str = "clamp",
    ) -> torch.Tensor:
    # "clamp": zero grad once EI underflows in fp32
    # "botorch": LogEI (Ament et al. 2023), grad survives much further
    if mode == "botorch":
        sigma = std + 1e-8
        u = _scaled_improvement(mean, sigma, best_f, maximize=True)
        return _log_ei_helper(u) + sigma.log()
    return torch.log(expected_improvement(mean, std, best_f).clamp_min(1e-30))


def probability_of_improvement(mean: torch.Tensor, std: torch.Tensor, best_f: torch.Tensor) -> torch.Tensor:
    sigma = std + 1e-8
    u = _scaled_improvement(mean, sigma, best_f, maximize=True)
    return Phi(u)


def upper_confidence_bound(mean: torch.Tensor, std: torch.Tensor, kappa: float) -> torch.Tensor:
    return mean + kappa * std


AMORTIZED_ACQUISITIONS = (
    "aae", "ei", "log_ei", "pi", "ucb", "uncertainty", "feedback", "thompson",
    )


def amortized_scores(
    acquisition: str, model_out: AttrDict, batch: AttrDict, *,
    best_f: torch.Tensor, kappa: float = 1.0, t: int = 0, T: int = 1,
    log_ei_mode: str = "clamp",
    ) -> torch.Tensor:
    # best_f [B], as evaluate_policy_vs_bo
    if acquisition == "aae":
        return model_out.design_out.logits

    posterior = model_out.posterior_out_query
    mean, std = weighted_mean_std(posterior.mixture_means, posterior.mixture_stds, posterior.mixture_weights)

    if acquisition == "ei":
        return expected_improvement(mean, std, best_f.unsqueeze(-1))
    if acquisition == "log_ei":
        return log_expected_improvement(mean, std, best_f.unsqueeze(-1), mode=log_ei_mode)
    if acquisition == "pi":
        return probability_of_improvement(mean, std, best_f.unsqueeze(-1))
    if acquisition == "ucb":
        return upper_confidence_bound(mean, std, kappa)
    if acquisition == "uncertainty":
        return calculate_gmm_variance(posterior.mixture_means, posterior.mixture_stds, posterior.mixture_weights)
    if acquisition == "feedback":
        return batch.query_feedback[:, :, 0]
    if acquisition == "thompson":
        return mean + std * torch.randn_like(mean)

    raise NotImplementedError(f"Unknown amortized acquisition: {acquisition}")


class BatchedGP(ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: GaussianLikelihood) -> None:
        super().__init__(train_x, train_y, likelihood)
        batch_shape = torch.Size([train_x.shape[0]])
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=train_x.shape[-1]),
            batch_shape=batch_shape,
            )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _normalize_x(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_mean = x.mean(dim=1, keepdim=True)
    x_std = x.std(dim=1, keepdim=True)
    safe = x_std > 1e-2
    x_std_safe = torch.where(safe, x_std, torch.ones_like(x_std))
    x_mean_out = torch.where(safe, x_mean, torch.zeros_like(x_mean))
    return (x - x_mean_out) / x_std_safe, x_mean_out, x_std_safe


def _normalize_y(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_mean = y.mean(dim=1, keepdim=True)
    y_std = y.std(dim=1, keepdim=True)
    fallback = y.abs().mean(dim=1, keepdim=True).clamp(min=1.0)
    y_std = torch.where(y_std < 1e-2, fallback, y_std)
    return (y - y_mean) / y_std, y_mean, y_std


def fit_batched_gp(
    train_x: torch.Tensor, train_y: torch.Tensor, n_iter: int = 30, lr: float = 0.1,
    prev_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BatchedGP, GaussianLikelihood, Optional[Dict[str, Any]], Dict[str, torch.Tensor]]:
    device = train_x.device
    B = train_x.shape[0]

    train_x_norm, x_mean, x_std = _normalize_x(train_x)
    train_y_norm, y_mean, y_std = _normalize_y(train_y)

    likelihood = GaussianLikelihood(batch_shape=torch.Size([B])).to(device)
    model = BatchedGP(train_x_norm, train_y_norm, likelihood).to(device)

    if prev_state is not None:
        try:
            model.load_state_dict(prev_state["model"], strict=False)
            likelihood.load_state_dict(prev_state["likelihood"], strict=False)
        except Exception:
            pass

    if any(torch.isnan(p).any() for p in model.parameters()):
        model = BatchedGP(train_x_norm, train_y_norm, likelihood).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    clean = True
    for _ in range(n_iter):
        optimizer.zero_grad()
        try:
            with gpytorch.settings.cholesky_jitter(1e-2):
                loss = -mll(model(train_x_norm), train_y_norm).mean()
        except Exception:
            clean = False
            break
        if not torch.isfinite(loss):
            clean = False
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    likelihood.eval()

    state = {"model": model.state_dict(), "likelihood": likelihood.state_dict()} if clean else None
    norm_stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, likelihood, state, norm_stats


def gp_posterior(
    model: BatchedGP, likelihood: GaussianLikelihood, x: torch.Tensor, norm_stats: Dict[str, torch.Tensor],
    differentiable: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # differentiable: graph thru x for refine_pool, hypers frozen
    x_norm = (x - norm_stats["x_mean"]) / norm_stats["x_std"]

    if differentiable:
        pred = likelihood(model(x_norm))
        mean_norm = pred.mean
        std_norm = pred.variance.clamp_min(1e-8).sqrt()
    else:
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
            pred = likelihood(model(x_norm))
            mean_norm = pred.mean
            std_norm = pred.variance.clamp_min(1e-8).sqrt()

    mean = mean_norm * norm_stats["y_std"] + norm_stats["y_mean"]
    std = std_norm * norm_stats["y_std"]

    if not differentiable:
        mean = torch.nan_to_num(mean, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0)
    return mean, std


def gp_scores(
    acquisition: str, mean: torch.Tensor, std: torch.Tensor, *,
    best_f: torch.Tensor, kappa: float = 1.0, log_ei_mode: str = "clamp",
    ) -> torch.Tensor:
    if acquisition == "ei":
        return expected_improvement(mean, std, best_f)
    if acquisition == "log_ei":
        return log_expected_improvement(mean, std, best_f, mode=log_ei_mode)
    if acquisition == "pi":
        return probability_of_improvement(mean, std, best_f)
    if acquisition == "ucb":
        return upper_confidence_bound(mean, std, kappa)
    if acquisition == "thompson":
        return mean + std * torch.randn_like(mean)

    raise NotImplementedError(f"Unknown GP acquisition: {acquisition}")


def _make_lbfgs(q: torch.Tensor) -> torch.optim.LBFGS:
    return torch.optim.LBFGS([q], max_iter=1, line_search_fn="strong_wolfe", history_size=10)


def refine_pool(
    x0: torch.Tensor, score_fn: Callable[[torch.Tensor], torch.Tensor], *,
    design_scale: float, n_steps: int = 10, objective: str = "sum",
    refresh_feedback: Optional[Callable[[torch.Tensor], None]] = None,
    x_fixed: Optional[torch.Tensor] = None,
    optimizer_name: str = "lbfgs", lr: float = 0.1, normalize_box: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # score_fn(x_visible) -> [B, N], differentiable, feedback only via caller closure
    # objective: "sum" multistart ascent, "winner" log_softmax at argmax, saturates
    # x_fixed [B, M, dim_x]: ballast, scored not moved, keeps query block near training pool
    # normalize_box: optimize x / design_scale, lr not per-bench
    # returns x, scores over [x0_refined, x_fixed]
    scale = design_scale if normalize_box else 1.0
    bound = 1.0 if normalize_box else design_scale

    q = (x0.detach() / scale).clone().requires_grad_(True)
    n_refine = q.shape[1]

    if optimizer_name == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam([q], lr=lr)
    else:
        optimizer = _make_lbfgs(q)

    def all_x(q_current: torch.Tensor) -> torch.Tensor:
        x_current = q_current * scale
        if x_fixed is None:
            return x_current
        return torch.cat([x_current, x_fixed], dim=1)

    def closure():
        optimizer.zero_grad()
        scores = score_fn(all_x(q))
        if objective == "winner":
            winner = scores.detach().argmax(dim=-1, keepdim=True)
            loss = -F.log_softmax(scores, dim=-1).gather(1, winner).sum()
        else:
            loss = -scores[:, :n_refine].sum()
        if not torch.isfinite(loss):
            # nan loss corrupts lbfgs line search for good, skip
            if q.grad is not None:
                q.grad.zero_()
            return torch.zeros((), device=q.device, dtype=q.dtype)
        loss.backward()
        if q.grad is not None:
            torch.nan_to_num_(q.grad, nan=0.0, posinf=0.0, neginf=0.0)
        return loss

    for _ in range(n_steps):
        optimizer.step(closure)
        with torch.no_grad():
            q.clamp_(-bound, bound)
            torch.nan_to_num_(q, nan=0.0)  # clamp_ keeps nan
        if refresh_feedback is not None:
            refresh_feedback(q.detach() * scale)
            if optimizer_name == "lbfgs":
                # objective moved, curvature history stale
                optimizer = _make_lbfgs(q)

    with torch.no_grad():
        x_all = all_x(q)
        final_scores = score_fn(x_all)
        final_scores = torch.nan_to_num(final_scores, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf"))
    return x_all.detach(), final_scores.detach()


def continuous_bo_step(
    experiment: Any, batch: AttrDict, *,
    model: Optional[Any] = None, acquisition: str = "aae",
    best_f: torch.Tensor, kappa: float = 1.0, t: int = 0, T: int = 1,
    feedback_mode: str = "frozen", concat_feedback: bool = False, n_refine_steps: int = 10,
    expert_feedback_adjustor: Optional[Callable] = None, turn_into_probs: bool = True,
    use_gp: bool = False, gp_state: Optional[Dict[str, Any]] = None, n_iter_gp: int = 30,
    time_token: Optional[torch.Tensor] = None,
    pool_size: int = 2000, n_refine_candidates: int = 100, score_chunk_size: int = 200,
    n_ballast_candidates: int = 0,
    candidate_mode: str = "pool", n_candidates: int = 100, n_background: int = 100,
    refine_optimizer: str = "lbfgs", refine_lr: float = 0.1, adam_step_multiplier: int = 4,
    normalize_box: bool = True, log_ei_mode: str = "clamp",
    ) -> Tuple[AttrDict, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    # one continuous BO step, PFNs4BO style, fresh candidates each step
    # candidate_mode: "pool" refines top n_refine_candidates of pool_size,
    #   "sobol" refines n_candidates sobol points, pool top-k collapses to 1-2 basins anyway
    # n_ballast_candidates / n_background: unoptimized sobol block, see refine_pool x_fixed
    # not experiment.dim_x, concat_feedback callers mutate it
    dim_x = batch.query_x.shape[-1]
    design_scale = experiment.design_scale
    batch_size = batch.query_x.shape[0]
    device = batch.query_x.device
    bi = torch.arange(batch_size, device=device)

    def full_x(x_visible: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] > 0:
            return torch.cat([x_visible, hidden], dim=-1)
        return x_visible

    def apply_adjustor(scores: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        if expert_feedback_adjustor is None:
            return scores
        fb = torch.softmax(feedback, dim=1) if turn_into_probs else feedback
        return expert_feedback_adjustor(scores, fb, n=t)

    gp_state_new = gp_state
    k = min(n_refine_candidates, pool_size)

    # random/feedback always pool, as pool baselines. sobol skips it, 2000 sim calls
    need_pool = candidate_mode == "pool" or acquisition in ("random", "feedback")
    if need_pool:
        x_pool_full = experiment.sample_data(batch_size, pool_size, total_dim=experiment.total_dim)
        x_pool_visible = x_pool_full[..., :dim_x]
        pool_hidden = x_pool_full[..., dim_x:]
        # simulator not net, ok unchunked
        pool_y_real, _, pool_feedback = experiment.observe_at(x_pool_full, time_token=time_token)

    if acquisition == "random":
        # k not 1, query_x must stay non-empty after update_batch
        refined_visible = x_pool_visible[:, :k]
        hidden_top = pool_hidden[:, :k]
        final_scores = torch.rand(batch_size, k, device=device)
        unrefined_best_y, boundary_frac, refine_dacq, ei_clamp_frac = None, float("nan"), float("nan"), float("nan")
    elif acquisition == "feedback":
        # no grad thru feedback, nothing to refine
        refined_visible = x_pool_visible[:, :k]
        hidden_top = pool_hidden[:, :k]
        final_scores = apply_adjustor(pool_feedback[:, :k, 0], pool_feedback[:, :k, 0])
        unrefined_best_y, boundary_frac, refine_dacq, ei_clamp_frac = None, float("nan"), float("nan"), float("nan")
    else:
        m = max(n_background, 0) if candidate_mode == "sobol" else max(n_ballast_candidates, 0)
        if m > 0:
            # fresh, not the pool's low tail
            x_ballast_full = experiment.sample_data_sobol(batch_size, m, total_dim=experiment.total_dim)
            x_ballast = x_ballast_full[..., :dim_x]
            hidden_ballast = x_ballast_full[..., dim_x:]
            y_ballast, _, feedback_ballast = experiment.observe_at(x_ballast_full, time_token=time_token)
        else:
            x_ballast, hidden_ballast, feedback_ballast, y_ballast = None, None, None, None

        feedback_box: Dict[str, torch.Tensor] = {"value": None}
        hidden_refine = None  # set below, before refresh runs

        def refresh(x_visible_detached: torch.Tensor) -> None:
            # only refined rows move, background feedback frozen
            with torch.no_grad():
                _, _, fb = experiment.observe_at(full_x(x_visible_detached, hidden_refine), time_token=time_token)
            feedback_box["value"] = fb if feedback_ballast is None else torch.cat([fb, feedback_ballast], dim=1)

        refresh_feedback = refresh if feedback_mode == "live" else None

        if use_gp:
            gp_model, gp_likelihood, gp_state_new, norm_stats = fit_batched_gp(
                batch.context_x, batch.context_y.squeeze(-1), n_iter=n_iter_gp, prev_state=gp_state,
                )

            def score_fn(x_visible: torch.Tensor) -> torch.Tensor:
                mean, std = gp_posterior(gp_model, gp_likelihood, x_visible, norm_stats, differentiable=True)
                scores = gp_scores(
                    acquisition, mean, std, best_f=best_f.unsqueeze(-1), kappa=kappa, log_ei_mode=log_ei_mode,
                    )
                return apply_adjustor(scores, feedback_box["value"][:, :, 0])

        else:
            # trained with feedback on context/target x too, fixed per step, widen once
            context_x_wide = torch.cat([batch.context_x, batch.context_feedback], dim=-1) if concat_feedback else None
            target_x_wide = torch.cat([batch.target_x, batch.target_feedback], dim=-1) if concat_feedback else None

            def score_fn(x_visible: torch.Tensor) -> torch.Tensor:
                batch_q = AttrDict(batch)
                if concat_feedback:
                    batch_q.query_x = torch.cat([x_visible, feedback_box["value"]], dim=-1)
                    batch_q.context_x = context_x_wide
                    batch_q.target_x = target_x_wide
                else:
                    batch_q.query_x = x_visible
                    batch_q.query_feedback = feedback_box["value"]
                out = model.forward(batch_q)
                scores = amortized_scores(
                    acquisition, out, batch_q, best_f=best_f, kappa=kappa, t=t, T=T, log_ei_mode=log_ei_mode,
                    )
                return apply_adjustor(scores, feedback_box["value"][:, :, 0])

        if candidate_mode == "sobol":
            n_cand = max(n_candidates, 0)
            x_cand_full = experiment.sample_data_sobol(batch_size, n_cand, total_dim=experiment.total_dim)
            x_top = x_cand_full[..., :dim_x]
            hidden_refine = x_cand_full[..., dim_x:]
            y_top, _, feedback_top = experiment.observe_at(x_cand_full, time_token=time_token)
        else:
            # candidates self-attend, shared background makes chunks comparable
            with torch.no_grad():
                score_chunks = []
                for i in range(0, pool_size, score_chunk_size):
                    chunk_x = x_pool_visible[:, i:i + score_chunk_size]
                    n_chunk = chunk_x.shape[1]
                    chunk_feedback = pool_feedback[:, i:i + score_chunk_size]
                    if x_ballast is not None:
                        feedback_box["value"] = torch.cat([chunk_feedback, feedback_ballast], dim=1)
                        chunk_scores = score_fn(torch.cat([chunk_x, x_ballast], dim=1))[:, :n_chunk]
                    else:
                        feedback_box["value"] = chunk_feedback
                        chunk_scores = score_fn(chunk_x)
                    score_chunks.append(chunk_scores)
                pool_scores = torch.cat(score_chunks, dim=1)

            _, top_idx = torch.topk(pool_scores, k=k, dim=1)
            x_top = torch.gather(x_pool_visible, 1, top_idx.unsqueeze(-1).expand(-1, -1, dim_x))
            hidden_refine = torch.gather(pool_hidden, 1, top_idx.unsqueeze(-1).expand(-1, -1, pool_hidden.shape[-1]))
            feedback_top = torch.gather(
                pool_feedback, 1, top_idx.unsqueeze(-1).expand(-1, -1, pool_feedback.shape[-1]),
                )
            y_top = torch.gather(pool_y_real, 1, top_idx.unsqueeze(-1).expand(-1, -1, pool_y_real.shape[-1]))

        # from here on [refined, background], refine_pool order
        if m > 0:
            hidden_top = torch.cat([hidden_refine, hidden_ballast], dim=1)
            feedback_box["value"] = torch.cat([feedback_top, feedback_ballast], dim=1)
            y_for_baseline = torch.cat([y_top, y_ballast], dim=1)
            x_for_init_scores = torch.cat([x_top, x_ballast], dim=1)
        else:
            hidden_top = hidden_refine
            feedback_box["value"] = feedback_top
            y_for_baseline = y_top
            x_for_init_scores = x_top

        # diag only, does refine help true y or just fool the head
        with torch.no_grad():
            init_scores = torch.nan_to_num(
                score_fn(x_for_init_scores), nan=-float("inf"), posinf=float("inf"), neginf=-float("inf"),
                )
        unrefined_best_y = y_for_baseline.squeeze(-1).max(dim=-1).values  # [B]

        # budget parity, strong_wolfe evals several times per step
        effective_refine_steps = n_refine_steps * adam_step_multiplier if refine_optimizer == "adam" else n_refine_steps

        refined_visible, final_scores = refine_pool(
            x_top, score_fn, design_scale=design_scale, n_steps=effective_refine_steps,
            objective="sum", refresh_feedback=refresh_feedback, x_fixed=x_ballast,
            optimizer_name=refine_optimizer, lr=refine_lr, normalize_box=normalize_box,
            )

        boundary_frac = (refined_visible.abs() >= design_scale - 1e-4).float().mean().item()
        refine_dacq = (final_scores.max(dim=-1).values - init_scores.max(dim=-1).values).mean().item()
        ei_clamp_frac = (
            (init_scores <= LOG_EI_CLAMP_FLOOR + 1e-3).float().mean().item() if acquisition == "log_ei" else float("nan")
            )

    winner = final_scores.argmax(dim=-1, keepdim=True)  # [B, 1]
    x_full_refined = full_x(refined_visible, hidden_top)
    expected_width = experiment.total_dim
    if x_full_refined.shape[-1] != expected_width:
        raise RuntimeError(
            f"observe_at about to receive width {x_full_refined.shape[-1]}, expected "
            f"experiment.total_dim={expected_width}. Diagnostics: concat_feedback={concat_feedback}, "
            f"dim_x(local)={dim_x}, experiment.dim_x={experiment.dim_x}, "
            f"refined_visible.shape={tuple(refined_visible.shape)}, hidden_top.shape={tuple(hidden_top.shape)}, "
            f"batch.query_x.shape={tuple(batch.query_x.shape)}, "
            f"batch.query_feedback.shape={tuple(batch.query_feedback.shape)}"
            )
    y_real, y_hidden, feedback_real = experiment.observe_at(x_full_refined, time_token=time_token)

    batch.query_x = refined_visible
    batch.query_y = y_real
    batch.query_feedback = feedback_real
    batch.y_hidden_query = y_hidden

    batch = experiment.update_batch(batch, winner)

    next_y = y_real[bi, winner.squeeze(-1)]
    best_f_new = torch.max(best_f, next_y.squeeze(-1))

    refine_dy = (next_y.squeeze(-1) - unrefined_best_y).mean().item() if unrefined_best_y is not None else float("nan")
    diagnostics = {
        "refine_dy": refine_dy, "refine_dacq": refine_dacq,
        "boundary_frac": boundary_frac, "ei_clamp_frac": ei_clamp_frac,
        }

    info = {"gp_state": gp_state_new, "final_scores": final_scores, "diagnostics": diagnostics}
    return batch, best_f_new, next_y, info


def _nanmean(values: List[float]) -> float:
    finite = [v for v in values if v == v]  # v == v is False only for nan
    return sum(finite) / len(finite) if finite else float("nan")


def evaluate_continuous(
    eval_set: AttrDict, experiment: Any, T: int, *,
    model: Optional[Any] = None, acquisition: str = "aae",
    n_theta: int = 0, concat_feedback: bool = False, feedback_mode: str = "frozen",
    n_refine_steps: int = 10, kappa: float = 1.0, use_gp: bool = False, n_iter_gp: int = 30,
    expert_feedback_adjustor: Optional[Callable] = None, turn_into_probs: bool = True,
    reference_optimum: Optional[Union[torch.Tensor, float]] = None, reference_n_samples: int = 20000,
    always_T_is_1: bool = False,
    pool_size: int = 2000, n_refine_candidates: int = 100, score_chunk_size: int = 200,
    n_ballast_candidates: int = 0,
    candidate_mode: str = "pool", n_candidates: int = 100, n_background: int = 100,
    refine_optimizer: str = "lbfgs", refine_lr: float = 0.1, adam_step_multiplier: int = 4,
    normalize_box: bool = True, log_ei_mode: str = "clamp",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    # continuous evaluate_policy_vs_bo, same 6 metrics
    if model is not None:
        model.eval()

    batch = AttrDict({k: (v.clone() if torch.is_tensor(v) else v) for k, v in eval_set.items()})
    device = batch.context_x.device
    batch_size = batch.context_x.shape[0]

    # as evaluate_policy_vs_bo
    batch.target_all = torch.cat(
        [batch.target_all[:, :experiment.n_target_data + n_theta - 1], batch.target_all[:, -1:]], dim=1,
        )

    if reference_optimum is None:
        probe = experiment.sample_data(batch_size, reference_n_samples, total_dim=experiment.total_dim)
        reference_optimum = experiment.forward(probe, experiment.benchmark_name).max(dim=1).values.squeeze(-1)
    elif torch.is_tensor(reference_optimum):
        reference_optimum = reference_optimum.to(device)

    selected_values = torch.zeros(T + 1, batch_size, device=device)
    best_so_far = torch.zeros(T + 1, batch_size, device=device)
    rmse_values = torch.zeros(T, batch_size, device=device)
    regret_values = torch.zeros(T, batch_size, device=device)
    cumulative_regret_values = torch.zeros(T, batch_size, device=device)
    regret_fraction_within_5_percent = torch.zeros(T, batch_size, device=device)

    best_f = batch.context_y.max(dim=1).values.squeeze(-1)
    selected_values[0] = best_f
    best_so_far[0] = best_f

    gp_state = None
    agreements_hard = []
    diagnostics_history: List[Dict[str, float]] = []

    for t in range(T):
        time_token = torch.tensor([1.0] if always_T_is_1 else [t / T], device=device)

        batch, best_f, next_y, info = continuous_bo_step(
            experiment, batch,
            model=model, acquisition=acquisition,
            best_f=best_f, kappa=kappa, t=t, T=T,
            feedback_mode=feedback_mode, concat_feedback=concat_feedback, n_refine_steps=n_refine_steps,
            expert_feedback_adjustor=expert_feedback_adjustor, turn_into_probs=turn_into_probs,
            use_gp=use_gp, gp_state=gp_state, n_iter_gp=n_iter_gp, time_token=time_token,
            pool_size=pool_size, n_refine_candidates=n_refine_candidates, score_chunk_size=score_chunk_size,
            n_ballast_candidates=n_ballast_candidates,
            candidate_mode=candidate_mode, n_candidates=n_candidates, n_background=n_background,
            refine_optimizer=refine_optimizer, refine_lr=refine_lr, adam_step_multiplier=adam_step_multiplier,
            normalize_box=normalize_box, log_ei_mode=log_ei_mode,
            )
        gp_state = info["gp_state"]
        diagnostics_history.append(info["diagnostics"])

        selected_values[t + 1] = next_y.squeeze(-1)
        best_so_far[t + 1] = best_f

        if not use_gp and model is not None:
            with torch.no_grad():
                diag_batch = batch
                if concat_feedback:
                    diag_batch = AttrDict(batch)
                    diag_batch.query_x = torch.cat([batch.query_x, batch.query_feedback], dim=-1)
                    diag_batch.context_x = torch.cat([batch.context_x, batch.context_feedback], dim=-1)
                    diag_batch.target_x = torch.cat([batch.target_x, batch.target_feedback], dim=-1)
                out = model.forward(diag_batch)
                target_gt = batch.target_all[:, :experiment.n_target_data].squeeze(-1)
                pred_mean, _ = weighted_mean_std(
                    out.posterior_out.mixture_means, out.posterior_out.mixture_stds, out.posterior_out.mixture_weights,
                    )
                rmse_values[t] = torch.sqrt(((pred_mean[:, :experiment.n_target_data] - target_gt) ** 2).mean(dim=-1))

                if acquisition == "aae":
                    agreements_hard.append(
                        (out.design_out.zt.argmax(dim=-1) == batch.query_feedback[..., 0].argmax(dim=-1)).float()
                        )

        regret_values[t] = reference_optimum - next_y.squeeze(-1)
        regret_best = reference_optimum - best_f
        regret_fraction_within_5_percent[t] = (regret_best <= 0.05 * reference_optimum).float()
        prev_cum = cumulative_regret_values[t - 1] if t > 0 else torch.zeros_like(regret_best)
        cumulative_regret_values[t] = prev_cum + regret_best

    stats = {
        "reference_optimum": (
            reference_optimum.detach().cpu() if torch.is_tensor(reference_optimum) else reference_optimum
            ),
        "model_policy_argmax_vs_expert": (
            [{"mean": v.mean().item()} for v in agreements_hard] if agreements_hard else []
            ),
        # nanmean, random/feedback never refine, all nan
        "diagnostics": {
            key: _nanmean([d[key] for d in diagnostics_history])
            for key in ("refine_dy", "refine_dacq", "boundary_frac", "ei_clamp_frac")
            },
        }

    return (
        selected_values.detach().cpu(),
        best_so_far.detach().cpu(),
        rmse_values.detach().cpu(),
        regret_values.detach().cpu(),
        cumulative_regret_values.detach().cpu(),
        regret_fraction_within_5_percent.detach().cpu(),
        stats,
        )


_BOTORCH_ANALYTIC_ACQF = {
    "ei": ExpectedImprovement,
    "log_ei": LogExpectedImprovement,
    "pi": ProbabilityOfImprovement,
    }


class _FeedbackAdjustedAcqf(AcquisitionFunction):
    # adjustor inside optimize_acqf search, feedback under no_grad
    def __init__(
        self, base_acqf: AcquisitionFunction, experiment: Any, adjustor: Callable, t: int, *,
        dim_x: int, total_dim: int, env_idx: torch.Tensor,
        turn_into_probs: bool = True, feedback_mode: str = "live",
        ) -> None:
        super().__init__(model=base_acqf.model)
        self.base_acqf = base_acqf
        self.experiment = experiment
        self.adjustor = adjustor
        self.t = t
        self.dim_x = dim_x
        self.total_dim = total_dim
        self.env_idx = env_idx
        self.turn_into_probs = turn_into_probs
        self.feedback_mode = feedback_mode
        self._cached_feedback: Optional[torch.Tensor] = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        scores = self.base_acqf(X)  # differentiable; shape = X's batch dims (q=1 squeezed off)
        # frozen: reuse per shape, raw_samples vs restarts differ
        if (
            self.feedback_mode == "frozen"
            and self._cached_feedback is not None
            and self._cached_feedback.shape == scores.shape
            ):
            feedback = self._cached_feedback
        else:
            with torch.no_grad():
                x_visible = X.detach().squeeze(-2)  # [..., dim_x], q=1
                if self.total_dim > self.dim_x:
                    n = x_visible.shape[0]
                    fresh = self.experiment.sample_data(1, n, total_dim=self.total_dim).squeeze(0)
                    x_full = torch.cat([x_visible, fresh[:, self.dim_x:]], dim=-1)
                else:
                    x_full = x_visible
                _, _, fb = self.experiment.observe_at(x_full.unsqueeze(0), env_idx=self.env_idx)
                feedback = fb[0, :, 0]
            if self.feedback_mode == "frozen":
                self._cached_feedback = feedback

        fb_in = torch.softmax(feedback, dim=0) if self.turn_into_probs else feedback
        return self.adjustor(scores.unsqueeze(0), fb_in.unsqueeze(0), n=self.t).squeeze(0)


def evaluate_botorch_gp_continuous(
    eval_set: AttrDict, experiment: Any, T: int, *,
    acquisition: str = "log_ei", kappa: float = 1.0, num_restarts: int = 5, raw_samples: int = 64,
    expert_feedback_adjustor: Optional[Callable] = None, turn_into_probs: bool = True,
    feedback_mode: str = "live", feedback_integration: str = "gradient",
    reference_optimum: Optional[Union[torch.Tensor, float]] = None, reference_n_samples: int = 20000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    # feedback_integration: "gradient" adjustor inside optimize_acqf,
    #   "posthoc_topk" legacy, adjustor picks among restart optima
    if feedback_integration not in ("gradient", "posthoc_topk"):
        raise ValueError(f"Unknown feedback_integration: {feedback_integration}")
    batch = AttrDict({k: (v.clone() if torch.is_tensor(v) else v) for k, v in eval_set.items()})
    device = batch.context_x.device
    dtype = batch.context_x.dtype
    B = batch.context_x.shape[0]
    dim_x = batch.context_x.shape[-1]
    total_dim = experiment.total_dim
    design_scale = experiment.design_scale

    if reference_optimum is None:
        probe = experiment.sample_data(B, reference_n_samples, total_dim=total_dim)
        reference_optimum = experiment.forward(probe, experiment.benchmark_name).max(dim=1).values.squeeze(-1)
    elif torch.is_tensor(reference_optimum):
        reference_optimum = reference_optimum.to(device)

    selected_values = torch.zeros(T + 1, B, device=device)
    best_so_far = torch.zeros(T + 1, B, device=device)
    rmse_values = torch.zeros(T, B, device=device)
    regret_values = torch.zeros(T, B, device=device)
    cumulative_regret_values = torch.zeros(T, B, device=device)
    regret_fraction_within_5_percent = torch.zeros(T, B, device=device)

    best_f = batch.context_y.max(dim=1).values.squeeze(-1)
    selected_values[0] = best_f
    best_so_far[0] = best_f

    bounds = torch.stack(
        [torch.full((dim_x,), -design_scale, device=device, dtype=dtype),
         torch.full((dim_x,), design_scale, device=device, dtype=dtype)],
        )

    for t in range(T):
        candidates = torch.zeros(B, dim_x, device=device, dtype=dtype)
        for b in range(B):
            gp = SingleTaskGP(
                batch.context_x[b].detach(), batch.context_y[b].detach(),
                input_transform=Normalize(d=dim_x, bounds=bounds),
                outcome_transform=Standardize(m=1),
                )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                cx, cy = batch.context_x[b], batch.context_y[b]
                print(
                    f"[DEBUG evaluate_botorch_gp_continuous] fit failed at t={t} b={b}: {type(e).__name__}: {e}\n"
                    f"  context_x: shape={tuple(cx.shape)} dtype={cx.dtype} "
                    f"has_nan={torch.isnan(cx).any().item()} has_inf={torch.isinf(cx).any().item()} "
                    f"per_dim_std={cx.std(dim=0).tolist()}\n"
                    f"  context_y: shape={tuple(cy.shape)} dtype={cy.dtype} "
                    f"has_nan={torch.isnan(cy).any().item()} has_inf={torch.isinf(cy).any().item()} "
                    f"min={cy.min().item()} max={cy.max().item()} std={cy.std().item()}\n"
                    f"  n_unique_rows={cx.unique(dim=0).shape[0]} of {cx.shape[0]}"
                    )
                raise

            best_f_b = best_f[b].item()
            if acquisition == "ucb":
                acqf = UpperConfidenceBound(gp, beta=kappa ** 2)
            elif acquisition in _BOTORCH_ANALYTIC_ACQF:
                acqf = _BOTORCH_ANALYTIC_ACQF[acquisition](gp, best_f=best_f_b)
            else:
                raise NotImplementedError(f"Unknown botorch acquisition: {acquisition}")

            if expert_feedback_adjustor is None:
                candidate, _ = optimize_acqf(
                    acqf, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
                    )
                candidates[b] = candidate.squeeze(0)

            elif feedback_integration == "gradient":
                wrapped_acqf = _FeedbackAdjustedAcqf(
                    acqf, experiment, expert_feedback_adjustor, t,
                    dim_x=dim_x, total_dim=total_dim, env_idx=torch.tensor([b], device=device),
                    turn_into_probs=turn_into_probs, feedback_mode=feedback_mode,
                    )
                candidate, _ = optimize_acqf(
                    wrapped_acqf, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
                    )
                candidates[b] = candidate.squeeze(0)

            else:  # feedback_integration == "posthoc_topk"
                # restarts stand in for a pool
                cands_b, vals_b = optimize_acqf(
                    acqf, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
                    return_best_only=False,
                    )
                cands_flat = cands_b.squeeze(1)  # [num_restarts, dim_x]
                if total_dim > dim_x:
                    fresh = experiment.sample_data(1, num_restarts, total_dim=total_dim).squeeze(0)
                    x_full_b = torch.cat([cands_flat, fresh[:, dim_x:]], dim=-1).unsqueeze(0)
                else:
                    x_full_b = cands_flat.unsqueeze(0)
                _, _, feedback_b = experiment.observe_at(
                    x_full_b, env_idx=torch.tensor([b], device=device),
                    )  # [1, num_restarts, dim_feedback]
                fb = feedback_b[0, :, 0].unsqueeze(0)  # [1, num_restarts]
                fb_in = torch.softmax(fb, dim=1) if turn_into_probs else fb
                adjusted = expert_feedback_adjustor(vals_b.unsqueeze(0), fb_in, n=t).squeeze(0)
                candidates[b] = cands_flat[adjusted.argmax()]

        # botorch blind to hidden dims, random draw
        if total_dim > dim_x:
            fresh = experiment.sample_data(B, 1, total_dim=total_dim).squeeze(1)
            x_full = torch.cat([candidates, fresh[:, dim_x:]], dim=-1).unsqueeze(1)
        else:
            x_full = candidates.unsqueeze(1)

        y_real, _, _ = experiment.observe_at(x_full)
        next_y = y_real.squeeze(1).squeeze(-1)

        selected_values[t + 1] = next_y
        best_f = torch.max(best_f, next_y)
        best_so_far[t + 1] = best_f

        batch.context_x = torch.cat([batch.context_x, x_full[..., :dim_x]], dim=1)
        batch.context_y = torch.cat([batch.context_y, y_real], dim=1)

        regret_values[t] = reference_optimum - next_y
        regret_best = reference_optimum - best_f
        regret_fraction_within_5_percent[t] = (regret_best <= 0.05 * reference_optimum).float()
        prev_cum = cumulative_regret_values[t - 1] if t > 0 else torch.zeros_like(regret_best)
        cumulative_regret_values[t] = prev_cum + regret_best

    return (
        selected_values.detach().cpu(),
        best_so_far.detach().cpu(),
        rmse_values.detach().cpu(),
        regret_values.detach().cpu(),
        cumulative_regret_values.detach().cpu(),
        regret_fraction_within_5_percent.detach().cpu(),
        {"reference_optimum": reference_optimum.detach().cpu() if torch.is_tensor(reference_optimum) else reference_optimum},
        )


def evaluate_gp_bo_botorch(
    eval_set: AttrDict, T: int, n_target_data: int, *,
    acquisition: str = "ei", kappa: float = 2.0,
    expert_feedback_adjustor: Optional[Callable] = None, turn_into_probs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    # pool-argmax version of evaluate_botorch_gp_continuous
    batch = AttrDict({k: (v.clone() if torch.is_tensor(v) else v) for k, v in eval_set.items()})
    device = batch.context_x.device
    dim_x = batch.context_x.shape[-1]
    B = batch.context_x.shape[0]

    selected_values = torch.zeros(T + 1, B, device=device)
    best_so_far = torch.zeros(T + 1, B, device=device)
    rmse_values = torch.zeros(T, B, device=device)
    regret_values = torch.zeros(T, B, device=device)
    cumulative_regret_values = torch.zeros(T, B, device=device)
    regret_fraction_within_5_percent = torch.zeros(T, B, device=device)

    best_y = batch.context_y.max(dim=1).values  # [B, 1]
    true_max = torch.cat([batch.context_y, batch.query_y], dim=1).max(dim=1).values.squeeze(-1)
    selected_values[0] = best_y.squeeze(-1)
    best_so_far[0] = best_y.squeeze(-1)

    for t in range(T):
        # BatchedGP kernel, compare fitting only, not prior
        batch_shape = torch.Size([B])
        covar_module = ScaleKernel(
            MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=dim_x), batch_shape=batch_shape,
            )
        likelihood = GaussianLikelihood(batch_shape=batch_shape)
        gp = SingleTaskGP(
            batch.context_x, batch.context_y,
            covar_module=covar_module, likelihood=likelihood,
            input_transform=Normalize(d=dim_x, batch_shape=batch_shape),
            outcome_transform=Standardize(m=1, batch_shape=batch_shape),
            )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except (NotPSDError, ModelFittingError):
            # no priors to resample, more jitter, else untrained hypers
            try:
                with gpytorch.settings.cholesky_jitter(1e-1):
                    fit_gpytorch_mll(mll)
            except (NotPSDError, ModelFittingError):
                gp.eval()

        with torch.no_grad():
            qry_post = gp.posterior(batch.query_x)
            mean = qry_post.mean.squeeze(-1)
            std = qry_post.variance.clamp_min(1e-8).sqrt().squeeze(-1)

            if acquisition == "ei":
                scores = expected_improvement(mean, std, best_y)
            elif acquisition == "log_ei":
                scores = log_expected_improvement(mean, std, best_y)
            elif acquisition == "pi":
                scores = probability_of_improvement(mean, std, best_y)
            elif acquisition == "ucb":
                scores = upper_confidence_bound(mean, std, kappa)
            elif acquisition == "random":
                scores = torch.rand_like(mean)
            else:
                raise NotImplementedError(f"Unknown acquisition: {acquisition}")

            if expert_feedback_adjustor is not None:
                feedback = batch.query_feedback[:, :, 0]
                fb_in = torch.softmax(feedback, dim=1) if turn_into_probs else feedback
                scores = expert_feedback_adjustor(scores, fb_in, n=t)

            idx = scores.argmax(dim=1, keepdim=True)
            mean_t = gp.posterior(batch.target_x[:, :n_target_data]).mean.squeeze(-1)

        batch_idx = torch.arange(B, device=device)
        idx_sq = idx.squeeze(-1)

        next_y = batch.query_y.squeeze(-1)[batch_idx, idx_sq].unsqueeze(-1)
        best_y = torch.max(best_y, next_y)

        selected_values[t + 1] = next_y.squeeze(-1)
        best_so_far[t + 1] = best_y.squeeze(-1)

        target_gt = batch.target_all[:, :n_target_data].squeeze(-1)
        rmse_values[t] = torch.sqrt(((mean_t - target_gt) ** 2).mean(dim=-1))

        regret_values[t] = true_max - next_y.squeeze(-1)
        regret_best = true_max - best_y.squeeze(-1)
        prev_cum = cumulative_regret_values[t - 1] if t > 0 else torch.zeros_like(regret_best)
        cumulative_regret_values[t] = prev_cum + regret_best
        regret_fraction_within_5_percent[t] = (regret_best <= 0.05 * true_max).float()

        mask = torch.ones(B, batch.query_x.shape[1], dtype=torch.bool, device=device)
        mask[batch_idx, idx_sq] = False
        next_x = batch.query_x[batch_idx, idx_sq].unsqueeze(1)
        next_feedback = batch.query_feedback[batch_idx, idx_sq].unsqueeze(1)

        batch.query_x = batch.query_x[mask].view(B, -1, dim_x)
        batch.query_y = batch.query_y[mask].view(B, -1, 1)
        batch.query_feedback = batch.query_feedback[mask].view(B, -1, batch.query_feedback.shape[-1])

        batch.context_x = torch.cat([batch.context_x, next_x], dim=1)
        batch.context_y = torch.cat([batch.context_y, next_y.unsqueeze(-1)], dim=1)
        batch.context_feedback = torch.cat([batch.context_feedback, next_feedback], dim=1)

    return (
        selected_values.detach().cpu(),
        best_so_far.detach().cpu(),
        rmse_values.detach().cpu(),
        regret_values.detach().cpu(),
        cumulative_regret_values.detach().cpu(),
        regret_fraction_within_5_percent.detach().cpu(),
        {},
        )

