import copy
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import gpytorch
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdictionary import AttrDict
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from hydra import compose, initialize_config_dir
from linear_operator.operators import DiagLinearOperator
from linear_operator.utils.errors import NotPSDError
from torch.distributions import Normal
from tqdm import tqdm

from model import FICBO
from utils import create_target_mask
from utils.misc import calculate_gmm_variance

# copied from notebooks/eval_with_expert.ipynb for batch runs, 
# but may be a tiny bit different after code changes

def seed_everything(seed: int = 42):
    import os
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    print(f"Seed set to {seed}")


def _batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def load_config_and_model(path, config_name="config.yaml", file_name="aae.pth"):
    """
    Load hydra cfg and FICBO checkpoint from a run dir.

    Returns: (cfg, model)
    """
    # not initialize(), resolves relative to caller file
    config_path = os.path.abspath(os.path.join(path, ".hydra"))
    print(config_path)
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name=config_name)

        embedder = hydra.utils.instantiate(cfg.embedder)
        encoder = hydra.utils.instantiate(cfg.encoder)
        head = hydra.utils.instantiate(cfg.head)
        #model = BaseTransformer(embedder, encoder, head)
        model = FICBO(embedder, encoder, head)

        # file_path = os.path.join(path, 'model', file_name)
        # model.load_state_dict(torch.load(file_path, map_location=torch.get_default_device(), weights_only=True))

        ckpt = torch.load(os.path.join(path, file_name), map_location=torch.get_default_device(), weights_only=False)

        try:
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint using ckpt['model'] from {file_name}")
        except KeyError:
            model.load_state_dict(ckpt)
            print(f"Loaded raw state dict from {file_name}")

    return cfg, model


@dataclass
class MethodConfig:
    """
    Per-method args for _run_method.

    name: key in results dict
    acquisition: "aae", "ucb", "ei", "log_ei", "pi", "random", "feedback", ...
    n_theta: None -> cfg.task.n_target_theta
    concat_feedback: append expert feedback to x
    experiment_dim_x_delta: temp offset on experiment.dim_x, +cfg.task.dim_feedback for concat_feedback
    use_gp: evaluate_gp_bo, hand-rolled BatchedGP
    gp_native: free_mode.evaluate_gp_bo_botorch, SingleTaskGP + fit_gpytorch_mll, ignores use_gp
    feedback_integration: botorch_continuous only. "gradient": adjustor inside optimize_acqf.
        "posthoc_topk": legacy, adjustor only reranks the restart optima
    advanced_stats: keep per-step prediction tensors, slow
    n_iter_gp: GP fit iters
    continuous: gradient-refine candidates (utils/free_mode.py) instead of pool argmax
    feedback_mode: "frozen" or "live" (re-query feedback each refine iter)
    use_multitask_gp: feedback as 2nd GP task, evaluate_multitask_gp_bo
    mtgp_kind: "icm" (feedback on full pool) or "kronecker" (same ICM, context-only data,
        not botorch KroneckerMultiTaskGP)
    use_rgpe: feedback as RGPE base task, evaluate_rgpe_bo
    rgpe_*: passed to evaluate_rgpe_bo, None -> its defaults
    use_imfbo: feedback as noisy obs of f, learned sigma_lf(x), evaluate_imfbo_bo
    imfbo_noise_model: "parametric" (MLP) or "gp" (per-row log-noise, paper eq. 12)
    imfbo_max_pool: feedback rows per step, Cholesky cubic in it. None -> default

    continuous only, None -> free_mode default in ():
    reference_optimum: shared regret baseline, e.g. pool true_max
    pool_size: fresh pool per step (2000)
    n_refine_candidates: top-k refined (100)
    score_chunk_size: scoring chunk (200 = n_query_init, scores not chunk-invariant)
    n_ballast_candidates: unoptimized sobol points in the query set (0). stops query block
        collapsing to high-acq points, aim refine + ballast ~ n_query_init
    candidate_mode: "pool" (refine top-k of pool) or "sobol" (refine n_candidates, n_background fixed)
    n_candidates, n_background: sobol block sizes (100 each)
    refine_optimizer: "lbfgs" or "adam" (adam_step_multiplier x steps, budget-matched)
    refine_lr: step size in [-1, 1] box coords (0.1)
    adam_step_multiplier: (4)
    log_ei_mode: "clamp" (zero grad on underflow) or "botorch" (LogEI formula). log_ei only
    """
    name: str
    model: Any
    cfg: Any
    acquisition: str = "aae"
    attend_to: str = "data"
    n_theta: Optional[int] = None
    concat_feedback: bool = False
    experiment_dim_x_delta: int = 0
    use_gp: bool = False
    gp_native: bool = False
    feedback_integration: str = "gradient"
    gp_model: Any = None
    expert_feedback_adjustor: Optional[Callable] = None
    advanced_stats: bool = False
    n_iter_gp: int = 100
    continuous: bool = False
    feedback_mode: str = "frozen"
    reference_optimum: Any = None
    pool_size: Optional[int] = None
    n_refine_candidates: Optional[int] = None
    score_chunk_size: Optional[int] = None
    n_ballast_candidates: Optional[int] = None
    candidate_mode: Optional[str] = None
    n_candidates: Optional[int] = None
    n_background: Optional[int] = None
    refine_optimizer: Optional[str] = None
    refine_lr: Optional[float] = None
    adam_step_multiplier: Optional[int] = None
    log_ei_mode: Optional[str] = None
    kappa: float = 1.0
    use_multitask_gp: bool = False
    mtgp_kind: str = "icm"
    use_rgpe: bool = False
    rgpe_max_pool: Optional[int] = None
    rgpe_num_posterior_samples: Optional[int] = None
    rgpe_target_noise_var: Optional[float] = None
    rgpe_feedback_noise_var: Optional[float] = None
    use_imfbo: bool = False
    imfbo_noise_model: str = "parametric"
    imfbo_max_pool: Optional[int] = None
 
    def effective_n_theta(self) -> int:
        return self.n_theta if self.n_theta is not None else self.cfg.task.n_target_theta
 
    def vary(self, **overrides) -> "MethodConfig":
        """Shallow copy with overrides, e.g. base.vary(name="ucb", acquisition="ucb")."""
        mc = copy.copy(self)
        for k, v in overrides.items():
            setattr(mc, k, v)
        return mc
 
 
def baseline_configs(
    template: MethodConfig,
    acquisitions: List[str] = ("random", "ucb", "log_ei", "pi"),
    name_prefix: str = "",
) -> List[MethodConfig]:
    """
    Classical BO baselines from a template, only acquisition and name vary.

        *baseline_configs(template, acquisitions=("random", "ucb", "log_ei", "feedback"))
    """
    return [
        template.vary(name=f"{name_prefix}{acq}", acquisition=acq)
        for acq in acquisitions
    ]


def get_normalized_best_so_far(results: dict, optima: dict) -> dict:
    """
    Best-so-far scaled to [0, 1], 0 = value at t=0, 1 = optimum.

    Returns: dict[method] -> [n_benchmarks, T+1, B]
    """
    benchmark_names = list(results.keys())
    metrics_data_list = aggregate_results(results, metric="best_so_far")
    y_opt = torch.stack([optima[name] for name in benchmark_names], dim=0)  # [n_benchmarks, batch_size]

    normalized = {}
    for method, data in metrics_data_list.items():
        y_init = data[:, 0, :].unsqueeze(1)  # [n_benchmarks, 1, batch_size]
        y_opt_exp = y_opt.unsqueeze(1)  # [n_benchmarks, 1, batch_size]
        normalized[method] = (data - y_init) / (y_opt_exp - y_init + 1e-8)

    return normalized


def get_normalized_cum_regret(results: dict, optima: dict) -> dict:
    norm_bsf = get_normalized_best_so_far(results, optima)
    normalized = {}
    for method, data in norm_bsf.items():
        simple_regret = 1 - data[:, 1:, :]  # [n_benchmarks, T, batch_size]
        normalized[method] = simple_regret.cumsum(dim=1)
    return normalized


def concat_feedback_to_batch_x(batch):
    """In place: {context,query,target}_x <- [x, feedback]. feedback [B, N, D_fb]"""
    if hasattr(batch, "context_feedback") and batch.context_feedback is not None:
        batch.context_x = torch.cat([batch.context_x, batch.context_feedback], dim=-1)

    if hasattr(batch, "query_feedback") and batch.query_feedback is not None:
        batch.query_x = torch.cat([batch.query_x, batch.query_feedback], dim=-1)

    if hasattr(batch, "target_feedback") and batch.target_feedback is not None:
        batch.target_x = torch.cat([batch.target_x, batch.target_feedback], dim=-1)
    batch.dim_x += batch.context_feedback.shape[-1]
    return batch


def print_batch_stats(batch, name="batch"):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    def stats(label, t):
        if t is None or t.numel() == 0:
            print(f"  {label:30s} : empty/None")
            return
        qs = torch.quantile(t.float().flatten(), torch.tensor([0.25, 0.5, 0.75, 0.95], device=t.device))

        print(f"  {label:30s} : shape={str(tuple(t.shape)):20s}  "
              f"mean={t.mean().item():7.3f}  std={t.std().item():7.3f}  "
              f"mode={t.flatten().mode().values.item():7.3f} \n"
              f"\t\t\t\t min={t.min().item():7.3f}  q25={qs[0].item():7.3f}  q50={qs[1].item():7.3f}  "
              f"q75={qs[2].item():7.3f}  q95={qs[3].item():7.3f}  max={t.max().item():7.3f}")

    stats("query_y", batch.query_y)
    stats("context_y", batch.context_y)
    stats("target_y", batch.target_y)

    print()
    query_maxes = batch.query_y.squeeze(-1).max(dim=1).values
    print(f"  {'query_y max (per batch)':30s} : "
          f"mean={query_maxes.mean().item():7.3f}  std={query_maxes.std().item():7.3f}  "
          f"min={query_maxes.min().item():7.3f}  max={query_maxes.max().item():7.3f}")
    query_means = batch.query_y.squeeze(-1).mean(dim=1)
    print(f"  {'query_y mean (per batch)':30s} : "
          f"mean={query_means.mean().item():7.3f}  std={query_means.std().item():7.3f}  "
          f"min={query_means.min().item():7.3f}  max={query_means.max().item():7.3f}")

    print()
    if getattr(batch, 'y_hidden_query', None) is not None:
        stats("y_hidden_query", batch.y_hidden_query)
        stats("y_hidden_context", batch.y_hidden_context)
        stats("y_hidden_target", batch.y_hidden_target)

    print()
    fb = getattr(batch, 'query_feedback', None)
    if fb is not None:
        stats("query_feedback", batch.query_feedback)
        stats("context_feedback", batch.context_feedback)
        stats("target_feedback", batch.target_feedback)

        # rank alignment, query_y vs first feedback dim
        y = batch.query_y.squeeze(-1)  # [B, N]
        f = fb[..., 0]  # [B, N]
        N = y.shape[1]

        # double argsort -> rank, 0 = best
        y_ranks = torch.argsort(torch.argsort(y, dim=1, descending=True), dim=1)  # [B, N]
        f_ranks = torch.argsort(torch.argsort(f, dim=1, descending=True), dim=1)  # [B, N]

        true_best_idx = y.argmax(dim=1)  # [B]
        bi = torch.arange(y.shape[0], device=y.device)
        true_best_fb_rank = f_ranks[bi, true_best_idx].float()  # [B]

        fb_best_idx = f.argmax(dim=1)  # [B]
        fb_best_y_rank = y_ranks[bi, fb_best_idx].float()  # [B]

        # spearman, 1 - 6 sum d^2 / (N (N^2 - 1))
        d_sq = (y_ranks.float() - f_ranks.float()).pow(2).sum(dim=1).float()
        spearman = 1.0 - 6.0 * d_sq / (N * (N**2 - 1) + 1e-8)  # [B]

        print()
        print(f"  -- rank alignment (query, N={N}) {'-'*18}")

        def rstat(label, vals):
            pct = vals / (N - 1) * 100
            print(f"  {label:40s} : "
                  f"mean={vals.mean().item():5.1f}  "
                  f"({pct.mean().item():4.1f}%)  "
                  f"std={vals.std().item():4.1f}  "
                  f"min={vals.min().item():4.0f}  max={vals.max().item():4.0f}")

        rstat("true-y-best rank in feedback  (0=perfect)", true_best_fb_rank)
        rstat("feedback-best rank in true y  (0=perfect)", fb_best_y_rank)
        print(f"  {'Spearman rho (y vs feedback)':40s} : "
              f"mean={spearman.mean().item():6.3f}  "
              f"std={spearman.std().item():6.3f}  "
              f"min={spearman.min().item():6.3f}  max={spearman.max().item():6.3f}")

    print()
    stats("target_all", batch.target_all)
    if getattr(batch, 'target_theta', None) is not None:
        stats("target_theta", batch.target_theta)
    print(f"\n  n_target_theta: {getattr(batch, 'n_target_theta', 'N/A')}")
    print(f"  expert_dim_x:   {getattr(batch, 'expert_dim_x', 'N/A')}")
    print(f"  dim_x:          {getattr(batch, 'dim_x', 'N/A')}")
    print(f"{'='*50}\n")


def weighted_alpha_bo(acq_scores, expert_probs, alpha=0.5, **kwargs):
    """
    alpha-BO style mix, (1 - alpha) * acq + alpha * expert.

    acq_scores, expert_probs: [B, n_options]
    alpha: expert weight in [0, 1]
    """
    return (1 - alpha) * acq_scores + alpha * expert_probs


def pi_bo(acq_scores, expert_probs, beta=.5, n = 1, **kwargs):
    """piBO style, acq * expert^(beta / n), prior decays over steps."""
    n= n+1  # n starts at 0
    return acq_scores * (expert_probs ** (beta/n))


def expert_override(acq_scores, expert_probs, scale=1.0, **kwargs):
    """Hard override, normalized expert probs, ignores acq_scores."""
    adjusted = expert_probs / expert_probs.sum(dim=-1, keepdim=True) * scale
    return adjusted


def topk_expert_bo(acq_scores, expert_probs, k=1, boost=1.0, **kwargs):
    """Add boost to acq scores of the expert's top-k."""
    adjusted = acq_scores.clone()
    topk_vals, topk_idx = torch.topk(expert_probs, k, dim=1)
    batch_idx = torch.arange(acq_scores.size(0))[:, None]
    adjusted[batch_idx, topk_idx] += boost
    return adjusted


# dont use, ground truth prefs, slightly incorrect
def collaborative_bo_selection(acq_scores, expert_scores, k=5, **kwargs):
    """
    Expert pick among the acq top-k.

    acq_scores, expert_scores: [B, n_options]
    Returns: selected indices [B]
    """
    batch_size, n_options = acq_scores.shape
    k = min(k, n_options)

    topk_acq_vals, topk_indices = torch.topk(acq_scores, k=k, dim=1)

    topk_expert_scores = torch.gather(
        expert_scores, 1, topk_indices
    )

    best_within_topk = torch.argmax(topk_expert_scores, dim=1)

    selected_indices = topk_indices[
        torch.arange(batch_size), best_within_topk
    ]

    return selected_indices



def topk_blend(acq_scores, expert_scores, k=5, lambda_h=0.5, normalize=True, **kwargs):
    """
    Soft top-k blend, acq + lambda_h * expert inside the acq top-k, 0 elsewhere.
    not CoExBO, no preference learning or trust decay

    acq_scores, expert_scores: [B, n_options], expert_scores is utility, not probs
    normalize: z-score both inside the top-k
    Returns: [B, n_options]
    """
    batch_size, n_options = acq_scores.shape
    k = min(k, n_options)

    _, topk_indices = torch.topk(acq_scores, k=k, dim=1)

    topk_acq = torch.gather(acq_scores, 1, topk_indices)
    topk_expert = torch.gather(expert_scores, 1, topk_indices)

    if normalize:
        topk_acq = (topk_acq - topk_acq.mean(dim=1, keepdim=True)) / (
            topk_acq.std(dim=1, keepdim=True) + 1e-8
        )
        topk_expert = (topk_expert - topk_expert.mean(dim=1, keepdim=True)) / (
            topk_expert.std(dim=1, keepdim=True) + 1e-8
        )

    topk_collab = topk_acq + lambda_h * topk_expert

    collaborative_scores = torch.zeros_like(acq_scores)
    collaborative_scores.scatter_(1, topk_indices, topk_collab)

    return collaborative_scores


def _acq_feedback_agreement(scores: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
    """Fraction of acq top-5 also in feedback top-5. [B]"""
    n_q = scores.shape[-1]
    k = max(1, min(5, n_q))

    top_scores_idx = scores.topk(k, dim=-1).indices  # [B, k]
    top_feedback_idx = feedback.topk(k, dim=-1).indices  # [B, k]

    matches = (top_scores_idx.unsqueeze(-1) == top_feedback_idx.unsqueeze(-2))  # [B, k, k]
    overlap = matches.any(dim=-1).sum(dim=-1).float()  # [B]

    return overlap / k


class _PrecomputedModel(Model):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self._mean = mean
        self._std = std

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X, observation_noise=False, **kwargs):
        mean = self._mean.unsqueeze(-1)
        var = self._std.pow(2).unsqueeze(-1)
        mvn = gpytorch.distributions.MultivariateNormal(
            mean, DiagLinearOperator(var)
        )
        return GPyTorchPosterior(mvn)
        

def evaluate_policy_vs_bo(
    eval_set, cfg, model, experiment, T,
    acquisition="aae", attend_to="data",
    always_T_is_1=False,
    n_theta=2,
    advanced_stats=False,
    concat_feedback=False,
    use_gp=False,
    gp_model=None,
    expert_feedback_adjustor=None,
    Turn_into_probs=True,
    mask_type="none",
    ):
    model.eval()
    batch = deepcopy(eval_set)

    batch.target_all = torch.cat([
        batch.target_all[:, :cfg.task.n_target_data + n_theta - 1],
        batch.target_all[:, -1:]
    ], dim=1)
    
    if concat_feedback:
        batch = concat_feedback_to_batch_x(batch)

    batch_size = batch.context_x.shape[0]
    
    n_tgt_actual = batch.target_y.shape[1]

    batch.target_all = torch.cat([
        batch.target_all[:, :n_tgt_actual + n_theta - 1],
        batch.target_all[:, -1:]
    ], dim=1)
    
    target_mask = create_target_mask(
        mask_type, cfg.task.embedding_type, n_tgt_actual,
        n_theta, cfg.task.n_selected_targets, cfg.task.predefined_masks,
        None, cfg.task.mask_index, attend_to,
    )

    batch.target_mask = target_mask if acquisition == "aae" else None

    selected_values = torch.zeros((T + 1, batch_size), device=batch.context_x.device)
    best_so_far = torch.zeros((T + 1, batch_size), device=batch.context_x.device)
    rmse_values = torch.zeros((T, batch_size), device=batch.context_x.device)
    regret_values = torch.zeros((T, batch_size), device=batch.context_x.device)
    regret_fraction_within_5_percent = torch.zeros((T, batch_size), device=batch.context_x.device)
    cumulative_regret_values = torch.zeros((T, batch_size), device=batch.context_x.device)

    best_y = batch.context_y.max(dim=1).values.squeeze(-1).to(batch.context_x.device)
    selected_values[0] = best_y
    best_so_far[0] = best_y
    best_y = best_y.unsqueeze(-1)
    
    acq_feedback_agreement_batch = []
    agreements_hard_batch = []
    agreements_soft_batch = []
    mean_y_diffs_batch = []
    mean_y_diffs_sampled_batch = []
    expert_best_hits_batch = []
    expert_rank_batch = []
    expert_feedback_stats = {}

    if advanced_stats:
        expert_feedback_stats["other"] = {
            "target_x": batch.target_x[:, :cfg.task.n_target_data].detach().cpu(),
            "target_y": batch.target_all[:, :cfg.task.n_target_data].detach().cpu(),
            "target_feedback": getattr(batch, "target_feedback", None).detach().cpu(),
            "thetas_true": batch.target_all[:, cfg.task.n_target_data:].detach().cpu(),
            "data_targets": {"pred_mean": [], "pred_var": []},
            "thetas": {"pred_mean": [], "pred_var": []},
            "gmm_feedback_log_likelihood": [],
            "gmm_feedback_ll_mask": [],
        }

    n_query_orig = batch.query_y.shape[1]
    true_max = torch.cat([batch.context_y, batch.query_y], dim=1).max(dim=1).values.squeeze(-1)

    gp_state = None

    for t in range(T):
        batch.t = torch.tensor([1.] if always_T_is_1 else [t / T], device=batch.context_x.device)

        if use_gp:
            gp_botorch, gp_state = _fit_botorch_gp(
                batch.context_x, batch.context_y.squeeze(-1), prev_state=gp_state
            )
            with torch.no_grad():
                qry_post = gp_botorch.posterior(batch.query_x)
                weighted_mean_query = qry_post.mean.squeeze(-1)
                weighted_std_query = qry_post.variance.clamp_min(1e-8).sqrt().squeeze(-1)

                tgt_post = gp_botorch.posterior(batch.target_x[:, :cfg.task.n_target_data])
                weighted_mean = tgt_post.mean.squeeze(-1)

            acq_model = gp_botorch
            design_out_idx = None

        else:
            outs = model.forward(batch)
            design_out = outs.design_out
            posterior = outs.posterior_out
            posterior_query = outs.posterior_out_query
            if advanced_stats:
                _feedback_vals_for_stats = batch.query_feedback[..., 0].detach()

            weighted_mean = (posterior.mixture_means *
                                posterior.mixture_weights).sum(dim=-1)
            weighted_std = torch.sqrt(
                (posterior.mixture_weights * posterior.mixture_stds ** 2).sum(dim=-1) + 1e-8
            )
            weighted_mean_query = (posterior_query.mixture_means *
                                    posterior_query.mixture_weights).sum(dim=-1)
            weighted_std_query = torch.sqrt(
                (posterior_query.mixture_weights *
                    posterior_query.mixture_stds ** 2).sum(dim=-1) + 1e-8
            )
            design_out_idx = design_out.idx

            acq_model = _PrecomputedModel(weighted_mean_query, weighted_std_query)

        X = batch.query_x.unsqueeze(-2)
        best_f = best_y.squeeze(-1)

        with torch.no_grad():
            if acquisition == "aae" and not use_gp:
                acquisition_scores = design_out.zt
                idx = design_out_idx

            elif acquisition == "log_ei":
                # botorch LogEI + _PrecomputedModel unreliable at this batch size
                from utils.free_mode import log_expected_improvement
                acquisition_scores = log_expected_improvement(weighted_mean_query, weighted_std_query, best_f.unsqueeze(-1))
                idx = acquisition_scores.argmax(dim=1, keepdim=True)
            
            elif acquisition == "ei":
                from utils.free_mode import expected_improvement
                acquisition_scores = expected_improvement(weighted_mean_query, weighted_std_query, best_f.unsqueeze(-1))
                idx = acquisition_scores.argmax(dim=1, keepdim=True)
            elif acquisition == "ucb":
                from utils.free_mode import upper_confidence_bound
                kappa = getattr(cfg, "bo_kappa", 1.0)
                acquisition_scores = upper_confidence_bound(weighted_mean_query, weighted_std_query, kappa)
                idx = acquisition_scores.argmax(dim=1, keepdim=True)
            
            elif acquisition == "pi":
                from utils.free_mode import probability_of_improvement
                acquisition_scores = probability_of_improvement(weighted_mean_query, weighted_std_query, best_f.unsqueeze(-1))
                idx = acquisition_scores.argmax(dim=1, keepdim=True)

            elif acquisition == "uncertainty":
                acquisition_scores = (
                    weighted_std_query if use_gp else
                    calculate_gmm_variance(
                        posterior_query.mixture_means,
                        posterior_query.mixture_stds,
                        posterior_query.mixture_weights,
                    )
                )
                idx = acquisition_scores.argmax(dim=1, keepdim=True)

            elif acquisition in ("feedback", "feedback_first"):
                acquisition_scores = batch.query_feedback[:, :, 0]
                idx = acquisition_scores.argmax(dim=1, keepdim=True)

            elif acquisition == "thompson":
                post = acq_model.posterior(batch.query_x)
                acquisition_scores = (
                    post.rsample(torch.Size([1])).squeeze(0).squeeze(-1)
                )
                acquisition_scores = torch.nan_to_num(acquisition_scores, nan=0.0)
                idx = acquisition_scores.argmax(dim=1, keepdim=True)

            elif acquisition == "random":
                acquisition_scores = torch.rand(batch_size, batch.query_x.shape[1], device=batch.context_x.device)
                idx = acquisition_scores.argmax(dim=1, keepdim=True)

            else:
                raise NotImplementedError(f"Unknown acquisition: {acquisition}")

        if acquisition_scores is not None:
            scores_2d = (acquisition_scores.mean(dim=-1)
                if acquisition_scores.dim() == 3
                else acquisition_scores)
            feedback_2d = batch.query_feedback[:, :acquisition_scores.shape[1], 0]
            acq_feedback_agreement_batch.append(
                _acq_feedback_agreement(
                    scores_2d.detach(), feedback_2d.detach()
                ).cpu()
            )

        if expert_feedback_adjustor is not None and acquisition_scores is not None:
            feedback = batch.query_feedback[:, :, 0]
            if Turn_into_probs:
                feedback = torch.softmax(feedback, 1)
            if acquisition_scores.dim() == 3:
                scores_for_adjustor = acquisition_scores.mean(dim=-1)
            else:
                scores_for_adjustor = acquisition_scores
            acquisition_scores_adjusted = expert_feedback_adjustor(
                scores_for_adjustor, feedback, n=t
            )
            idx = acquisition_scores_adjusted.argmax(dim=1, keepdim=True)

        next_y = torch.gather(batch.query_y.squeeze(-1), 1, idx)
        selected_values[t + 1] = next_y.squeeze(-1)
        best_y = torch.max(best_y, next_y)
        best_so_far[t + 1] = best_y.squeeze(-1)

        rmse_values[t] = torch.sqrt(
            ((weighted_mean[:, :cfg.task.n_target_data] -
                batch.target_all[:, :cfg.task.n_target_data].squeeze(-1)) ** 2
                ).mean(dim=-1)
        )

        best_available_t = true_max
        regret_values[t] = best_available_t - next_y.squeeze(-1)
        regret_values_best = best_available_t - best_y.squeeze(-1)
        regret_fraction_within_5_percent[t] = (regret_values_best <= 0.05 * best_available_t).float()
        cumulative_regret_values[t] = cumulative_regret_values[t - 1] + regret_values_best

        if getattr(cfg.task, "use_expert_model", False):
            with torch.no_grad():
                expert_probs = batch.query_feedback[..., 0]
                if getattr(cfg.task, "normalize_outputs_to_max_one", False):
                    expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True)

                policy_actions = (
                    idx.squeeze(-1) if use_gp
                    else design_out.zt.argmax(dim=-1)
                )
                expert_actions = expert_probs.argmax(dim=-1)

                agreements_hard_batch.append((policy_actions == expert_actions).float())
                if not use_gp:
                    agreements_soft_batch.append(
                        (design_out.zt * expert_probs).sum(dim=-1)
                    )

                qy_full = batch.query_y
                model_y = qy_full[torch.arange(qy_full.size(0)), policy_actions].squeeze(-1)
                expert_y = qy_full[torch.arange(qy_full.size(0)), expert_actions].squeeze(-1)
                mean_y_diffs_batch.append(model_y - expert_y)
                mean_y_diffs_sampled_batch.append(next_y.squeeze(-1) - expert_y)

                qy = batch.query_y.squeeze(-1)
                expert_best_hits_batch.append((expert_actions == qy.argmax(dim=1)).float())
                expert_rank_batch.append(
                    ((qy > expert_y.unsqueeze(-1)).sum(dim=1) + 1).float()
                )

        batch = (
            experiment.update_batch(batch, idx)
            if acquisition != "feedback_first" else batch
        )

        if advanced_stats and not use_gp:
            other = expert_feedback_stats["other"]
            n_data = cfg.task.n_target_data
            n_total = batch.target_all.shape[1]
            pred_mean = weighted_mean[:, :n_total]
            pred_var = weighted_std[:, :n_total] ** 2
            other["data_targets"]["pred_mean"].append(pred_mean[:, :n_data].detach().cpu())
            other["data_targets"]["pred_var"].append(pred_var[:, :n_data].detach().cpu())
            other["thetas"]["pred_mean"].append(pred_mean[:, n_data:].detach().cpu())
            other["thetas"]["pred_var"].append(pred_var[:, n_data:].detach().cpu())

            feedback_vals = _feedback_vals_for_stats
            means = posterior_query.mixture_means
            stds = posterior_query.mixture_stds
            weights = posterior_query.mixture_weights
            y_exp = feedback_vals.unsqueeze(-1)
            comp_log_prob = Normal(means, stds + 1e-8).log_prob(y_exp)
            log_mixture = torch.logsumexp(comp_log_prob + torch.log(weights + 1e-8), dim=-1)

            pad_size = n_query_orig - log_mixture.shape[1]
            if pad_size > 0:
                log_mixture_padded = F.pad(log_mixture, (0, pad_size), value=0.0)
                mask = torch.cat([
                    torch.ones(log_mixture.shape, dtype=torch.bool),
                    torch.zeros((*log_mixture.shape[:-1], pad_size), dtype=torch.bool),
                ], dim=-1)
            else:
                log_mixture_padded = log_mixture
                mask = torch.ones(log_mixture.shape, dtype=torch.bool)

            other["gmm_feedback_log_likelihood"].append(log_mixture_padded.detach().cpu())
            other["gmm_feedback_ll_mask"].append(mask.cpu())

    def step_stats(values_list):
        return [
            {"mean": v.mean().item(), "variance": v.var(unbiased=False).item(),
                "min": v.min().item(), "max": v.max().item()}
            for v in values_list
        ]

    expert_feedback_stats.update({
        "model_policy_argmax_vs_expert": [{"mean": v.mean().item()} for v in agreements_hard_batch],
        "model_policy_distribution_vs_expert": step_stats(agreements_soft_batch),
        "model_predictions_minus_expert": step_stats(mean_y_diffs_batch),
        "sampled_model_outputs_minus_expert": step_stats(mean_y_diffs_sampled_batch),
        "fraction_expert_selected_true_best": [{"mean": v.mean().item()} for v in expert_best_hits_batch],
        "rank_of_expert_choice_among_all_options": step_stats(expert_rank_batch),
        "acq_feedback_agreement": (
            torch.stack(acq_feedback_agreement_batch, dim=0)
            if acq_feedback_agreement_batch else
            torch.full((T, batch_size), float("nan"))
        ),  # [T, B]
    })

    if advanced_stats:
        other = expert_feedback_stats["other"]
        for group in ["data_targets", "thetas"]:
            for stat in ["pred_mean", "pred_var"]:
                other[group][stat] = torch.stack(other[group][stat], dim=0)
        other["gmm_feedback_log_likelihood"] = torch.stack(
            other["gmm_feedback_log_likelihood"], dim=0
        )
        other["gmm_feedback_ll_mask"] = torch.stack(
            other["gmm_feedback_ll_mask"], dim=0
        )

    return (
        selected_values.detach().cpu(),
        best_so_far.detach().cpu(),
        rmse_values.detach().cpu(),
        regret_values.detach().cpu(),
        cumulative_regret_values.detach().cpu(),
        regret_fraction_within_5_percent.detach().cpu(),
        expert_feedback_stats,
    )


class GMMPosterior(Posterior):
    """Botorch posterior over a GMM. means, stds, weights: [B, n_q, K]"""
    def __init__(self, mixture_means, mixture_stds, mixture_weights):
        super().__init__()
        self._means = mixture_means
        self._stds = mixture_stds
        self._weights = mixture_weights

    @property
    def device(self):
        return self._means.device

    @property
    def dtype(self):
        return self._means.dtype

    @property
    def event_shape(self):
        return torch.Size([1])

    @property
    def mean(self):
        # [B, n_q, 1]
        return (self._weights * self._means).sum(dim=-1, keepdim=True)

    @property
    def variance(self):
        mix_mean = (self._weights * self._means).sum(dim=-1, keepdim=True)
        mix_var = (self._weights * (self._stds**2 + self._means**2)).sum(dim=-1, keepdim=True) \
                   - mix_mean**2
        return mix_var.clamp(min=1e-8)

    def rsample(self, sample_shape=torch.Size()):
        B, n_q, K = self._means.shape
        n_samples = sample_shape.numel() if len(sample_shape) > 0 else 1
        weights_flat = self._weights.view(B * n_q, K)
        comp_idx = torch.multinomial(weights_flat, n_samples, replacement=True)  # [B*n_q, n_samples]
        comp_idx = comp_idx.view(B, n_q, n_samples)  # [B, n_q, n_samples]

        means_exp = self._means.unsqueeze(-1).expand(B, n_q, K, n_samples)
        stds_exp = self._stds.unsqueeze(-1).expand(B, n_q, K, n_samples)
        idx_exp = comp_idx.unsqueeze(2)  # [B, n_q, 1, n_samples]

        chosen_mean = means_exp.gather(2, idx_exp).squeeze(2)  # [B, n_q, n_samples]
        chosen_std = stds_exp.gather(2, idx_exp).squeeze(2)

        samples = chosen_mean + chosen_std * torch.randn_like(chosen_mean)  # [B, n_q, n_samples]
        # botorch wants [n_samples, B, n_q, 1]
        return samples.permute(2, 0, 1).unsqueeze(-1)


class GMMModel(Model):
    """Thin BoTorch Model wrapper around GMMPosterior."""
    _num_outputs = 1

    def __init__(self, mixture_means, mixture_stds, mixture_weights):
        super().__init__()
        self._means = mixture_means
        self._stds = mixture_stds
        self._weights = mixture_weights

    @property
    def num_outputs(self):
        return 1

    def posterior(self, X, observation_noise=False, **kwargs):
        # X ignored, precomputed
        return GMMPosterior(self._means, self._stds, self._weights)


class _GMMModel:
    def __init__(self, mixture_means, mixture_stds, mixture_weights):
        self.means = mixture_means  # [B, n_q, K]
        self.stds = mixture_stds
        self.weights = mixture_weights

    def ei(self, best_f):
        f = best_f.unsqueeze(-1)
        z = (self.means - f) / (self.stds + 1e-8)
        normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        ei_k = (self.means - f) * normal.cdf(z) + self.stds * torch.exp(normal.log_prob(z))
        return (self.weights * ei_k).sum(dim=-1)

    def log_ei(self, best_f):
        f = best_f.unsqueeze(-1)
        z = (self.means - f) / (self.stds + 1e-8)
        normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        log_phi = normal.log_prob(z)
        log_Phi = torch.log(normal.cdf(z).clamp(min=torch.finfo(z.dtype).tiny))
        log_std = torch.log(self.stds + 1e-8)
        log_ei_k = log_std + log_phi + torch.log1p(z * torch.exp(log_Phi - log_phi))
        log_ei_k = log_ei_k.clamp(min=torch.finfo(z.dtype).min)
        log_w = torch.log(self.weights + 1e-8)
        return torch.logsumexp(log_w + log_ei_k, dim=-1)

    def pi(self, best_f):
        f = best_f.unsqueeze(-1)
        z = (self.means - f) / (self.stds + 1e-8)
        normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        return (self.weights * normal.cdf(z)).sum(dim=-1)

    def ucb(self, kappa):
        mix_mean = (self.weights * self.means).sum(dim=-1)
        mix_var = (self.weights * (self.stds**2 + self.means**2)).sum(dim=-1) - mix_mean**2
        mix_std = mix_var.clamp(min=1e-8).sqrt()
        return mix_mean + kappa * mix_std


class BatchedGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        batch_shape = torch.Size([train_x.shape[0]])
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=train_x.shape[-1]),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        return MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


def _normalize_x(x):
    """Per-batch standardize, identity on flat dims."""
    x_mean = x.mean(dim=1, keepdim=True)
    x_std = x.std(dim=1, keepdim=True)
    # flat dim -> rank-1 kernel -> NotPSD
    safe = x_std > 1e-2
    x_std_safe = torch.where(safe, x_std, torch.ones_like(x_std))
    x_mean_out = torch.where(safe, x_mean, torch.zeros_like(x_mean))
    return (x - x_mean_out) / x_std_safe, x_mean_out, x_std_safe


def _normalize_y(y):
    """Per-batch standardize. [B, N] -> [B, N], stats [B, 1]"""
    y_mean = y.mean(dim=1, keepdim=True)  # [B, 1]
    y_std = y.std(dim=1, keepdim=True)  # [B, 1]
    # N=1 -> std 0
    fallback = y.abs().mean(dim=1, keepdim=True).clamp(min=1.0)
    y_std = torch.where(y_std < 1e-2, fallback, y_std)
    return (y - y_mean) / y_std, y_mean, y_std


def _fit_batched_gp(train_x, train_y, n_iter=30, lr=0.1, prev_state=None):
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

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    clean = True
    for _ in range(n_iter):
        optimizer.zero_grad()
        try:
            with gpytorch.settings.cholesky_jitter(1e-2):
                loss = -mll(model(train_x_norm), train_y_norm).mean()
        except Exception:  # NotPSDError
            clean = False
            break
        if not torch.isfinite(loss):
            clean = False
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval(); likelihood.eval()

    state = (
        {"model": model.state_dict(), "likelihood": likelihood.state_dict()}
        if clean else None
    )
    norm_stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, likelihood, state, norm_stats


def _batched_posterior(model, likelihood, x, norm_stats):
    """Mean, std [B, N] in original y scale. x [B, N, d]"""
    x_norm = (x - norm_stats["x_mean"]) / norm_stats["x_std"]
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
        pred = likelihood(model(x_norm))
        mean_norm = pred.mean  # [B, N]
        std_norm = pred.variance.clamp(min=1e-8).sqrt()

    mean = mean_norm * norm_stats["y_std"] + norm_stats["y_mean"]
    std = std_norm * norm_stats["y_std"]

    mean = torch.nan_to_num(mean, nan=0.0)
    std = torch.nan_to_num(std, nan=1.0)
    return mean, std


def _update_batch(batch, idx):
    B = batch.query_x.shape[0]
    device = batch.query_x.device
    batch_idx = torch.arange(B, device=device)
    idx_sq = idx.squeeze(-1)

    next_x = batch.query_x[batch_idx, idx_sq].unsqueeze(1)
    next_y = batch.query_y[batch_idx, idx_sq].unsqueeze(1)
    next_f = batch.query_feedback[batch_idx, idx_sq].unsqueeze(1)

    mask = torch.ones(B, batch.query_x.shape[1], dtype=torch.bool, device=device)
    mask[batch_idx, idx_sq] = False

    batch.query_x = batch.query_x[mask].view(B, -1, batch.query_x.shape[-1])
    batch.query_y = batch.query_y[mask].view(B, -1, 1)
    batch.query_feedback = batch.query_feedback[mask].view(B, -1, 1)

    batch.context_x = torch.cat([batch.context_x, next_x], dim=1)
    batch.context_y = torch.cat([batch.context_y, next_y], dim=1)
    batch.context_feedback = torch.cat([batch.context_feedback, next_f], dim=1)

    return batch


def _init_tracking(batch, T):
    B = batch.context_x.shape[0]
    device = batch.context_x.device
    best_y = batch.context_y.max(dim=1).values.squeeze(-1)
    true_max = torch.cat([batch.context_y, batch.query_y], dim=1).max(dim=1).values.squeeze(-1)

    selected_values = torch.zeros(T + 1, B, device=device)
    best_so_far = torch.zeros(T + 1, B, device=device)
    rmse_values = torch.zeros(T, B, device=device)
    regret_values = torch.zeros(T, B, device=device)
    cumulative_regret_values = torch.zeros(T, B, device=device)
    regret_fraction_within_5_percent = torch.zeros(T, B, device=device)

    selected_values[0] = best_y
    best_so_far[0] = best_y

    return (
        best_y.unsqueeze(-1), true_max,
        selected_values, best_so_far,
        rmse_values, regret_values,
        cumulative_regret_values, regret_fraction_within_5_percent,
    )


def _acq_scores(mean, std, best_y, acquisition, kappa):
    """mean, std [B, N], best_y [B, 1] -> [B, N]. log_ei in log space, same argmax as ei"""
    if acquisition == "ei":
        z = (mean - best_y) / (std + 1e-8)
        normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        return (mean - best_y) * normal.cdf(z) + std * torch.exp(normal.log_prob(z))
    elif acquisition == "log_ei":
        z = (mean - best_y) / (std + 1e-8)
        normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        log_phi = normal.log_prob(z)
        log_Phi = torch.log(normal.cdf(z).clamp(min=torch.finfo(z.dtype).tiny))
        log_std = torch.log(std + 1e-8)
        return log_std + log_phi + torch.log1p(z * torch.exp(log_Phi - log_phi))
    elif acquisition == "ucb":
        return mean + kappa * std
    else:
        raise ValueError(f"Unknown acquisition: {acquisition}")


def _step_update(
    batch, idx, best_y, true_max,
    selected_values, best_so_far,
    rmse_values, regret_values,
    cumulative_regret_values, regret_fraction_within_5_percent,
    t, gp_mean_target, n_target_data,
    ):
    B = batch.context_x.shape[0]
    device = batch.context_x.device
    batch_idx = torch.arange(B, device=device)

    next_y = batch.query_y.squeeze(-1)[batch_idx, idx.squeeze(-1)].unsqueeze(-1)
    best_y = torch.max(best_y, next_y)

    selected_values[t + 1] = next_y.squeeze(-1)
    best_so_far[t + 1] = best_y.squeeze(-1)

    target_gt = batch.target_all[:, :n_target_data].squeeze(-1)  # [B, n_target]
    rmse_values[t] = torch.sqrt(((gp_mean_target - target_gt) ** 2).mean(dim=-1))

    regret_values[t] = true_max - next_y.squeeze(-1)

    # cum regret on best-so-far, not selected
    regret_values_best = true_max - best_y.squeeze(-1)
    prev_cum = cumulative_regret_values[t - 1] if t > 0 else torch.zeros_like(regret_values_best)
    cumulative_regret_values[t] = prev_cum + regret_values_best

    regret_fraction_within_5_percent[t] = (regret_values_best <= 0.05 * true_max).float()

    batch = _update_batch(batch, idx)
    return batch, best_y


def evaluate_gp_bo(
    eval_set, T, n_target_data,
    acquisition="log_ei",
    kappa=1.0,
    n_iter=30,
    expert_feedback_adjustor=None,
    turn_into_probs=True,
    ):
    """
    GP-BO, batched gpytorch surrogate.

    acquisition: "ucb", "ei", "log_ei", "pi", "thompson", "random"
    expert_feedback_adjustor: (acq_scores, feedback, n=t) -> scores, as in evaluate_policy_vs_bo
    turn_into_probs: softmax feedback first
    """
    batch = deepcopy(eval_set)
    B = batch.context_x.shape[0]
    device = batch.context_x.device

    (
        best_y, true_max,
        selected_values, best_so_far,
        rmse_values, regret_values,
        cumulative_regret_values, regret_fraction_within_5_percent,
    ) = _init_tracking(batch, T)

    hf_state = None

    for t in tqdm(range(T), desc=f"GP-BO ({acquisition})"):
        ctx_x = batch.context_x  # [B, n_ctx, d]
        ctx_y = batch.context_y.squeeze(-1)  # [B, n_ctx]
        qry_x = batch.query_x  # [B, n_q, d]
        tgt_x = batch.target_x[:, :n_target_data]  # [B, n_target, d]

        model, lik, hf_state, norm_stats = _fit_batched_gp(
            ctx_x, ctx_y, n_iter=n_iter, prev_state=hf_state
        )

        mean, std = _batched_posterior(model, lik, qry_x, norm_stats)  # [B, n_q]
        mean_t, _ = _batched_posterior(model, lik, tgt_x, norm_stats)  # [B, n_target]

        if acquisition == "ucb":
            scores = mean + kappa * std
        elif acquisition == "ei":
            z = (mean - best_y) / (std + 1e-8)
            normal = Normal(torch.zeros_like(z), torch.ones_like(z))
            scores = (mean - best_y) * normal.cdf(z) + std * torch.exp(normal.log_prob(z))
        elif acquisition == "log_ei":
            z = (mean - best_y) / (std + 1e-8)
            normal = Normal(torch.zeros_like(z), torch.ones_like(z))
            log_phi = normal.log_prob(z)
            log_Phi = torch.log(normal.cdf(z).clamp(min=torch.finfo(z.dtype).tiny))
            log_std = torch.log(std + 1e-8)
            scores = log_std + log_phi + torch.log1p(z * torch.exp(log_Phi - log_phi))
            scores = scores.clamp(min=torch.finfo(z.dtype).min)
        elif acquisition == "pi":
            z = (mean - best_y) / (std + 1e-8)
            normal = Normal(torch.zeros_like(z), torch.ones_like(z))
            scores = normal.cdf(z)
        elif acquisition == "thompson":
            eps = torch.randn_like(mean)
            scores = mean + std * eps
        elif acquisition == "random":
            scores = torch.rand_like(mean)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")

        if expert_feedback_adjustor is not None:
            feedback = batch.query_feedback[:, :, 0]  # [B, n_q]
            if turn_into_probs:
                feedback = torch.softmax(feedback, dim=1)
            #scores = torch.softmax(scores, dim=1)

            scores = expert_feedback_adjustor(scores,feedback, n=t)

        idx = scores.argmax(dim=1, keepdim=True)

        batch, best_y = _step_update(
            batch, idx, best_y, true_max,
            selected_values, best_so_far,
            rmse_values, regret_values,
            cumulative_regret_values, regret_fraction_within_5_percent,
            t, mean_t, n_target_data,
        )

    return (
        selected_values.cpu(),
        best_so_far.cpu(),
        rmse_values.cpu(),
        regret_values.cpu(),
        cumulative_regret_values.cpu(),
        regret_fraction_within_5_percent.cpu(),
        {},
    )


def _summarize_mtgp_diags(diags, label):
    """Fit summary of utils.multitask_gp diags, its failures are silent."""
    if not diags:
        return f"    [{label}] no fit diagnostics"

    n = len(diags)
    trained = sum(d["trained"] for d in diags)
    skipped = sum(d["steps_skipped"] for d in diags)
    taken = sum(d["steps_taken"] for d in diags)
    fallbacks = sum(d["posterior_fallback"] for d in diags)
    max_jitter = max(d["max_jitter"] for d in diags)
    last = diags[-1]

    ls = last["lengthscale"]
    ls_str = "n/a" if ls is None else "[" + ", ".join(f"{v:.3f}" for v in ls) + "]"
    noise = last["noise"]
    noise_str = "n/a" if noise is None else "[" + ", ".join(f"{v:.4g}" for v in noise) + "]"
    corr_str = "n/a" if last["task_corr"] is None else f"{last['task_corr']:+.3f}"

    errors = [e for d in diags for e in d["errors"]]
    msg = (
        f"    [{label}] fits={trained}/{n} adam_steps={taken} skipped={skipped} "
        f"posterior_fallbacks={fallbacks} max_jitter={max_jitter:g}\n"
        f"    [{label}] final lengthscale={ls_str} noise={noise_str} task_corr={corr_str}"
    )
    if errors:
        uniq = list(dict.fromkeys(errors))
        msg += f"\n    [{label}] {len(errors)} error(s), e.g. {uniq[0][:160]}"
    return msg


def evaluate_multitask_gp_bo(
    eval_set, T, n_target_data,
    acquisition="log_ei",
    kappa=1.0,
    mtgp_kind="icm",
    rank=None,
    max_pool=400,  # 64
    n_iter=50,
    warm_start=True,
    expert_feedback_adjustor=None,
    turn_into_probs=True,
    verbose_diagnostics=True,
    ):
    """
    GP-BO with a multi-task (objective, feedback) GP, batched over B, utils/multitask_gp.py.

    mtgp_kind: "icm": feedback task on full pool, objective task on context.
        "kronecker": same ICM on context-only data, not botorch KroneckerMultiTaskGP.
        never sees pool feedback, so a no-feedback baseline. prefer "icm"
    n_iter: Adam steps per fit
    verbose_diagnostics: print fit summary, keep on, fit fails silently
    """
    from utils import multitask_gp
    from utils.free_mode import gp_scores

    batch = deepcopy(eval_set)

    (
        best_y, true_max,
        selected_values, best_so_far,
        rmse_values, regret_values,
        cumulative_regret_values, regret_fraction_within_5_percent,
    ) = _init_tracking(batch, T)

    states = None
    diags = []

    for t in tqdm(range(T), desc=f"MTGP-BO ({mtgp_kind}/{acquisition})"):
        qry_x = batch.query_x  # [B, n_q, d]
        tgt_x = batch.target_x[:, :n_target_data]  # [B, n_target, d]

        fits, states, diag = multitask_gp.fit_multitask_gps(
            batch.context_x, batch.context_y, batch.context_feedback,
            qry_x, batch.query_feedback,
            kind=mtgp_kind, rank=rank, max_pool=max_pool, n_iter=n_iter,
            prev_states=states if warm_start else None,
        )
        diags.append(diag)

        mean, std = multitask_gp.multitask_posterior(fits, qry_x)  # [B, n_q]
        mean_t, _ = multitask_gp.multitask_posterior(fits, tgt_x)  # [B, n_target]

        if acquisition == "random":
            scores = torch.rand_like(mean)
        else:
            scores = gp_scores(acquisition, mean, std, best_f=best_y, kappa=kappa)

        if expert_feedback_adjustor is not None:
            feedback = batch.query_feedback[:, :, 0]
            if turn_into_probs:
                feedback = torch.softmax(feedback, dim=1)
            scores = expert_feedback_adjustor(scores, feedback, n=t)

        idx = scores.argmax(dim=1, keepdim=True)

        batch, best_y = _step_update(
            batch, idx, best_y, true_max,
            selected_values, best_so_far,
            rmse_values, regret_values,
            cumulative_regret_values, regret_fraction_within_5_percent,
            t, mean_t, n_target_data,
        )

    if verbose_diagnostics:
        print(_summarize_mtgp_diags(diags, f"mtgp_{mtgp_kind}_{acquisition}"))
        print(f"    [mtgp_{mtgp_kind}_{acquisition}] trust (avg task corr) = {diags[-1]['task_corr']:+.3f}")

    return (
        selected_values.cpu(),
        best_so_far.cpu(),
        rmse_values.cpu(),
        regret_values.cpu(),
        cumulative_regret_values.cpu(),
        regret_fraction_within_5_percent.cpu(),
        {"diags": diags},
    )


def _summarize_imfbo_diags(diags, label):
    """
    Fit summary of utils.imfbo_gp diags, its failures are silent.

    lf_snr: a^2 * outputscale / mean(sigma_lf^2). >~1 feedback informative in visible x,
        << 1 feedback dropped out, plain GP. expected on 3-D benches, expert coord hidden
    lf_share: fraction of posterior mean moved by feedback rows
    gamma: NVUCB factor sigma_t / sqrt(sigma_t^2 + sigma_hf^2). ~1 -> nvucb = nucb = ucb
    ls_floor: frac of lengthscales at floor. high with high lf_snr -> interpolating noise
    """
    if not diags:
        return f"    [{label}] no fit diagnostics"

    n = len(diags)
    trained = sum(d["trained"] for d in diags)
    skipped = sum(d["steps_skipped"] for d in diags)
    taken = sum(d["steps_taken"] for d in diags)
    fallbacks = sum(d["posterior_fallback"] for d in diags)
    max_jitter = max(d["max_jitter"] for d in diags)
    last = diags[-1]

    def _f(key, fmt="{:.4g}"):
        v = last.get(key)
        return "n/a" if v is None else fmt.format(v)

    ls = last["lengthscale"]
    ls_str = "n/a" if ls is None else "[" + ", ".join(f"{v:.3f}" for v in ls) + "]"

    msg = (
        f"    [{label}] fits={trained}/{n} adam_steps={taken} skipped={skipped} "
        f"posterior_fallbacks={fallbacks} max_jitter={max_jitter:g} "
        f"rows={last['n_hf']}hf/{last['n_lf']}lf\n"
        f"    [{label}] lengthscale={ls_str} (floor_frac={_f('lengthscale_floor_frac', '{:.2f}')}) "
        f"outputscale={_f('outputscale')} a={_f('scale', '{:+.3f}')} b={_f('offset', '{:+.3f}')}\n"
        f"    [{label}] hf_noise={_f('hf_noise')} lf_noise={_f('lf_noise_mean')} "
        f"(p05={_f('lf_noise_p05')} p95={_f('lf_noise_p95')} "
        f"floor_frac={_f('lf_noise_floor_frac', '{:.2f}')})\n"
        f"    [{label}] TRUST: lf_snr={_f('lf_snr')} lf_share={_f('lf_share', '{:.3f}')} "
        f"gamma={_f('gamma_mean', '{:.3f}')}"
    )

    errors = [e for d in diags for e in d["errors"]]
    if errors:
        uniq = list(dict.fromkeys(errors))
        msg += f"\n    [{label}] {len(errors)} error(s), e.g. {uniq[0][:160]}"
    return msg


def evaluate_imfbo_bo(
    eval_set, T, n_target_data,
    acquisition="nvucb",
    kappa=1.0,
    noise_model="parametric",
    max_pool=192,
    n_iter=50,
    warm_start=True,
    expert_feedback_adjustor=None,
    turn_into_probs=True,
    verbose_diagnostics=True,
    ):
    """
    GP-BO with iMFBO surrogate (Fan et al., UAI 2024), utils/imfbo_gp.py.

    feedback = free noisy obs of f at every pool point, fb = a*f + b, trust via learned sigma_lf(x).
    only y is queried, so no source selection or cost model

    acquisition: "nvucb" (eq. 13), "nucb" (eq. 17), "ucb" (control, same surrogate), "random"
    noise_model: "parametric" (MLP sigma_lf) or "gp" (per-row log-noise, eq. 12)
    max_pool: feedback rows per step, Cholesky cubic in it
    verbose_diagnostics: print fit summary, see _summarize_imfbo_diags
    """
    from utils import imfbo_gp

    batch = deepcopy(eval_set)

    (
        best_y, true_max,
        selected_values, best_so_far,
        rmse_values, regret_values,
        cumulative_regret_values, regret_fraction_within_5_percent,
    ) = _init_tracking(batch, T)

    state = None
    diags = []

    for t in tqdm(range(T), desc=f"iMFBO ({noise_model}/{acquisition})"):
        qry_x = batch.query_x  # [B, n_q, d]
        tgt_x = batch.target_x[:, :n_target_data]  # [B, n_target, d]

        fits, state, diag = imfbo_gp.fit_imfbo_gp(
            batch.context_x, batch.context_y, batch.context_feedback,
            qry_x, batch.query_feedback,
            noise_model=noise_model, max_pool=max_pool, n_iter=n_iter,
            prev_state=state if warm_start else None,
        )
        diags.append(diag)

        mean, std = imfbo_gp.imfbo_posterior(fits, qry_x)  # [B, n_q]
        mean_t, _ = imfbo_gp.imfbo_posterior(fits, tgt_x)  # [B, n_target]
        sigma_hf = imfbo_gp.imfbo_query_noise(fits)  # [B, 1]

        if acquisition == "random":
            scores = torch.rand_like(mean)
        else:
            scores = imfbo_gp.imfbo_scores(acquisition, mean, std, sigma_hf=sigma_hf, kappa=kappa)

        # per step, needs this step's posterior
        diag["gamma_mean"] = imfbo_gp.imfbo_gamma(std, sigma_hf).mean().item()
        diag["lf_share"] = imfbo_gp.imfbo_lf_share(fits, qry_x).mean().item()

        if expert_feedback_adjustor is not None:
            feedback = batch.query_feedback[:, :, 0]
            if turn_into_probs:
                feedback = torch.softmax(feedback, dim=1)
            scores = expert_feedback_adjustor(scores, feedback, n=t)

        idx = scores.argmax(dim=1, keepdim=True)

        batch, best_y = _step_update(
            batch, idx, best_y, true_max,
            selected_values, best_so_far,
            rmse_values, regret_values,
            cumulative_regret_values, regret_fraction_within_5_percent,
            t, mean_t, n_target_data,
        )

    if verbose_diagnostics:
        print(_summarize_imfbo_diags(diags, f"imfbo_{noise_model}_{acquisition}"))

    return (
        selected_values.cpu(),
        best_so_far.cpu(),
        rmse_values.cpu(),
        regret_values.cpu(),
        cumulative_regret_values.cpu(),
        regret_fraction_within_5_percent.cpu(),
        {"diags": diags},
    )


def _summarize_rgpe_diags(diags, label):
    """Mean rank weights of evaluate_rgpe_bo, degenerate split is silent."""
    if not diags:
        return f"    [{label}] no fit diagnostics"

    n = len(diags)
    feedback_weight = sum(d["feedback_weight"] for d in diags) / n
    target_weight = sum(d["target_weight"] for d in diags) / n
    errors = [e for d in diags for e in d["errors"]]

    msg = (
        f"    [{label}] mean rank weight: feedback={feedback_weight:.3f} target={target_weight:.3f} "
        f"(averaged over {n} steps x {diags[0]['B']} environments)"
    )
    if errors:
        uniq = list(dict.fromkeys(errors))
        msg += f"\n    [{label}] {len(errors)} error(s), e.g. {uniq[0][:160]}"
    return msg


def _slice_state_dict(state_dict, b, B):
    # env b slice, lets get_fitted_model skip refit
    return {
        name: (t[b] if t.dim() > 0 and t.shape[0] == B else t)
        for name, t in state_dict.items()
        }


def evaluate_rgpe_bo(
    eval_set, T, n_target_data,
    acquisition="log_ei",
    kappa=1.0,
    max_pool=400,
    num_posterior_samples=256,
    target_noise_var=1e-4,
    feedback_noise_var=1e-4,
    expert_feedback_adjustor=None,
    turn_into_probs=True,
    verbose_diagnostics=True,
    ):
    """
    GP-BO with RGPE surrogate (Feurer et al. 2018), target GP on context y
    ensembled with base GP on pool-wide feedback.

    utils/rgpe.py is the botorch RGPE tutorial, unmodified. here: both fits batched
    over B, rank weights per env, pool argmax instead of optimize_acqf

    target_noise_var, feedback_noise_var: fixed train_Yvar per task, as in tutorial
    """
    from utils import rgpe
    from utils.free_mode import gp_scores

    batch = deepcopy(eval_set)
    B = batch.context_x.shape[0]
    device = batch.context_x.device
    dtype = batch.context_x.dtype

    (
        best_y, true_max,
        selected_values, best_so_far,
        rmse_values, regret_values,
        cumulative_regret_values, regret_fraction_within_5_percent,
    ) = _init_tracking(batch, T)

    diags = []

    for t in tqdm(range(T), desc=f"RGPE-BO ({acquisition})"):
        qry_x = batch.query_x  # [B, n_q, d]
        tgt_x = batch.target_x[:, :n_target_data]  # [B, n_target, d]
        n_q = qry_x.shape[1]
        n_ctx = batch.context_x.shape[1]

        mean = torch.zeros(B, n_q, device=device, dtype=dtype)
        std = torch.ones(B, n_q, device=device, dtype=dtype)
        mean_t = torch.zeros(B, tgt_x.shape[1], device=device, dtype=dtype)
        step_diag = {"feedback_weight": 0.0, "target_weight": 0.0, "B": B, "errors": []}

        if n_ctx < 2:
            # LOOCV needs 2 points, n_ctx same for all envs
            step_diag["errors"].append(f"skipped step: only {n_ctx} context point(s)")
        else:
            ctx_x = batch.context_x.double()
            ctx_y = batch.context_y.squeeze(-1).double()
            ctx_fb = batch.context_feedback[:, :, 0].double()
            qry_x_d = qry_x.double()
            qry_fb = batch.query_feedback[:, :, 0].double()
            tgt_x_d = tgt_x.double()

            all_x = torch.cat([ctx_x, qry_x_d], dim=1)  # [B, n_total, d]
            all_fb = torch.cat([ctx_fb, qry_fb], dim=1)  # [B, n_total]
            lo = all_x.min(dim=1, keepdim=True).values
            hi = all_x.max(dim=1, keepdim=True).values
            span = (hi - lo).clamp_min(1e-8)

            def _norm(x):
                return ((x - lo) / span).clamp(0.0, 1.0)

            try:
                target_x_train = _norm(ctx_x)
                target_y_train = ctx_y.unsqueeze(-1)
                target_yvar = torch.full_like(target_y_train, target_noise_var)
                target_model = rgpe.get_fitted_model(target_x_train, target_y_train, target_yvar)

                # as in multitask_gp task 1
                n_total = all_x.shape[1]
                if n_total > max_pool:
                    keep = torch.argsort(torch.rand(B, n_total, device=device), dim=1)[:, :max_pool]
                    fb_x_raw = torch.gather(all_x, 1, keep.unsqueeze(-1).expand(-1, -1, all_x.shape[-1]))
                    fb_y_raw = torch.gather(all_fb, 1, keep)
                else:
                    fb_x_raw, fb_y_raw = all_x, all_fb
                fb_x_train = _norm(fb_x_raw)
                fb_y_train = fb_y_raw.unsqueeze(-1)
                fb_yvar = torch.full_like(fb_y_train, feedback_noise_var)
                base_model = rgpe.get_fitted_model(fb_x_train, fb_y_train, fb_yvar)

                # loop ok, no fitting
                target_state = target_model.state_dict()
                base_state = base_model.state_dict()
                weights = torch.full((B, 2), 0.5, device=device, dtype=torch.double)

                for b in range(B):
                    try:
                        target_model_b = rgpe.get_fitted_model(
                            target_x_train[b], target_y_train[b], target_yvar[b],
                            state_dict=_slice_state_dict(target_state, b, B),
                            )
                        base_model_b = rgpe.get_fitted_model(
                            fb_x_train[b], fb_y_train[b], fb_yvar[b],
                            state_dict=_slice_state_dict(base_state, b, B),
                            )
                        weights[b] = rgpe.compute_rank_weights(
                            target_x_train[b], target_y_train[b], target_yvar[b],
                            [base_model_b], target_model_b, num_posterior_samples,
                            )
                    except Exception as exc:
                        step_diag["errors"].append(f"env {b}: {type(exc).__name__}: {exc}")
                    step_diag["feedback_weight"] += weights[b, 0].item() / B
                    step_diag["target_weight"] += weights[b, 1].item() / B

                with torch.no_grad():
                    post_q_fb = base_model.posterior(_norm(qry_x_d))
                    post_q_tg = target_model.posterior(_norm(qry_x_d))
                    post_t_fb = base_model.posterior(_norm(tgt_x_d))
                    post_t_tg = target_model.posterior(_norm(tgt_x_d))

                    w_fb = weights[:, 0].view(B, 1)
                    w_tg = weights[:, 1].view(B, 1)

                    mean_q = w_fb * post_q_fb.mean.squeeze(-1) + w_tg * post_q_tg.mean.squeeze(-1)
                    var_q = (
                        w_fb**2 * post_q_fb.variance.clamp_min(1e-8).squeeze(-1)
                        + w_tg**2 * post_q_tg.variance.clamp_min(1e-8).squeeze(-1)
                        )
                    mean_tt = w_fb * post_t_fb.mean.squeeze(-1) + w_tg * post_t_tg.mean.squeeze(-1)

                    mean = mean_q.to(dtype)
                    std = var_q.clamp_min(1e-8).sqrt().to(dtype)
                    mean_t = mean_tt.to(dtype)
            except Exception as exc:
                step_diag["errors"].append(f"batched fit failed: {type(exc).__name__}: {exc}")

        mean = torch.nan_to_num(mean, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0).clamp_min(1e-8)
        mean_t = torch.nan_to_num(mean_t, nan=0.0)
        diags.append(step_diag)

        if acquisition == "random":
            scores = torch.rand_like(mean)
        else:
            scores = gp_scores(acquisition, mean, std, best_f=best_y, kappa=kappa)

        if expert_feedback_adjustor is not None:
            feedback = batch.query_feedback[:, :, 0]
            if turn_into_probs:
                feedback = torch.softmax(feedback, dim=1)
            scores = expert_feedback_adjustor(scores, feedback, n=t)

        idx = scores.argmax(dim=1, keepdim=True)

        batch, best_y = _step_update(
            batch, idx, best_y, true_max,
            selected_values, best_so_far,
            rmse_values, regret_values,
            cumulative_regret_values, regret_fraction_within_5_percent,
            t, mean_t, n_target_data,
        )

    if verbose_diagnostics:
        print(_summarize_rgpe_diags(diags, f"rgpe_{acquisition}"))

    return (
        selected_values.cpu(),
        best_so_far.cpu(),
        rmse_values.cpu(),
        regret_values.cpu(),
        cumulative_regret_values.cpu(),
        regret_fraction_within_5_percent.cpu(),
        {"diags": diags},
    )


def _run_method(
    mc: MethodConfig,
    bench_sample,
    experiment,
    T: int,
    n_target_data: int,
    evaluate_policy_vs_bo: Callable,
    evaluate_gp_bo: Callable,
    ) -> tuple:

    if mc.experiment_dim_x_delta:
        experiment.dim_x += mc.experiment_dim_x_delta

    def _gp_sample():
        sample = copy.deepcopy(bench_sample)
        return concat_feedback_to_batch_x(sample) if mc.concat_feedback else sample

    try:
        if getattr(mc, "botorch_continuous", False):
            # SingleTaskGP + optimize_acqf, no pool
            from utils import free_mode
            out = free_mode.evaluate_botorch_gp_continuous(
                copy.deepcopy(bench_sample), experiment, T,
                acquisition=mc.acquisition,
                expert_feedback_adjustor=mc.expert_feedback_adjustor,
                feedback_integration=getattr(mc, "feedback_integration", "gradient"),
                reference_optimum=getattr(mc, "reference_optimum", None),
            )
        elif getattr(mc, "continuous", False):
            from utils import free_mode
            pool_kwargs = {
                k: v for k, v in (
                    ("pool_size", getattr(mc, "pool_size", None)),
                    ("n_refine_candidates", getattr(mc, "n_refine_candidates", None)),
                    ("score_chunk_size", getattr(mc, "score_chunk_size", None)),
                    ("n_ballast_candidates", getattr(mc, "n_ballast_candidates", None)),
                    ("candidate_mode", getattr(mc, "candidate_mode", None)),
                    ("n_candidates", getattr(mc, "n_candidates", None)),
                    ("n_background", getattr(mc, "n_background", None)),
                    ("refine_optimizer", getattr(mc, "refine_optimizer", None)),
                    ("refine_lr", getattr(mc, "refine_lr", None)),
                    ("adam_step_multiplier", getattr(mc, "adam_step_multiplier", None)),
                    ("normalize_box", getattr(mc, "normalize_box", None)),
                    ("log_ei_mode", getattr(mc, "log_ei_mode", None)),
                    )
                if v is not None
                }
            out = free_mode.evaluate_continuous(
                copy.deepcopy(bench_sample), experiment, T,
                model=mc.model, acquisition=mc.acquisition,
                n_theta=mc.effective_n_theta(), concat_feedback=mc.concat_feedback,
                feedback_mode=getattr(mc, "feedback_mode", "frozen"),
                use_gp=mc.use_gp, n_iter_gp=mc.n_iter_gp,
                expert_feedback_adjustor=mc.expert_feedback_adjustor,
                reference_optimum=getattr(mc, "reference_optimum", None),
                **pool_kwargs,
            )
        elif getattr(mc, "use_multitask_gp", False):
            out = evaluate_multitask_gp_bo(
                copy.deepcopy(bench_sample), T=T, n_target_data=n_target_data,
                acquisition=mc.acquisition, mtgp_kind=getattr(mc, "mtgp_kind", "icm"),
                kappa=mc.kappa,
                n_iter=mc.n_iter_gp,
                expert_feedback_adjustor=mc.expert_feedback_adjustor,
            )
        elif getattr(mc, "use_imfbo", False):
            imfbo_kwargs = {
                k: v for k, v in (("max_pool", getattr(mc, "imfbo_max_pool", None)),) if v is not None
                }
            out = evaluate_imfbo_bo(
                copy.deepcopy(bench_sample), T=T, n_target_data=n_target_data,
                acquisition=mc.acquisition, kappa=mc.kappa,
                noise_model=getattr(mc, "imfbo_noise_model", "parametric"),
                n_iter=mc.n_iter_gp,
                expert_feedback_adjustor=mc.expert_feedback_adjustor,
                **imfbo_kwargs,
            )
        elif getattr(mc, "use_rgpe", False):
            rgpe_kwargs = {
                k: v for k, v in (
                    ("max_pool", getattr(mc, "rgpe_max_pool", None)),
                    ("num_posterior_samples", getattr(mc, "rgpe_num_posterior_samples", None)),
                    ("target_noise_var", getattr(mc, "rgpe_target_noise_var", None)),
                    ("feedback_noise_var", getattr(mc, "rgpe_feedback_noise_var", None)),
                    )
                if v is not None
                }
            out = evaluate_rgpe_bo(
                copy.deepcopy(bench_sample), T=T, n_target_data=n_target_data,
                acquisition=mc.acquisition, kappa=mc.kappa,
                expert_feedback_adjustor=mc.expert_feedback_adjustor,
                **rgpe_kwargs,
            )
        elif getattr(mc, "gp_native", False):
            from utils import free_mode
            out = free_mode.evaluate_gp_bo_botorch(
                _gp_sample(), T=T, n_target_data=n_target_data,
                acquisition=mc.acquisition, kappa=mc.kappa,
                expert_feedback_adjustor=mc.expert_feedback_adjustor,
            )
        elif not mc.use_gp:
            out = evaluate_policy_vs_bo(
                copy.deepcopy(bench_sample), mc.cfg, mc.model, experiment, T,
                acquisition=mc.acquisition, attend_to=mc.attend_to,
                n_theta=mc.effective_n_theta(), concat_feedback=mc.concat_feedback,
                gp_model=mc.gp_model, expert_feedback_adjustor=mc.expert_feedback_adjustor,
                advanced_stats=mc.advanced_stats,
            )
        else:
            try:
                out = evaluate_gp_bo(
                    _gp_sample(), T=T, n_target_data=n_target_data,
                    acquisition=mc.acquisition, n_iter=mc.n_iter_gp, kappa=mc.kappa,
                    expert_feedback_adjustor=mc.expert_feedback_adjustor,
                )
            except NotPSDError:
                print(f"    [GP fallback 1] {mc.name} -- retrying with increased jitter")
                try:
                    with gpytorch.settings.cholesky_jitter(1e-1):
                        out = evaluate_gp_bo(
                            _gp_sample(), T=T, n_target_data=n_target_data,
                            acquisition=mc.acquisition, n_iter=mc.n_iter_gp, kappa=mc.kappa,
                            expert_feedback_adjustor=mc.expert_feedback_adjustor,
                        )
                except NotPSDError:
                    print(f"    [GP fallback 2] {mc.name} -- retrying with fixed kernel (n_iter=0)")
                    try:
                        out = evaluate_gp_bo(
                            _gp_sample(), T=T, n_target_data=n_target_data,
                            acquisition=mc.acquisition, n_iter=0, kappa=mc.kappa,
                            expert_feedback_adjustor=mc.expert_feedback_adjustor,
                        )
                    except NotPSDError:
                        print(f"    [GP FAILED] {mc.name} -- all fallbacks exhausted, skipping.")
                        return None

    finally:
        if mc.experiment_dim_x_delta:
            experiment.dim_x -= mc.experiment_dim_x_delta

    selected, best, rmse, regret, regret_cum, top5, _ = out
    return (selected, best, rmse, regret, top5, regret_cum)


def run_benchmark_evaluation(
    benchmarks: List[str],
    method_configs: List[MethodConfig],
    experiment,
    evaluate_policy_vs_bo: Callable,
    evaluate_gp_bo: Callable,
    BenchmarkFeedbackTask,
    cfg,
    batch_size: int,
    T: int = 30,
    # BenchmarkFeedbackTask kwargs
    expert_dim: int = 2,
    n_marginal_samples: int = 1,
    design_scale: float = 5.0,
    noise_scale: float = 0.1,
    expert_bias_model=None,
    expert_generator=None,
    n_expert_extra_min: int = 50,
    n_expert_extra_max: int = 50,
    expert_min_centers: int = 5,
    expert_max_centers: int = 5,
    expert_cluster_std_min: float = 0.5,
    expert_cluster_std_max: float = 1.5,
    uniform_fraction: float = 1,
    n_optimum_points: int = 1,
    optimum_jitter: float = 0.0,
    fake_peaks: bool = False,
    fake_peak_k: int = 3,
    fake_peak_height_mult: float = 1.5,
    fake_peak_decay_range: tuple = (0.5, 2.0),
    device=None,
    seed_fn: Optional[Callable] = None,
    verbose: int = 1,
    n_query_init: int =300,
    regret_reference: str = "pool",
    ) -> Dict[str, Dict[str, tuple]]:
    """
    Run all method_configs on all benchmarks.

    regret_reference: "pool" (vs pool max) or "global" (shared reference, for continuous methods)
    Returns: (results, optima)
        results[bench][method] = (selected, best_so_far, rmse, regret, top5, regret_cum),
            selected, best_so_far [T+1, B], rest [T, B]
        optima[bench]: [B]
    """
    results = {}
    optima = {}
    for bench in benchmarks:
        if verbose:
            print(f"\n{'='*60}\nBenchmark: {bench}\n{'='*60}")
 
        if seed_fn is not None:
            seed_fn()
 
        benchmark_kwargs = dict(
            benchmark_name=bench,
            dim_x=cfg.task.dim_x,
            dim_y=cfg.task.dim_y,
            n_context_init=1,
            n_query_init=n_query_init,
            n_target_data=cfg.task.n_target_data if bench not in ["nasbench201", "airfoil_lift_drag", "airfoil_nn",] else 1,
            n_target_theta=cfg.task.n_target_theta,
            design_scale=design_scale,
            noise_scale=noise_scale,
            expert_dim=expert_dim,
            n_marginal_samples=n_marginal_samples,
            concat_bias_generated_certainty=False,
            embedding_type="mix",
            normalize_y=True,
            expert_bias_model=expert_bias_model,
            expert_generator=expert_generator,
            n_expert_extra_min=n_expert_extra_min,
            n_expert_extra_max=n_expert_extra_max,
            expert_min_centers=expert_min_centers,
            expert_max_centers=expert_max_centers,
            expert_cluster_std_min=expert_cluster_std_min,
            expert_cluster_std_max=expert_cluster_std_max,
            uniform_fraction=uniform_fraction,
            n_optimum_points=n_optimum_points,
            optimum_jitter=optimum_jitter,
            fake_peaks=fake_peaks,
            fake_peak_k=fake_peak_k,
            fake_peak_height_mult=fake_peak_height_mult,
            fake_peak_decay_range=fake_peak_decay_range,
        )
        if device is not None:
            benchmark_kwargs["device"] = device
 
        benchmark = BenchmarkFeedbackTask(**benchmark_kwargs)
        bench_sample = benchmark.sample_batch(batch_size=batch_size, mode="mix")
        if verbose >1:
            print_batch_stats(bench_sample, name=bench)
        results[bench] = {}
        true_max = torch.cat([bench_sample.context_y, bench_sample.query_y], dim=1).max(dim=1).values.squeeze(-1).cpu()
        optima[bench] = true_max
 
        for mc in method_configs:
            t0 = time.time()
            if getattr(mc, "continuous", False):
                mc.reference_optimum = true_max
            res = _run_method(
                mc, bench_sample, benchmark, T,
                cfg.task.n_target_data,
                evaluate_policy_vs_bo, evaluate_gp_bo,
            )
            results[bench][mc.name] = res
            if verbose:
                status = "FAILED" if res is None else f"{time.time() - t0:.1f}s"
                print(f"  {mc.name}: {status}")

        # continuous can beat pool max, rescore all vs max(polished opt, best found)
        if regret_reference == "global":
            per_method = {name: res for name, res in results[bench].items() if res is not None}
            if per_method:
                from utils.free_mode import estimate_reference_optimum

                empirical_best = max(res[1][-1].max().item() for res in per_method.values())
                polished = estimate_reference_optimum(
                    benchmark, n_samples=20000, num_restarts=20, n_polish_steps=150,
                )
                reference = max(polished, empirical_best)
                reference_t = torch.tensor(reference)

                for name, (selected, best, rmse, regret, top5, regret_cum) in per_method.items():
                    regret_best = reference_t - best[1:]
                    regret_r = reference_t - selected[1:]
                    cum_r = torch.cumsum(regret_best, dim=0)
                    top5_r = (regret_best <= 0.05 * reference_t.abs().clamp_min(1e-8)).float()
                    results[bench][name] = (selected, best, rmse, regret_r, top5_r, cum_r)

                optima[bench] = torch.full_like(true_max, reference)

                if verbose:
                    print(f"\n  Reference optimum for {bench}: polished={polished:.4f} "
                          f"empirical_best={empirical_best:.4f} -> reference={reference:.4f}")
                    print("  Final cumulative regret ranking (lower = better):")
                    finals = {
                        name: metrics[5][-1].mean().item()
                        for name, metrics in results[bench].items()
                        if metrics is not None
                    }
                    for rank, (name, val) in enumerate(
                        sorted(finals.items(), key=lambda x: x[1]), start=1
                    ):
                        print(f"    {rank}. {name:<35} {val:.4f}")
                    assert all(v >= -1e-4 for v in finals.values()), \
                        f"negative regret in {bench}: {finals}"
        elif verbose:
            print(f"\n  Final cumulative regret ranking (lower = better):")
            finals = {
                name: (
                    metrics[5][-1].mean().item()
                    if torch.is_tensor(metrics[5])
                    else float(metrics[5][-1].mean())
                )
                for name, metrics in results[bench].items()
            }
            for rank, (name, val) in enumerate(
                sorted(finals.items(), key=lambda x: x[1]), start=1
            ):
                print(f"    {rank}. {name:<35} {val:.4f}")

    return results, optima


def aggregate_results(results: dict, metric: str = "best_so_far") -> dict:
    """
    Stack one metric over benchmarks.

    metric: "selected", "best_so_far", "rmse", "regret", "top5", "regret_cum"
    Returns: dict[method] -> [n_benchmarks, T, B]
    """
    metric_idx = {
        "selected": 0,
        "best_so_far": 1,
        "rmse": 2,
        "regret": 3,
        "top5": 4,
        "regret_cum": 5,
    }
    idx = metric_idx[metric]
 
    all_methods = set(m for bench in results for m in results[bench])
    return {
        method: torch.stack([
            results[bench][method][idx]
            for bench in results
            if method in results[bench]
        ], dim=0)
        for method in all_methods
    }


class AveragedPolicyModel(nn.Module):
    """Averages the softmax policy distributions of two FICBO models."""

    def __init__(self, model_a: nn.Module, model_b: nn.Module) -> None:
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, batch: AttrDict) -> AttrDict:
        outs_a = self.model_a.forward(batch)
        outs_b = self.model_b.forward(batch)

        zt_avg = 0.5 * (outs_a.design_out.zt + outs_b.design_out.zt)
        log_prob, idx_next = torch.max(torch.log(zt_avg.clamp_min(1e-30)), -1)
        idx_next = idx_next.unsqueeze(1)

        return AttrDict(
            posterior_out_query=outs_a.posterior_out_query,
            posterior_out=outs_a.posterior_out,
            design_out=AttrDict(
                idx=idx_next,
                log_prob=log_prob,
                zt=zt_avg,
                logits=torch.log(zt_avg.clamp_min(1e-30)),
            ),
        )
