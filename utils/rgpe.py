import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import GP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import LikelihoodList
from gpytorch.priors.utils import BUFFERED_PREFIX
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from linear_operator.operators import PsdSumLinearOperator
from torch.nn import ModuleList
from typing import Any, Dict, List, Optional


def get_fitted_model(
    train_X: torch.Tensor, train_Y: torch.Tensor, train_Yvar: torch.Tensor,
    state_dict: Optional[Dict[str, Any]] = None,
    ) -> SingleTaskGP:
    # fit unless state_dict given
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
    if state_dict is None:
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_mll(mll)
    else:
        model.load_state_dict(state_dict, strict=False)
    return model


def roll_col(X: torch.Tensor, shift: int) -> torch.Tensor:
    # rotate right
    return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)


def compute_ranking_loss(f_samps: torch.Tensor, target_y: torch.Tensor) -> torch.Tensor:
    n = target_y.shape[0]
    if f_samps.ndim == 3:
        # target model, cartesian product of target_y
        cartesian_y = torch.cartesian_prod(
            target_y.squeeze(-1),
            target_y.squeeze(-1),
            ).view(n, n, 2)
        # diagonal = out-of-sample LOO preds
        rank_loss = (
            (
                (f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps)
                ^ (cartesian_y[..., 0] < cartesian_y[..., 1])
                )
            .sum(dim=-1)
            .sum(dim=-1)
            )
    else:
        rank_loss = torch.zeros(
            f_samps.shape[0], dtype=torch.long, device=target_y.device,
            )
        y_stack = target_y.squeeze(-1).expand(f_samps.shape)
        for i in range(1, target_y.shape[0]):
            rank_loss += (
                (roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack)
                ).sum(dim=-1)
    return rank_loss


# non-parameter buffers, not expanded
NON_EXPAND_ATTRIBUTES = [
    f"covar_module.lengthscale_prior.{BUFFERED_PREFIX}loc",
    f"covar_module.lengthscale_prior.{BUFFERED_PREFIX}scale",
    "covar_module.raw_lengthscale_constraint.lower_bound",
    "covar_module.raw_lengthscale_constraint.upper_bound",
    "outcome_transform._is_trained",
    ]


def get_target_model_loocv_sample_preds(
    train_x: torch.Tensor, train_y: torch.Tensor, train_yvar: torch.Tensor,
    target_model: SingleTaskGP, num_samples: int,
    ) -> torch.Tensor:
    # batch LOOCV GP, joint sample on target task
    # returns [num_samples, n LOO models, n points]
    device = train_x.device
    batch_size = len(train_x)
    masks = torch.eye(len(train_x), dtype=torch.uint8, device=device).bool()
    train_x_cv = torch.stack([train_x[~m] for m in masks])
    train_y_cv = torch.stack([train_y[~m] for m in masks])
    train_yvar_cv = torch.stack([train_yvar[~m] for m in masks])
    state_dict = target_model.state_dict()
    state_dict_expanded = {
        name: (
            t.expand(batch_size, *[-1 for _ in range(t.ndim)])
            if name not in NON_EXPAND_ATTRIBUTES
            else t
            )
        for name, t in state_dict.items()
        }
    model = get_fitted_model(
        train_x_cv, train_y_cv, train_yvar_cv, state_dict=state_dict_expanded,
        )
    with torch.no_grad():
        posterior = model.posterior(train_x)
        # posterior.sample() is [num_samples, n, n, 1]
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        return sampler(posterior).squeeze(-1)


def compute_rank_weights(
    train_x: torch.Tensor, train_y: torch.Tensor, train_yvar: torch.Tensor,
    base_models: List[SingleTaskGP], target_model: SingleTaskGP, num_samples: int,
    ) -> torch.Tensor:
    # LOOCV for target, no weight dilution handling, few base models
    # train_yvar explicit, tutorial used a notebook global
    ranking_losses = []
    for task in range(len(base_models)):
        model = base_models[task]
        posterior = model.posterior(train_x)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
        ranking_losses.append(compute_ranking_loss(base_f_samps, train_y))
    target_f_samps = get_target_model_loocv_sample_preds(
        train_x, train_y, train_yvar, target_model, num_samples,
        )
    ranking_losses.append(compute_ranking_loss(target_f_samps, train_y))
    ranking_loss_tensor = torch.stack(ranking_losses)
    best_models = torch.argmin(ranking_loss_tensor, dim=0)
    rank_weights = (
        best_models.bincount(minlength=len(ranking_losses)).type_as(train_x)
        / num_samples
        )
    return rank_weights


class RGPE(GP, GPyTorchModel):
    # rank-weighted GP ensemble

    _num_outputs = 1  # metadata for botorch

    def __init__(self, models: List[SingleTaskGP], weights: torch.Tensor) -> None:
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "RGPE currently only supports models that have a likelihood "
                    "(e.g. ExactGPs)"
                    )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.weights = weights
        self.to(weights)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        weighted_means = []
        weighted_covars = []
        # skip zero weights, cov weights are weight**2
        non_zero_weight_indices = (self.weights**2 > 0).nonzero()
        non_zero_weights = self.weights[non_zero_weight_indices]
        non_zero_weights = non_zero_weights / non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            model = self.models[raw_idx]
            posterior = model.posterior(x)
            posterior_mean = posterior.mean.squeeze(-1)
            posterior_cov = posterior.mvn.lazy_covariance_matrix
            weight = non_zero_weights[non_zero_weight_idx]
            weighted_means.append(weight * posterior_mean)
            weighted_covars.append(posterior_cov * weight**2)
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLinearOperator(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)

