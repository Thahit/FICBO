import torch
import torch.nn as nn
from attrdictionary import AttrDict
from typing import Optional
from expert_models.base_expert import Expert


def print_nested_keys(d, indent=1):
    """
    Recursively prints all keys in a nested dictionary or AttrDict.
    """
    for k, v in d.items():
        print("  " * indent + str(k))
        if isinstance(v, dict) or isinstance(v, AttrDict):
            print_nested_keys(v, indent + 1)


class AmortizedExpert(Expert):
    """
    Wrapper for an amortized GP-prior ICBO model used as feedback source.
    """

    def __init__(self, amortized_model: nn.Module, batch: AttrDict, embedding_type:str , adaptive: bool = True, expert_noise: float = 0.0,
                 normalize_outputs_to_max_one: bool = False, return_uncertainty: bool = False, predict_epistemic_var: bool= False,
                  mode:str = "policy", kappa:float=1, **kwargs):
        """
        Initialize with only expert points.
        batch: AttrDict containing:
            - batch.expert_x, batch.expert_y: used as initial internal context
        """
        self.model = amortized_model
        self.model.requires_grad_(False) 
        self.mode = mode
        self.kappa = kappa  # for ucb
        self.expert_dim = batch.expert_dim_x
        self.full_dim = self.model.embedder.dim_x  # total feature dim
        self.adaptive = adaptive
        self.embedding_type = embedding_type
        self.normalize_outputs_to_max_one = normalize_outputs_to_max_one
        self.expert_noise = expert_noise
        self.return_uncertainty = return_uncertainty
        self.predict_epistemic_var = predict_epistemic_var
        expert_x = self._pad_to_full_dim(batch.expert_x[..., -self.expert_dim:])
        self.batch = AttrDict()
        self.batch.context_x = expert_x  # no clone needed
        self.batch.context_y = batch.expert_y

        self.batch.query_x = torch.zeros(
            expert_x.shape[0], 
            0,                    
            self.full_dim,         
            device=expert_x.device,
            dtype=expert_x.dtype
        )
        self.batch.target_x = torch.zeros(
            expert_x.shape[0],    
            0,                    
            self.full_dim,   
            device=expert_x.device,
            dtype=expert_x.dtype
        )
        self.batch.target_all = torch.zeros(
            expert_x.shape[0],    
            0,                    
            1,   
            device=expert_x.device,
            dtype=expert_x.dtype
        )
        if self.embedding_type == "mix":
            self.batch.target_all = torch.zeros(
            expert_x.shape[0],
            self.full_dim+1,
            1,   
            device=expert_x.device,
            dtype=expert_x.dtype
        )
        self.batch.t = torch.tensor([0], device=expert_x.device)  

    def _pad_to_full_dim(self, x: torch.Tensor):
        """Right-pad input tensor along the last dimension with zeros to match model.embedder.dim_x."""
        pad_dim = self.full_dim - x.shape[-1]
        if pad_dim > 0:
            pad = torch.zeros(*x.shape[:-1], pad_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        return x

    def normalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Takes raw logits (e.g., pred_context_y.design_out.zt),
        applies softmax, and optionally rescales so the max value = 1.

        Expected shape: [B, N]
        """
        # softmax first (stable)
        logits = self._add_noise(logits)
        probs = torch.softmax(logits, dim=-1)

        if self.normalize_outputs_to_max_one:
            # Avoid divide-by-zero when all probs are zero for whatever reason
            max_vals = probs.max(dim=-1, keepdim=True).values
            max_vals = torch.clamp(max_vals, min=1e-10)
            probs = probs / max_vals

        return probs

    def initial_predict(self, batch: AttrDict, augment_queries: bool = False, **kwargs):
        query_x_from_context = self._pad_to_full_dim(batch.context_x[..., -self.expert_dim:])
        n_context = query_x_from_context.shape[1]
        if augment_queries:
            self.batch.query_x = torch.cat([query_x_from_context,
                                            self._pad_to_full_dim(batch.query_x[..., -self.expert_dim:])], dim=1)
        else:
            self.batch.query_x = query_x_from_context

        with torch.no_grad():
            pred_context_y = self.model(self.batch)
            posterior_query = pred_context_y.posterior_out_query
            weighted_mean, weighted_std = self._aggregate_posterior(posterior_query)

            # Depending on mode, return learned policy or acquisition-function-based policy
            if self.mode == "policy":
                probs = self.normalize_logits(pred_context_y.design_out.zt).unsqueeze(-1)
            elif self.mode in ["ucb", "ei", "pi", "mean"]:
                probs = self._acquisition_policy_tensor(weighted_mean, weighted_std, self.batch)
            else:
                raise NotImplementedError(f"Unknown mode {self.mode}")
        if augment_queries:
            probs = probs[:, :query_x_from_context.shape[1]]
        out = probs

        if self.adaptive:
            self.batch.context_x = torch.cat([self.batch.context_x, query_x_from_context], dim=1)
            self.batch.context_y = torch.cat([self.batch.context_y, batch.context_y], dim=1)

        if self.return_uncertainty:
            # optionally reuse weighted_mean / std if you want
            uncertainty = self._calc_uncertainty(posterior_query)[:,:n_context]
            out = torch.cat([out, uncertainty], dim=-1)

        return out

    def predict(self, x: torch.Tensor, time_token: Optional[torch.Tensor] = None, **kwargs):
        """Predict using the current internal context."""
        self.batch.query_x = self._pad_to_full_dim(x[..., -self.expert_dim:])
        self.batch.t = time_token
        if self.batch.query_x.shape[1] == 0:
            # return empty tensor of correct shape
            return torch.empty(self.batch.context_x.shape[0], 0, 1, device=self.batch.context_x.device)

        with torch.no_grad():
            preds = self.model(self.batch)

        posterior_query = preds.posterior_out_query
        weighted_mean, weighted_std = self._aggregate_posterior(posterior_query)

        if self.mode == "policy":
            # learned pol
            out = self.normalize_logits(preds.design_out.zt).unsqueeze(-1)
        elif self.mode in ["ucb", "ei", "pi", "mean"]:
            # acquisition-function-based pol
            out = self._acquisition_policy_tensor(weighted_mean, weighted_std, self.batch)
        else:
            raise NotImplementedError(f"Unknown mode {self.mode}")

        # Optionally append uncertainty
        if self.return_uncertainty:
            uncertainty = self._calc_uncertainty(posterior_query)
            out = torch.cat([out, uncertainty], dim=-1)

        return out  # shape [B, N_query, 1] (or [B, N_query, 2] with uncertainty)
    
    def _acquisition_policy_tensor(self, weighted_mean, weighted_std, batch):
        """Return acquisition-based scores as a vector over query points (not AttrDict)."""
        if self.mode == "ucb":
            score = weighted_mean + self.kappa * weighted_std
        elif self.mode == "ei":
            best_y = batch.context_y.max(dim=1, keepdim=True).values  # [B,1,1]
            # remove trailing dim :(
            best_y = best_y.squeeze(-1)  # [B,1]
            mean = weighted_mean.squeeze(-1)  # [B,N]
            std = weighted_std.squeeze(-1)  # [B,N]

            z = (mean - best_y) / (std + 1e-8)  # [B,N]

            normal = torch.distributions.Normal(0.0, 1.0)
            ei = (mean - best_y) * normal.cdf(z) + std * torch.exp(normal.log_prob(z))
            score = ei  # [B,N]
        elif self.mode == "pi":
            best_y = batch.context_y.max(dim=1, keepdim=True).values  # [B,1,1]

            best_y = best_y.squeeze(-1)  # [B,1]
            mean = weighted_mean.squeeze(-1)  # [B,N]
            std = weighted_std.squeeze(-1)  # [B,N]

            z = (mean - best_y) / (std + 1e-8)
            normal = torch.distributions.Normal(0.0, 1.0)
            pi = normal.cdf(z)
            score = pi  # [B,N]

        elif self.mode == "mean":
            score = weighted_mean
        else:
            raise NotImplementedError(f"Unknown mode {self.mode}")

        policy_probs = score

        # Return as a tensor of shape [B, N_query, 1] to match predict
        return policy_probs.unsqueeze(-1)

    def _aggregate_posterior(self, posterior):
        """Weighted mean/std of the query-point mixture posterior, [B,N,K] -> [B,N]."""
        mean = posterior.mixture_means  # [B, N, K]
        std = posterior.mixture_stds  # [B, N, K]
        weights = posterior.mixture_weights  # [B, N, K]

        weighted_mean = (mean * weights).sum(dim=-1)  # [B, N]
        weighted_var = (weights * (std**2 + mean**2)).sum(dim=-1) - weighted_mean**2
        weighted_std = torch.sqrt(weighted_var + 1e-8)
        return weighted_mean, weighted_std

    def add_data(self, x_new: torch.Tensor, y_new: torch.Tensor, **kwargs):
        """Add new points to the internal context set."""
        if not self.adaptive:
            return
        x_new = self._pad_to_full_dim(x_new[..., -self.expert_dim:])
        self.batch.context_x = torch.cat([self.batch.context_x, x_new], dim=1)
        self.batch.context_y = torch.cat([self.batch.context_y, y_new], dim=1)

    def _calc_uncertainty(self, posterior):
        """Mixture variance from the posterior, [B,N,K] -> [B,N,1]."""
        mean = posterior.mixture_means  # [B, N, K]
        std = posterior.mixture_stds  # [B, N, K]
        weights = posterior.mixture_weights  # [B, N, K]

        weighted_mean = (weights * mean).sum(dim=-1)  # [B, N]

        if self.predict_epistemic_var:
            var = (weights * (mean**2)).sum(dim=-1) - weighted_mean**2  # epistemic only
        else:
            var = (weights * (std**2 + mean**2)).sum(dim=-1) - weighted_mean**2  # aleatoric + epistemic

        uncertainty = var.unsqueeze(-1)   # [B, N, 1]
        return uncertainty

    def _add_noise(self, y: torch.Tensor):
        """Add Gaussian noise to the outputs."""
        if self.expert_noise > 0.0:
            noise = torch.randn_like(y) * self.expert_noise
            return y + noise
        return y
