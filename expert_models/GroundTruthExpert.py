import torch
from typing import Optional
from expert_models.base_expert import Expert
from types import SimpleNamespace as AttrDict


class GroundTruthExpert(Expert):
    def __init__(self, 
                 mode: str = "policy", 
                 temperature: float = 1.0,
                 normalize_outputs_to_max_one: bool = False,
                 expert_noise: float = 0.0,
                 truth_mode: str= "y", # "y", "y_hidden"
                 return_uncertainty: bool = False, **kwargs):
        self.mode = mode
        self.temperature = temperature
        self.normalize_outputs_to_max_one = normalize_outputs_to_max_one
        self.expert_noise = expert_noise
        self.return_uncertainty = return_uncertainty
        self.truth_mode = truth_mode
        assert self.truth_mode in ["y", "y_hidden"], f"Unknown truth_mode {self.truth_mode}"

    def _add_noise(self, y: torch.Tensor):
        if self.expert_noise > 0.0:
            return y + torch.randn_like(y) * self.expert_noise
        return y

    def normalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self._add_noise(logits)
        probs = torch.softmax(logits / self.temperature, dim=1)
        if self.normalize_outputs_to_max_one:
            max_vals = probs.max(dim=1, keepdim=True).values
            max_vals = torch.clamp(max_vals, min=1e-10)
            probs = probs / max_vals
        return probs

    def initial_predict(self, batch: AttrDict, augment_queries: bool = True, **kwargs):
        if self.truth_mode == "y_hidden":
            y_values = batch.y_hidden_context  # [B, N_context, 1]
        else:
            y_values = batch.context_y  # [B, N_context, 1]
        
        n = batch.context_y.shape[1]
        # optionally concatenate query_y if augment_queries
        if augment_queries and hasattr(batch, "query_y"):
            y_values = torch.cat([y_values, batch.query_y], dim=1)  # [B, N_context + N_query, 1]

        # compute output depending on mode
        if self.mode == "value":
            out = self._add_noise(y_values)
        elif self.mode == "policy":
            out = self.normalize_logits(y_values)
        else:
            raise NotImplementedError(f"Unknown mode {self.mode}")

        # query_y was only appended to calibrate the normalization jointly; return context portion only
        if augment_queries and hasattr(batch, "query_y"):
            out = out[:, :n]  # keep only context points
        
        # optionally append uncertainty (dummy for now, can be improved, in case useful...)
        if self.return_uncertainty:
            uncertainty = torch.zeros_like(out)
            out = torch.cat([out, uncertainty], dim=-1)  # [B, N_query, 2]

        return out

    def predict(self, ground_truth_y: torch.Tensor, **kwargs):
        if self.mode == "value":
            out = self._add_noise(ground_truth_y)
        elif self.mode == "policy":
            out = self.normalize_logits(ground_truth_y)
        else:
            raise NotImplementedError(f"Unknown mode {self.mode}")

        if self.return_uncertainty:
            uncertainty = torch.zeros_like(out)
            out = torch.cat([out, uncertainty], dim=-1)  # [B, N_query, 2]

        return out
    
    def add_data(self, x_new, y_new, **kwargs):
        pass
