from expert_models.base_expert import Expert
import torch
from typing import Optional


class TimeEncodedExpertWrapper(Expert):
    """
    Wraps any Expert and appends a time token to all prediction outputs.
    """

    def __init__(self, expert: Expert, time_encoded: bool = True, **kwargs):
        self.expert = expert
        self.time_encoded = time_encoded
        # proxy frequently accessed attributes
        self.adaptive = expert.adaptive

    def _append_time_token(self, output: torch.Tensor, time_token: Optional[torch.Tensor]) -> torch.Tensor:
        """Appends time token as an extra output dimension. output: [B, N, D]"""
        if not self.time_encoded or time_token is None:
            return output
        B, N, _ = output.shape
        t = time_token.to(output.device)
        # scalar or [1] -> [B, N, 1]
        t_expanded = t.reshape(1, 1, -1).expand(B, N, -1)
        return torch.cat([output, t_expanded], dim=-1)

    def initial_predict(self, batch, **kwargs):
        out = self.expert.initial_predict(batch, **kwargs)
        time_token = getattr(batch, "t", None)
        return self._append_time_token(out, time_token)

    def predict(self, x: torch.Tensor, time_token: Optional[torch.Tensor] = None, **kwargs):
        out = self.expert.predict(x, time_token=time_token, **kwargs)
        return self._append_time_token(out, time_token)

    def add_data(self, x_new: torch.Tensor, y_new: torch.Tensor, **kwargs):
        self.expert.add_data(x_new, y_new, **kwargs)

    def __getattr__(self, name):
        return getattr(self.expert, name)
