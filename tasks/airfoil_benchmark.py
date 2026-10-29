from __future__ import annotations

import numpy as np
import torch
from attrdictionary import AttrDict
from tasks.base_task import Task, Expert_Task

try:  # neuralfoil gives an ImportError on import tasks -> 
    #breaks every task, not only the airfoil task :(
    import neuralfoil as nf
    import aerosandbox as asb
    NEURALFOIL_AVAILABLE = True
except ImportError:
    NEURALFOIL_AVAILABLE = False


class AirfoilBenchmarkBase(Expert_Task):
    """
    Shared infrastructure for airfoil benchmarks.

    Subclasses implement:
        _eval_true(x_phys)      [N, 3] -> [N]   (unnormalised)
        _eval_feedback(x_phys)  [N, 3] -> [N]   (unnormalised)
    """

    _BOUNDS = np.array([
        [-2.0, 15.0],  # alpha [deg]
        [ 0.08, 0.20],  # thickness [chord frac]
        [ 0.00, 0.06],  # camber    [chord frac]
    ])  # (3, 2)

    RE = 1e6  # Reynolds number (fixed)
    MODEL_SIZE = "large"  # NeuralFoil model size

    # NeuralFoil input: alpha (angle of attack), airfoil shape (encoded from the
    # NACA string - thickness + camber essentially), Re (Reynolds number).
    # output: CL (lift coefficient), CD (drag coefficient), CM (moment coefficient),
    # plus some boundary layer quantities you don't use.

    def __init__(
        self,
        dim_x: int = 3,
        dim_y: int = 1,
        n_context_init: int = 5,
        n_query_init: int = 10,
        n_target_theta: int = 0,
        n_target_data: int = 5,
        design_scale: float = 5.0,
        noise_scale: float = 0.0,
        normalize_y: bool = True,
        n_norm_samples: int = 2000,
        expert_generator = None,
        **kwargs,
        ):
        Expert_Task.__init__(self, dim_x=dim_x, dim_y=dim_y,
                             design_scale=design_scale, **kwargs)

        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_data = n_target_data
        self.noise_scale = noise_scale
        self.expert_dim = dim_x
        self.total_dim = dim_x
        self.n_target_theta = n_target_theta
        self.expert_model = None
        self.expert_generator = expert_generator
        self._init_expert_model_from_batch = None

        # airfoil object cache: naca_str -> asb.Airfoil
        self._airfoil_cache: dict[str, "asb.Airfoil"] = {}

        # normalisation stats (mean, std)
        self._y_norm: tuple[float, float] = (0.0, 1.0)
        self._fb_norm: tuple[float, float] = (0.0, 1.0)

        if normalize_y:
            self._setup_normalizers(n_norm_samples)

    def _x_to_physical(self, x_norm: torch.Tensor) -> np.ndarray:
        """[B, N, 3] or [N, 3] normalised -> [B*N, 3] physical."""
        lo = self._BOUNDS[:, 0]
        hi = self._BOUNDS[:, 1]
        mid = (hi + lo) / 2.0
        half = (hi - lo) / 2.0
        flat = x_norm.detach().cpu().numpy().reshape(-1, x_norm.shape[-1])
        return (flat / self.design_scale) * half + mid  # [B*N, 3]

    def _naca_str(self, x_phys: np.ndarray) -> np.ndarray:
        """
        (alpha, thickness, camber) -> NACA 4-digit strings e.g. '2412'.
        Camber position fixed at 40% chord (middle digit = 4).
        """
        camber = np.clip(x_phys[:, 2], 0.00, 0.09)
        thickness = np.clip(x_phys[:, 1], 0.04, 0.40)
        c_d = np.round(camber * 100).astype(int).clip(0, 9)
        t_d = np.round(thickness * 100).astype(int).clip(4, 40)
        return np.array([f"{c}4{t:02d}" for c, t in zip(c_d, t_d)])

    def _get_airfoil(self, naca: str) -> "asb.Airfoil":
        """Return cached aerosandbox Airfoil for this NACA code."""
        if naca not in self._airfoil_cache:
            self._airfoil_cache[naca] = asb.Airfoil(f"naca{naca}")
        return self._airfoil_cache[naca]

    def _run_neuralfoil(self, x_phys: np.ndarray,
                    model_size: str = None) -> tuple[np.ndarray, np.ndarray]:
        model_size = model_size or self.MODEL_SIZE  # falls back to class default
        nacas = self._naca_str(x_phys)
        alphas = x_phys[:, 0]
        N = len(x_phys)
        CL_out = np.full(N, np.nan)
        CD_out = np.full(N, np.nan)

        for naca in np.unique(nacas):
            mask = nacas == naca
            try:
                af = self._get_airfoil(naca)
                out = nf.get_aero_from_airfoil(
                    af, alpha=alphas[mask], Re=self.RE,
                    model_size=model_size,
                )
                CL_out[mask] = np.atleast_1d(out["CL"]).ravel()
                CD_out[mask] = np.atleast_1d(out["CD"]).ravel()
            except Exception:
                pass

        return CL_out, CD_out
    
    @staticmethod
    def _fill_nan(arr: np.ndarray) -> np.ndarray:
        """Replace NaN with the minimum finite value (conservative penalty)."""
        finite = arr[np.isfinite(arr)]
        col_min = float(finite.min()) if len(finite) > 0 else 0.0
        return np.where(np.isfinite(arr), arr, col_min)

    def _eval_true(self, x_phys: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _eval_feedback(self, x_phys: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _setup_normalizers(self, n: int = 2000):
        rng = np.random.default_rng(0)
        lo, hi = self._BOUNDS[:, 0], self._BOUNDS[:, 1]
        x_phys = rng.uniform(lo, hi, (n, 3))

        y_raw = self._eval_true(x_phys)
        fb_raw = self._eval_feedback(x_phys)

        y_v = y_raw [np.isfinite(y_raw )]
        fb_v = fb_raw[np.isfinite(fb_raw)]

        self._y_norm = (float(y_v.mean()), max(float(y_v.std()), 1e-8))
        self._fb_norm = (float(fb_v.mean()), max(float(fb_v.std()), 1e-8))

        print(f"  [{type(self).__name__}] y  normalizer: "
              f"mean={self._y_norm[0]:.3f},  std={self._y_norm[1]:.3f}")
        print(f"  [{type(self).__name__}] fb normalizer: "
              f"mean={self._fb_norm[0]:.3f}, std={self._fb_norm[1]:.3f}")

    def _apply_norm(self, arr: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
        return (arr - stats[0]) / stats[1]

    def _forward(self, x: torch.Tensor, eval_fn, norm_stats: tuple) -> torch.Tensor:
        """[B, N, 3] -> [B, N, 1]"""
        B, N, _ = x.shape
        x_phys = self._x_to_physical(x)
        raw = eval_fn(x_phys)
        normed = self._apply_norm(raw, norm_stats)
        return (torch.tensor(normed, dtype=x.dtype, device=x.device)
                .reshape(B, N, 1))

    def _forward_true(self, x: torch.Tensor) -> torch.Tensor:
        t = self._forward(x, self._eval_true, self._y_norm)
        if self.noise_scale > 0:
            t = t + self.noise_scale * torch.randn_like(t)
        return t

    def _forward_feedback(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x, self._eval_feedback, self._fb_norm)

    def sample_data(self, batch_size: int, n_data: int, total_dim=None) -> torch.Tensor:
        device = getattr(self, 'device', 'cpu')
        d = total_dim if total_dim is not None else self.dim_x
        return (torch.rand(batch_size, n_data, d, device=device)
                * 2 * self.design_scale - self.design_scale)

    @torch.no_grad()
    def sample_batch(self, batch_size: int, mode: str = "data", **kwargs) -> AttrDict:
        n_ctx = self.n_context_init
        n_qry = self.n_query_init
        n_tgt = self.n_target_data

        x = self.sample_data(batch_size, n_ctx + n_qry + n_tgt)
        y = self._forward_true(x)
        feedback = self._forward_feedback(x)

        sl_ctx = slice(None, n_ctx)
        sl_qry = slice(n_ctx, n_ctx + n_qry)
        sl_tgt = slice(n_ctx + n_qry, n_ctx + n_qry + n_tgt)

        batch = AttrDict()

        batch.context_x = x[:, sl_ctx]; batch.context_y = y[:, sl_ctx]
        batch.query_x = x[:, sl_qry]; batch.query_y = y[:, sl_qry]
        batch.target_x = x[:, sl_tgt]; batch.target_y = y[:, sl_tgt]

        batch.y_hidden_context = feedback[:, sl_ctx]
        batch.y_hidden_query = feedback[:, sl_qry]
        batch.y_hidden_target = feedback[:, sl_tgt]

        batch.context_feedback = batch.y_hidden_context
        batch.query_feedback = batch.y_hidden_query
        batch.target_feedback = batch.y_hidden_target

        # empty expert slots (API compat)
        batch.expert_x = x[:, n_ctx + n_qry + n_tgt:]
        batch.expert_y = feedback[:, n_ctx + n_qry + n_tgt:]

        if mode == "mix":
            batch.target_theta = torch.zeros(
                batch_size, self.n_target_theta, 1, device=y.device)
            batch.target_all = torch.cat([batch.target_y, batch.target_theta], dim=1)
            batch.n_target_theta = self.n_target_theta
        else:  # "data"
            batch.target_all = batch.target_y
            batch.target_theta = None
            batch.n_target_theta = 0

        batch.expert_dim_x = self.expert_dim
        batch.dim_x = self.dim_x

        device = getattr(self, 'device', 'cpu')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        return batch

    def update_batch(self, batch: AttrDict, idx: torch.Tensor) -> AttrDict:
        """Move selected query point(s) into context. idx : [B, 1]"""
        idx_sq = idx.squeeze(-1)
        B = batch.query_x.size(0)
        bi = torch.arange(B, device=idx.device)

        next_x = batch.query_x [bi, idx_sq].unsqueeze(1)
        next_y = batch.query_y [bi, idx_sq].unsqueeze(1)
        next_y_hidden = batch.y_hidden_query[bi, idx_sq].unsqueeze(1)
        next_feedback = batch.query_feedback[bi, idx_sq].unsqueeze(1)

        batch.query_x = self.update_batch_query(batch.query_x, idx)
        batch.query_y = self.update_batch_query(batch.query_y, idx)
        batch.query_feedback = self.update_batch_query(batch.query_feedback, idx)
        batch.y_hidden_query = self.update_batch_query(batch.y_hidden_query, idx)

        batch.context_x = self.update_batch_context(batch.context_x, next_x)
        batch.context_y = self.update_batch_context(batch.context_y, next_y)
        batch.context_feedback = self.update_batch_context(batch.context_feedback, next_feedback)

        if self.expert_model is not None and getattr(self.expert_model, "adaptive", False):
            self.expert_model.add_data(next_x, next_y_hidden)
            with torch.no_grad():
                batch.query_feedback = self.expert_model.predict(
                    x=batch.query_x,
                    time_token=getattr(batch, "t", None),
                )

        return batch

    def init_expert_model_from_batch(self, batch, expert_dim_x, y_hidden, **kwargs):
        n_ctx = self.n_context_init
        n_qry = self.n_query_init
        n_tgt = self.n_target_data
        return (
            y_hidden[:, :n_ctx],
            y_hidden[:, n_ctx: n_ctx + n_qry],
            y_hidden[:, n_ctx + n_qry: n_ctx + n_qry + n_tgt],
        )


# Variant 1: Cl/Cd vs Cl
class AirfoilLiftDragTask(AirfoilBenchmarkBase):
    """
    True y   = Cl/Cd  (NeuralFoil)
    Feedback = Cl     (NeuralFoil - same high-fi, different output quantity)

    Why feedback is misleading:
      Cl climbs with alpha until hard stall. Cd stays low in a drag bucket
      then explodes near stall. So Cl/Cd peaks ~5-8 degrees earlier in alpha -
      an agent trusting Cl overshoots into the high-drag regime.
    """

    def __init__(self, **kwargs):
        if not NEURALFOIL_AVAILABLE:
            raise ImportError("pip install neuralfoil aerosandbox")
        super().__init__(**kwargs)

    def _eval_true(self, x_phys: np.ndarray) -> np.ndarray:
        CL, CD = self._run_neuralfoil(x_phys)
        return self._fill_nan(np.where(CD > 1e-6, CL / CD, np.nan))

    def _eval_feedback(self, x_phys: np.ndarray) -> np.ndarray:
        CL, _ = self._run_neuralfoil(x_phys)
        return self._fill_nan(CL)

    def __str__(self):
        return (f"AirfoilLiftDragTask(true=Cl/Cd, feedback=Cl | NeuralFoil, "
                f"n_ctx={self.n_context_init}, n_qry={self.n_query_init}, "
                f"n_tgt={self.n_target_data}, noise={self.noise_scale})")


class AirfoilNNFidelityTask(AirfoilBenchmarkBase):
    """
    True y   = Cl/Cd  (NeuralFoil large  - high-fidelity)
    Feedback = Cl/Cd  (NeuralFoil small  - low-fidelity NN)

    Unlike AirfoilMultiFidelityTask, both models share the same
    physics-informed architecture and stall model so the proxy
    optimum is likely at the same location, just noisier/less accurate.

    This represents the 'noisy same-objective' case instead of the
    'structurally misaligned proxy' case. This can be useful as a control/baseline
    to compare against the structurally biased variants.

    Available model sizes (increasing accuracy and cost):
        "xxsmall", "xsmall", "small", "medium", "large", "xlarge"
    """

    def __init__(self, feedback_model_size: str = "small", **kwargs):
        self.feedback_model_size = feedback_model_size
        if not NEURALFOIL_AVAILABLE:
            raise ImportError("pip install neuralfoil aerosandbox")
        super().__init__(**kwargs)

    def _eval_true(self, x_phys: np.ndarray) -> np.ndarray:
        CL, CD = self._run_neuralfoil(x_phys, model_size="large")
        return self._fill_nan(np.where(CD > 1e-6, CL / CD, np.nan))

    def _eval_feedback(self, x_phys: np.ndarray) -> np.ndarray:
        CL, CD = self._run_neuralfoil(x_phys, model_size=self.feedback_model_size)
        return self._fill_nan(np.where(CD > 1e-6, CL / CD, np.nan))

    def __str__(self):
        return (f"AirfoilNNFidelityTask("
                f"true=NF_large_Cl/Cd, feedback=NF_{self.feedback_model_size}_Cl/Cd, "
                f"n_ctx={self.n_context_init}, n_qry={self.n_query_init}, "
                f"n_tgt={self.n_target_data}, noise={self.noise_scale})")
