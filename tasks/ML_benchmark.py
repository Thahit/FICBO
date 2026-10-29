from __future__ import annotations
import re
import numpy as np
import torch
import pandas as pd
from attrdictionary import AttrDict
from tasks.base_task import Expert_Task


OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
OP2INT = {op: i for i, op in enumerate(OPS)}
N_OPS = 5
DIM_X = 3

_OP_LO = 0.0
_OP_HI = float(N_OPS - 1)

# 0->nor_conv_3x3, 3->skip_connect, 5->nor_conv_3x3
# I set to these becase they seem to be pretty good
# Free edges: 1, 2, 4  ->  5^3 = 125 architectures
_FIXED = {0: 3, 3: 1, 5: 3}
_FREE = [1, 2, 4]
_FREE_COLS = ['op1', 'op2', 'op4']


def _parse_arch(arch_str):
    tokens = re.findall(r'(\w+)~\d', arch_str)
    assert len(tokens) == 6
    return [OP2INT[t] for t in tokens]


def _load_pth_to_df(pth_path: str) -> pd.DataFrame:
    import torch as _torch
    _orig = _torch.load
    _torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
    try:
        from nas_201_api import NASBench201API as API
        print(f"  [NASBench201Task] Loading {pth_path} ...")
        api = API(pth_path, verbose=False)
    finally:
        _torch.load = _orig

    rows, skipped = [], 0
    for idx in range(len(api)):
        try:
            ops = _parse_arch(api[idx])
            if any(ops[e] != v for e, v in _FIXED.items()):
                continue
            c10 = api.get_more_info(idx, 'cifar10', None, hp='200', is_random=False)
            inet = api.get_more_info(idx, 'ImageNet16-120', None, hp='200', is_random=False)
            rows.append({
                'idx': idx,
                'op1': ops[1], 'op2': ops[2], 'op4': ops[4],
                'cifar10': float(c10['test-accuracy']),
                'imagenet': float(inet['test-accuracy']),
            })
        except Exception:
            skipped += 1

    print(f"  [NASBench201Task] Extracted {len(rows)} architectures (skipped {skipped})")
    return pd.DataFrame(rows)


def _build_batch(x, y, feedback, n_ctx, n_qry, n_tgt, n_theta, device):
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

    if n_theta > 0:
        zeros = torch.zeros(x.size(0), n_theta, 1, dtype=y.dtype, device=device)
        batch.target_theta = zeros
        batch.target_all = torch.cat([batch.target_y, zeros], dim=1)
    else:
        batch.target_theta = None
        batch.target_all = batch.target_y

    batch.n_target_theta = n_theta
    batch.expert_x = x[:, n_ctx + n_qry + n_tgt:]
    batch.expert_y = feedback[:, n_ctx + n_qry + n_tgt:]
    return batch


class NASBench201Task(Expert_Task):
    """
    NAS-Bench-201 3D benchmark.

    Search space:
    Edges 0, 3, 5 are fixed; free edges 1, 2, 4 each take 5 values,
    giving 125 unique architectures total.

    True objective  : ImageNet16-120 test accuracy
    Proxy/feedback  : CIFAR-10 test accuracy

    Sampling:
    Batches are drawn WITHOUT replacement from the 125 architectures.
    Each call to sample_batch draws n_ctx + n_qry + n_tgt unique rows
    per batch element by cycling through a per-batch-element permutation
    that is re-shuffled each full pass.

    Constraint: n_context_init + n_query_init + n_target_data <= 125.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        expert_generator = None,
        n_expert: int = 0,
        n_context_init: int = 5,
        n_query_init: int = 10,
        n_target_data: int = 5,
        n_target_theta: int = 0,
        dim_x: int = DIM_X,
        dim_y: int = 1,
        design_scale: float = 5.0,
        noise_scale: float = 0.0,
        normalize_y: bool = True,
        device: str = "cpu",
        **kwargs,
        ):
        Expert_Task.__init__(self, dim_x=dim_x, dim_y=dim_y,
                             design_scale=design_scale, **kwargs)

        required = _FREE_COLS + ['cifar10', 'imagenet']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        n_per = n_context_init + n_query_init + n_target_data
        if n_per > len(df):
            raise ValueError(
                f"n_context_init + n_query_init + n_target_data = {n_per} "
                f"exceeds number of architectures ({len(df)}). Reduce one of them."
            )

        self._df = df[required].reset_index(drop=True)
        self._n = len(self._df)
        self._expert_generator = expert_generator
        self.n_expert = n_expert
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_data = n_target_data
        self.n_target_theta = n_target_theta
        self.noise_scale = noise_scale
        self.expert_dim = dim_x
        self.total_dim = dim_x
        self.expert_model = None
        self.device = device
        self._y_norm = (0.0, 1.0)
        self._fb_norm = (0.0, 1.0)
        self._rng = np.random.default_rng(0)

        # Per-batch-element cyclic permutation state.
        # Initialised lazily in sample_batch once batch_size is known.
        self._perm: np.ndarray | None = None  # (B, n) permutation indices
        self._pos: np.ndarray | None = None  # (B,)   current position per element

        if normalize_y:
            y_all = self._df['imagenet'].values.astype(np.float32)
            fb_all = self._df['cifar10'].values.astype(np.float32)
            self._y_norm = (float(y_all.mean()), max(float(y_all.std()), 1e-8))
            self._fb_norm = (float(fb_all.mean()), max(float(fb_all.std()), 1e-8))

    @classmethod
    def from_pth(cls, pth_path: str, cache_csv: str = None, **kwargs) -> "NASBench201Task":
        df = _load_pth_to_df(pth_path)
        if cache_csv is not None:
            df.to_csv(cache_csv, index=False)
            print(f"  [NASBench201Task] Cached to {cache_csv}")
        return cls(df=df, **kwargs)

    @classmethod
    def from_csv(cls, csv_path: str, **kwargs) -> "NASBench201Task":
        df = pd.read_csv(csv_path)
        return cls(df=df, **kwargs)

    def _phys_to_norm(self, ops: np.ndarray) -> np.ndarray:
        mid, half = (_OP_LO + _OP_HI) / 2.0, (_OP_HI - _OP_LO) / 2.0
        return ((ops.astype(np.float32) - mid) / half * self.design_scale)

    def _norm_to_phys(self, x_norm: np.ndarray) -> np.ndarray:
        mid, half = (_OP_LO + _OP_HI) / 2.0, (_OP_HI - _OP_LO) / 2.0
        ops = (x_norm / self.design_scale) * half + mid
        return np.clip(np.round(ops), 0, N_OPS - 1).astype(int)

    def _init_perm(self, batch_size: int):
        """Initialise independent permutations for each batch element."""
        self._perm = np.stack([
            self._rng.permutation(self._n) for _ in range(batch_size)
        ])  # (B, n)
        self._pos = np.zeros(batch_size, dtype=int)

    def _draw_without_replacement(self, batch_size: int, n_per: int) -> np.ndarray:
        """
        Draw n_per unique indices per batch element, cycling through
        independent permutations.  Returns (B, n_per) index array.
        """
        if self._perm is None or self._perm.shape[0] != batch_size:
            self._init_perm(batch_size)

        out = np.empty((batch_size, n_per), dtype=int)
        for b in range(batch_size):
            needed = n_per
            filled = 0
            while filled < needed:
                remaining = self._n - self._pos[b]
                take = min(needed - filled, remaining)
                out[b, filled:filled + take] = \
                    self._perm[b, self._pos[b]: self._pos[b] + take]
                self._pos[b] += take
                filled += take
                if self._pos[b] >= self._n:  # exhausted, reshuffle
                    self._perm[b] = self._rng.permutation(self._n)
                    self._pos[b] = 0
        return out  # (B, n_per)

    def sample_data(self, batch_size, n_data, total_dim=None):
        """Used by base class internals - samples with replacement."""
        idx = self._rng.integers(0, self._n, size=batch_size * n_data)
        rows = self._df.iloc[idx]
        ops = rows[_FREE_COLS].values
        x_norm = self._phys_to_norm(ops).reshape(batch_size, n_data, DIM_X)
        return torch.tensor(x_norm, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def sample_batch(self, batch_size: int, mode: str = "data", **kwargs):
        n_ctx = self.n_context_init
        n_qry = self.n_query_init
        n_tgt = self.n_target_data
        n_exp = self.n_expert
        n_per = n_ctx + n_qry + n_tgt

        draw_idx = self._draw_without_replacement(batch_size, n_per)  # (B, n_per)

        ops = np.stack([self._df[_FREE_COLS].values[draw_idx[b]] for b in range(batch_size)])
        y_raw = np.stack([self._df['imagenet'].values[draw_idx[b]] for b in range(batch_size)])
        fb_raw = np.stack([self._df['cifar10'].values[draw_idx[b]] for b in range(batch_size)])

        x_norm = self._phys_to_norm(ops.reshape(-1, DIM_X)).reshape(batch_size, n_per, DIM_X)
        y_norm = ((y_raw - self._y_norm[0]) / self._y_norm[1]).astype(np.float32)
        fb_norm = ((fb_raw - self._fb_norm[0]) / self._fb_norm[1]).astype(np.float32)

        if self.noise_scale > 0:
            y_norm = y_norm + self._rng.normal(0, self.noise_scale, y_norm.shape).astype(np.float32)

        x = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_norm, dtype=torch.float32, device=self.device).unsqueeze(-1)
        fb = torch.tensor(fb_norm, dtype=torch.float32, device=self.device).unsqueeze(-1)

        batch = _build_batch(x, y, fb, n_ctx, n_qry, n_tgt,
                             self.n_target_theta, self.device)
        batch.dim_x = self.dim_x
        batch.expert_dim_x = self.expert_dim

        if n_exp > 0:
            exp_idx = self._rng.integers(0, self._n, size=batch_size * n_exp)
            exp_rows = self._df.iloc[exp_idx]
            exp_ops = exp_rows[_FREE_COLS].values.astype(np.float32)
            exp_fb_raw = exp_rows['cifar10'].values.astype(np.float32)
            exp_x_norm = self._phys_to_norm(exp_ops).reshape(batch_size, n_exp, DIM_X)
            exp_fb_norm = ((exp_fb_raw - self._fb_norm[0]) / self._fb_norm[1]).reshape(batch_size, n_exp)

            batch.expert_x = torch.tensor(exp_x_norm, dtype=torch.float32, device=self.device)
            batch.expert_y = torch.tensor(exp_fb_norm, dtype=torch.float32, device=self.device).unsqueeze(-1)

            if self._expert_generator is not None:
                self.expert_model = self._expert_generator(batch=batch)

        return batch

    def update_batch(self, batch, idx):
        idx_sq = idx.squeeze(-1)
        B = batch.query_x.size(0)
        bi = torch.arange(B, device=idx.device)

        next_x = batch.query_x [bi, idx_sq].unsqueeze(1)
        next_y = batch.query_y [bi, idx_sq].unsqueeze(1)
        next_fb = batch.query_feedback[bi, idx_sq].unsqueeze(1)

        batch.query_x = self.update_batch_query(batch.query_x, idx)
        batch.query_y = self.update_batch_query(batch.query_y, idx)
        batch.query_feedback = self.update_batch_query(batch.query_feedback, idx)
        batch.y_hidden_query = self.update_batch_query(batch.y_hidden_query, idx)

        batch.context_x = self.update_batch_context(batch.context_x, next_x)
        batch.context_y = self.update_batch_context(batch.context_y, next_y)
        batch.context_feedback = self.update_batch_context(batch.context_feedback, next_fb)

        if self.expert_model is not None:
            self.expert_model.add_data(
                next_x[..., -self.expert_dim:], next_fb
            )
            with torch.no_grad():
                batch.query_feedback = self.expert_model.predict(
                    x=batch.query_x,
                    time_token=getattr(batch, "t", None),
                )
        return batch

    def init_expert_model_from_batch(self, batch, expert_dim_x,
                                     y_hidden, **kwargs):
        n, q, t = self.n_context_init, self.n_query_init, self.n_target_data
        return (y_hidden[:, :n],
                y_hidden[:, n:n+q],
                y_hidden[:, n+q:n+q+t])

    def print_stats(self):
        y_all = self._df['imagenet'].values
        fb_all = self._df['cifar10'].values
        ti = int(np.argmax(y_all))
        pi = int(np.argmax(fb_all))
        bt, bp = self._df.iloc[ti], self._df.iloc[pi]
        spearman = float(np.corrcoef(
            y_all.argsort().argsort(), fb_all.argsort().argsort()
        )[0, 1])
        print(f"\n  NASBench201Task (3D subspace, 125 architectures)")
        print(f"  Fixed: edge0=nor_conv_3x3  edge3=skip_connect  edge5=nor_conv_3x3")
        print(f"  Free : edges 1, 2, 4  (5 ops each)")
        print(f"  n_ctx/qry/tgt : {self.n_context_init}/{self.n_query_init}/{self.n_target_data}")
        print(f"  n_theta       : {self.n_target_theta}")
        print(f"  True optimum  (ImageNet16-120):")
        print(f"    ops (1,2,4) : {[OPS[int(bt[c])] for c in _FREE_COLS]}")
        print(f"    imagenet    : {bt['imagenet']:.2f}%   cifar10: {bt['cifar10']:.2f}%")
        print(f"  Proxy optimum (CIFAR-10):")
        print(f"    ops (1,2,4) : {[OPS[int(bp[c])] for c in _FREE_COLS]}")
        print(f"    cifar10     : {bp['cifar10']:.2f}%   imagenet: {bp['imagenet']:.2f}%")
        print(f"  Same?         : {ti == pi}")
        print(f"  Spearman rho  : {spearman:.3f}\n")

    def __str__(self):
        return (
            f"NASBench201Task(n_arch={self._n}, dim=3, "
            f"n_ctx={self.n_context_init}, n_qry={self.n_query_init}, "
            f"n_tgt={self.n_target_data}, n_theta={self.n_target_theta})"
        )
