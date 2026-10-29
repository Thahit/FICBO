import torch
import numpy as np
from typing import Optional, Tuple
from attrdictionary import AttrDict
from tasks.base_task import Task, Expert_Task


# we have many more benchmarks here, which we did not test. many came from ALINE code
class BenchmarkMixin:
    """
    Pure logic mixin for benchmark functions. No nn.Module concerns.
    Provides the function registry, input scaling, parsing, normalization and sampling.
    Can be combined with Task or Expert_Task.

    Naming convention:
    Append 'n' to negate the output, e.g.:
        'branin2d'   -> raw Branin (minimisation landscape)
        'branin2dn'  -> negated Branin (maximisation landscape)
        'ackley4d'   -> raw Ackley
        'ackley4dn'  -> negated Ackley
    All functions are stored in their standard canonical (minimisation) form.
    """

    normalize_y: bool = False
    _y_mean: float = 0.0
    _y_std: float = 1.0

    _REACTOR_A1 = 1e10; _REACTOR_Ea1 = 60_000.0
    _REACTOR_A2 = 1e13; _REACTOR_Ea2 = 80_000.0
    _REACTOR_R = 8.314

    @property
    def _registry(self):
        return {
            'forrester': {
                'type': 'fixed', 'dim': 1,
                'domain': (0.0, 1.0),
                'func': self._forrester,
            },
            'forrester_fb': {
                'type': 'fixed', 'dim': 1,
                'domain': (0.0, 1.0),
                'func': self._forrester,
                'feedback_func': self._forrester_lf,
            },
            'branin': {
                'type': 'fixed', 'dim': 2,
                'domain': [(0.0, 1.0), (0.0, 1.0)],
                'func': self._branin,
            },
            'branin_fb': {
                'type': 'fixed', 'dim': 2,
                'domain': [(0.0, 1.0), (0.0, 1.0)],
                'func': self._branin,
                'feedback_func': self._branin_lf,
            },
            'gramacy1d': {
                'type': 'fixed', 'dim': 1,
                'domain': (0.5, 2.5),
                'func': self._gramacy1d,
            },
            'gramacy': {
                'type': 'fixed', 'dim': 2,
                'domain': [(-2.0, 6.0), (-2.0, 6.0)],
                'func': self._gramacy2d,
            },
            'higdon': {
                'type': 'fixed', 'dim': 1,
                'domain': (0.0, 20.0),
                'func': self._higdon,
            },
            'three_hump_camel': {
                'type': 'fixed', 'dim': 2,
                'domain': [(-2.0, 2.0), (-2.0, 2.0)],
                'func': self._three_hump_camel,
            },
            'holder_table': {
                'type': 'fixed', 'dim': 2,
                'domain': [(-10.0, 10.0), (-10.0, 10.0)],
                'func': self._holder_table,
            },
            'goldstein_price': {
                'type': 'fixed', 'dim': 2,
                'domain': [(-2.0, 2.0), (-2.0, 2.0)],
                'func': self._goldstein_price,
            },
            'ackley': {
                'type': 'any', 'min_dim': 1,
                'domain': (-5.0, 5.0),
                'func': self._ackley,
            },
            'rosenbrock': {
                'type': 'any', 'min_dim': 2,
                'domain': (-2.0, 2.0),
                'func': self._rosenbrock,
            },
            'rastrigin': {
                'type': 'any', 'min_dim': 1,
                'domain': (-5.12, 5.12),
                'func': self._rastrigin,
            },
            'levy': {
                'type': 'any', 'min_dim': 1,
                'domain': (-10.0, 10.0),
                'func': self._levy,
            },
            'sphere': {
                'type': 'any', 'min_dim': 1,
                'domain': (-5.12, 5.12),
                'func': self._sphere,
            },
            'griewank': {
                'type': 'any', 'min_dim': 1,
                'domain': (-600.0, 600.0),
                'func': self._griewank,
            },
            'schwefel': {
                'type': 'any', 'min_dim': 1,
                'domain': (-500.0, 500.0),
                'func': self._schwefel,
            },
            'dixon_price': {
                'type': 'any', 'min_dim': 2,
                'domain': (-10.0, 10.0),
                'func': self._dixon_price,
            },
            'reactor': {
                'type': 'fixed', 'dim': 2,
                'domain': [(300.0, 600.0), (0.01, 2.0)],
                'func': self._reactor_yield,
                'feedback_func': self._reactor_conversion,
            },
            'hartmann': {
                'type': 'fixed', 'dim': 3,
                'domain': [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                'func': self._hartmann3d,
            },
        }

    # fixed-dim functions - all in canonical minimisation form
    def _hartmann3d(self, x):
        # Canonical Hartmann-3  (global min ~ -3.863 at [0.1146, 0.5556, 0.8526])
        alpha = [1.0, 1.2, 3.0, 3.2]
        A = [[3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0]]
        P = [[0.36890, 0.11700, 0.26730],
            [0.46990, 0.43870, 0.74700],
            [0.10910, 0.87320, 0.55470],
            [0.03815, 0.57430, 0.88280]]

        result = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for i in range(4):
            inner = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
            for j in range(3):
                inner = inner + A[i][j] * (x[..., j] - P[i][j]) ** 2
            result = result + alpha[i] * torch.exp(-inner)

        return -result / 4.0  # scale to ~ [-1, 0]; minimum ~ -0.966

    def _forrester(self, x):
        return (torch.pow(6 * x[..., 0] - 2, 2) * torch.sin(12 * x[..., 0] - 4)) / 5

    def _forrester_lf(self, x):
        v = x[..., 0]
        f_high_raw = torch.pow(6 * v - 2, 2) * torch.sin(12 * v - 4)
        f_low_raw = 0.5 * f_high_raw + 10 * (v - 0.5) - 5
        return f_low_raw / 5

    def _branin(self, x):
        x1p = 15.0 * x[..., 0]
        x2p = 15.0 * x[..., 1]
        a, b, c, r, s, t = 1.0, 5.1/(4.0*np.pi**2), 5.0/np.pi, 6.0, 10.0, 1.0/(8.0*np.pi)
        term1 = a * (x2p - b*x1p**2 + c*x1p - r)**2
        term2 = s * (1.0 - t) * torch.cos(x1p)
        return (term1 + term2 + s - 44.81) / 51.95

    def _branin_lf(self, x, a=0.75):  # a in [0, 1], set by hand; see https://link.springer.com/article/10.1007/s00158-014-1209-5
        x1p = 15.0 * x[..., 0]
        x2p = 15.0 * x[..., 1]

        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8.0 * np.pi)

        term1 = (x2p - b * x1p**2 + c * x1p - r)**2
        term2 = s * (1.0 - t) * torch.cos(x1p)
        hf = term1 + term2 + s

        lf = hf - (a + 0.5) * term1
        # normalization constants below are for a=0.75; recorded mean/std by a:
        # 0.0: std=23.2223 | 0.25: mean=20.4082, std=12.8069 | 0.5: mean=10.3843, std=6.6569
        # 0.75: mean=0.4484, std=13.5236 | 1.0: mean=-9.4129, std=24.0372
        return (lf - 0.4484) / 13.5236

    def _gramacy1d(self, x):
        v = x[..., 0]
        return (torch.sin(10 * np.pi * v) / (2 * v) + (v - 1)**4) / 3

    def _gramacy2d(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        return x1 * torch.exp(-x1**2 - x2**2)

    def _higdon(self, x):
        v = x[..., 0]
        result = torch.zeros_like(v)
        m = v < 10
        result[m] = torch.sin(np.pi * v[m] / 5) + 0.2 * torch.cos(4 * np.pi * v[m] / 5)
        result[~m] = v[~m] / 10 - 1
        return result

    def _three_hump_camel(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        return 2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2

    def _holder_table(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        term1 = torch.sin(x1) * torch.cos(x2)
        term2 = torch.exp(torch.abs(1 - torch.sqrt(x1**2 + x2**2) / np.pi))
        return -torch.abs(term1 * term2)

    def _goldstein_price(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        p1 = 1 + (x1+x2+1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        p2 = 30 + (2*x1-3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        return (torch.log(p1 * p2) - 6) / 4

    # any-dim functions - all in canonical minimisation form

    def _ackley(self, x):
        d = x.shape[-1]
        sum_sq = (x**2).sum(dim=-1)
        sum_cos = torch.cos(2 * np.pi * x).sum(dim=-1)
        term1 = -20 * torch.exp(-0.2 * torch.sqrt(sum_sq / d))
        term2 = -torch.exp(sum_cos / d)
        return (term1 + term2 + 20 + np.e) / 5

    def _rosenbrock(self, x):
        xi = x[..., :-1]
        xi1 = x[..., 1:]
        raw = (100 * (xi1 - xi**2)**2 + (xi - 1)**2).sum(dim=-1)
        return (torch.log1p(raw) - 3) / 3

    def _rastrigin(self, x):
        d = x.shape[-1]
        raw = 10*d + (x**2 - 10*torch.cos(2*np.pi*x)).sum(dim=-1)
        return (raw / (10*d) - 0.5) * 2

    def _levy(self, x):
        w = 1 + (x - 1) / 4
        term1 = torch.sin(np.pi * w[..., 0])**2
        term_mid = ((w[..., :-1] - 1)**2 * (1 + 10*torch.sin(np.pi*w[..., :-1]+1)**2)).sum(dim=-1)
        term_end = (w[..., -1] - 1)**2 * (1 + torch.sin(2*np.pi*w[..., -1])**2)
        return (torch.log1p(term1 + term_mid + term_end) - 1) / 2

    def _sphere(self, x):
        d = x.shape[-1]
        raw = (x**2).sum(dim=-1)
        return (raw / (d * 5.12**2) * 2 - 1)

    def _griewank(self, x):
        sum_term = (x**2).sum(dim=-1) / 4000
        idx = torch.arange(1, x.shape[-1]+1, dtype=x.dtype, device=x.device)
        prod_term = torch.cos(x / idx.sqrt()).prod(dim=-1)
        return (sum_term - prod_term + 1)

    def _schwefel(self, x):
        d = x.shape[-1]
        raw = 418.9829*d - (x * torch.sin(torch.sqrt(torch.abs(x)))).sum(dim=-1)
        return ((raw / (418.9829*d)) * 2 - 1)

    def _dixon_price(self, x):
        term1 = (x[..., 0] - 1)**2
        i = torch.arange(2, x.shape[-1]+1, dtype=x.dtype, device=x.device)
        term_sum = (i * (2*x[..., 1:]**2 - x[..., :-1])**2).sum(dim=-1)
        return (torch.log1p(term1 + term_sum) - 2) / 3

    def _reactor_yield(self, x):
        T = x[..., 0]
        tau = x[..., 1]
        k1 = self._REACTOR_A1 * torch.exp(-self._REACTOR_Ea1 / (self._REACTOR_R * T))
        k2 = self._REACTOR_A2 * torch.exp(-self._REACTOR_Ea2 / (self._REACTOR_R * T))
        denom = k2 - k1 + 1e-12
        yield_B = (k1 / denom) * (torch.exp(-k1 * tau) - torch.exp(-k2 * tau))
        return yield_B.clamp(0.0, 1.0)

    def _reactor_conversion(self, x):
        T = x[..., 0]
        tau = x[..., 1]
        k1 = self._REACTOR_A1 * torch.exp(-self._REACTOR_Ea1 / (self._REACTOR_R * T))
        return (1.0 - torch.exp(-k1 * tau)).clamp(0.0, 1.0)

    def _parse_benchmark_name(self, benchmark_name: str):
        """
        Parse 'name[N]d[n]' into (base, dim, spec, negate).

        Examples:
        'branin2d'       -> base='branin',  dim=2, negate=False
        'branin2dn'      -> base='branin',  dim=2, negate=True
        'ackley4d'       -> base='ackley',  dim=4, negate=False
        'ackley4dn'      -> base='ackley',  dim=4, negate=True
        'forrester_fb1d' -> base='forrester_fb', dim=1, negate=False
        """
        import re
        match = re.fullmatch(r'([a-zA-Z_]+?)(\d+)d(n?)', benchmark_name)
        if match:
            base = match.group(1)
            requested_dim = int(match.group(2))
            negate = match.group(3) == 'n'
        else:
            # no dim suffix - try bare name, no negation
            negate = benchmark_name.endswith('n') and len(benchmark_name) > 1
            base = benchmark_name[:-1] if negate else benchmark_name
            requested_dim = None

        if base not in self._registry:
            raise ValueError(
                f"Unknown benchmark base name '{base}' (from '{benchmark_name}'). "
                f"Available: {list(self._registry.keys())}"
            )

        spec = self._registry[base]

        if requested_dim is None:
            requested_dim = spec['dim'] if spec['type'] == 'fixed' else spec['min_dim']

        if spec['type'] == 'fixed':
            if requested_dim != spec['dim']:
                raise ValueError(
                    f"'{base}' is fixed-dim={spec['dim']}, "
                    f"but '{benchmark_name}' requests dim={requested_dim}."
                )
        else:
            if requested_dim < spec['min_dim']:
                raise ValueError(
                    f"'{base}' requires dim >= {spec['min_dim']}, "
                    f"but '{benchmark_name}' requests dim={requested_dim}."
                )

        return base, requested_dim, spec, negate

    def _scale_input_to_domain(self, x_norm, spec: dict):
        domain = spec['domain']
        if isinstance(domain, tuple):
            lower, upper = domain
            half = (upper - lower) / 2
            mid = (upper + lower) / 2
            return (x_norm / self.design_scale) * half + mid
        else:
            x_scaled = torch.zeros_like(x_norm)
            for i, (lower, upper) in enumerate(domain):
                half = (upper - lower) / 2
                mid = (upper + lower) / 2
                x_scaled[..., i] = (x_norm[..., i] / self.design_scale) * half + mid
            return x_scaled

    def _setup_normalizer(self, benchmark_name: str, total_dim: int,
                          normalize_y: bool, n_samples: int = 10000):
        self.normalize_y = normalize_y
        self._y_mean = 0.0
        self._y_std = 1.0
        if not normalize_y:
            return
        _, _, spec, negate = self._parse_benchmark_name(benchmark_name)
        device = getattr(self, 'device', 'cpu')
        x = (torch.rand(1, n_samples, total_dim, device=device)
             * 2 * self.design_scale - self.design_scale)
        x_scaled = self._scale_input_to_domain(x, spec)
        y = spec['func'](x_scaled)
        if negate:
            y = -y
        if y.dim() < 3:
            y = y.unsqueeze(-1)
        self._y_mean = y.mean().item()
        self._y_std = y.std().item()
        print(f"  y normalizer for '{benchmark_name}': "
              f"mean={self._y_mean:.3f}, std={self._y_std:.3f}")

    def _normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        if not self.normalize_y:
            return y
        normed= (y - self._y_mean) / (self._y_std + 1e-8)
        #p95 = torch.quantile(normed, 0.9)
        #normed = torch.clamp(normed, max=p95)
        #normed = torch.sign(normed) * torch.log1p(normed.abs())
        return normed


    def sample_data(self, batch_size, n_data, total_dim=None):
        device = getattr(self, 'device', 'cpu')
        d = total_dim if total_dim is not None else self.dim_x
        return (torch.rand(batch_size, n_data, d, device=device)
                * 2 * self.design_scale - self.design_scale)

    def forward(self, xi, benchmark_name: str):
        _, _, spec, negate = self._parse_benchmark_name(benchmark_name)
        x_scaled = self._scale_input_to_domain(xi, spec)

        is_single = len(x_scaled.shape) == 2
        if is_single:
            x_scaled = x_scaled.unsqueeze(1)

        y = spec['func'](x_scaled)
        if negate:
            y = -y
        if len(y.shape) < 3:
            y = y.unsqueeze(-1)

        y = self._normalize_y(y)
        y = y + self.noise_scale * torch.randn_like(y)

        if is_single:
            return y.squeeze(1)
        return y

    def list_functions(self, print_output=True):
        lines = ["\n=== Fixed-dim functions ==="]
        for name, spec in self._registry.items():
            if spec['type'] == 'fixed':
                lines.append(f"  {name}  ->  '{name}{spec['dim']}d' or '{name}{spec['dim']}dn'")
        lines.append("=== Any-dim functions ===")
        for name, spec in self._registry.items():
            if spec['type'] == 'any':
                lines.append(
                    f"  {name}  ->  '{name}Nd' or '{name}Ndn'  (N >= {spec['min_dim']})"
                )
        if print_output:
            print('\n'.join(lines))
        return lines


class BenchmarkTask(BenchmarkMixin, Task):
    def __init__(
            self,
            benchmark_name: str = "forrester1d",
            dim_x: int = 1,
            dim_y: int = 1,
            n_context_init: int = 5,
            n_query_init: int = 10,
            n_target_data: int = 5,
            design_scale: float = 5.0,
            noise_scale: float = 0.1,
            normalize_y: bool = False,
            **kwargs
    ) -> None:
        Task.__init__(self, dim_x=dim_x, dim_y=dim_y,
                      design_scale=design_scale, **kwargs)

        self.benchmark_name = benchmark_name
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_data = n_target_data
        self.noise_scale = noise_scale

        self._setup_normalizer(benchmark_name, total_dim=dim_x, normalize_y=normalize_y)

    def sample_batch(self, batch_size, benchmark_name=None, mode="data"):
        benchmark_name = benchmark_name or self.benchmark_name
        self._parse_benchmark_name(benchmark_name)

        n_total = self.n_context_init + self.n_query_init + self.n_target_data
        x = self.sample_data(batch_size, n_total, total_dim=self.dim_x)
        y = self.forward(x, benchmark_name)

        batch = AttrDict()
        batch.context_x = x[:, :self.n_context_init]
        batch.context_y = y[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init: self.n_context_init + self.n_query_init]
        batch.query_y = y[:, self.n_context_init: self.n_context_init + self.n_query_init]
        batch.target_x = x[:, self.n_context_init + self.n_query_init:]
        batch.target_y = y[:, self.n_context_init + self.n_query_init:]

        if mode == "data":
            batch.target_all = batch.target_y
            batch.target_theta = None
            batch.n_target_theta = 0
        elif mode == "mix":
            batch.target_theta = torch.zeros(batch_size, self.n_target_theta, 1)
            batch.target_all = torch.cat([batch.target_y, batch.target_theta], dim=1)

        device = getattr(self, 'device', 'cpu')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        return batch

    def __str__(self):
        return (f"BenchmarkTask(benchmark={self.benchmark_name}, dim_x={self.dim_x}, "
                f"n_context={self.n_context_init}, n_query={self.n_query_init}, "
                f"n_target={self.n_target_data}, noise={self.noise_scale}, "
                f"normalize_y={self.normalize_y})")


class BenchmarkFeedbackTask(BenchmarkMixin, Expert_Task):
    def __init__(
            self,
            benchmark_name: str = "ackley2d",
            dim_x: int = 1,
            dim_y: int = 1,
            n_context_init: int = 5,
            n_query_init: int = 10,
            n_target_data: int = 5,
            design_scale: float = 5.0,
            noise_scale: float = 0.1,
            expert_dim: int = 1,
            n_marginal_samples: int = 100,
            expert_bias_model=None,
            use_expert_thetas: bool = False,
            concat_bias_generated_certainty: bool = False,
            embedding_type: str = "data",
            init_expert_model_from_batch=None,
            n_target_theta: int = 0,
            normalize_y: bool = False,
            expert_generator=None,
            truth_mode: str = "y_hidden",
            n_expert_extra_min: int = 0,
            n_expert_extra_max: int = 0,
            uniform_fraction: float = 1.0,
            expert_cluster_std_min: float = 0.5,
            expert_cluster_std_max: float = 1.5,
            expert_min_centers: int = 1,
            expert_max_centers: int = 5,
            fake_peaks: bool = False,
            fake_peak_k: int = 3,
            fake_peak_height_mult: float = 1.5,
            fake_peak_decay_range: Tuple[float, float] = (0.5, 2.0),
            **kwargs
    ) -> None:
        Expert_Task.__init__(self, dim_x=dim_x, dim_y=dim_y,
                             design_scale=design_scale, **kwargs)

        _, total_dim, _, _ = self._parse_benchmark_name(benchmark_name)

        assert expert_dim > 0
        assert expert_dim <= total_dim
        assert dim_x <= total_dim

        self.benchmark_name = benchmark_name
        self.total_dim = total_dim
        self.expert_dim = expert_dim
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_data = n_target_data
        self.noise_scale = noise_scale
        self.n_marginal_samples = n_marginal_samples
        self.expert_bias_model = expert_bias_model
        self.use_expert_thetas = use_expert_thetas
        self.concat_bias_generated_certainty = concat_bias_generated_certainty
        self.embedding_type = embedding_type
        self._init_expert_model_from_batch = init_expert_model_from_batch
        self.n_target_theta = n_target_theta
        self.expert_generator = expert_generator
        self.truth_mode = truth_mode
        self.n_expert_extra_min = n_expert_extra_min
        self.n_expert_extra_max = n_expert_extra_max
        self.expert_model = None
        self.uniform_fraction = uniform_fraction
        self.expert_cluster_std_min = expert_cluster_std_min
        self.expert_cluster_std_max = expert_cluster_std_max
        self.expert_min_centers = expert_min_centers
        self.expert_max_centers = expert_max_centers
        self.fake_peaks = fake_peaks
        self.fake_peak_k = fake_peak_k
        self.fake_peak_height_mult = fake_peak_height_mult
        self.fake_peak_decay_range = fake_peak_decay_range
        self._fake_peak_centers = None
        self._fake_peak_decay = None


        self._setup_normalizer(benchmark_name, total_dim=total_dim, normalize_y=normalize_y)
        self._y_true_global_max = (
            self._estimate_true_global_max(benchmark_name, total_dim) if fake_peaks else None
            )

    def _estimate_true_global_max(self, benchmark_name: str, total_dim: int, n_samples: int = 10000) -> float:
        _, _, spec, negate = self._parse_benchmark_name(benchmark_name)
        x = (torch.rand(1, n_samples, total_dim, device=self.device)
             * 2 * self.design_scale - self.design_scale)
        x_scaled = self._scale_input_to_domain(x, spec)
        with torch.no_grad():
            y = spec['func'](x_scaled)
            if negate:
                y = -y
            if y.dim() < 3:
                y = y.unsqueeze(-1)
            y = self._normalize_y(y)
        return y.max().item()

    def _sample_fake_peaks(self, batch_size: int) -> None:
        k = self.fake_peak_k
        self._fake_peak_centers = (
            torch.rand(batch_size, k, self.expert_dim, device=self.device) * 2 - 1
            ) * self.design_scale
        low, high = self.fake_peak_decay_range
        self._fake_peak_decay = (
            torch.rand(batch_size, 1, k, device=self.device) * (high - low) + low
            )

    def _apply_fake_peaks(
        self, y_hidden: torch.Tensor, x_expert_only: torch.Tensor, ref_max, env_idx: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        # Bump shape mirrors ExpertBiasEngine._local_distortion (exp(-dist*decay)), summed across the
        # k centers, scaled so each peak reaches fake_peak_height_mult * ref_max at its own center.
        # env_idx lets a caller with a sliced/reordered batch subset say which original environment(s)
        # it corresponds to, instead of silently broadcasting wrong against the full cached batch.
        if env_idx is not None:
            centers = self._fake_peak_centers[env_idx]
            decay = self._fake_peak_decay[env_idx]
        else:
            assert x_expert_only.shape[0] == self._fake_peak_centers.shape[0], (
                f"fake_peaks: x batch dim ({x_expert_only.shape[0]}) doesn't match the "
                f"cached peak batch dim ({self._fake_peak_centers.shape[0]}); pass env_idx "
                f"if this call intentionally uses a reduced/reordered batch."
                )
            centers, decay = self._fake_peak_centers, self._fake_peak_decay

        xi = x_expert_only.unsqueeze(2)  # [B,N,1,expert_dim]
        xj = centers.unsqueeze(1)  # [B,1,k,expert_dim]
        dist = torch.sqrt(((xi - xj) ** 2).sum(dim=-1) + 1e-8)  # [B,N,k]
        influence = torch.exp(-dist * decay)  # [B,N,k]
        height = self.fake_peak_height_mult * ref_max  # [B,1,1] or python float
        bump = (influence * height).sum(dim=2, keepdim=True)  # [B,N,1]
        return y_hidden + bump.expand_as(y_hidden)

    @torch.no_grad()
    def sample_expert_positions(self, batch_size, n_points, expert_dim):# from gp task
        K = int(torch.randint(self.expert_min_centers, self.expert_max_centers + 1, (1,)))
        centers = torch.rand(batch_size, K, expert_dim, device=self.device) * 2 * self.design_scale - self.design_scale
        cluster_std = torch.rand(batch_size, 1, 1, device=self.device) * (self.expert_cluster_std_max - self.expert_cluster_std_min) + self.expert_cluster_std_min

        if self.uniform_fraction > 0:
            n_uniform = int(n_points * self.uniform_fraction)
            n_clustered = n_points - n_uniform
        else:
            n_clustered = n_points
            n_uniform = 0

        if self.uniform_fraction == 1:
            x_expert = (torch.rand(batch_size, n_uniform, expert_dim, device=self.device) * 2 - 1) * self.design_scale
        else:
            weights = torch.ones(batch_size, K, device=self.device) / K
            cluster_idx = torch.multinomial(weights, n_clustered, replacement=True)
            selected_centers = centers.gather(1, cluster_idx.unsqueeze(-1).expand(-1, -1, expert_dim))
            x_clustered = selected_centers + torch.randn_like(selected_centers) * cluster_std

            if n_uniform > 0:
                x_uniform = (torch.rand(batch_size, n_uniform, expert_dim, device=self.device) * 2 - 1) * self.design_scale
                x_expert = torch.cat([x_clustered, x_uniform], dim=1)
            else:
                x_expert = x_clustered

        return x_expert.clamp(-self.design_scale, self.design_scale)
        
    def sample_data(self, batch_size, n_data, total_dim=None):
        device = getattr(self, 'device', 'cpu')
        d = total_dim if total_dim is not None else self.total_dim
        return (torch.rand(batch_size, n_data, d, device=device)
                * 2 * self.design_scale - self.design_scale)

    @torch.no_grad()
    def sample_data_sobol(
        self, batch_size: int, n_data: int, total_dim: Optional[int] = None, scramble: bool = True
        ) -> torch.Tensor:
        # Space-filling drop-in for sample_data (same box, device, dtype and [B, n_data, d] shape).
        # The Sobol net is drawn once and randomized per batch element with a Cranley-Patterson
        # toroidal shift: every batch element gets an independently randomized set while each one
        # stays equidistributed, and it vectorizes (a per-batch SobolEngine loop does not).
        device = getattr(self, 'device', 'cpu')
        d = total_dim if total_dim is not None else self.total_dim
        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=scramble)
        u = engine.draw(n_data).to(device)  # [n_data, d]
        if scramble:
            u = (u.unsqueeze(0) + torch.rand(batch_size, 1, d, device=device)) % 1.0
        else:
            u = u.unsqueeze(0).expand(batch_size, -1, -1)
        return u * 2 * self.design_scale - self.design_scale

    def _marginal_domain_samples(self, spec: dict, n: int) -> torch.Tensor:
        device = getattr(self, 'device', 'cpu')
        main_dim = self.total_dim - self.expert_dim
        domain = spec['domain']
        if isinstance(domain, tuple):
            lower, upper = domain
            return torch.rand(n, main_dim, device=device) * (upper - lower) + lower
        else:
            samples = torch.zeros(n, main_dim, device=device)
            for i in range(main_dim):
                lower, upper = domain[i]
                samples[:, i] = torch.rand(n, device=device) * (upper - lower) + lower
            return samples

    def _compute_y_hidden(self, x: torch.Tensor, benchmark_name: str) -> torch.Tensor:
        _, _, spec, negate = self._parse_benchmark_name(benchmark_name)

        if 'feedback_func' in spec:
            x_scaled = self._scale_input_to_domain(x, spec)
            y = spec['feedback_func'](x_scaled)
            if negate:
                y = -y
            if y.dim() < 3:
                y = y.unsqueeze(-1)
            return y

        B, N, _ = x.shape
        main_dim = self.total_dim - self.expert_dim
        x_expert = x[..., main_dim:]

        x_main_original = x[..., :main_dim].unsqueeze(2)  # [B, N, 1, main_dim]
        n_random = self.n_marginal_samples - 1
        if n_random > 0:
            x_main_random = self._marginal_domain_samples(spec, n_random).to(x.device)
            x_main_random = x_main_random.unsqueeze(0).unsqueeze(0).expand(B, N, n_random, main_dim)
            x_main_all = torch.cat([x_main_original, x_main_random], dim=2)  # [B, N, n_marginal, main_dim]
        else:
            x_main_all = x_main_original  # n_marginal_samples=1, just use original

        x_expert_exp = x_expert.unsqueeze(2).expand(B, N, self.n_marginal_samples, self.expert_dim)
        x_main_exp = x_main_all
        
        x_full = torch.cat([x_main_exp, x_expert_exp], dim=-1)

        x_flat = x_full.reshape(B * N * self.n_marginal_samples, self.total_dim).unsqueeze(1)
        x_scaled = self._scale_input_to_domain(x_flat, spec)
        y_flat = spec['func'](x_scaled).squeeze(1)

        if negate:
            y_flat = -y_flat
        if y_flat.dim() == 1:
            y_flat = y_flat.unsqueeze(-1)

        y_flat = self._normalize_y(y_flat)
        return y_flat.reshape(B, N, self.n_marginal_samples, 1).mean(dim=2)

    def _build_expert_batch(self, x_full, y_hidden, y_total, n_ctx, n_qry, n_tgt):
        main_dim = self.total_dim - self.expert_dim
        eb = AttrDict()
        eb.context_x = x_full[:, :n_ctx, main_dim:]
        eb.query_x = x_full[:, n_ctx: n_ctx + n_qry, main_dim:]
        eb.target_x = x_full[:, n_ctx + n_qry: n_ctx + n_qry + n_tgt, main_dim:]
        eb.expert_x = x_full[:, n_ctx + n_qry + n_tgt:, main_dim:]
        eb.context_y = y_hidden[:, :n_ctx]
        eb.query_y = y_hidden[:, n_ctx: n_ctx + n_qry]
        eb.target_y = y_hidden[:, n_ctx + n_qry: n_ctx + n_qry + n_tgt]
        eb.expert_y = y_hidden[:, n_ctx + n_qry + n_tgt:]
        eb.y_hidden_context = y_hidden[:, :n_ctx]
        eb.y_hidden_query = y_hidden[:, n_ctx: n_ctx + n_qry]
        eb.y_hidden_target = y_hidden[:, n_ctx + n_qry: n_ctx + n_qry + n_tgt]
        eb.expert_dim_x = self.expert_dim
        eb.dim_x = self.expert_dim
        eb.target_all = torch.zeros(x_full.shape[0], 0, 1, device=x_full.device, dtype=x_full.dtype)
        eb.target_theta = None
        eb.t = torch.tensor([0])
        return eb

    def init_expert_model_from_batch(self, batch, expert_dim_x, y_hidden, **kwargs):
        n_ctx = self.n_context_init
        n_qry = self.n_query_init
        n_tgt = self.n_target_data

        if self.expert_generator is not None:
            x_full = self._last_x_full
            expert_batch = self._build_expert_batch(x_full, y_hidden, None, n_ctx, n_qry, n_tgt)

            if self.expert_model is not None:
                del self.expert_model
                self.expert_model = None

            self.expert_model = self.expert_generator(batch=expert_batch)
            time_token = torch.tensor([0], device=self.device)
            context_feedback = self.expert_model.initial_predict(batch=expert_batch)

            gt_query = expert_batch.y_hidden_query if self.truth_mode == "y_hidden" else expert_batch.query_y
            gt_target = expert_batch.y_hidden_target if self.truth_mode == "y_hidden" else expert_batch.target_y

            with torch.no_grad():
                query_feedback = self.expert_model.predict(x=expert_batch.query_x, ground_truth_y=gt_query, time_token=time_token)
                target_feedback = self.expert_model.predict(x=expert_batch.target_x, ground_truth_y=gt_target, time_token=time_token)

            return context_feedback, query_feedback, target_feedback

        if self._init_expert_model_from_batch is not None:
            return self._init_expert_model_from_batch(batch, expert_dim_x=expert_dim_x, y_hidden=y_hidden, **kwargs)

        return (
            y_hidden[:, :n_ctx],
            y_hidden[:, n_ctx: n_ctx + n_qry],
            y_hidden[:, n_ctx + n_qry: n_ctx + n_qry + n_tgt],
        )

    def observe_at(
        self, x: torch.Tensor, benchmark_name: Optional[str] = None, time_token: Optional[torch.Tensor] = None,
        env_idx: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x is full-width (total_dim); the model only sees x[..., :dim_x], so callers refining a
        # candidate must keep x[..., dim_x:] fixed and splice in the refined visible part before
        # calling this. Returns (y_real, y_hidden, feedback), mirroring sample_batch()'s fields.
        # env_idx is only needed when fake_peaks is on and x's batch is a reduced/reordered subset of
        # the batch sample_batch() last ran with.
        benchmark_name = benchmark_name or self.benchmark_name
        main_dim = self.total_dim - self.expert_dim

        with torch.no_grad():
            y_real = self.forward(x, benchmark_name)
            y_hidden_truth = self._compute_y_hidden(x, benchmark_name)
            x_expert_only = x[..., main_dim:]

            if self.expert_bias_model is not None:
                y_hidden = self.expert_bias_model.apply(y=y_hidden_truth, x=x_expert_only)
            else:
                y_hidden = y_hidden_truth

            if self.fake_peaks:
                if self._fake_peak_centers is None:  # defensive: observe_at called before sample_batch
                    self._sample_fake_peaks(x.shape[0] if env_idx is None else int(env_idx.max()) + 1)
                y_hidden = self._apply_fake_peaks(
                    y_hidden, x_expert_only, self._y_true_global_max, env_idx=env_idx,
                    )

            if self.expert_model is None:
                if self._init_expert_model_from_batch is not None:
                    raise NotImplementedError(
                        "observe_at cannot re-query a custom init_expert_model_from_batch callback "
                        "at an arbitrary continuous x; only expert_generator-based experts (or no "
                        "expert at all) are supported"
                        )
                return y_real, y_hidden, y_hidden

            ground_truth_y = y_hidden if self.truth_mode == "y_hidden" else y_real
            if time_token is None:
                time_token = torch.tensor([0], device=x.device)

            feedback = self.expert_model.predict(x=x_expert_only, ground_truth_y=ground_truth_y, time_token=time_token)

            if self.concat_bias_generated_certainty and self.expert_bias_model is not None:
                uncertainty, _ = self.expert_bias_model.certainty_from_delta(y_hidden_truth, y_hidden)
                feedback = torch.cat([feedback, uncertainty], dim=2)

            return y_real, y_hidden, feedback

    def query_hidden_tail(self, dim_x: Optional[int] = None) -> torch.Tensor:
        # The x columns appart dim_x the model never sees, sliced from the full x sample_batch() last
        # drew (only ok for that same call's query pool). Pass dim_x explicitly if self.dim_x might
        # be temporarily mutated (e.g. _run_method's concat_feedback bump), which would otherwise silently slice past the end of the tensor.
        if dim_x is None:
            dim_x = self.dim_x
        n_ctx, n_qry = self.n_context_init, self.n_query_init
        return self._last_x_full[:, n_ctx: n_ctx + n_qry, dim_x:]

    def update_batch(self, batch, idx: torch.Tensor):
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
                batch.query_feedback = self.expert_model.predict(x=batch.query_x, time_token=getattr(batch, "t", None))

        return batch

    @torch.no_grad()
    def sample_batch(self, batch_size: int, benchmark_name: str = None, mode: str = "data"):
        benchmark_name = benchmark_name or self.benchmark_name
        _, total_dim, spec, _ = self._parse_benchmark_name(benchmark_name)

        assert self.expert_dim <= total_dim
        assert self.dim_x <= total_dim

        main_dim = total_dim - self.expert_dim
        n_ctx, n_qry, n_tgt = self.n_context_init, self.n_query_init, self.n_target_data
        n_total = n_ctx + n_qry + n_tgt

        x = self.sample_data(batch_size, n_total)
        y_total = self.forward(x, benchmark_name)

        # compute y_hidden for main points,  only marginalize if needed...
        if self.truth_mode == "y_hidden":
            y_hidden_truth = self._compute_y_hidden(x, benchmark_name)
        else:
            y_hidden_truth = y_total.clone()

        # append extra expert-only points when an expert model will be used
        if self.expert_generator is not None and self.n_expert_extra_max > 0:
            n_expert_extra = int(torch.randint(
                self.n_expert_extra_min, self.n_expert_extra_max + 1, (1,)
            ).item())
            if n_expert_extra > 0:
                x_extra = self.sample_expert_positions(batch_size, n_expert_extra, total_dim)

                if self.truth_mode == "y_hidden":
                    y_extra = self._compute_y_hidden(x_extra, benchmark_name)
                else:
                    y_extra = self.forward(x_extra, benchmark_name)

                x = torch.cat([x, x_extra], dim=1)
                y_hidden_truth = torch.cat([y_hidden_truth, y_extra], dim=1)
                # y_total only covers context/query/...; extra points are expert-only so we do not extend y_total here

        expert_thetas = None
        if self.expert_bias_model is not None:
            x_expert_only = x[..., main_dim:]
            if self.use_expert_thetas:
                y_hidden, expert_thetas = self.expert_bias_model.apply(y=y_hidden_truth, x=x_expert_only, return_params=True)
            else:
                y_hidden = self.expert_bias_model.apply(y=y_hidden_truth, x=x_expert_only)
        else:
            y_hidden = y_hidden_truth

        if self.fake_peaks:
            self._sample_fake_peaks(batch_size)
            ref_max = y_total.amax(dim=1, keepdim=True)  # true-y pool max: context+query+target
            y_hidden = self._apply_fake_peaks(y_hidden, x[..., main_dim:], ref_max)

        self._last_x_full = x

        batch = AttrDict()
        batch.context_x = x[:, :n_ctx, :self.dim_x]
        batch.context_y = y_total[:, :n_ctx]
        batch.query_x = x[:, n_ctx: n_ctx + n_qry, :self.dim_x]
        batch.query_y = y_total[:, n_ctx: n_ctx + n_qry]
        batch.target_x = x[:, n_ctx + n_qry: n_ctx + n_qry + n_tgt, :self.dim_x]
        batch.target_y = y_total[:, n_ctx + n_qry: n_ctx + n_qry + n_tgt]

        # everything after n_total lands in expert_x/y (empty if no extra points were added)
        batch.expert_x = x[:, n_ctx + n_qry + n_tgt:, main_dim:]
        batch.expert_y = y_hidden[:, n_ctx + n_qry + n_tgt:]

        batch.y_hidden_context = y_hidden[:, :n_ctx]
        batch.y_hidden_query = y_hidden[:, n_ctx: n_ctx + n_qry]
        batch.y_hidden_target = y_hidden[:, n_ctx + n_qry: n_ctx + n_qry + n_tgt]

        batch.expert_dim_x = self.expert_dim
        batch.dim_x = self.dim_x

        target_theta = torch.zeros(batch_size, self.n_target_theta, 1, device=x.device)
        if self.use_expert_thetas and expert_thetas is not None:
            target_theta = torch.cat([target_theta, expert_thetas], dim=1)

        batch.target_theta = target_theta if self.embedding_type in ["theta", "mix"] else None

        if self.embedding_type == "mix":
            batch.target_all = torch.cat([batch.target_y, target_theta], dim=1)
        elif self.embedding_type == "data":
            batch.target_all = batch.target_y
        else:
            batch.target_all = target_theta

        batch.context_feedback, batch.query_feedback, batch.target_feedback = \
            self.init_expert_model_from_batch(batch, expert_dim_x=self.expert_dim, y_hidden=y_hidden)
        batch.n_target_theta = self.n_target_theta

        if self.concat_bias_generated_certainty and self.expert_bias_model is not None:
            uncertainty_all, theta_for_uncertainty_noise = \
                self.expert_bias_model.certainty_from_delta(y_hidden_truth, y_hidden)
            batch.context_feedback = torch.cat([batch.context_feedback, uncertainty_all[:, :n_ctx]], dim=2)
            batch.query_feedback = torch.cat([batch.query_feedback, uncertainty_all[:, n_ctx: n_ctx + n_qry]], dim=2)
            batch.target_feedback = torch.cat([batch.target_feedback, uncertainty_all[:, n_ctx + n_qry: n_ctx + n_qry + n_tgt]], dim=2)
            if self.use_expert_thetas and batch.target_theta is not None:
                batch.target_theta = torch.cat([batch.target_theta, theta_for_uncertainty_noise], dim=1)

        device = getattr(self, 'device', 'cpu')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        return batch

    def __str__(self):
        expert_str = type(self.expert_model).__name__ if self.expert_model is not None else "raw_y_hidden"
        return (f"BenchmarkFeedbackTask(benchmark={self.benchmark_name}, "
                f"total_dim={self.total_dim}, dim_x={self.dim_x}, "
                f"expert_dim={self.expert_dim}, expert={expert_str}, "
                f"n_context={self.n_context_init}, n_query={self.n_query_init}, "
                f"n_target={self.n_target_data}, noise={self.noise_scale}, "
                f"normalize_y={self.normalize_y}, "
                f"n_marginal_samples={self.n_marginal_samples}, "
                f"fake_peaks={self.fake_peaks})")


def make_model_expert_generator(
    sklearn_model_or_cfg,
    fit_on=("context", "expert"),
    as_probs: bool = False,
    temperature: float = 1.0,
    n_jobs: int = -1,
    adaptive: bool = False,
    expert_noise: float = 0.0,
    normalize_outputs_to_max_one: bool = False,
    mode: str = "mean",
    kappa: float = 1.0,
    **kwargs,
    ):
    """
    Returns a callable(batch) -> ModelExpert, suitable for passing
    to BenchmarkFeedbackTask(expert_generator=...).

    sklearn_model_or_cfg : either an ExpertModel instance (e.g.
        SklearnExpertModel(RandomForestRegressor())) or a config object
        that build_sklearn_model() understands.

    Example:
    from sklearn.ensemble import RandomForestRegressor

    generator = make_model_expert_generator(
        SklearnExpertModel(RandomForestRegressor(n_estimators=50)),
        fit_on=["context"],
        adaptive=True,
    )
    task = BenchmarkFeedbackTask(
        benchmark_name="branin2d",
        dim_x=1,
        expert_dim=1,
        expert_generator=generator,
    )
    """
    # Accept either a ready ExpertModel or a raw sklearn config
    if isinstance(sklearn_model_or_cfg, ExpertModel):
        base_model = sklearn_model_or_cfg
    else:
        base_model = SklearnExpertModel(build_sklearn_model(sklearn_model_or_cfg))

    fit_on = list(fit_on)

    def _build(batch, **_ignored):
        return ModelExpert(
            model=base_model,
            batch=batch,
            fit_on=fit_on,
            as_probs=as_probs,
            temperature=temperature,
            n_jobs=n_jobs,
            adaptive=adaptive,
            expert_noise=expert_noise,
            normalize_outputs_to_max_one=normalize_outputs_to_max_one,
            mode=mode,
            kappa=kappa,
            **kwargs,
        )
    return _build
