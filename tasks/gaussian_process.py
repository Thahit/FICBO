import torch
import numpy as np
import torch.nn as nn
import torch.distributions as dist
from abc import ABC, abstractmethod
from attrdictionary import AttrDict
from tasks.base_task import Task, Expert_Task
from expert_models import AmortizedExpert, GroundTruthExpert, BaseBias, ModelExpert, build_sklearn_model, SklearnExpertModel, TimeEncodedExpertWrapper
from typing import Optional
from functools import partial
import math

# congratz for going this far into the repo
# this file is a bit of a mess, too full, depends too much on ALINEs code
# further extending without refactoring will be painful

def tensor_stats(name, x):
    if x is None:
        print(f"{name}: None")
        return

    if x.numel() == 0:
        print(f"{name}: shape={tuple(x.shape)}, empty tensor")
        return

    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()

    finite_mask = torch.isfinite(x)
    if finite_mask.any():
        finite_vals = x[finite_mask]
        min_val = finite_vals.min().item()
        max_val = finite_vals.max().item()
        mean_val = finite_vals.mean().item()
        var_val = finite_vals.var().item()
    else:
        min_val = "no finite values"
        max_val = "no finite values"

    print(
        f"{name}: "
        f"shape={tuple(x.shape)}, "
        f"min={min_val}, "
        f"max={max_val}, "
        f"has_nan={has_nan}, "
        f"has_inf={has_inf}, "
        f"mean={mean_val}, "
        f"var={var_val}"
    )


class GPTask(Task):
    """
    Gaussian process data generation for active learning with support for isotropic/anisotropic kernels

    Args:
        name (str): Name of the task
        dim_x (int): Dimension of input
        dim_y (int): Dimension of output
        embedding_type (str): Mode of the experiment: "data", "theta", or "mix"
        n_context_init (int): Number of initial context points
        n_query_init (int): Number of initial query points
        n_target_theta (int): Number of parameters: [lengthscale, variance]
        n_target_data (int): Number of target points (for data and mix modes)
        design_scale: Scale of the design space
        noise_scale: Noise scale
        p_iso: Probability of isotropic kernel
        kernel_weights: Weights for different kernel types
        lengthscale_lower: Lower bound for lengthscale
        lengthscale_upper: Upper bound for lengthscale
    """

    def __init__(
            self,
            name: str = "AL_mix",
            dim_x: int = 1,  # dimension of input
            dim_y: int = 1,  # dimension of output
            embedding_type="mix",  # mode of the experiment: "data", "theta", or "mix"
            n_context_init: int = 5,  # number of initial context points
            n_query_init: int = 10,  # number of initial query points
            n_target_theta: int = 2,  # number of parameters: [lengthscale, variance]
            n_target_data: int = 5,  # number of target points (for data and mix modes)
            design_scale = None,  # scale of the design space
            noise_scale: float = 0.01,  # noise scale
            p_iso: float = 0.5,  # probability of isotropic kernel
            kernel_weights = None,
            lengthscale_lower: float = 0.1,
            lengthscale_upper: float = 2.0,
            mineffective_dims: Optional[int] = None,

            **kwargs
    ) -> None:
        super().__init__(dim_x=dim_x, dim_y=dim_y)

        self.name = name
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_theta = n_target_theta
        self.n_target_data = n_target_data
        self.mineffective_dims = mineffective_dims if mineffective_dims is not None else dim_x
        assert self.mineffective_dims <= self.dim_x, "mineffective_dims must be less than or equal to dim_x"

        self.embedding_type = embedding_type
        self.jitter = 1e-5
        self.p_iso = p_iso  # probability of isotropic kernel
        self.kernel_weights = kernel_weights if kernel_weights is not None else [1/3, 0, 1/3, 1/3]
        self.kernel_types = ["rbf", "matern12", "matern32", "matern52"]

        if self.embedding_type in ["mix", "theta"]:
            if self.n_target_theta != dim_x + 1:
                raise ValueError("n_target_theta must be equal to dim_x + 1 for theta or mix embedding type")
        else:
            self.n_target_theta = 0

        # Define priors for GP hyperparameters
        # Lengthscale prior (now for each dimension)
        base_ls_factor = torch.sqrt(torch.tensor(dim_x, dtype=torch.float))
        self.lengthscale_lower = lengthscale_lower * base_ls_factor
        self.lengthscale_upper = lengthscale_upper * base_ls_factor

        # Variance
        self.scale_lower = 0.1
        self.scale_upper = 1.0

        self.noise_scale = noise_scale

        self.design_scale = torch.tensor(design_scale, device=self.device) if design_scale is not None else torch.tensor(5.0, device=self.device)

    @torch.no_grad()
    def sample_theta(self, batch_size):
        """Sample hyperparameters from the prior, supporting both isotropic and anisotropic kernels"""
        # Sample per-dimension lengthscales
        lengthscale_range = self.lengthscale_upper - self.lengthscale_lower
        length_scales = self.lengthscale_lower + lengthscale_range * torch.rand(batch_size, self.dim_x)

        # Determine if each batch uses isotropic kernel (same lengthscale for all dimensions)
        is_iso = torch.bernoulli(torch.ones(batch_size) * self.p_iso).bool()

        # If isotropic, set all lengthscales to be the same as the first dimension's lengthscale
        length_scales[is_iso] = length_scales[is_iso, 0].unsqueeze(1)

        # Sample output scale
        output_scale = self.scale_lower + (self.scale_upper - self.scale_lower) * torch.rand(batch_size)

        # Stack all parameters: [lengthscales, variance]
        theta = torch.cat([
            length_scales,  # [B, D]
            output_scale.unsqueeze(1),  # [B, 1]
        ], dim=1)

        return theta.unsqueeze(2)  # [B, D+1, 1]

    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """
        Sample input points from the domain

        Args:
            batch_size (int): Number of batches to generate
            n_data (int): Number of data points to generate per batch

        Returns:
            Input points [batch_size, n_data, dim_x]
        """
        # Simple uniform random sampling in the design space
        return torch.rand(batch_size, n_data, self.dim_x, device=self.device) * 2 * self.design_scale - self.design_scale
    
    @torch.no_grad()
    def sample_data_sobol(self, batch_size, n_data, scramble=True):
        """
        Generate points using the Sobol sequence within the design space.

        Args:
            batch_size (int): Number of batches to generate.
            n_data (int): Number of data points per batch.
            scramble (bool, optional): Whether to scramble the sequence.

        Returns:
            Tensor: Generated points in the design space, shape [batch_size, n_data, dim_x].
        """
        device = self.design_scale.device

        # Define lower and upper bounds for the design space
        lb = -self.design_scale * torch.ones(self.dim_x, device=device)
        ub = self.design_scale * torch.ones(self.dim_x, device=device)

        # Initialize result tensor
        points = torch.zeros(batch_size, n_data, self.dim_x, device=device)

        # Generate points for each batch
        for b in range(batch_size):
            # Create a Sobol engine for this batch
            soboleng = torch.quasirandom.SobolEngine(dimension=self.dim_x, scramble=scramble)

            # Draw points and scale to the design space
            batch_points = soboleng.draw(n_data)
            # Move batch_points to the correct device before operations
            batch_points = batch_points.to(device)
            batch_points = batch_points * (ub - lb) + lb

            # Store in result tensor
            points[b] = batch_points

            # Optional: Apply random permutation to introduce randomness while preserving uniformity
            if scramble:
                for d in range(self.dim_x):
                    perm_idx = torch.randperm(n_data, device=device)
                    points[b, :, d] = points[b, perm_idx, d]

        return points

    def to_design_space(self, xi):
        """
        Convert normalized design to actual input domain

        Args:
            xi (torch.Tensor): Normalized design points [batch_size, n_data, dim_x]

        Returns:
            Design points [batch_size, n_data, dim_x]
        """
        return xi * self.design_scale

    def normalise_outcomes(self, y):
        """
        Normalize outcomes if needed

        Args:
            y (torch.Tensor): Outcomes [batch_size, n_data, dim_y]

        Returns:
            Normalized outcomes [batch_size, n_data, dim_y]
        """

        return y
    
    @torch.no_grad()
    def rbf_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute RBF kernel between x1 and x2 with support for both isotropic and anisotropic kernels

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x] (can be all same for isotropic)
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting: [N, 1, dim_x] - [1, M, dim_x]
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate squared distance with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Sum across feature dimensions for RBF kernel
        sq_dist = weighted_sq_diff.sum(dim=-1)  # [N, M]

        # Apply RBF kernel formula
        kernel = scale * torch.exp(-0.5 * sq_dist)

        return kernel  # [N, M]

    @torch.no_grad()
    def matern12_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 1/2 kernel (exponential kernel) between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate weighted distances with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Compute Euclidean distance
        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        # Apply Matérn 1/2 kernel formula
        kernel = scale * torch.exp(-dist)

        return kernel  # [N, M]

    @torch.no_grad()
    def matern32_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 3/2 kernel between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate weighted distances with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Compute Euclidean distance
        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        # Apply Matérn 3/2 kernel formula
        sqrt3 = torch.sqrt(torch.tensor(3.0, device=x1.device))
        kernel = scale * (1 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)

        return kernel  # [N, M]

    @torch.no_grad()
    def matern52_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 5/2 kernel between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate weighted distances with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Compute Euclidean distance
        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        # Apply Matérn 5/2 kernel formula
        sqrt5 = torch.sqrt(torch.tensor(5.0, device=x1.device))
        kernel = scale * (1 + sqrt5 * dist + (5.0 / 3.0) * dist ** 2) * torch.exp(-sqrt5 * dist)

        return kernel  # [N, M]

    def compute_kernel_matrix(self, x1, x2, lengthscales, scale, kernel_type):
        """
        Compute kernel matrix based on the specified kernel type

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)
            kernel_type: one of "rbf", "matern12", "matern32", "matern52"

        Returns:
            kernel matrix [N, M]
        """
        if kernel_type == "rbf":
            return self.rbf_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern12":
            return self.matern12_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern32":
            return self.matern32_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern52":
            return self.matern52_kernel(x1, x2, lengthscales, scale)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def sample_kernel_type(self, batch_size):
        """
        Sample kernel types for each batch element based on weights

        Args:
            batch_size: Number of batches to generate kernel types for

        Returns:
            List of kernel types, one for each batch element
        """
        # Convert weights to PyTorch tensor and normalize
        weights = torch.tensor(self.kernel_weights, dtype=torch.float)
        weights = weights / weights.sum()

        # Sample indices based on weights
        indices = torch.multinomial(weights, batch_size, replacement=True)

        # Map indices to kernel types
        kernel_types = [self.kernel_types[idx] for idx in indices]

        return kernel_types

    def forward(self, xi, theta):
        """
        Generate observations from a GP

        Args:
            xi: normalized input locations [B, N, dim_x] or [B, dim_x]
            theta: GP hyperparameters [B, dim_x+2, 1]

        Returns:
            noisy observations [B, N, 1] or [B, 1]
        """
        # Convert to actual input domain
        x = self.to_design_space(xi)

        # Check if input is for single points per batch or multiple points
        is_single_point_per_batch = len(x.shape) == 2

        if is_single_point_per_batch:
            # Handle single point per batch case
            batch_size = x.shape[0]
            x = x.unsqueeze(1)  # [B, 1, dim_x]

            # Generate data
            y = self.generate_gp_data(x, theta)

            # Return with original shape
            return y.squeeze(1)
        else:
            # Handle multiple points per batch case
            return self.generate_gp_data(x, theta)

    def sample_batch(self, batch_size):
        batch = AttrDict()

        # Sample theta on correct device
        theta = self.sample_theta(batch_size).to(self.device)

        # Choose total n based on mode
        if self.embedding_type == "theta":
            n_total = self.n_context_init + self.n_query_init
        else:
            n_total = self.n_context_init + self.n_query_init + self.n_target_data

        # Sample input points on device
        x = self.sample_data(batch_size, n_total).to(self.device)

        # Generate observations
        y = self.generate_gp_data(x, theta)

        # Optional feature masking
        if self.embedding_type == "theta" and self.mineffective_dims < self.dim_x:
            feature_dim = torch.randint(
                self.mineffective_dims,
                self.dim_x,
                (x.shape[0],),
                device=self.device,
            )
            # mask out trailing dims
            for b in range(x.shape[0]):
                x[b, :, feature_dim[b]:] = 0.0

        # Splits
        c0 = self.n_context_init
        q0 = c0 + self.n_query_init

        batch.context_x = x[:, :c0]
        batch.context_y = y[:, :c0]
        batch.query_x = x[:, c0:q0]
        batch.query_y = y[:, c0:q0]

        if self.embedding_type == "theta":
            batch.target_x = None
            batch.target_y = None
            batch.target_theta = theta
            batch.target_all = theta

        elif self.embedding_type == "data":
            batch.target_x = x[:, q0:]
            batch.target_y = y[:, q0:]
            batch.target_theta = None
            batch.target_all = batch.target_y

        else:  # mix
            batch.target_x = x[:, q0:]
            batch.target_y = y[:, q0:]
            batch.target_theta = theta
            batch.target_all = torch.cat([batch.target_y, theta], dim=1)

        batch.n_target_theta = self.n_target_theta
        return batch

    @torch.no_grad()
    def generate_gp_data(self, x, theta):
        """
        Same behavior as your first version: device-safe sampling,
        use kernel types per batch element, proper jitter, etc.
        """
        batch_size, n_points, _ = x.shape

        lengthscales = theta[:, :self.dim_x, 0].to(self.device)
        scale = theta[:, self.dim_x, 0].to(self.device)

        kernel_types = self.sample_kernel_type(batch_size)
        self.last_kernel_types = kernel_types

        samples = torch.zeros(batch_size, n_points, 1, device=self.device)
        eye = torch.eye(n_points, device=self.device)

        for b in range(batch_size):
            K = self.compute_kernel_matrix(
                x[b], x[b], lengthscales[b], scale[b], kernel_types[b]
            )

            K = K + self.jitter * eye

            try:
                L = torch.linalg.cholesky(K)
                z = torch.randn(n_points, device=self.device)
                f = L @ z
            except:
                mvn = dist.MultivariateNormal(
                    loc=torch.zeros(n_points, device=self.device),
                    covariance_matrix=K
                )
                f = mvn.sample()

            samples[b, :, 0] = f + self.noise_scale * torch.randn(n_points, device=self.device)

        return samples

    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]
        return f"Active learning task with GP prior data({', '.join('{}={}'.format(key, val) for key, val in info.items())})"


class Expert_GPTask(Expert_Task, ABC):
    def __init__(
        self,
        name: str = "Expert_GPTask",
        dim_x: int = 1,
        dim_y: int = 1,
        expert_dim_x_min: int = 1,
        expert_dim_x_max: int = 1,
        embedding_type="mix",
        n_context_init: int = 5,
        n_query_init: int = 10,
        n_target_data: int = 5,
        n_target_theta: int = 2,
        use_expert_thetas: bool = False,
        design_scale=None,
        noise_scale: float = 0.01,
        p_iso: float = 0.5,
        kernel_weights=None,
        lengthscale_lower: float = 0.1,
        lengthscale_upper: float = 2.0,
        expert_extra_data_min: int = 10,
        expert_extra_data_max: int = 50,
        expert_theta_error : float = 0.0,
        uniform_fraction: float = 0.0,
        expert_cluster_std_min: float = 0.25,
        expert_cluster_std_max: float = 1.5,
        expert_min_centers: int = 3,
        expert_max_centers: int = 10,
        expert_noise: float = 0.0,
        dim_feedback: int = 1,
        expert_mode: str = "policy",
        kappa: float = 1.0,# for ucb
        adaptive_expert: bool = True,
        expert_returns_uncertainty: bool = False,
        expert_predict_epistemic_var: bool = False,
        min_overlap_dims: int = 0,
        max_overlap_dims: Optional[int] = None,
        normalize_outputs_to_max_one: bool = False,# this is the expert feedback
        provide_model_params: bool = False,
        expert_model_type: str = "amortized",
        expert_model: Optional[torch.nn.Module] = None,
        expert_bias_model: Optional[BaseBias] = None,# currently only works with child class
        concat_bias_generated_certainty: bool = False, # whether to concatenate the bias and generated uncertainty for the expert input
        model_expert_cfg: Optional[dict] = None, # config for model expert (e.g. sklearn model config)
        **kwargs
        ):
        super().__init__(dim_x=dim_x, dim_y=dim_y)
        self.name = name
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.expert_dim_x_min = expert_dim_x_min
        self.expert_dim_x_max = expert_dim_x_max
        self.dim_feedback = dim_feedback
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_data = n_target_data
        self.n_target_theta = n_target_theta
        self.use_expert_thetas = use_expert_thetas
        self.embedding_type = embedding_type
        self.jitter = 1e-5
        self.uniform_fraction = uniform_fraction
        self.expert_min_centers = expert_min_centers
        self.expert_max_centers = expert_max_centers
        self.expert_cluster_std_max = expert_cluster_std_max
        self.expert_cluster_std_min = expert_cluster_std_min
        self.p_iso = p_iso
        self.expert_model_type = expert_model_type
        self.kernel_weights = kernel_weights if kernel_weights is not None else [1/3, 0, 1/3, 1/3]
        self.kernel_types = ["rbf", "matern12", "matern32", "matern52"]
        if self.embedding_type in ["mix", "theta"]:
            #if self.n_target_theta != dim_x + self.expert_dim_x_max + 1:
            #    raise ValueError(f"n_target_theta ({self.n_target_theta}) must be equal to dim_x ({dim_x}) + 1 for theta or mix embedding type")
            pass
        else:
            self.n_target_theta = 0
        base_ls_factor = torch.sqrt(torch.tensor(dim_x, dtype=torch.float))
        self.lengthscale_lower = lengthscale_lower * base_ls_factor
        self.lengthscale_upper = lengthscale_upper * base_ls_factor
        self.scale_lower = 0.1
        self.scale_upper = 1.0
        self.noise_scale = noise_scale
        self.design_scale = torch.tensor(design_scale, device=self.device) if design_scale is not None else torch.tensor(5.0, device=self.device)
        self.expert_extra_data_min = expert_extra_data_min
        self.expert_extra_data_max = expert_extra_data_max
        self.expert_theta_error = expert_theta_error
        self.expert_model_batch = None
        self.provide_model_params = provide_model_params
        self.expert_noise = expert_noise
        self.min_overlap_dims = min_overlap_dims
        self.max_overlap_dims = max_overlap_dims if max_overlap_dims is not None else min(self.expert_dim_x_min, self.dim_x)
        assert min_overlap_dims <= self.expert_dim_x_min and min_overlap_dims <= self.dim_x, "min_overlap_dims must be less than or equal to expert_dim_x_min and dim_x"
        assert self.min_overlap_dims <= self.max_overlap_dims, "min_overlap_dims must be less than or equal to max_overlap_dims"
        print(f"Expert GP Task with expert input dimension in [{self.expert_dim_x_min}, {self.expert_dim_x_max}] and overlap in [{self.min_overlap_dims}, {self.max_overlap_dims}]")

        if expert_model_type == "truth":
            self.expert_generator = partial(GroundTruthExpert, mode=expert_mode, expert_noise=self.expert_noise, 
                    normalize_outputs_to_max_one=normalize_outputs_to_max_one,)
        elif expert_model_type == "amortized":
            self.expert_generator = partial(AmortizedExpert, amortized_model=expert_model, embedding_type=expert_model.embedder.embedding_type, 
                            expert_noise=self.expert_noise, normalize_outputs_to_max_one=normalize_outputs_to_max_one,
                            return_uncertainty=expert_returns_uncertainty, predict_epistemic_var=expert_predict_epistemic_var,
                            adaptive=adaptive_expert, mode=expert_mode, kappa=kappa)  # Placeholder for amortized expert model
        elif expert_model_type == "sklearn":
            sklearn_model = build_sklearn_model(model_expert_cfg.sklearn_model)
            self.expert_generator = partial(
                ModelExpert,
                model    = SklearnExpertModel(sklearn_model),
                fit_on   = model_expert_cfg.get("fit_on", ["expert"]),
                n_jobs   = -1,
                adaptive = adaptive_expert,
                normalize_outputs_to_max_one = normalize_outputs_to_max_one,
                mode     = expert_mode,
                kappa    = kappa,
            )
        else:
            raise ValueError(f"Unknown expert model type: {expert_model_type}")
        
        self.expert_bias_model = expert_bias_model # not used here currently
        self.concat_bias_generated_certainty = concat_bias_generated_certainty


    def compute_kernel_matrix(self, x1, x2, lengthscales, scale, kernel_type):
        """Still fully vectorized - the fastest part of the code."""
        if x1.dim() == 2: x1 = x1.unsqueeze(0)
        if x2.dim() == 2: x2 = x2.unsqueeze(0)
        B, N, _ = x1.shape
        M = x2.shape[1]

        if lengthscales.dim() == 1: lengthscales = lengthscales.unsqueeze(0)
        s_val = scale.view(-1, 1, 1) if torch.is_tensor(scale) else torch.full((B, 1, 1), scale, device=x1.device)

        x1_s = x1 / (lengthscales.unsqueeze(1) + 1e-8)
        x2_s = x2 / (lengthscales.unsqueeze(1) + 1e-8)
        
        dist_sq = torch.cdist(x1_s, x2_s, p=2)**2
        dist = torch.sqrt(dist_sq + 1e-12) 
        
        K = torch.zeros((B, N, M), device=x1.device)
        k_list = [kernel_type] * B if isinstance(kernel_type, str) else kernel_type

        unique_types = set(k_list)
        for kt in unique_types:
            mask = torch.tensor([t == kt for t in k_list], device=x1.device)
            s, d2, d = s_val[mask], dist_sq[mask], dist[mask]
            if kt == "rbf": K[mask] = s * torch.exp(-0.5 * d2)
            elif kt == "matern12": K[mask] = s * torch.exp(-d)
            elif kt == "matern32":
                s3 = 1.73205080757
                K[mask] = s * (1 + s3 * d) * torch.exp(-s3 * d)
            elif kt == "matern52":
                s5 = 2.2360679775
                K[mask] = s * (1 + s5 * d + (5/3) * d2) * torch.exp(-s5 * d)
        return K
    
    def sample_kernel_type(self, batch_size):
        """
        Sample kernel types for each batch element based on weights

        Args:
            batch_size: Number of batches to generate kernel types for

        Returns:
            List of kernel types, one for each batch element
        """
        # Convert weights to PyTorch tensor and normalize
        weights = torch.tensor(self.kernel_weights, dtype=torch.float)
        weights = weights / weights.sum()

        # Sample indices based on weights
        indices = torch.multinomial(weights, batch_size, replacement=True)

        # Map indices to kernel types
        kernel_types = [self.kernel_types[idx] for idx in indices]

        return kernel_types
    
    @torch.no_grad()
    def generate_dims(self):
        expert_dim = int(torch.randint(self.expert_dim_x_min, self.expert_dim_x_max + 1, (1,)))

        min_total = self.dim_x + expert_dim - self.max_overlap_dims
        max_total = self.dim_x + expert_dim - self.min_overlap_dims

        min_total = max(min_total, max(self.dim_x, expert_dim))

        total = int(torch.randint(min_total, max_total + 1, (1,)))

        return total, expert_dim

    @torch.no_grad()
    def sample_theta(self, batch_size, num_theta, lengthscale_lower=None, lengthscale_upper=None, p_iso=None):
        # use provided or default
        ls_lower = lengthscale_lower if lengthscale_lower is not None else self.lengthscale_lower
        ls_upper = lengthscale_upper if lengthscale_upper is not None else self.lengthscale_upper
        p_iso = p_iso if p_iso is not None else self.p_iso

        lengthscale_range = ls_upper - ls_lower
        length_scales = ls_lower + lengthscale_range * torch.rand(batch_size, num_theta)

        # isotropic main dims if p_iso > 0
        if p_iso > 0 and num_theta > 0:
            is_iso = torch.bernoulli(torch.ones(batch_size) * p_iso).bool()
            if is_iso.any():
                length_scales[is_iso] = length_scales[is_iso, 0].unsqueeze(1)

        output_scale = self.scale_lower + (self.scale_upper - self.scale_lower) * torch.rand(batch_size)
        theta = torch.cat([length_scales, output_scale.unsqueeze(1)], dim=1)

        return theta.unsqueeze(2)

    @torch.no_grad()
    def sample_data(self, batch_size, n_data, total_x_dim):
        return torch.rand(batch_size, n_data, total_x_dim, device=self.device) * 2 * self.design_scale - self.design_scale
    
    @torch.no_grad()
    def sample_expert_positions(self, batch_size, n_points, expert_dim):
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

    @torch.no_grad()
    def generate_gp_data(self, x, theta):
        batch_size, n_points, _ = x.shape
        eye = torch.eye(n_points, device=self.device)
        
        lengthscales = theta[:, :-1, 0].to(self.device)
        scale = theta[:, -1, 0].to(self.device)
        kernel_types = self.sample_kernel_type(batch_size)
        self.last_kernel_types = kernel_types

        K_batch = self.compute_kernel_matrix(x, x, lengthscales, scale, kernel_type=kernel_types)
        
        f = torch.zeros(batch_size, n_points, 1, device=self.device)
        for b in range(batch_size):
            Kb = K_batch[b] + self.jitter * eye
            success = False
            for multiplier in [1, 10, 100, 1000]:
                try:
                    Lb = torch.linalg.cholesky(Kb + (multiplier * self.jitter) * eye)
                    f[b] = Lb @ torch.randn(n_points, 1, device=self.device)
                    success = True
                    break
                except torch._C._LinAlgError:
                    continue
            
            if not success:
                print(f"Warning: all Cholesky attempts failed for b={b}, using diagonal fallback")
                var = torch.diagonal(Kb) + self.jitter
                print(f"  var min={var.min():.6f}, max={var.max():.6f}, has_neg={(var < 0).any()}")
                var = torch.clamp(var, min=1e-8)  # prevent NaN from sqrt of negative
                f[b] = torch.sqrt(var).unsqueeze(-1) * torch.randn(n_points, 1, device=self.device)

        noise = self.noise_scale * torch.randn(batch_size, n_points, 1, device=self.device)
        result = f + noise
        result = torch.nan_to_num(result, nan=0.0)# safety
        return result
        
    @abstractmethod
    def sample_batch(self, batch_size):
        ...


class WeightedMultiGP_Expert_GPTask(Expert_GPTask):
    """
    Child of Expert_GPTask where:
    - y_total = mean of K_real independent GPs (equal weights 1/K_real)
    - Expert feedback = weighted sum of all GPs (real + decoy) with Dirichlet weights
      where first component is always active, rest selected via Binomial
    - No expert model needed - feedback is directly the reweighted y_hidden
    - x_expert = x_total (full overlap enforced)
    - K_real: fixed number of real GP components  
    - K_decoy: sampled U(k_decoy_min, k_decoy_max) per batch
    """

    def __init__(
        self,
        k_real: int = 3,
        k_decoy_min: int = 0,
        k_decoy_max: int = 2,
        expert_weight_alpha: float = 0.9,
        expert_weight_p: float = 0.8,
        share_lengthscale_across_gps: bool = True,
        **kwargs
        ):
        super().__init__(**kwargs)

        #assert self.min_overlap_dims == self.dim_x, f"WeightedMultiGP requires min_overlap_dims == dim_x ({self.dim_x}), got {self.min_overlap_dims}. Main model only features are handled via expert weights, not input dimension."
        #assert self.expert_dim_x_min >= self.dim_x, f"WeightedMultiGP requires expert_dim_x_min >= dim_x ({self.dim_x})"
        #self.min_overlap_dims = self.dim_x
        #self.expert_dim_x_min = self.dim_x
        #print("overwwriting expert_dim_x_min to be equal to dim_x for full overlap in WeightedMultiGP")
        self.k_real = k_real
        self.k_decoy_min = k_decoy_min
        self.k_decoy_max = k_decoy_max
        self.expert_weight_alpha = expert_weight_alpha
        self.expert_weight_p = expert_weight_p
        self.expert_generator = None
        # True (legacy default): an isotropic draw leaks GP 0's lengthscale onto every other GP in
        # the batch row instead of sampling each GP independently. Set False for correct behavior.
        self.share_lengthscale_across_gps = share_lengthscale_across_gps

    def _sample_expert_weights(self, batch_size: int, k_total: int) -> torch.Tensor:
        """
        Sample expert weights for all K_real + K_decoy components.
        - First component always active
        - Remaining k_total-1 sampled via Binomial(p=expert_weight_p)
        - Active weights from symmetric Dirichlet(expert_weight_alpha)
        Shape: [B, k_total, 1]
        """
        rest_mask = torch.bernoulli(
            torch.full((batch_size, k_total - 1), self.expert_weight_p, device=self.device)
        )
        mask = torch.cat([
            torch.ones(batch_size, 1, device=self.device),
            rest_mask
        ], dim=1)  # [B, k_total]

        gammas = torch.distributions.Gamma(
            concentration=torch.full((batch_size, k_total), self.expert_weight_alpha, device=self.device),
            rate=torch.ones(batch_size, k_total, device=self.device)
        ).sample()

        gammas = gammas * mask
        w = gammas / (gammas.sum(dim=1, keepdim=True) + 1e-8) # = dirichlet

        w = w / (torch.sqrt((w ** 2).sum(dim=1, keepdim=True)) + 1e-8)

        return w.unsqueeze(-1)  # [B, k_total, 1]

    def _build_thetas(self, batch_size: int, n_gps: int, total_dim: int) -> torch.Tensor:
        """
        Sample and assemble stacked thetas for n_gps independent GPs.
        Returns [B, n_gps * (total_dim + 1), 1]
        """
        theta_size = total_dim + 1
        thetas = torch.zeros(batch_size, n_gps * theta_size, 1, device=self.device)

        if self.share_lengthscale_across_gps:
            # legacy behavior: a single sample_theta call across all n_gps GPs at once,
            # so the isotropic draw (when it fires) leaks GP 0's lengthscale onto every GP
            ls = self.sample_theta(batch_size, n_gps * total_dim).to(self.device)  # [B, n_gps*total_dim(+1), 1]
            for k in range(n_gps):
                scale = self.scale_lower + (self.scale_upper - self.scale_lower) * \
                        torch.rand(batch_size, 1, 1, device=self.device)
                thetas[:, k * theta_size: (k + 1) * theta_size - 1, :] = ls[:, k * total_dim: (k + 1) * total_dim, :]
                thetas[:, (k + 1) * theta_size - 1: (k + 1) * theta_size, :] = scale
        else:
            # each GP gets its own independent sample_theta call, so the isotropic draw
            # only ever applies within that GP's own dimensions, never across GPs
            for k in range(n_gps):
                ls_k = self.sample_theta(batch_size, total_dim).to(self.device)[:, :total_dim, :]  # [B, total_dim, 1]
                scale = self.scale_lower + (self.scale_upper - self.scale_lower) * \
                        torch.rand(batch_size, 1, 1, device=self.device)
                thetas[:, k * theta_size: (k + 1) * theta_size - 1, :] = ls_k
                thetas[:, (k + 1) * theta_size - 1: (k + 1) * theta_size, :] = scale
        return thetas

    def _sample_gps(self, x: torch.Tensor, thetas: torch.Tensor, n_gps: int) -> list:
        """
        Sample n_gps independent GPs from stacked thetas.
        Returns list of [B, N, 1] tensors.
        """
        batch_size, n_points, total_dim = x.shape
        theta_size = total_dim + 1
        eye = torch.eye(n_points, device=self.device)
        gps = []

        for k in range(n_gps):
            theta_k = thetas[:, k * theta_size: (k + 1) * theta_size, :]
            ls    = theta_k[:, :total_dim, 0]   # [B, total_dim]
            scale = theta_k[:, total_dim, 0]    # [B]

            K = self.compute_kernel_matrix(x, x, ls, scale, kernel_type=self.last_kernel_types)
            f = torch.zeros(batch_size, n_points, 1, device=self.device)

            for b in range(batch_size):
                Kb = K[b] + self.jitter * eye
                for mult in [1, 10, 100, 1000]:
                    try:
                        Lb = torch.linalg.cholesky(Kb + mult * self.jitter * eye)
                        f[b] = Lb @ torch.randn(n_points, 1, device=self.device)
                        break
                    except torch._C._LinAlgError:
                        continue
                else:
                    var = torch.diagonal(Kb) + self.jitter
                    f[b] = torch.sqrt(var).unsqueeze(-1) * torch.randn(n_points, 1, device=self.device)

            f += self.noise_scale * torch.randn_like(f)
            f = torch.nan_to_num(f, nan=0.0)
            gps.append(f)

        return gps

    @torch.no_grad()
    def generate_gp_data(self, x, thetas_real, thetas_decoy):
        """
        Returns:
            y_total  : [B, N, 1] - equal-weight mean of real GPs
            y_expert : [B, N, 1] - expert's biased weighted sum over all GPs
            w_expert : [B, k_total, 1]
        """
        batch_size, n_points, total_dim = x.shape
        k_decoy = thetas_decoy.shape[1] // (total_dim + 1) if thetas_decoy.shape[1] > 0 else 0
        k_total = self.k_real + k_decoy

        self.last_kernel_types = self.sample_kernel_type(batch_size)

        real_gps  = self._sample_gps(x, thetas_real, self.k_real)
        decoy_gps = self._sample_gps(x, thetas_decoy, k_decoy) if k_decoy > 0 else []

        # y_total: equal weight mean of real GPs only
        y_total = torch.stack(real_gps, dim=0).mean(dim=0)*math.sqrt(self.k_real)  # [B, N, 1], rescale for same variance

        # expert feedback: Dirichlet-weighted sum over all GPs
        w_expert = self._sample_expert_weights(batch_size, k_total)   # [B, k_total, 1]
        all_gps  = torch.stack(real_gps + decoy_gps, dim=1)           # [B, k_total, N, 1]
        y_expert = (w_expert.unsqueeze(2) * all_gps).sum(dim=1)       # [B, N, 1]
        # maybe change?
        
        return y_total, y_expert, w_expert

    @torch.no_grad()
    def sample_batch(self, batch_size):
        batch = AttrDict()
        total_dim, expert_dim = self.generate_dims()

        #assert expert_dim >= self.dim_x,  f"expert_dim ({expert_dim}) must be >= dim_x ({self.dim_x})"

        k_decoy = int(torch.randint(self.k_decoy_min, self.k_decoy_max + 1, (1,)))

        thetas_real  = self._build_thetas(batch_size, self.k_real, total_dim)
        thetas_decoy = self._build_thetas(batch_size, k_decoy, total_dim) if k_decoy > 0 \
                       else torch.empty(batch_size, 0, 1, device=self.device)

        n_expert_extra = int(torch.randint(self.expert_extra_data_min, self.expert_extra_data_max + 1, (1,)))
        n_total = self.n_context_init + self.n_query_init + self.n_target_data
        x       = self.sample_data(batch_size, n_total, total_dim).to(self.device)
        #x_extra = self.sample_expert_positions(batch_size, n_expert_extra, total_dim).to(self.device)
        #x       = torch.cat([x, x_extra], dim=1)

        y_total, y_expert, w_expert = self.generate_gp_data(x, thetas_real, thetas_decoy)

        # optionally apply bias model
        if self.expert_bias_model is not None:
            if self.use_expert_thetas:
                y_expert, expert_thetas = self.expert_bias_model.apply(
                    y=y_expert, x=x[:, :, :expert_dim], return_params=True
                )
            else:
                y_expert = self.expert_bias_model.apply(y=y_expert, x=x[:, :, :expert_dim])
        elif self.use_expert_thetas:
            expert_thetas = torch.empty(batch_size, 0, 1, device=self.device)

        nc = self.n_context_init
        nq = self.n_query_init
        nt = self.n_target_data

        batch.context_x = x[:, :nc, :self.dim_x]
        batch.context_y = y_total[:, :nc]
        batch.query_x   = x[:, nc: nc + nq, :self.dim_x]
        batch.query_y   = y_total[:, nc: nc + nq]
        batch.target_x  = x[:, nc + nq: nc + nq + nt, :self.dim_x]
        batch.target_y  = y_total[:, nc + nq: nc + nq + nt]

        #batch.expert_x  = x[:, nc + nq + nt:, :expert_dim]
        #batch.expert_y  = y_expert[:, nc + nq + nt:]

        # feedback is directly the expert's weighted sum - no model needed
        batch.context_feedback = y_expert[:, :nc]
        batch.query_feedback   = y_expert[:, nc: nc + nq]
        batch.target_feedback  = y_expert[:, nc + nq: nc + nq + nt]

        batch.y_hidden_query   = y_expert[:, nc: nc + nq]
        batch.y_hidden_context = y_expert[:, :nc]
        batch.y_hidden_target  = y_expert[:, nc + nq: nc + nq + nt]

        batch.expert_dim_x = expert_dim
        batch.dim_x        = self.dim_x

        # theta for model: lengthscales of first real GP + its output scale
        theta_size = total_dim + 1
        theta_for_model = torch.cat([
            thetas_real[:, :self.dim_x, :],
            thetas_real[:, theta_size - 1: theta_size, :],
        ], dim=1)

        if self.use_expert_thetas:
            theta_for_model = torch.cat([theta_for_model, w_expert], dim=1)
            if self.expert_bias_model is not None:
                theta_for_model = torch.cat([theta_for_model, expert_thetas], dim=1)

        if self.embedding_type in ["theta", "mix"]:
            batch.target_theta = theta_for_model
        else:
            batch.target_theta = None

        if self.embedding_type == "mix":
            batch.target_all = torch.cat([batch.target_y, batch.target_theta], dim=1)
        elif self.embedding_type == "data":
            batch.target_all = batch.target_y
        else:
            batch.target_all = theta_for_model

        batch.n_target_theta = self.n_target_theta

        return batch
    
    def update_batch(self, batch, idx):
        """ Update the batch of data
        
        Args:
            batch: Current batch
            idx: Indices of selected query points
            
        Returns:
            updated_batch: Updated batch with selected query points moved to context
        """

        # get the next data points
        idx_squeezed = idx.squeeze(-1)  # [B, 1] -> [B]
        B = batch.query_x.size(0)
        batch_indices = torch.arange(B, device=idx.device)# could even precompute
        next_x        = batch.query_x[batch_indices, idx_squeezed].unsqueeze(1)       # [B, 1, dim_x]
        next_y        = batch.query_y[batch_indices, idx_squeezed].unsqueeze(1)       # [B, 1, dim_y]
        #next_y_simple = batch.y_hidden_query[batch_indices, idx_squeezed].unsqueeze(1) # [B, 1, dim_y]
        next_feedback = batch.query_feedback[batch_indices, idx_squeezed].unsqueeze(1) # [B, 1, dim_feedback]

        # remove from query set
        batch.query_x = self.update_batch_query(batch.query_x, idx)
        batch.query_y = self.update_batch_query(batch.query_y, idx)
        batch.query_feedback = self.update_batch_query(batch.query_feedback, idx)
        batch.y_hidden_query = self.update_batch_query(batch.y_hidden_query, idx)

        batch.context_x = self.update_batch_context(batch.context_x, next_x)
        batch.context_y = self.update_batch_context(batch.context_y, next_y)
        batch.context_feedback = self.update_batch_context(batch.context_feedback, next_feedback)

        if getattr(self.expert_model, "adaptive", False):# otherwise safe the compute
            with torch.no_grad():
                self.expert_model.add_data(next_x, next_y if self.truth_mode == "y" else next_feedback, time_token=getattr(batch, "t", None))
                batch.expert_eval_query = self.expert_model.predict(batch.query_x, time_token=getattr(batch, "t", None))

        return batch


class Additive_Expert_GPTask(Expert_GPTask):
    def __init__(
        self,
        lengthscale_lower_hidden: float = 0.8,
        lengthscale_upper_hidden: float = 3.0,
        expert_weight_min: float = 0.2,
        expert_weight_max: float = 0.8,
        expert_truth_mode: str = "y_hidden", # "y", "y_hidden"
        time_encoded_feedback: bool = False,
        **kwargs
        ):
        super().__init__(**kwargs)
        # hidden/expert GP lengthscales
        self.lengthscale_lower_hidden = lengthscale_lower_hidden
        self.lengthscale_upper_hidden = lengthscale_upper_hidden
        # expert weighting
        self.expert_weight_min = expert_weight_min
        self.expert_weight_max = expert_weight_max

        self.expert_generator = partial(self.expert_generator, truth_mode=expert_truth_mode)
        self.truth_mode = expert_truth_mode
        self.time_encoded_feedback = time_encoded_feedback

    @torch.no_grad()
    def generate_gp_data(self, x, theta_main, theta_hidden, dim_of_main_gp=1):
        batch_size, n_points, _ = x.shape
        eye = torch.eye(n_points, device=self.device)

        w_expert = self.expert_weight_min + (self.expert_weight_max - self.expert_weight_min) * \
                   torch.rand([batch_size, 1, 1], device=self.device)

        if theta_main is not None and theta_main.shape[1] > 0:
            f_main = super().generate_gp_data(x[:,:,:dim_of_main_gp], theta_main)
        else:# no main only stuff
            f_main = torch.zeros(batch_size, n_points, 1, device=self.device)
            self.last_kernel_types = self.sample_kernel_type(batch_size)
            w_expert = torch.ones_like(w_expert)

        num_hidden_dims = theta_hidden.shape[1] - 1
        x_hidden = x[:, :, -num_hidden_dims:] 
        h_ls = theta_hidden[:, :-1, 0].to(self.device)
        h_s = theta_hidden[:, -1, 0].to(self.device)

        K_batch_h = self.compute_kernel_matrix(x_hidden, x_hidden, h_ls, h_s, kernel_type=self.last_kernel_types)
        
        f_hidden = torch.zeros(batch_size, n_points, 1, device=self.device)
        for b in range(batch_size):# vectorizing this breaks the numerics
            Kb_h = K_batch_h[b] + self.jitter * eye
            success = False
            for multiplier in [1, 10, 100, 1000]:
                try:
                    Lb = torch.linalg.cholesky(Kb_h + (multiplier * self.jitter) * eye)
                    f_hidden[b] = Lb @ torch.randn(n_points, 1, device=self.device)
                    if torch.isnan(f_hidden[b]).any():
                        print(f"b={b}, mult={multiplier}: NaN after matmul, Lb max={Lb.abs().max():.4f}, Kb max={Kb_h.max():.4f}, scale={h_s[b]:.4f}, ls={h_ls[b]}")
                        success = False  # treat as failure, try next multiplier
                        continue
                    success = True
                    break

                except torch._C._LinAlgError:
                    continue
            
            if not success:
                # Diagonal fallback for hidden GP
                var_h = torch.diagonal(Kb_h) + self.jitter
                var_h = torch.clamp(var_h, min=1e-8)
                f_hidden[b] = torch.sqrt(var_h).unsqueeze(-1) * torch.randn(n_points, 1, device=self.device)
                print("Warning: Fallback to diagonal sampling for hidden GP")
            
            f_hidden[b] += self.noise_scale * torch.randn(n_points, 1, device=self.device)

        f_hidden = torch.nan_to_num(f_hidden, nan=0.0)# there is some rare numerical issues, and ttrying to fix has been unsucessful
        f_main = torch.nan_to_num(f_main, nan=0.0)

        scale = 1.0 / torch.sqrt((1 - w_expert)**2 + w_expert**2)
        y_total = ((1 - w_expert) * f_main + w_expert * f_hidden) * scale # rescale to same var as other
        return y_total, f_hidden, w_expert
    
    def init_expert_model_from_batch(self, batch, expert_dim_x, y_hidden, thetas=None):
        """
        Prepare expert batch so that existing initial_predict works unchanged.
        """
        batch_size, n_points, _ = batch.expert_x.shape

        # Optionally set fixed hyperparams from expert portion of theta
        if self.provide_model_params and thetas is not None:
            theta_hidden = thetas[:, -expert_dim_x-1:, :]  # last dims + output scale
            theta_hidden = theta_hidden + self.expert_theta_error * torch.randn_like(theta_hidden)

            fixed_hyperparams = [#useless
                {
                    "kernel_type": self.last_kernel_types[i],
                    "lengthscale": theta_hidden[i, :-1, 0],
                    "outputscale": theta_hidden[i, -1, 0],
                    "noise": self.noise_scale
                }
                for i in range(batch_size)
            ]
        else:
            fixed_hyperparams = None

        # Build a batch “view” for the expert so initial_predict works
        #expert_batch  = copy.deepcopy(batch)#memory issues
        expert_batch = AttrDict()
        expert_batch.update(batch)  # shallow copy, no tensor duplication, safes some mem.

        ctx_len = batch.context_x.shape[1]
        qry_len = batch.query_x.shape[1]
        tgt_len = batch.target_x.shape[1]
        exp_len = batch.expert_x.shape[1]
        #if self.truth_mode != "y":
        expert_batch.context_y = y_hidden[:, :ctx_len]
        expert_batch.query_y   = y_hidden[:, ctx_len: ctx_len + qry_len]
        expert_batch.target_y  = y_hidden[:, ctx_len + qry_len: ctx_len + qry_len + tgt_len]
        expert_batch.expert_y  = y_hidden[:, ctx_len + qry_len + tgt_len:]

        # Save expert dim so padding works
        #self.expert_dim = expert_dim_x

        # create expert model
        if hasattr(self, 'expert_model') and self.expert_model is not None:
            del self.expert_model
            self.expert_model = None

        self.expert_model = self.expert_generator(
            batch=expert_batch,
            fixed_hyperparams=fixed_hyperparams,
        )
        if self.time_encoded_feedback:
            self.expert_model = TimeEncodedExpertWrapper(self.expert_model)

        # generate feedback
        expert_batch.t = torch.tensor([0.], device=self.device)
        time_token = expert_batch.t
        # feed in all that might be required, some versions don´t use everything
        with torch.no_grad():
            expert_feedback_context = self.expert_model.initial_predict(batch=expert_batch)
            query_feedback = self.expert_model.predict(x=expert_batch.query_x, 
                            ground_truth_y=expert_batch.query_y,  # used for groundtruth expert
                            time_token=time_token)
            target_feedback = self.expert_model.predict(x=expert_batch.target_x, 
                            ground_truth_y=expert_batch.target_y, # used for groundtruth expert
                            time_token=time_token)

        return expert_feedback_context, query_feedback, target_feedback
    
    def update_batch(self, batch, idx):
        """ Update the batch of data
        
        Args:
            batch: Current batch
            idx: Indices of selected query points
            
        Returns:
            updated_batch: Updated batch with selected query points moved to context
        """

        # get the next data points
        idx_squeezed = idx.squeeze(-1)  # [B, 1] -> [B]
        B = batch.query_x.size(0)
        batch_indices = torch.arange(B, device=idx.device)# could even precompute
        next_x        = batch.query_x[batch_indices, idx_squeezed].unsqueeze(1)       # [B, 1, dim_x]
        next_y        = batch.query_y[batch_indices, idx_squeezed].unsqueeze(1)       # [B, 1, dim_y]
        #next_y_simple = batch.y_hidden_query[batch_indices, idx_squeezed].unsqueeze(1) # [B, 1, dim_y]
        next_feedback = batch.query_feedback[batch_indices, idx_squeezed].unsqueeze(1) # [B, 1, dim_feedback]

        # remove from query set
        batch.query_x = self.update_batch_query(batch.query_x, idx)
        batch.query_y = self.update_batch_query(batch.query_y, idx)
        batch.query_feedback = self.update_batch_query(batch.query_feedback, idx)
        batch.y_hidden_query = self.update_batch_query(batch.y_hidden_query, idx)

        batch.context_x = self.update_batch_context(batch.context_x, next_x)
        batch.context_y = self.update_batch_context(batch.context_y, next_y)
        batch.context_feedback = self.update_batch_context(batch.context_feedback, next_feedback)

        if getattr(self.expert_model, "adaptive", False):# otherwise safe the compute
            with torch.no_grad():
                #self.expert_model.add_data(next_x, next_y if self.truth_mode == "y" else next_feedback, time_token=getattr(batch, "t", None)) # don´t think it makes sense to adapt to wrong things
                self.expert_model.add_data(next_x, next_y, time_token=getattr(batch, "t", None))
                batch.expert_eval_query = self.expert_model.predict(batch.query_x, time_token=getattr(batch, "t", None))

        return batch

    @torch.no_grad()
    def sample_batch(self, batch_size):
        """
        Override: sample main and hidden GP, return batch with y_total and expert-only y
        """
        batch = AttrDict()
        total_dim, expert_dim = self.generate_dims()
        dim_of_main_gp = total_dim - expert_dim
        if dim_of_main_gp > 0:
            theta_main = self.sample_theta(batch_size, total_dim - expert_dim).to(self.device)
        else:
            # zero main dims, just keep batch_size x 0 x 1 for compatibility
            theta_main = torch.empty(batch_size, 0, 1, device=self.device)

        theta_hidden = self.sample_theta(batch_size, expert_dim).to(self.device)

        # sample X
        n_expert_extra = int(torch.randint(self.expert_extra_data_min, self.expert_extra_data_max + 1, (1,)))
        n_total = self.n_context_init + self.n_query_init + self.n_target_data
        x = self.sample_data(batch_size, n_total, total_dim).to(self.device)
        x_extra = self.sample_expert_positions(batch_size, n_expert_extra, total_dim).to(self.device)
        x = torch.cat([x, x_extra], dim=1)
        # generate additive GP data
        y_total, y_hidden_truth, w_expert = self.generate_gp_data(x, theta_main, theta_hidden, dim_of_main_gp=dim_of_main_gp)
        #assert not torch.isnan(y_hidden_truth).any(), f"NaN in y_hidden, y_hidden={y_hidden_truth}, x shape={x.shape}"
        #assert not torch.isnan(y_total).any(), f"NaN in y_total"

        if self.truth_mode == "y":
            y_hidden_truth = y_total.clone()

        # post process this to introduce biases,   can do param_tensor=True to have extra thetas to learn
        if self.use_expert_thetas:
            y_hidden, expert_thetas = self.expert_bias_model.apply(y=y_hidden_truth, x=x[:,:,-expert_dim:], return_params=True) if (self.expert_bias_model is not None) else (y_hidden_truth, torch.empty((batch_size, 0, 1), device=self.device))

        else:
            y_hidden = self.expert_bias_model.apply(y=y_hidden_truth, x=x[:,:,-expert_dim:]) if self.expert_bias_model is not None else y_hidden_truth

        # slice batches as before
        batch.context_x = x[:, :self.n_context_init] #, :self.dim_x
        batch.context_y = y_total[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init: self.n_context_init+self.n_query_init]#, :self.dim_x
        batch.query_y = y_total[:, self.n_context_init: self.n_context_init+self.n_query_init]
        batch.target_x = x[:, self.n_context_init+self.n_query_init: self.n_context_init+self.n_query_init+self.n_target_data]#, :self.dim_x
        batch.target_y = y_total[:, self.n_context_init+self.n_query_init: self.n_context_init+self.n_query_init+self.n_target_data]

        # expert sees hidden dims only
        batch.expert_x = x[:, self.n_context_init+self.n_query_init+self.n_target_data:]#, -expert_dim:
        batch.expert_y = y_hidden[:, self.n_context_init+self.n_query_init+self.n_target_data:] # could rename

        # for adaptive mode we need to save the other y_hidden to add them
        batch.y_hidden_query = y_hidden[:, self.n_context_init: self.n_context_init+self.n_query_init]
        batch.y_hidden_context = y_hidden[:, : self.n_context_init]# for ground truth expert 
        batch.y_hidden_target = y_hidden[:, self.n_context_init+self.n_query_init: self.n_context_init+self.n_query_init+self.n_target_data]

        batch.expert_dim_x = expert_dim
        batch.dim_x = self.dim_x

        # handle theta embeddings, that part is a bit strange
        #batch.w_expert = w_expert
        #batch.knowledge_ratio = n_expert_extra/self.expert_extra_data_max

        if dim_of_main_gp > 0:# now output lengthscale is in first one
            theta_for_model = torch.concat((theta_main[:, :dim_of_main_gp], theta_hidden[:,  :self.dim_x- dim_of_main_gp], theta_main[:, -1:]), dim=1)
            #theta_for_model = torch.concat((theta_main[:, :dim_of_main_gp], theta_main[:, dim_of_main_gp :self.dim_x], theta_main[:, -1:]), dim=1)

        else:
            theta_for_model = torch.concat((theta_hidden[:, :self.dim_x], theta_hidden[:, -1:]), dim=1)
        if self.use_expert_thetas:
            #  might want to add expert thetas used to generate gp data, but the number of them varies, but it is possible, one could post-pend 0
            # but since the dimensions are sometimes the same and sometimes not, it gets too complicated
    
            theta_for_model = torch.concat((theta_for_model, w_expert), dim=1) 
            # , (n_expert_extra/self.expert_extra_data_max) *torch.ones_like(theta_hidden[:, -1:]) # this is not relevant anymore I guess, but was a theta before
            if self.expert_bias_model is not None:
                theta_for_model = torch.cat([theta_for_model, expert_thetas], dim=1)
        
        if self.embedding_type in ["theta", "mix"]:
            batch.target_theta = theta_for_model
        else:
            batch.target_theta = None
        if self.embedding_type == "mix":
            batch.target_all = torch.cat([batch.target_y, batch.target_theta], dim=1)
            
        elif self.embedding_type == "data":
            batch.target_all = batch.target_y
        else:
            batch.target_all = theta_for_model

        # init expert feedback using hidden GP only
        batch.context_feedback, batch.query_feedback, batch.target_feedback = self.init_expert_model_from_batch(batch, 
                                                                            expert_dim_x=expert_dim,
                                                                            y_hidden=y_hidden,
                                                                            thetas=theta_hidden)
        batch.n_target_theta = self.n_target_theta
        
        if self.concat_bias_generated_certainty:
            uncertainty_all, theta_for_uncertainty_noise = self.expert_bias_model.certainty_from_delta(y_hidden_truth, y_hidden)
            batch.context_feedback = torch.cat([batch.context_feedback, uncertainty_all[:, :self.n_context_init]], dim=2)
            batch.query_feedback = torch.cat([batch.query_feedback, uncertainty_all[:, self.n_context_init: self.n_context_init+self.n_query_init]], dim=2)
            batch.target_feedback = torch.cat([batch.target_feedback, uncertainty_all[:, self.n_context_init+self.n_query_init: self.n_context_init+self.n_query_init+self.n_target_data]], dim=2)

            if self.use_expert_thetas:
                batch.target_theta = torch.cat([batch.target_theta, theta_for_uncertainty_noise], dim=1)
        
        return batch


class MixedPrior_Expert_GPTask(Expert_GPTask):
    """
    Meta prior over the two expert priors: each batch is drawn entirely from
    Additive_Expert_GPTask (probability p_additive) or WeightedMultiGP_Expert_GPTask otherwise.
    Both sub-tasks share one flat kwargs config. Mixing per batch (not per row) keeps each batch
    homogeneous, so the training loop's batchwise reward normalization stays valid.
    """

    def __init__(
        self, p_additive: float = 0.5, **kwargs
        ):
        super().__init__(**kwargs)
        if not 0.0 <= p_additive <= 1.0:
            raise ValueError(f"p_additive must be in [0, 1], got {p_additive}")
        if kwargs.get("use_expert_thetas", False):
            # Additive appends a single w_expert theta, WeightedMultiGP appends one per GP
            # (k_real + k_decoy, resampled every batch), so n_target_theta would neither
            # match between the priors nor stay constant across batches.
            raise ValueError(
                "MixedPrior_Expert_GPTask requires use_expert_thetas=False: the two priors "
                "append a different number of expert thetas, so n_target_theta would not be "
                "constant and the embedder's theta tokens would not line up"
            )
        self.p_additive = p_additive
        self.sub_tasks = nn.ModuleList([
            Additive_Expert_GPTask(**kwargs),
            WeightedMultiGP_Expert_GPTask(**kwargs),
        ])
        self.active_prior_idx = 0
        print(f"Mixed prior over [Additive, WeightedMultiGP] with p(additive)={self.p_additive}")

    @property
    def active_task(self) -> Expert_GPTask:
        return self.sub_tasks[self.active_prior_idx]

    def _sync_shared_state(self) -> None:
        """Push the fields the training loop mutates on the task down to both sub-tasks."""
        for task in self.sub_tasks:
            task.n_query_init = self.n_query_init
            task.n_context_init = self.n_context_init
            task.n_target_data = self.n_target_data

    @torch.no_grad()
    def sample_batch(self, batch_size):
        self.active_prior_idx = 0 if torch.rand(1).item() < self.p_additive else 1
        self._sync_shared_state()
        return self.active_task.sample_batch(batch_size)

    def update_batch(self, batch, idx):
        """ Update the batch of data with the prior that generated it

        Args:
            batch: Current batch
            idx: Indices of selected query points

        Returns:
            updated_batch: Updated batch with selected query points moved to context
        """
        return self.active_task.update_batch(batch, idx)


if __name__ == "__main__":
    # Import visualization libraries
    import matplotlib.pyplot as plt
    import argparse
    from utils.plot_config import apply_style
    apply_style(use_tex=False)

    parser = argparse.ArgumentParser(description='GPTask demonstration or offline data generation')
    parser.add_argument('--dim_x', type=int, default=3, help='Input dimension')
    parser.add_argument('--embedding_type', type=str, default='data', help='Embedding type: data, theta, or mix')
    args = parser.parse_args()

    # Create the GP task
    task = GPTask(
        dim_x=args.dim_x,
        embedding_type=args.embedding_type,
        n_context_init=1,
        n_query_init=500,
        n_target_data=200,
        noise_scale=0.01,
        design_scale=5
    )

    batch = task.sample_batch(20)

    plt.figure(figsize=(15, 10))

    # Only visualize 1D GPs
    if args.dim_x == 1:
        for i in range(20):
            # Create subplot
            plt.subplot(5, 4, i + 1)

            # Plot the sampled points
            # plt.scatter(batch.context_x[i], batch.context_y[i], c='blue', label='Context' if i == 0 else "", s=10)

            query_x_i = batch.query_x[i].squeeze().numpy()  # Squeeze to 1D numpy array
            query_y_i = batch.query_y[i].squeeze().numpy()  # Squeeze to 1D numpy array

            sorted_indices = np.argsort(query_x_i)
            query_x_sorted = query_x_i[sorted_indices]
            query_y_sorted = query_y_i[sorted_indices]
            plt.plot(query_x_sorted, query_y_sorted, c='C0', label='Context' if i == 0 else "")

            # plt.plot(batch.query_x[i], batch.query_y[i], c='red', label='Query' if i == 0 else "", s=10)
            plt.ylim(-3, 3)
            # if hasattr(batch, 'target_x'):
            #     plt.scatter(batch.target_x[i], batch.target_y[i], c='green', label='Target' if i == 0 else "", s=10)

            # Add title with hyperparameters
            lengthscale = batch.target_theta[i, 0, 0].item()
            variance = batch.target_theta[i, 1, 0].item()
            plt.title(f'Sample {i + 1}: ls = {lengthscale:.2f}, scale = {variance:.2f}')
            # plt.text(-5, 2, f'lengthscale={lengthscale:.2f}', fontsize=10)
            # plt.text(-5, 1, f'scale={variance:.2f}', fontsize=10)

            # if i == 0:
            #     plt.legend()

            plt.grid(False)

        plt.tight_layout()
        plt.savefig("../figures/gp_samples.pdf", dpi=300)
        plt.show()
