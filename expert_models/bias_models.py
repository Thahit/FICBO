import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


# Most of the individual bias types in ExpertBiasEngine/ProbExpertBiasEngine (anchoring,
# feature_weight, expectation_bias, hard_easy, simplification, regression_to_mean,
# hetero_noise, rank_normalize, binning, rounding, ...) are unexplored work from an
# earlier direction where we thought about building a prior over human cognitive biases. They are
# not part of the current active pipeline (see config/expert_bias/expert_bias_with_switch.yaml
# for what's actually enabled) and are largely untested. Left in in case they're useful later.
# The uncertainty-generation helpers below (certainty_from_delta, certainty_from_local_delta)
# are similar: thez are from an earlier direction where an user can specify how certain thez are


class BaseBias(ABC):
    """
    Abstract helper class for expert biases.

    Intended usage:
      - Each bias is a small module with:
          * enable flag
          * parameter ranges
          * per-batch sampling
          * apply() method

      - Your ExpertBiasEngine can hold a list of BaseBias instances and apply them sequentially.
    """

    def __init__(
        self,
        enabled: bool = True,
        device: str | torch.device = "cpu",
        name: Optional[str] = None,
        ):
        self.enabled = enabled
        self.device = torch.device(device)
        self.name = name or self.__class__.__name__

    @abstractmethod
    def n_params(self) -> int:
        """
        How many scalar parameters this bias returns per batch element
        (for return_params=True aggregation).
        """
        raise NotImplementedError

    @abstractmethod
    def sample_params(self, batch_size: int) -> torch.Tensor:
        """
        Sample per-batch parameters.

        Returns:
            params: [B, 1, n_params]
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        return_params=False,
    ) -> torch.Tensor:
        """
        Apply the bias to y.

        Args:
            y: [B,N,D]
            x: [B,N,X_dim]
            params: [B,1,n_params] sampled from sample_params()
            ctx: optional precomputed values (e.g. dist2)

        Returns:
            y_biased: [B,N,D]
        """
        raise NotImplementedError


class ExpertBiasEngine:
    def __init__(
        self,
        noise_level=(0.1, 1.),
        local_averaging=True, local_averaging_alpha=(0.05, 0.5), local_averaging_lengthscale=(.3, 1),# expensive
        anchoring=False, anchor_alpha=(0.2, 0.5), # Useless for BO
        feature_weight=True, feature_coef=(0.1, 0.4),
        expectation_bias=False, expectation_alpha=(0.05, 0.15), # useless for BO
        hard_easy=False, hard_easy_alpha=(0.1, 0.3), # useless for BO
        simplification=False, simplification_alpha=(0.1, 0.3),# maybe wated effort
        regression_to_mean=False, regression_alpha=(0.2, 0.5), #useless for BO
        cross_dim_confusion=False, confusion_prob=(0.03, 0.1),
        perceived_max_bias=True, perceived_max_alpha=(0.05, 0.2),
        knowledge_center=False, knowledge_center_noise_radiation_scale=(0.1, 0.5),
        hetero_noise=True, hetero_alpha=(0.01, 0.05), hetero_n_centers=(2,5), hetero_decay=(0.5, 2.0),
        add_gp_bias=True, add_gp_max_std=(0.3, 5), add_gp_lengthscale=(1, 3.0), add_gp_scale=(0.05, 0.3),
        distortion_centers=(2,5), distortion_range=(0.1, 0.5), distortion_decay=(0.5, 2.0), local_distortion=True,
        binning=False, binning_k= 7, binning_y_min=None, binning_y_max=None,
        rank_normalize=False,
        constant_shift_bias=False, constant_shift_bias_range=(-0.5, 0.5),
        rounding_scale=1, rounding=False,
        catastrophic=False,
        device='cpu', uncertainty_noise_std_range=(0.1, 1),
        ):
        self.device = device

        self.noise_level = noise_level

        # Store ranges as tuples
        self.local_averaging = local_averaging
        self.local_averaging_alpha = local_averaging_alpha
        self.local_averaging_lengthscale = local_averaging_lengthscale

        self.anchoring = anchoring
        self.anchor_alpha = anchor_alpha

        self.feature_weight = feature_weight
        self.feature_coef = feature_coef

        self.expectation_bias = expectation_bias
        self.expectation_alpha = expectation_alpha

        self.hard_easy = hard_easy
        self.hard_easy_alpha = hard_easy_alpha

        self.simplification = simplification
        self.simplification_alpha = simplification_alpha

        self.regression_to_mean = regression_to_mean
        self.regression_alpha = regression_alpha

        self.cross_dim_confusion = cross_dim_confusion
        self.confusion_prob = confusion_prob

        self.perceived_max_bias = perceived_max_bias
        self.perceived_max_alpha = perceived_max_alpha

        self.hetero_noise = hetero_noise
        self.hetero_alpha = hetero_alpha
        self.hetero_n_centers = hetero_n_centers
        self.hetero_decay = hetero_decay

        self.knowledge_center = knowledge_center
        self.knowledge_center_noise_radiation_scale = knowledge_center_noise_radiation_scale

        self.binning = binning
        self.binning_k = binning_k
        self.binning_y_min = binning_y_min
        self.binning_y_max = binning_y_max

        self.constant_shift_bias = constant_shift_bias
        self.constant_shift_bias_range = constant_shift_bias_range

        self.rank_normalize = rank_normalize
        self.jitter = 1e-6

        # basically also hetero noise, but more complicated, and expensive
        self.add_gp_bias = add_gp_bias            
        self.add_gp_max_std = add_gp_max_std         
        self.add_gp_lengthscale = add_gp_lengthscale
        self.add_gp_scale = add_gp_scale

        self.local_distortion = local_distortion
        self.distortion_centers = distortion_centers
        self.distortion_range = distortion_range
        self.distortion_decay = distortion_decay

        self.rounding = rounding
        self.rounding_scale = rounding_scale

        self.catastrophic = catastrophic

        self.uncertainty_noise_std_range = uncertainty_noise_std_range

    def _safe_y(self, y: torch.Tensor, clip: float = 10) -> torch.Tensor:
        """
        Replace NaN/Inf with 0 and hard-clip to [-clip, clip].
        Mirrors the nan_to_num pattern used in GP sampling.
        """
        y = torch.nan_to_num(y, nan=0.0, posinf=clip, neginf=-clip)
        return y.clamp(-clip, clip)


    def sample_param(self, param_range, batch_size, num_el=1):
        """Sample one value per batch element from min/max range."""
        min_val, max_val = param_range
        return (min_val + (max_val - min_val) * torch.rand(batch_size, 1, num_el, device=self.device))
    
    def _add_noise(self, y):
        """
        Add Gaussian noise to y.

        Args:
            y: [B,N,D]
            noise_level: tuple/list (min_std, max_std)

        Returns:
            y_noisy: [B,N,D]
        """
        B, N, D = y.shape
        std = self.sample_param(self.noise_level, B)
        noise = torch.randn_like(y) * std
        return y + noise, std

    def apply(self, y, x, return_params=False):
        """
        Apply all active biases to y.
        
        Args:
            y: [B, N, D] tensor of expert outputs
            x: [B, N, X_dim] tensor of expert inputs
            return_params: if True, also return a tensor with all sampled alpha values
        
        Returns:
            y_biased: biased outputs
            param_tensor (optional): [B,1,n_params] tensor with sampled alphas for each batch element
        """
        B, N, D = y.shape
        y, std = self._add_noise(y.clone())

        xi = x.unsqueeze(2)  # [B, N, 1, X_dim]
        xj = x.unsqueeze(1)  # [B, 1, N, X_dim]
        dist2 = ((xi - xj) ** 2).sum(dim=3)  # [B, N, N]
        dist = torch.sqrt(dist2 + 1e-8)      # optional, can pass dist2 if some methods need squared distances

        
        param_values = [std]

        if self.catastrophic:# add extreme noise
            y=  torch.randn_like(y) * 1.5
            param_values.append(torch.ones(B, 1, 1, device=y.device)) # just a flag to indicate it was active, magnitude is not learnable

        # Apply each bias if activated
        if self.cross_dim_confusion: # should be done early to not undo other biases
            alpha = self.sample_param(self.confusion_prob, B)
            y = self._cross_dim_confusion(y, alpha)
        
        if self.hard_easy:
            alpha = self.sample_param(self.hard_easy_alpha, B)
            param_values.append(alpha)
            y = self._hard_easy_bias(y, alpha)
        
        if self.regression_to_mean:
            alpha = self.sample_param(self.regression_alpha, B)
            param_values.append(alpha)
            y = self._regression_to_mean(y, alpha)

        if self.anchoring:
            alpha = self.sample_param(self.anchor_alpha, B)
            param_values.append(alpha)
            y = self._anchoring(y, alpha)

        if self.feature_weight:
            alpha = self.sample_param(self.feature_coef, B)
            param_values.append(alpha)
            y = self._feature_weight_bias(y, x, alpha)

        if self.expectation_bias:
            alpha = self.sample_param(self.expectation_alpha, B)
            param_values.append(alpha)
            y = self._expectation_bias(y, alpha)

        if self.simplification:
            alpha = self.sample_param(self.simplification_alpha, B)
            param_values.append(alpha)
            y = self._simplification_bias(y, x, alpha)
        
        if self.hetero_noise:
            # sample per-batch parameters
            n_centers = int(torch.randint(self.hetero_n_centers[0],
                        self.hetero_n_centers[1]+1, (1,)))  # #centers
            # topk below needs n_centers <= N (the number of points passed to
            # apply() this call, which can be very small: example. a single
            # continuous candidate scored during gradient-based acquisition optimization, 
            # not just a full training-time batch)
            n_centers = min(n_centers, N)

            alpha = self.sample_param(self.hetero_alpha, B, n_centers)            # strength
            decay = self.sample_param(self.hetero_decay, B, n_centers)            # decay per batch
            param_values.append(torch.ones(B, 1, 1, device=y.device) * n_centers)
            y = self._hetero_noise_bias(y, x, alpha, n_centers, decay, dist)

        if self.add_gp_bias:
            # Sample lengthscale and scale per batch
            ls = self.sample_param(self.add_gp_lengthscale, B)  # [B,1,1]
            sc = self.sample_param(self.add_gp_scale, B)        # [B,1,1]
            theta_hidden = torch.cat([ls, sc], dim=1)             # [B,2,1]
            max_std_batch = self.sample_param(self.add_gp_max_std, B)
            param_values.append(max_std_batch)  # optionally record only that
            y = self._add_gp_bias(y, dist2, theta_hidden, max_std=max_std_batch)
        
        if self.local_distortion:
            n_centers = int(torch.randint(self.distortion_centers[0], self.distortion_centers[1], (1,)))
            n_centers = min(n_centers, N)  # topk below needs n_centers <= N, see hetero_noise above
            mag = self.sample_param(self.distortion_range, B, num_el=n_centers)
            mag *= torch.where(torch.rand_like(mag) < 0.5, -1.0, 1.0)  # random sign
            decay = self.sample_param(self.distortion_decay, B, num_el=n_centers)
            param_values.append(torch.ones(B, 1, 1, device=y.device) * n_centers)
            y = self._local_distortion(y, x, mag, n_centers, decay, dist)

        if self.knowledge_center:
            noise_radiation_level = self.sample_param(self.knowledge_center_noise_radiation_scale, B)
            param_values.append(noise_radiation_level)
            y = self._linear_distance_noise_bias(y, noise_radiation_level=noise_radiation_level, dist_all=dist)
            
        if self.local_averaging:# could be before the noise, both are valid in a way? i guess like this is better because it is actually averaging which will not be undone
            alpha = self.sample_param(self.local_averaging_alpha, B)
            param_values.append(alpha)
            lengthscale = self.sample_param(self.local_averaging_lengthscale, B)
            y = self._local_averaging(y, x, alpha, dist2, lengthscale=lengthscale)
        
        if self.perceived_max_bias:# order matters to some extent, and this should be last
            alpha = self.sample_param(self.perceived_max_alpha, B)
            param_values.append(alpha)
            y = self._perceived_max_bias(y, alpha, dist2)
        
        if self.constant_shift_bias:
            alpha = self.sample_param(self.constant_shift_bias_range, B)
            param_values.append(alpha)
            y = self._constant_shift_bias(y, alpha)

        if self.rounding:
            y = self._rounding(y)
        if self.binning:
            if self.binning_y_max is not None and self.binning_y_min is not None:
                y = self._binning_static(y)
            else:
                y = self._binning_dynamic(y)
        
        if self.rank_normalize:
             y = self._rank_normalize(y)

        y = self._safe_y(y)
        if return_params:
            if param_values:
                # Concatenate sampled parameters along dim=2
                param_tensor = torch.cat(param_values, dim=2)  # [B,1,n_params]
            else:
                param_tensor = torch.empty(B,1,0, device=self.device)
            return y, param_tensor
        else:
            return y

    def _local_averaging(self, y, x, alpha, dist2, lengthscale=0.3):
        """
        Illusion of validity: smooths y toward the local trend in x-space, so points near
        others get pulled toward their neighborhood average (shorter lngthscale -> stronger
        pull). Mimics experts overestimating patterns that look locally consistent.
        """
        B, N, D = y.shape

        # Compute pairwise squared distances in x-space: [B, N, N]
        # Using broadcasting: (x_i - x_j)^2 summed over features RBF kernel weights
        weights = torch.exp(-dist2 / (2 * lengthscale**2))  # [B, N, N]
        weights = weights / weights.sum(dim=2, keepdim=True)  # normalize along neighbors

        # Compute local trend via batched matmul: [B, N, N] @ [B, N, D] -> [B, N, D]
        trend = torch.bmm(weights, y)

        # Apply alpha per batch
        alpha = alpha.view(B, 1, 1)  # [B,1,1]
        y_biased = y + alpha * (trend - y)

        return y_biased

    def _anchoring(self, y, alpha): #  useless for BO
        """Anchoring: pulls every later point toward the first value in the sequence, like
        people relying too heavily on the first piece of information they see."""
        anchor = y[:, :1, :]
        B, N, D = y.shape
        y_new = y.clone()
        for i in range(B):
            y_new[i,1:,:] = alpha[i,0,0] * anchor[i] + (1-alpha[i,0,0]) * y[i,1:,:]
        return y_new

    def _feature_weight_bias(self, y, x, alpha):
        """Feature weight bias: overweights one input feature by alpha, the way experts
        often lean on a favorite feature more than it actually deserves."""
        feat = x[:,:,-1:]
        alpha = alpha.expand(-1, y.shape[1], y.shape[2])
        return y + alpha * feat

    def _expectation_bias(self, y, alpha):# not useful for bo
        """Expectation bias: pushes values above the mean a bit higher and values below the
        mean a bit lower, exaggerating highs and lows the way expectations color judgment."""
        threshold = y.mean(dim=1, keepdim=True)
        mask = y > threshold
        alpha = alpha.expand(-1, y.shape[1], y.shape[2])
        return torch.where(mask, y * (1+alpha), y * (1-alpha))

    def _hard_easy_bias(self, y, alpha):# translation is probably useless
        """Hard-easy bias: adds a small positive offset when the sequence looks "easy" (low
        variance), since people tend to overrate themselves on easy tasks and underrate on hard ones."""
        smoothness = torch.var(y, dim=1, keepdim=True)
        bias = alpha * (smoothness < smoothness.mean())
        return y + bias

    def _simplification_bias_linear(self, y, x, alpha):# probably too simplistic
        """Simplification bias: fits a straight line through the first feature and blends
        it into y, standing in for people reaching for a simple heuristic instead of doing
        the actual computation."""
        coef = np.polyfit(x[:,:,-1].cpu().numpy().flatten(), y.cpu().numpy().flatten(), 1)[0]
        linear_est = torch.tensor(coef, device=y.device) * x[:,:,-1:]

        alpha = alpha.expand(-1, y.shape[1], y.shape[2])
        return (1-alpha) * y + alpha * linear_est

    def _simplification_bias(self, y, x, alpha, degree=1):
        B, N, D = y.shape

        # Flatten last feature and y across batch and points
        x_last = x[:, :, -1].cpu().numpy().flatten()
        y_flat = y.cpu().numpy().flatten()

        # Fit low-dim polynomial
        coef = np.polyfit(x_last, y_flat, degree) 

        # Evaluate polynomial
        x_last_tensor = x[:, :, -1: ]  # [B,N,1]
        poly_vals = sum(
            torch.tensor(c, device=y.device) * x_last_tensor ** i
            for i, c in enumerate(reversed(coef))
        )  # [B,N,1]

        alpha = alpha.expand(-1, N, y.shape[2])
        return (1 - alpha) * y + alpha * poly_vals

    def _regression_to_mean(self, y, alpha):# useless
        """Regression to the mean: nudges outputs toward the global mean, since people
        expect extreme values to revert toward average."""
        global_mean = y.mean(dim=1, keepdim=True)
        alpha = alpha.expand(-1, y.shape[1], y.shape[2])
        return alpha * y + (1-alpha) * global_mean

    def _cross_dim_confusion(self, y, prob):# probably not worth the comp cost
        """Cross-dimensional confusion: randomly swaps values between output dimensions,
        mimicking an expert confusing correlated features or misattributing influence."""
        B, N, D = y.shape
        y_biased = y.clone()
        idx1, idx2 = torch.triu_indices(D, D, offset=1)
        num_pairs = idx1.shape[0]
        p = prob.view(B, 1, 1)
        swap_mask = torch.rand(B, N, num_pairs, device=y.device) < p
        y1 = y_biased[:, :, idx1]
        y2 = y_biased[:, :, idx2]
        y_biased[:, :, idx1] = torch.where(swap_mask, y2, y1)
        y_biased[:, :, idx2] = torch.where(swap_mask, y1, y2)
        return y_biased

    def _perceived_max_bias_multiplicative_unused(self, y, alpha, dist2):
        """
        UNUSED — superseded by the subtractive _perceived_max_bias below, which used to
        share this name and silently shadowed it. Kept for reference.

        Perceived max bias, multiplicative version: for each output dim, shrinks y
        toward zero in proportion to how far a point is (in x-space) from the current
        best point in that dim.

        Args:
            y:     [B, N, D]
            alpha: [B, 1, 1]
            dist2: [B, N, N] pairwise squared distances in x-space
        """
        B, N, D = y.shape

        # index of best point per batch and output dim
        y_max_idx = y.argmax(dim=1)  # [B, D]

        # gather distances from each i to max point (per d)
        # dist2[b, i, j] where j = y_max_idx[b,d]
        idx = y_max_idx.unsqueeze(1)  # [B,1,D]
        dist2_to_max = torch.gather(dist2, 2, idx.expand(-1, N, -1))  # [B,N,D]

        # normalize distances per batch+dim to [0,1]
        dist_to_max = torch.sqrt(dist2_to_max + 1e-8)  # [B,N,D]
        max_dist = dist_to_max.max(dim=1, keepdim=True).values + 1e-8
        norm_dist = dist_to_max / max_dist  # [B,N,D]

        alpha = alpha.expand(-1, N, D)

        # penalize farther points (multiplicative shrink)
        y_biased = y * (1.0 - alpha * norm_dist).clamp(min=0.0)

        return y_biased

    def _hetero_noise_bias(self, y, x, alpha, n_centers, decay, dist_all):
        """Heteroscedastic noise: picks n_centers random points and adds noise around them
        that decays with distance (rate set by `decay`), so some regions end up noisier
        than others instead of one uniform noise level everywhere."""
        B, N, D = y.shape
        y_biased = y.clone()

        rand_uniform = torch.rand(B, N, device=y.device)
        _, centers_idx = torch.topk(rand_uniform, n_centers, dim=1)

        centers_idx_exp = centers_idx.unsqueeze(1).expand(B, N, n_centers)  # [B, N, n_centers]
        dist_to_centers = torch.gather(dist_all, 2, centers_idx_exp)        # [B, N, n_centers]

        # Compute influence
        influence = torch.exp(-dist_to_centers * decay)                  # [B, N, n_centers]
        influence = influence.sum(dim=2, keepdim=True)                       # [B, N, 1]
        influence = influence / (influence.max(dim=1, keepdim=True).values + 1e-8)

        # Scale noise
        # alpha is sampled per-center ([B,1,n_centers], one strength draw per
        # center); reduce it to a single per-batch scalar before multiplying
        # against influence ([B,N,1], already summed over centers). 
        # without this, the broadcast silently blows y's last dim up to n_centers.
        alpha = alpha.mean(dim=2, keepdim=True)        # [B,1,1]
        noise_scale = alpha * influence               # [B,N,1]
        noise = torch.randn(B, N, D, device=y.device) * noise_scale          # broadcast to [B,N,D]

        return y_biased + noise

    def _linear_distance_noise_bias(self, y, noise_radiation_level, dist_all):
        """Single-center distance noise: picks one random "center of competence" per batch
        and adds noise that grows linearly with distance from it."""

        B, N, D = y.shape
        y_biased = y.clone()

        # Pick 1 center per batch
        centers_idx = torch.randint(0, N, (B, 1), device=y.device)  # [B, 1]

        # Gather distance to that center
        centers_idx_exp = centers_idx.unsqueeze(1).expand(B, N, 1)  # [B, N, 1]
        dist_to_center = torch.gather(dist_all, 2, centers_idx_exp) # [B, N, 1]

        # Normalize distance (recommended)
        dist_norm = dist_to_center / (dist_to_center.max(dim=1, keepdim=True).values + 1e-8)

        # Linear noise scale
        noise_scale = noise_radiation_level * dist_norm

        noise = torch.randn(B, N, D, device=y.device) * noise_scale

        return y_biased + noise

    def _local_distortion(self, y, x, mag, n_centers, decay, dist_all):
        """Local distortion: picks n_centers random points, each with its own +/- magnitude,
        and adds their effect to nearby points with strength decaying by distance."""
        B, N, D = y.shape
        y_distorted = y.clone()

        # Select centers
        rand_uniform = torch.rand(B, N, device=y.device)
        _, centers_idx = torch.topk(rand_uniform, n_centers, dim=1)

        # Compute distance to centers
        centers_idx_exp = centers_idx.unsqueeze(1).expand(B, N, n_centers)  # [B,N,n_centers]
        dist_to_centers = torch.gather(dist_all, 2, centers_idx_exp)        # [B,N,n_centers]

        # Influence decays with distance
        influence = torch.exp(-dist_to_centers * decay)                 # [B,N,n_centers]

        influence = influence * mag#.view(B,1,n_centers)                                        # broadcast effect
        total_distortion = influence.sum(dim=2, keepdim=True)               # [B,N,1]

        # Broadcast to output dim if needed (shouldn´t be needed currently)
        total_distortion = total_distortion.expand(B, N, D)

        return y_distorted + total_distortion

    def _constant_shift_bias(self, y, alpha):
        """Constant shift bias: adds the same constant alpha to every output, like an
        expert who's just generally pessimistic or optimistic across the board."""
        return y + alpha

    def _perceived_max_bias(self, y, alpha, dist2):
        """
        Perceived max bias, subtractive version: points far from the current best get a
        distance-proportional penalty subtracted off, which can push them into negative
        territory so they look actively bad instead of just merely worse.

        Args:
            y:     [B, N, D]
            alpha: [B, 1, 1]
            dist2: [B, N, N] pairwise squared distances in x-space
        """
        B, N, D = y.shape

        # index of best point per batch and output dim
        y_max_idx = y.argmax(dim=1)  # [B, D]

        # get distances from each point to the perceived max
        idx = y_max_idx.unsqueeze(1)                                      # [B, 1, D]
        dist2_to_max = torch.gather(dist2, 2, idx.expand(-1, N, -1))     # [B, N, D]
        dist_to_max = torch.sqrt(dist2_to_max + 1e-8)                    # [B, N, D]

        # normalize distances to [0, 1]
        max_dist = dist_to_max.max(dim=1, keepdim=True).values + 1e-8
        norm_dist = dist_to_max / max_dist                                # [B, N, D]

        # scale penalty by typical feedback magnitude so it's meaningful
        scale = y.abs().mean(dim=1, keepdim=True)                        # [B, 1, D]

        # subtract distance-proportional penalty
        y_biased = y - alpha * norm_dist * scale

        return y_biased

    def _catastrophic_bias(self, y, dist2):
        """
        Catastrophic expert failure: replace y with an independent GP draw.
        The result looks like a plausible smooth function, but has zero
        correlation with the true y — simulating a confidently wrong expert.
        """
        B, N, D = y.shape

        # Sample fresh GP hyperparams (wide range so the fake function varies)
        ls  = self.sample_param((0.1, 2.0), B)   # [B,1,1]
        sc  = self.sample_param((0.5, 2.0), B)   # [B,1,1]
        theta = torch.cat([ls, sc], dim=1)        # [B,2,1]

        fake = torch.zeros_like(y)
        eye  = torch.eye(N, device=y.device)

        for b in range(B):
            lengthscale = theta[b, 0, 0].clamp(min=1e-4)
            scale       = theta[b, 1, 0].clamp(min=1e-8)
            K = (scale**2) * torch.exp(-0.5 * dist2[b] / lengthscale**2) + self.jitter * eye

            for mult in [1, 10, 100, 1000]:
                try:
                    L = torch.linalg.cholesky(K + mult * self.jitter * eye)
                    fake[b] = L @ torch.randn(N, 1, device=y.device)
                    break
                except torch._C._LinAlgError:
                    continue
            else:
                # fallback: diagonal GP :(
                fake[b] = torch.sqrt(torch.diagonal(K).clamp(1e-8)).unsqueeze(-1) \
                        * torch.randn(N, 1, device=y.device)

        # Rescale to match y's typical magnitude so it's not detectable by scale alone
        fake_std = fake.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_std    = y.std(dim=1, keepdim=True).clamp(min=1e-8)
        fake     = fake / fake_std * y_std

        return fake.expand(-1, -1, D)
        
    @torch.no_grad()
    def _add_gp_bias(self, y, dist2, theta_hidden, max_std=0.2, jitter=1e-6):
        B, N, D = y.shape
        eye = torch.eye(N, device=y.device)
        y_add = torch.zeros(B, N, D, device=y.device)

        for b in range(B):
            lengthscale = theta_hidden[b, 0, 0].clamp(min=1e-4)
            scale = theta_hidden[b, 1, 0].clamp(min=1e-8)

            Kb = (scale**2) * torch.exp(-0.5 * dist2[b] / (lengthscale**2))
            Kb = Kb + jitter * eye

            success = False
            for mult in [1, 10, 100, 1000]:
                try:
                    L = torch.linalg.cholesky(Kb + (mult * jitter) * eye)
                    y_add[b] = L @ torch.randn(N, 1, device=y.device)
                    success = True
                    break
                except torch._C._LinAlgError:
                    continue

            if not success:
                var = torch.diagonal(Kb) + jitter
                var = torch.clamp(var, min=1e-8)   
                y_add[b] = torch.sqrt(var).unsqueeze(-1) * torch.randn(N, 1, device=y.device)

        # normalize to max_std per batch
        cur_std = y_add.std(dim=1, keepdim=True) + 1e-8
        y_add = y_add / cur_std * max_std

        y_add = torch.nan_to_num(y_add, nan=0.0, posinf=0.0, neginf=0.0)  # ← belt-and-suspenders
        return y + y_add.expand(-1, -1, D)

    def _rank_normalize(self, y):
        """Replaces each value with its rank (per batch and output dim), then rescales
        ranks analytically to mean 0, unit spread — no separate mean/var calls needed."""
        B, N, D = y.shape

        # Get ranks (argsort twice trick)
        ranks = (y.argsort(dim=1).argsort(dim=1).float() + 1)  
        # normalization
        mean = (N + 1) / 2.0
        #std = ((N**2 - 1) / 12.0) ** 0.5

        y_ranked = (ranks - mean) / (N / 2.0) # to make the biggest be 1

        return y_ranked
    
    def _binning_dynamic(self, y):
        """Bins y into `binning_k` buckets using its own min/max, then snaps each value
        to its bucket's center."""
        B, N, D = y.shape

        y_min = y.min(dim=1, keepdim=True).values
        y_max = y.max(dim=1, keepdim=True).values

        # avoid division by zero
        denom = (y_max - y_min).clamp(min=1e-8)

        # Normalize to [0,1]
        y_norm = (y - y_min) / denom

        # Bin indices
        bins = torch.clamp((y_norm * self.binning_k).long(), max=self.binning_k-1)

        # Map to bin centers
        bin_centers = (bins.float() + 0.5) / self.binning_k

        # Back to original scale
        y_binned = y_min + bin_centers * denom

        return y_binned
    
    def _binning_static(self, y):
        """
        Binning with fixed global range:
        - Uses predefined min/max instead of data-dependent

        Args:
            y: [B, N, D]
        """
        B, N, D = y.shape

        denom = (self.binning_y_max - self.binning_y_min)

        y_norm = (y - self.binning_y_min) / denom
        y_norm = y_norm.clamp(0.0, 1.0)

        bins = torch.clamp((y_norm * self.binning_k).long(), max=self.binning_k-1)

        bin_centers = (bins.float() + 0.5) / self.binning_k

        y_binned = self.binning_y_min + bin_centers * denom

        return y_binned
        
    def _rounding(self, y):
        """
        Simple rounding:
        - Rounds values to a fixed number of decimals

        Args:
            y: [B, N, D]
        """
        scale =  self.rounding_scale
        return torch.round(y * scale) / scale

    @property
    def n_params(self):
        n = 1 # noise

        #if self.cross_dim_confusion:
        #    n += 1   # only if you decide it should output a param

        if self.anchoring:
            n += 1

        if self.feature_weight:
            n += 1

        if self.expectation_bias:
            n += 1

        if self.hard_easy:
            n += 1

        if self.simplification:
            n += 1

        if self.regression_to_mean:
            n += 1

        if self.hetero_noise:
            n += 1

        if self.add_gp_bias:
            n += 1

        if self.local_distortion:
            n += 1
        
        if self.local_averaging:
            n += 1

        if self.perceived_max_bias:
            n += 1
        
        if self.catastrophic:
            n += 1

        if self.knowledge_center:
            n += 1

        if self.constant_shift_bias:
            n += 1

        return n

    def certainty_from_delta(self, y_orig, y_final, tau=0.5):
        """
        Compute a per-point certainty estimate based on the average distortion
        between the original and final values.

        Certainty is higher where the distortion is smaller and decays smoothly
        according to an exponential mapping. Random noise is added to mimic
        imperfect uncertainty estimation.

        Args:
            y_orig (torch.Tensor): Original values, shape [B, N].
            y_final (torch.Tensor): Distorted or post-processed values, shape [B, N].
            tau (float, optional): Temperature controlling how quickly certainty decays with distortion.
                                Default: 0.5. Must be > 0.

        Returns:
            torch.Tensor: Per-point certainty scores in (0, 1], shape [B, N, 1].
                        Higher values indicate higher certainty.
            torch.Tensor: noise level used for disturbing uncertainty, shape [B, 1, 1].
        """
        # Compute average per-point distortion
        delta_true = torch.norm(y_final - y_orig, dim=-1)  # [B,N]
        std_multillier = self.sample_param(self.uncertainty_noise_std_range, y_orig.shape[0])
        delta = (delta_true + torch.randn_like(delta_true) * std_multillier.squeeze(-1)).abs()

        # Map distortion to certainty
        c = torch.exp(-delta / tau)  # in (0,1]

        return c.unsqueeze(-1), std_multillier

    
    def certainty_from_local_delta(self, y_orig, y_final, dist2, lengthscale=0.3, tau=0.25, eps=1e-8):
        """
        Estimate a per-point certainty measure based on local distortions.

        The certainty for each point is computed from the average absolute difference
        between the original and distorted values (`delta`), optionally perturbed with noise,
        and smoothed locally using a distance-based weighting.

        Args:
            y_orig (torch.Tensor): Original values, shape [B, N, D].
            y_final (torch.Tensor): Distorted or post-processed values, shape [B, N, D].
            dist2 (torch.Tensor): Squared distances in input space, shape [B, N, N].
            lengthscale (float, optional): Spatial smoothing scale for local weighting. Default: 0.3
            tau (float, optional): Temperature controlling sharpness of certainty mapping. Default: 0.25
            eps (float, optional): Small constant for numerical stability. Default: 1e-8

        Returns:
            torch.Tensor: Per-point certainty scores in [0, 1], shape [B, N, 1].
                        Higher values indicate higher certainty.
        """

        # dist2: [B,N,N] squared distances in x-space
        delta = torch.norm(y_final - y_orig, dim=-1).unsqueeze(-1)  # [B,N,1]
        std_multillier = self.sample_param(self.uncertainty_noise_std_range, y_orig.shape[0])
        delta = delta + torch.randn_like(delta) * std_multillier


        W = torch.exp(-dist2 / (2 * lengthscale**2))                # [B,N,N]
        W = W / (W.sum(dim=2, keepdim=True) + eps)

        delta_local = torch.bmm(W, delta)                            # [B,N,1]
        c = torch.exp(-delta_local / (tau + eps))
        return c


class ProbExpertBiasEngine(ExpertBiasEngine):
    """
    ExpertBiasEngine where each bias is applied per batch element
    with an independent activation probability.
    Inactive biases still return zeros for parameter tracking.
    """

    def __init__(self, bias_probs=None, **kwargs):
        """
        Args:
            bias_probs: dict mapping bias_name -> probability of activation per batch
                        e.g. {"local_averaging":0.8, "add_gp_bias":0.5, ...}
                        If None, all biases always active.
        """
        super().__init__(**kwargs)
        self.bias_probs = bias_probs or {}

    def apply(self, y, x, return_params=False):
        B, N, D = y.shape
        y, std = self._add_noise(y.clone())
        xi = x.unsqueeze(2)
        xj = x.unsqueeze(1)
        dist2 = ((xi - xj)**2).sum(dim=3)
        dist = torch.sqrt(dist2 + 1e-8)

        param_values = [std]

        def active_bool(name):
            """Return a single bool per bias per apply call"""
            p = self.bias_probs.get(name, 0.0)
            return torch.rand(1).item() < p

        if self.catastrophic:
            active = active_bool("catastrophic")
            if active:
                y = self._catastrophic_bias(y, dist2)
                param_values.append(torch.ones(B, 1, 1, device=y.device)) # just a flag to indicate it was active, magnitude is not learnable
            else:
                param_values.append(torch.zeros(B, 1, 1, device=y.device))
        
        if self.cross_dim_confusion:  # should be done early to not undo other biases
            active = active_bool("cross_dim_confusion")
            if active:
                alpha = self.sample_param(self.confusion_prob, B)
                y = self._cross_dim_confusion(y, alpha)
            # no params tracked for cross_dim_confusion, consistent with base ExpertBiasEngine

        if self.anchoring:
            active = active_bool("anchoring")
            if active:
                alpha = self.sample_param(self.anchor_alpha, B)
                param_values.append(alpha)
                y = self._anchoring(y, alpha)
            else:
                alpha = torch.zeros(B,1,1, device=y.device)
                param_values.append(alpha)

        if self.feature_weight:
            active = active_bool("feature_weight")
            if active:
                alpha = self.sample_param(self.feature_coef, B)
                param_values.append(alpha)
                y = self._feature_weight_bias(y, x, alpha)
            else:
                alpha = torch.zeros(B,1,1, device=y.device)
                param_values.append(alpha)

        if self.expectation_bias:
            active = active_bool("expectation_bias")
            if active:
                alpha = self.sample_param(self.expectation_alpha, B)
                param_values.append(alpha)
                y = self._expectation_bias(y, alpha)
            else:
                alpha = torch.zeros(B,1,1, device=y.device)
                param_values.append(alpha)

        if self.hard_easy:
            active = active_bool("hard_easy")
            if active:
                alpha = self.sample_param(self.hard_easy_alpha, B)
                param_values.append(alpha)
                y = self._hard_easy_bias(y, alpha)
            else:
                alpha = torch.zeros(B,1,1, device=y.device)
                param_values.append(alpha)

        if self.simplification:
            active = active_bool("simplification")
            if active:
                alpha = self.sample_param(self.simplification_alpha, B)
                param_values.append(alpha)
                y = self._simplification_bias(y, x, alpha)
            else:
                alpha = torch.zeros(B,1,1, device=y.device)
                param_values.append(alpha)

        if self.regression_to_mean:
            active = active_bool("regression_to_mean")
            if active:
                alpha = self.sample_param(self.regression_alpha, B)
                param_values.append(alpha)
                y = self._regression_to_mean(y, alpha)
            else:
                alpha = torch.zeros(B,1,1, device=y.device)
                param_values.append(alpha)

        if self.hetero_noise:
            active = active_bool("hetero_noise")
            if active:
                alpha = self.sample_param(self.hetero_alpha, B)
                decay = self.sample_param(self.hetero_decay, B)
                n_centers = int(torch.randint(# impossible to learn with few points i guess
                    self.hetero_n_centers[0],
                    self.hetero_n_centers[1] + 1,
                    (1,)
                ))
                n_centers = min(n_centers, y.shape[1])  # topk needs n_centers <= N
                param_values.append(torch.ones(B,1,1, device=y.device)*n_centers)
                y = self._hetero_noise_bias(y, x, alpha, n_centers, decay, dist)
            else:
                n_centers = torch.zeros(B,1,1, device=y.device)
                param_values.append(n_centers)

        if self.add_gp_bias:
            active = active_bool("add_gp_bias")
            if active:
                ls = self.sample_param(self.add_gp_lengthscale, B)
                sc = self.sample_param(self.add_gp_scale, B)
                theta_hidden = torch.cat([ls, sc], dim=1)
                max_std_batch = self.sample_param(self.add_gp_max_std, B)
                param_values.append(max_std_batch)
                y = self._add_gp_bias(y, dist2, theta_hidden, max_std=max_std_batch)
            else:
                max_std_batch = torch.zeros(B,1,1, device=y.device)
                param_values.append(max_std_batch)

        if self.local_averaging:
            active = active_bool("local_averaging")
            if active:
                alpha = self.sample_param(self.local_averaging_alpha, B)
                param_values.append(alpha)
                y = self._local_averaging(y, x, alpha, dist2)
            else:
                alpha = torch.zeros(B,1,1, device=y.device)
                param_values.append(alpha)

        if self.local_distortion:
            active = active_bool("local_distortion")
            if active:
                n_centers = int(torch.randint(self.distortion_centers[0], self.distortion_centers[1], (1,)))
                n_centers = min(n_centers, y.shape[1])  # topk needs n_centers <= N
                mag = self.sample_param(self.distortion_range, B, num_el=n_centers)
                mag *= torch.where(torch.rand_like(mag) < 0.5, -1.0, 1.0)
                decay = self.sample_param(self.distortion_decay, B, num_el=n_centers)
                param_values.append(torch.ones(B,1,1, device=y.device) *n_centers)
                y = self._local_distortion(y, x, mag, n_centers, decay, dist)
            else:
                n_centers = torch.zeros(B,1,1, device=y.device)
                param_values.append(n_centers)
        
        if self.perceived_max_bias:
            active = active_bool("perceived_max_bias")
            if active:
                alpha = self.sample_param(self.perceived_max_alpha, B)
                param_values.append(alpha)
                y = self._perceived_max_bias(y, alpha, dist2)
            else:
                alpha = torch.zeros(B,1,1, device=y.device)
                param_values.append(alpha)

        if self.knowledge_center:
            active = active_bool("knowledge_center")
            if active:
                noise_radiation_level = self.sample_param(self.knowledge_center_noise_radiation_scale, B)
                param_values.append(noise_radiation_level)
                y = self._linear_distance_noise_bias(y, noise_radiation_level=noise_radiation_level, dist_all=dist)
            else:
                noise_radiation_level = torch.zeros(B,1,1, device=y.device)
                param_values.append(noise_radiation_level)
        
        if self.rounding:
            active = active_bool("rounding")
            if active:
                y = self._rounding(y)
            # no params to track for rounding currently
        if self.constant_shift_bias:
            active = active_bool("constant_shift_bias")
            if active:
                alpha = self.sample_param(self.constant_shift_bias_range, B)
                param_values.append(alpha)
                y = self._constant_shift_bias(y, alpha)
            else:
                alpha = torch.zeros(B,1,1, device=y.device)
                param_values.append(alpha)
        
        if self.binning:
            active = active_bool("binning")
            if active:
                if self.binning_y_min is not None and self.binning_y_max is not None:
                    y = self._binning_static(y)
                else:
                    y = self._binning_dynamic(y)
            # no params to track for binning currently
        
        if self.rank_normalize:
            active = active_bool("rank_normalize")
            if active:
                y = self._rank_normalize(y)
            # no params to track for rank normalization currently
        
        y = self._safe_y(y)
        
        if return_params:
            param_tensor = torch.cat(param_values, dim=1)# dim1 beacuse they will be treated like datapoints
            return y, param_tensor

        return y
