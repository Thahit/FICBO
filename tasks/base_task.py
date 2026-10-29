import torch
import torch.nn as nn
from typing import Optional
from attrdictionary import AttrDict
from expert_models.base_expert import Expert


class Task(nn.Module): # ALINE stuff
    """
    Base class for all tasks that provides common functionality:
    - Converting between normalized and real design spaces
    - Sampling batches with context, query, and target points
    - Updating batches when query points are selected
    - Supporting different modes: data, theta, mix
    """
    
    def __init__(
        self,
        dim_x: int = 2,
        dim_y: int = 1,
        dim_theta: int = 0,
        mode: str = "data",
        design_scale: float = 1.0,
        outcome_scale: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_theta = dim_theta
        self.mode = mode
        self.design_scale = design_scale
        self.outcome_scale = outcome_scale
        self.device = device
        
        # Validate mode and dimension consistency
        if mode in ["theta", "mix"] and dim_theta <= 0:
            raise ValueError(f"dim_theta must be positive for mode '{mode}'")
    
    @torch.no_grad()
    def sample_theta(self, size, **kwargs):
        """
        Sample theta values from the prior distribution.
        Must be implemented by child classes for theta and mix modes.
        
        Args:
            size: Size of theta batch to generate
            
        Returns:
            theta: Sampled theta values [size, dim_theta]
        """
        raise NotImplementedError("Child classes must implement sample_theta")
        
    def to_design_space(self, xi):
        """Convert from normalized to real design space."""
        return xi * self.design_scale
    
    def normalise_design(self, x):
        """Convert from real to normalized design space."""
        return x / self.design_scale
    
    def unnormalise_design(self, x):
        """Convert from normalized to real design space."""
        return x * self.design_scale
        
    def normalise_outcomes(self, y):
        """Normalize outcomes for consistent loss calculation."""
        return y / self.outcome_scale
    
    @torch.no_grad()
    def forward(self, xi, theta):
        """
        Forward function that maps xi (design) and theta to y (observation).
        
        Args:
            xi [B, D]: normalized design
            theta [B, K, D]: latent variables to learn from the experiments
            
        Returns:
            observations: [B, 1]
        """
        xi = self.to_design_space(xi)  # [B, D]
        raise NotImplementedError("Child classes must implement forward")
    
    def log_likelihood(self, y, xi, theta):
        """
        Log likelihood from gaussian noise
        
        Args:
            y [B, 1]
            xi [B, D]: real designs
            theta [B, K, D]
            
        Returns:
            log_prob: log likelihoods, [B, 1]
        """
        raise NotImplementedError("Child classes must implement log_likelihood")
    
    def update_batch_query(self, query, idx):
        """
        Update the batch of query points

        Args:
            query: Current query points [B, N, D]
            idx: Indices of query points to remove
        Returns:
            updated_query: Updated query points with selected points removed
        
        """
        B, Nt, D = query.shape
        mask = torch.ones((B, Nt), dtype=torch.bool)
        mask[torch.arange(B).unsqueeze(1), idx] = False
        query = query[mask].view(B, -1, D)
        return query


    def update_batch_context(self, context, new):
        """ Update the batch of context points

        Args:
            context: Current context points [B, N, D]
            new: New context points [B, 1, D]
        Returns:
            updated_context: Updated context points with new points added
        """
        context = torch.cat([context, new], dim=1)
        return context
    
    def update_batch(self, batch, idx):
        """ Update the batch of data
        
        Args:
            batch: Current batch
            idx: Indices of selected query points
            
        Returns:
            updated_batch: Updated batch with selected query points moved to context
        """

        # get the next data points
        next_x = torch.gather(batch.query_x, 1, idx.unsqueeze(2).expand(-1, 1, self.dim_x))
        next_y = torch.gather(batch.query_y, 1, idx.unsqueeze(2).expand(-1, 1, self.dim_y))

        batch.query_x = self.update_batch_query(batch.query_x, idx)
        batch.query_y = self.update_batch_query(batch.query_y, idx)

        batch.context_x = self.update_batch_context(batch.context_x, next_x)
        batch.context_y = self.update_batch_context(batch.context_y, next_y)

        return batch


class Expert_Task(Task):
    """
    Base class for tasks that utilize expert models.
    Extends the Task class with expert-specific functionality.
    """
    def __init__(
        self,
        dim_x: int = 2,
        dim_y: int = 1,
        dim_theta: int = 0,
        mode: str = "data",
        design_scale: float = 1.0,
        outcome_scale: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_theta=dim_theta,
            mode=mode,
            design_scale=design_scale,
            outcome_scale=outcome_scale,
            device=device,
            **kwargs
        )
        self.expert_model: Expert = None
        self.dim_feedback = 1 # maybe we add something more interesting later
    
    def add_expert(self, expert_model: Expert):
        """ Add an expert model to the task
        
        Args:
            expert_model: Expert model to be used
        """
        self.expert_model = expert_model
        self.expert_generator = lambda *args, **kwargs: self.expert_model
        
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
        next_feedback = batch.query_feedback[batch_indices, idx_squeezed].unsqueeze(1) # [B, 1, dim_feedback]

        if self.expert_model is not None:
            self.expert_model.add_data(next_x, next_y)

        # remove from query set
        batch.query_x = self.update_batch_query(batch.query_x, idx)
        batch.query_y = self.update_batch_query(batch.query_y, idx)
        batch.query_feedback = self.update_batch_query(batch.query_feedback, idx)

        batch.context_x = self.update_batch_context(batch.context_x, next_x)
        batch.context_y = self.update_batch_context(batch.context_y, next_y)
        batch.context_feedback = self.update_batch_context(batch.context_feedback, next_feedback)

        if getattr(self.expert_model, "adaptive", False):# otherwise safe the compute
            with torch.no_grad():
                batch.expert_eval_query = self.expert_model.predict(batch.query_x, time_token=getattr(batch, "t", None))

        return batch
