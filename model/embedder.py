import torch
import torch.nn as nn
from typing import Any
from attrdictionary import AttrDict

class Embedder(nn.Module):
    """
    Embedder module that processes batches in three different modes:
    - 'theta': For predicting latent variables only
    - 'data': For predicting data only
    - 'mix': Combination of data and theta prediction
    
    The embedder takes a batch with context, query, and target data,
    and produces embeddings in the order: context, query, target.
    """
    
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        n_target_theta: int = 0,
        embedding_type: str = "data",
        **kwargs: Any
    ) -> None:
        """
        Args:
            dim_x: Dimension of input x
            dim_y: Dimension of output y
            dim_embedding: Dimension of embedding vectors
            dim_feedforward: Dimension of feedforward layer
            n_target_theta: Number of latent variables (theta) to create theta identifiers
            embedding_type: Type of embedding ('data', 'theta', or 'mix')
        """
        super().__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_embedding = dim_embedding
        self.n_target_theta = n_target_theta
        self.embedding_type = embedding_type
        
        # Create embedders for x and y
        self.x_embedder = nn.Sequential(
            nn.Linear(dim_x, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding)
        )
        
        self.y_embedder = nn.Sequential(
            nn.Linear(dim_y, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding)
        )
        
        # Create theta tokens for theta and mix modes
        if embedding_type in ["theta", "mix"]:
            if self.n_target_theta <= 0:
                raise ValueError("dim_theta must be positive for theta or mix embedding type")
            
            # Create learnable theta tokens (one for each dimension of theta)
            self.theta_tokens = nn.Parameter(torch.randn(self.n_target_theta, dim_embedding))
    
    def forward(self, batch: AttrDict) -> torch.Tensor:
        """
        Args:
            batch: context/query/target x, y, feedback for one task batch.
        Returns:
            embeddings: [B, N, dim_embedding], dispatched to the mode-specific _embed_* method.
        """

        batch_size = batch.context_x.shape[0]
        
        # Extract dimensions
        n_context = batch.context_x.shape[1]
        n_query = batch.query_x.shape[1]

        # Process according to embedding type
        if self.embedding_type == "data":
            embeddings = self._embed_data_mode(batch, batch_size, n_context, n_query)
        elif self.embedding_type == "theta":
            embeddings = self._embed_theta_mode(batch, batch_size, n_context, n_query)
        elif self.embedding_type == "mix":
            embeddings = self._embed_mix_mode(batch, batch_size, n_context, n_query)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

        return embeddings

    def _embed_data_mode(self, batch: AttrDict, _batch_size: int, n_context: int, _n_query: int) -> torch.Tensor:
        x_all = torch.cat(
            [batch.context_x[:, :, :self.dim_x],
            batch.query_x[:, :, :self.dim_x],
            batch.target_x[:, :, :self.dim_x]],
            dim=1
        )
        embeddings = self.x_embedder(x_all)
        y_embeddings_context = self.y_embedder(batch.context_y)
        embeddings[:, :n_context] = embeddings[:, :n_context] + y_embeddings_context
        return embeddings

    def _embed_theta_mode(self, batch: AttrDict, batch_size: int, n_context: int, _n_query: int) -> torch.Tensor:
        """No target x here - just context/query embeddings followed by the theta tokens."""
        # Embed x (context and query only)
        x_context_query = torch.cat(
            [batch.context_x[:,:,:self.dim_x], 
             batch.query_x[:,:,:self.dim_x]], 
            dim=1
        ) # just in case there are extra dims for feedback or something
        x_embeddings = self.x_embedder(x_context_query)  # [B, n_context+n_query, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        # Add x and y embeddings for context
        context_embeddings = x_embeddings[:, :n_context] + y_embeddings_context
        
        # Extract query embeddings
        query_embeddings = x_embeddings[:, n_context:]
        
        # Create theta tokens (expanded to batch size)
        theta_embeddings = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, dim_theta, dim_embedding]
        
        # Concatenate all embeddings in order: context, query, theta tokens
        embeddings_list = [context_embeddings, query_embeddings, theta_embeddings]

        embeddings = torch.cat(embeddings_list, dim=1)  # [B, N, dim_embedding]
        
        return embeddings
    
    def _embed_mix_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        """Like data mode but also appends the theta tokens after the target embeddings."""

        # Embed x (concatenate all x tasks)
        x_all = torch.cat(
            [batch.context_x[:,:,:self.dim_x],
             batch.query_x[:,:,:self.dim_x],
             batch.target_x[:,:,:self.dim_x]],
            dim=1
        )# just in case there are extra dims for feedback or something
        x_embeddings = self.x_embedder(x_all)  # [B, n_context+n_query+n_target, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        # Add x and y embeddings for context
        context_embeddings = x_embeddings[:, :n_context] + y_embeddings_context
        
        # Extract query and target x embeddings
        query_embeddings = x_embeddings[:, n_context:n_context+n_query]
        target_embeddings = x_embeddings[:, n_context+n_query:]
        
        # Simply expand theta tokens to batch size
        theta_embeddings = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, dim_theta, dim_embedding]
        
        # Combine target embeddings by concatenation (not addition)
        # Concatenate target_x_embeddings and theta_embeddings
        embeddings_list = [context_embeddings, query_embeddings, target_embeddings, theta_embeddings]
        
        embeddings = torch.cat(embeddings_list, dim=1)  # [B, N, dim_embedding]
        
        return embeddings


class SimpleExpertEmbedder(Embedder): # not as great
    """Same as Embedder, but folds expert feedback into x by concatenating it before embedding."""
    
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        dim_feedback: int = 1,
        n_target_theta: int = 0,
        embedding_type: str = "data",
        **kwargs: Any
    ) -> None:
        """
        Args:
            dim_x: Dimension of input x
            dim_y: Dimension of output y
            dim_feedback: Dimension of expert feedback
            dim_embedding: Dimension of embedding vectors
            dim_feedforward: Dimension of feedforward layer
            n_target_theta: Number of latent variables (theta) to create theta identifiers
            embedding_type: Type of embedding ('data', 'theta', or 'mix')
        """
        super().__init__(
            dim_x=dim_x + dim_feedback,
            dim_y=dim_y,
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            embedding_type=embedding_type,
            n_target_theta=n_target_theta,
            **kwargs
        )

        
        self.dim_feedback = dim_feedback
        self.dim_x_child = dim_x

    def forward(self, batch: AttrDict) -> torch.Tensor:
        """
        Args:
            batch: same as Embedder.forward, plus context/query/target_feedback.
        Returns:
            embeddings: [B, N, dim_embedding]. Concatenates feedback onto x, then
            delegates to Embedder.forward.
        """
        # just add the feedback to the x inputs
        batch.context_x_with_feedback = torch.cat([batch.context_x[:,:,:self.dim_x_child], batch.context_feedback], dim=-1)
        batch.query_x_with_feedback = torch.cat([batch.query_x[:,:,:self.dim_x_child], batch.query_feedback], dim=-1)
        batch.target_x_with_feedback = torch.cat([batch.target_x[:,:,:self.dim_x_child], batch.target_feedback], dim=-1)
        return super().forward(batch)

    # a lot of copy paste, but otherwise bigger refactors are needed
    def _embed_data_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        """Same as Embedder._embed_data_mode, but x already has feedback concatenated in."""
        # Embed x
        x_all = torch.cat(
            [batch.context_x_with_feedback,
             batch.query_x_with_feedback,
             batch.target_x_with_feedback], 
            dim=1
        )
        embeddings = self.x_embedder(x_all)  # [B, n_context+n_query+n_target, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        embeddings[:, :n_context] = embeddings[:, :n_context] + y_embeddings_context

        return embeddings
    
    def _embed_theta_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:# duplicate i guess
        """Same as Embedder._embed_theta_mode, but x already has feedback concatenated in."""
        # Embed x (context and query only)
        x_context_query = torch.cat(
            [batch.context_x_with_feedback, 
             batch.query_x_with_feedback], 
            dim=1
        )
        x_embeddings = self.x_embedder(x_context_query)  # [B, n_context+n_query, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        # Add x and y embeddings for context
        context_embeddings = x_embeddings[:, :n_context] + y_embeddings_context
        
        # Extract query embeddings
        query_embeddings = x_embeddings[:, n_context:]
        
        # Create theta tokens (expanded to batch size)
        theta_embeddings = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, dim_theta, dim_embedding]
        
        # Concatenate all embeddings in order: context, query, theta tokens
        embeddings_list = [context_embeddings, query_embeddings, theta_embeddings]

        embeddings = torch.cat(embeddings_list, dim=1)  # [B, N, dim_embedding]
        
        return embeddings
    
    def _embed_mix_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        """Same as Embedder._embed_mix_mode, but x already has feedback concatenated in."""

        # Embed x (concatenate all x tasks)
        x_all = torch.cat(
            [batch.context_x_with_feedback,
             batch.query_x_with_feedback,
             batch.target_x_with_feedback], 
            dim=1
        )
        x_embeddings = self.x_embedder(x_all)  # [B, n_context+n_query+n_target, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        # Add x and y embeddings for context
        context_embeddings = x_embeddings[:, :n_context] + y_embeddings_context
        
        # Extract query and target x embeddings
        query_embeddings = x_embeddings[:, n_context:n_context+n_query]
        target_embeddings = x_embeddings[:, n_context+n_query:]
        
        # Simply expand theta tokens to batch size
        theta_embeddings = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, dim_theta, dim_embedding]
        
        # Combine target embeddings by concatenation (not addition)
        # Concatenate target_x_embeddings and theta_embeddings
        embeddings_list = [context_embeddings, query_embeddings, target_embeddings, theta_embeddings]
        
        embeddings = torch.cat(embeddings_list, dim=1)  # [B, N, dim_embedding]
        
        return embeddings


class ExpertAwareEmbedder(nn.Module):
    """
    Embedder that handles expert feedback separately.
    Supports 'data', 'theta', 'mix' modes and two fusion modes: 'concat' or 'add'.

    - 'concat': concatenate x and feedback embeddings (half-size each)
    - 'add': add x and feedback embeddings (must have same dim)
    """
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        dim_feedback: int = 1,
        n_target_theta: int = 0,
        embedding_type: str = "data",
        feedback_mode: str = "concat",  # 'concat' or 'add'
        **kwargs: Any
    ):
        super().__init__()
        assert feedback_mode in ["concat", "add"], "feedback_mode must be 'concat' or 'add'"
        self.embedding_type = embedding_type
        self.feedback_mode = feedback_mode
        self.use_concat = feedback_mode == "concat"
        self.dim_feedback = dim_feedback
        self.dim_x = dim_x

        # For concat mode, split embedding in half
        x_emb_dim = dim_embedding // 2 if self.use_concat else dim_embedding
        f_emb_dim = dim_embedding // 2 if self.use_concat else dim_embedding

        # x encoder
        self.x_embedder = nn.Sequential(
            nn.Linear(dim_x, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, x_emb_dim)
        )
        # feedback encoder
        self.feedback_embedder = nn.Sequential(
            nn.Linear(dim_feedback, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, f_emb_dim)
        )
        # y encoder
        self.y_embedder = nn.Sequential(
            nn.Linear(dim_y, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding)
        )

        self.n_target_theta = n_target_theta
        if embedding_type in ["theta", "mix"]:
            self.theta_tokens = nn.Parameter(torch.randn(self.n_target_theta, dim_embedding))

    def fuse_embeddings(self, x_emb: torch.Tensor, f_emb: torch.Tensor) -> torch.Tensor:
        """Fuse x and feedback embeddings"""
        if self.feedback_mode == "concat":
            return torch.cat([x_emb, f_emb], dim=-1)
        elif self.feedback_mode == "add":
            return x_emb + f_emb

    def forward(self, batch: AttrDict) -> torch.Tensor:
        batch_size = batch.context_x.shape[0]
        n_context = batch.context_x.shape[1]
        n_query = batch.query_x.shape[1]

        if self.embedding_type == "data":
            embeddings = self._embed_data_mode(batch, batch_size, n_context, n_query)
        elif self.embedding_type == "theta":
            embeddings = self._embed_theta_mode(batch, batch_size, n_context, n_query)
        elif self.embedding_type == "mix":
            embeddings = self._embed_mix_mode(batch, batch_size, n_context, n_query)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        return embeddings

    def _embed_data_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        # Embed x and feedback separately
        x_all = torch.cat([batch.context_x[:,:,:self.dim_x], batch.query_x[:,:,:self.dim_x], batch.target_x[:,:,:self.dim_x]], dim=1)
        x_emb = self.x_embedder(x_all)
        f_all = torch.cat([batch.context_feedback, batch.query_feedback, batch.target_feedback], dim=1)
        f_emb = self.feedback_embedder(f_all)
        embeddings = self.fuse_embeddings(x_emb, f_emb)

        # Embed y for context and add
        y_emb_context = self.y_embedder(batch.context_y)# not a fan of this
        embeddings[:, :n_context] += y_emb_context
        return embeddings

    def _embed_theta_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        x_cq = torch.cat([batch.context_x[:,:,:self.dim_x], batch.query_x[:,:,:self.dim_x]], dim=1)
        x_emb = self.x_embedder(x_cq)
        f_cq = torch.cat([batch.context_feedback, batch.query_feedback], dim=1)
        f_emb = self.feedback_embedder(f_cq)
        x_emb = self.fuse_embeddings(x_emb, f_emb)

        y_emb_context = self.y_embedder(batch.context_y)
        context_emb = x_emb[:, :n_context] + y_emb_context
        query_emb = x_emb[:, n_context:]
        
        theta_emb = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = torch.cat([context_emb, query_emb, theta_emb], dim=1)
        return embeddings

    def _embed_mix_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        x_all = torch.cat([batch.context_x[:,:,:self.dim_x], batch.query_x[:,:,:self.dim_x], batch.target_x[:,:,:self.dim_x]], dim=1)
        x_emb = self.x_embedder(x_all)
        f_all = torch.cat([batch.context_feedback, batch.query_feedback, batch.target_feedback], dim=1)
        f_emb = self.feedback_embedder(f_all)
        x_emb = self.fuse_embeddings(x_emb, f_emb)

        y_emb_context = self.y_embedder(batch.context_y)
        context_emb = x_emb[:, :n_context] + y_emb_context
        query_emb = x_emb[:, n_context:n_context+n_query]
        target_emb = x_emb[:, n_context+n_query:]
        theta_emb = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = torch.cat([context_emb, query_emb, target_emb, theta_emb], dim=1)
        return embeddings

