import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
from typing import Optional
from torch.utils.checkpoint import checkpoint


class EfficientTransformerEncoderLayer(TransformerEncoderLayer):
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False,
    ) -> Tensor:
        num_ctx = torch.sum(attn_mask[0, :] == 0).item()

        # Split into context and non-context parts
        context = x[:, :num_ctx, :]
        non_context = x[:, num_ctx:, :]

        # Process context with self-attention
        # NOTE: reverted from a shortcut that assumed context rows only ever need to
        # attend to the context+query block (attn_mask=None over `context` only). That
        # assumption breaks when target_mask grants query rows extra attendance to
        # selected target columns (e.g. mask_type="split"/"all"/"partial"): those rows
        # were silently losing access to their allowed targets. Old version kept below
        # for revert:
        # context_out = self.self_attn(
        #     context, context, context,
        #     attn_mask=None,
        #     key_padding_mask=key_padding_mask[:, :num_ctx] if key_padding_mask is not None else None,
        #     need_weights=False,
        #     is_causal=is_causal,
        # )[0]
        context_out = self.self_attn(
            context, x, x,
            attn_mask=attn_mask[:num_ctx, :],
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]

        # Process non-context with attention to the full sequence based on the mask
        non_context_out = self.self_attn(
            non_context,  # Query: non-context elements
            x,  # Key: full sequence
            x,  # Value: full sequence
            attn_mask=attn_mask[num_ctx:, :],  # Use the relevant part of the attention mask
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]

        # Combine results
        x_out = torch.cat([context_out, non_context_out], dim=1)

        return self.dropout1(x_out)


class Encoder(nn.Module):
    """
    Encoder module that processes batches in three different modes:
    - 'theta': For predicting latent variables only
    - 'data': For predicting data only
    - 'mix': Combination of data and theta prediction
    
    """
    def __init__(
            self,
            dim_embedding,
            dim_feedforward,
            n_head,
            dropout,
            num_layers,
            attend_to_queries: bool = False,
            use_gradient_checkpointing: bool = False,
    ):
        """
        Initialize the encoder module
        
        Args:
            dim_embedding: Dimension of embedding vectors
            dim_feedforward: Dimension of feedforward layer
            n_head: Number of attention heads   
            dropout: Dropout rate
            num_layers: Number of layers
        """
        super().__init__()
        # Create the encoder layer
        if attend_to_queries:
            tranformer_variant = EfficientTransformerEncoderLayer
        else: 
            tranformer_variant = TransformerEncoderLayer
        encoder_layer = tranformer_variant(
            dim_embedding, n_head, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.attend_to_queries = attend_to_queries
        self.use_gradient_checkpointing = use_gradient_checkpointing


    def create_mask(self, batch):
        """
        Create a mask for the encoder
        
        Args:
            batch: Batch containing context_x, query_x, and target_all

        Returns:
            mask: Mask for the encoder
        """
        # Get base dimensions
        num_context = batch.context_x.shape[1]
        num_query = batch.query_x.shape[1]
        num_target = batch.target_all.shape[1]
        num_all = num_context + num_query + num_target
        # Create mask where all positions initially can't attend to any position
        mask = torch.zeros(num_all, num_all, device=self.device).fill_(float("-inf"))

        query_start = num_context
        query_end = query_start + num_query
        target_start = query_end

        # All positions can attend to context positions
        mask[:, :num_context] = 0.0

        # Query points can attend to target points based on target_mask
        if hasattr(batch, 'target_mask') and batch.target_mask is not None:
            # Get the target mask (same for all batch elements)
            target_mask = batch.target_mask  # Shape: [num_target]

            # Find indices of selected targets
            selected_targets = torch.where(target_mask)[0]

            # Map selected target indices to positions in the full sequence
            target_positions = selected_targets + target_start

            # Enable attention from all queries to all selected targets at once
            mask[query_start:query_end, target_positions] = 0.0
        else:
            pass

        if self.attend_to_queries:
            mask[:, query_start:query_end] = 0.0
        
        return mask

    def forward(self, batch, embeddings):
        """
        Forward pass through the encoder
        
        Args:
            batch: Batch containing context_x, query_x, and target_all
            embeddings: Embeddings for the encoder

        Returns:
            out: Output of the encoder
        """
        mask = self.create_mask(batch)

        if self.use_gradient_checkpointing:
            # checkpoint across transformer layers
            def custom_forward(x):
                return self.encoder(x, mask=mask)
            out = checkpoint(custom_forward, embeddings, use_reentrant=False)
        else:
            out = self.encoder(embeddings, mask=mask)
        return out


class EncoderWithTime(nn.Module):
    def __init__(
        self,
        dim_embedding,
        dim_feedforward,
        n_head,
        dropout,
        num_layers,
        attend_to_queries: bool=False,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            dim_embedding,
            n_head,
            dim_feedforward,
            dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # projection for scalar time -> token vector
        self.time_proj = nn.Linear(1, dim_embedding)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.attend_to_queries = attend_to_queries

    def create_mask(self, batch, has_time, num_context, num_query, num_target):
        """
        Create attention mask given whether time token exists and sizes.
        This function expects indices consistent with prepending the time token.
        """
        num_global = 1 if has_time else 0
        num_all = num_global + num_context + num_query + num_target

        mask = torch.zeros(num_all, num_all, device=self.device).fill_(float("-inf"))

        context_start = 1 if has_time else 0
        context_end = context_start + num_context
        query_start = context_end
        query_end = query_start + num_query
        target_start = query_end

        # context tokens can attend to each other
        mask[context_start:context_end, context_start:context_end] = 0.0

        # everyone can attend to context
        mask[:, context_start:context_end] = 0.0

        if has_time:
            mask[query_start:query_end, 0] = 0.0

        # query tokens can attend selected target tokens if target_mask provided
        if hasattr(batch, 'target_mask') and batch.target_mask is not None:
            # target_mask expected shape: (num_target,) boolean or 1D tensor
            selected = torch.where(batch.target_mask)[0]
            if selected.numel() > 0:
                target_positions = selected + target_start
                mask[query_start:query_end, target_positions] = 0.0
        else:
            # allow queries -> all targets
            mask[query_start:query_end, target_start:] = 0.0
        if self.attend_to_queries:
            mask[:, query_start:query_end] = 0.0
        
        return mask

    def forward(self, batch, embeddings):
        """
        embeddings: (B, seq_len_without_time, dim_embedding)
        batch.t: either
            - None,
            - a Python float,
            - a 0-d or 1-d torch tensor,
            - a numpy scalar,
            - or a 1-D tensor of shape (B,)
        """

        B = embeddings.shape[0]
        has_time = hasattr(batch, "t") and batch.t is not None

        if has_time:
            t = batch.t

            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=self.device, dtype=embeddings.dtype)

            t = t.to(device=self.device, dtype=embeddings.dtype)

            if t.dim() == 0:
                t = t.unsqueeze(0).expand(B) 
            elif t.dim() == 1 and t.numel() == 1:
                t = t.expand(B) 

            if t.dim() == 1:
                t = t.view(B, 1)         
            elif t.dim() == 2 and t.shape[1] != 1:
                t = t.view(B, 1)

            # project and add at the end
            t_emb = self.time_proj(t)   
            t_emb = t_emb.unsqueeze(1)    
            embeddings = torch.cat([t_emb, embeddings], dim=1)

        num_context = batch.context_x.shape[1]
        num_query = batch.query_x.shape[1]
        num_target = batch.target_all.shape[1]

        mask = self.create_mask(batch, has_time, num_context, num_query, num_target)

        out = self.encoder(embeddings, mask=mask)
        return out[:, 1:, :]  # remove the time token