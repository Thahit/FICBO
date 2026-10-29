import torch
import torch.nn as nn
from attrdictionary import AttrDict

from .embedder import Embedder
from .encoder import Encoder
from .head import OutputHead



class FICBO(nn.Module):
    """Embedder -> Encoder -> Head pipeline.

    Attributes:
        embedder (Embedder): turns raw tasks into embeddings.
        encoder (Encoder): attention over the embeddings.
        head (OutputHead): produces predictions or log-likelihoods.
    """

    def __init__(
        self, embedder: Embedder, encoder: Encoder, head: OutputHead
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        batch: AttrDict,
    ) -> torch.Tensor:
        """
        Args:
            batch (AttrDict): batch of tasks.

        Returns:
            torch.Tensor: predictions or log-likelihoods from the head.
        """
        embedding = self.embedder(batch)
        encoding = self.encoder(batch, embedding)

        return self.head(batch, encoding)
