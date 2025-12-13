import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityEmbeddingLayer(nn.Module):
    """Embedding layer for high-cardinality categorical variables."""
    def __init__(self, num_categories: int, embedding_dim: int):
        """Initialize the embedding layer.

        Args:
            num_categories: Number of unique categorical values.
            embedding_dim: Size of the dense embedding vector.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings for discrete category indices.

        Args:
            x: Tensor of category indices with shape `(batch, 1)` or `(batch,)`.

        Returns:
            Tensor of embeddings with shape `(batch, embedding_dim)`.
        """
        return self.embedding(x)

    def forward_soft(self, probs: torch.Tensor) -> torch.Tensor:
        """Blend embeddings using soft probabilities for differentiability.

        Args:
            probs: Probability/logit tensor of shape `(batch, num_categories)`.

        Returns:
            Weighted embedding vectors allowing gradient flow through probabilities.
        """
        return torch.matmul(probs, self.embedding.weight)

class ResidualLayer(nn.Module):
    """Residual MLP block used in CTGAN generator/discriminator."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        """Construct a residual MLP block.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            dropout: Dropout probability applied after activation.
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear -> BN -> ReLU -> dropout with optional skip connection.

        Args:
            x: Input tensor of shape `(batch, input_dim)`.

        Returns:
            Tensor transformed by the residual block; includes skip when dims match.
        """
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        # Skip connection: only possible if dimensions match
        if x.shape[1] == out.shape[1]:
            return x + out
        else:
            return out

class Discriminator(nn.Module):
    """Simple MLP discriminator used by CTGAN."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """Initialize the discriminator network.

        Args:
            input_dim: Flattened feature dimension of real/fake samples.
            hidden_dim: Width of hidden layers.
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute discriminator logits for a batch of samples.

        Args:
            x: Input tensor of shape `(batch, input_dim)`.

        Returns:
            Tensor of shape `(batch, 1)` with realism scores.
        """
        return self.seq(x)
