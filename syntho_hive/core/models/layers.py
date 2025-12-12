import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityEmbeddingLayer(nn.Module):
    """
    Embedding layer for high-cardinality categorical variables.
    """
    def __init__(self, num_categories: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def forward_soft(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using soft probabilities (logits) for differentiability.
        probs: (batch_size, num_categories)
        """
        # Linear combination of embeddings: probs @ weights
        return torch.matmul(probs, self.embedding.weight)

class ResidualLayer(nn.Module):
    """
    Residual layer used in CTGAN generator/discriminator.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
