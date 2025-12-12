from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

class GenerativeModel(ABC):
    """Abstract base class for all generative models."""

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Train the model on the provided data."""
        pass

    @abstractmethod
    def sample(self, num_rows: int, **kwargs) -> pd.DataFrame:
        """Generate synthetic data."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

class ConditionalGenerativeModel(GenerativeModel):
    """Abstract base class for conditional generative models (supporting parent context)."""

    @abstractmethod
    def fit(self, data: pd.DataFrame, context: Optional[pd.DataFrame] = None, **kwargs) -> None:
        """Train the model with optional context."""
        pass

    @abstractmethod
    def sample(self, num_rows: int, context: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """Generate synthetic data conditioned on context."""
        pass
