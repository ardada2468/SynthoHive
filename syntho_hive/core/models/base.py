from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

class GenerativeModel(ABC):
    """Base contract for any tabular generative model."""

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs: Any) -> None:
        """Train the model on the provided data.

        Args:
            data: Training dataframe for the model to learn from.
            **kwargs: Model-specific hyperparameters or training options.
        """
        pass  # pragma: no cover

    @abstractmethod
    def sample(self, num_rows: int, **kwargs: Any) -> pd.DataFrame:
        """Generate synthetic rows using the learned model.

        Args:
            num_rows: Number of synthetic rows to generate.
            **kwargs: Optional sampling controls (e.g., temperature).

        Returns:
            DataFrame containing synthetic samples.
        """
        pass  # pragma: no cover

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to disk.

        Args:
            path: Filesystem path where the model artifacts are written.
        """
        pass  # pragma: no cover

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved model from disk.

        Args:
            path: Filesystem path pointing to saved model artifacts.
        """
        pass  # pragma: no cover

class ConditionalGenerativeModel(GenerativeModel):
    """Contract for models that condition on parent context during training/sampling.

    Constructor convention:
        Custom model classes passed as ``model_cls`` to ``StagedOrchestrator``
        must accept the following constructor signature::

            def __init__(self, metadata, batch_size=500, epochs=300, **kwargs):
                ...

        The ``metadata`` positional argument and ``batch_size``/``epochs`` keyword
        arguments are forwarded by the orchestrator during ``fit_all()``. Additional
        keyword arguments are forwarded from ``fit_all(**model_kwargs)``.

        Python ABCs cannot enforce constructor signatures; this convention is
        documented here so custom implementations know what is expected.
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame, context: Optional[pd.DataFrame] = None, **kwargs: Any) -> None:
        """Train the model with optional parent context.

        Args:
            data: Child table data to learn from.
            context: Optional parent attributes used for conditioning.
            **kwargs: Model-specific training options.
        """
        pass  # pragma: no cover

    @abstractmethod
    def sample(self, num_rows: int, context: Optional[pd.DataFrame] = None, **kwargs: Any) -> pd.DataFrame:
        """Generate synthetic rows with optional conditioning context.

        Args:
            num_rows: Number of rows to generate.
            context: Optional parent attributes aligned to the requested rows.
            **kwargs: Additional sampling controls.

        Returns:
            DataFrame of synthetic samples aligned to the provided context (if any).
        """
        pass  # pragma: no cover
