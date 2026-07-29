"""SynthoHive — synthetic tabular data engine with relational integrity."""

from typing import TYPE_CHECKING

from .interface.config import Metadata, PrivacyConfig, TableConfig, Constraint
from syntho_hive.exceptions import (
    SynthoHiveError,
    SchemaError,
    SchemaValidationError,
    TrainingError,
    SerializationError,
    ConstraintViolationError,
    GenerationError,
    PrivacyError,
)

if TYPE_CHECKING:  # pragma: no cover
    from .interface.synthesizer import Synthesizer

__version__ = "2.0.0"

__all__ = [
    "Synthesizer",
    "Metadata",
    "PrivacyConfig",
    "TableConfig",
    "Constraint",
    "SynthoHiveError",
    "SchemaError",
    "SchemaValidationError",
    "TrainingError",
    "SerializationError",
    "ConstraintViolationError",
    "GenerationError",
    "PrivacyError",
    "__version__",
]


def __getattr__(name: str):
    # Lazy import: Synthesizer pulls in torch, which is heavy — keep plain
    # `import syntho_hive` (metadata/config use) fast.
    if name == "Synthesizer":
        from .interface.synthesizer import Synthesizer

        return Synthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
