"""Public interface: Synthesizer facade and configuration models."""

from .config import Metadata, PrivacyConfig, TableConfig, Constraint

__all__ = ["Synthesizer", "Metadata", "PrivacyConfig", "TableConfig", "Constraint"]


def __getattr__(name: str):
    if name == "Synthesizer":
        from .synthesizer import Synthesizer

        return Synthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
