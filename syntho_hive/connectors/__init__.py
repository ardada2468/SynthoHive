"""IO backends and sampling connectors."""

from .local_io import LocalIO

__all__ = ["LocalIO", "SparkIO"]


def __getattr__(name: str):
    if name == "SparkIO":
        from .spark_io import SparkIO

        return SparkIO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
