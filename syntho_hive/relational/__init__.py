"""Relational synthesis: dependency graph, linkage, and orchestration."""

from .graph import SchemaGraph
from .linkage import LinkageModel

__all__ = ["SchemaGraph", "LinkageModel", "StagedOrchestrator"]


def __getattr__(name: str):
    if name == "StagedOrchestrator":
        from .orchestrator import StagedOrchestrator

        return StagedOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
