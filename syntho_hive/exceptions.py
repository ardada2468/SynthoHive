"""
SynthoHive typed exception hierarchy.

All internal modules raise subclasses of SynthoHiveError.
The Synthesizer public API boundary wraps untyped exceptions into the
appropriate subclass using raise ... from exc to preserve the full traceback.
"""


class SynthoHiveError(Exception):
    """Base exception for all SynthoHive errors."""
    pass


class SchemaError(SynthoHiveError):
    """
    Raised for invalid metadata, missing FK definitions, unsupported column
    types, or invalid identifier names (e.g., SQL injection attempt via
    database/table name).
    """
    pass


class SchemaValidationError(SchemaError):
    """
    Raised by validate_schema() when FK type mismatches, missing FK columns,
    or invalid FK references are detected. Collects all errors before raising
    so callers see the complete problem list in a single exception.
    """
    pass


class TrainingError(SynthoHiveError):
    """
    Raised for NaN loss, training divergence, GPU OOM, or any other failure
    that occurs during Synthesizer.fit().
    """
    pass


class SerializationError(SynthoHiveError):
    """
    Raised for save/load failures, corrupt checkpoints, missing checkpoint
    components, or version mismatches that prevent successful loading.
    """
    pass


class ConstraintViolationError(SynthoHiveError):
    """
    Raised when generated output violates numeric constraints (min, max,
    dtype) defined in the table Metadata.
    """
    pass
