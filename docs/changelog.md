---
title: Changelog
---

# Changelog

## 1.3.0

### Added
- Pluggable model architecture: `Synthesizer` and `StagedOrchestrator` accept a `model_cls` parameter for custom `ConditionalGenerativeModel` implementations.
- `enforce_constraints` parameter on `CTGAN.sample()` to raise `ConstraintViolationError` when generated data violates min/max constraints.
- Training observability: structured logging events (`training_start`, `epoch_end`, `training_complete`) with metrics.
- Validation-metric checkpointing: best model saved based on validation metric computed at configurable intervals.
- `SchemaValidationError` with comprehensive FK validation (type mismatches, missing columns, invalid references).
- Typed exception hierarchy: `SynthoHiveError`, `SchemaError`, `TrainingError`, `SerializationError`, `ConstraintViolationError`.
- SQL injection protection in `save_to_hive()` via identifier allowlist.

### Changed
- Version bump from 1.2.3 to 1.3.0.
- `LinkageModel` default method changed from GaussianMixture to empirical histogram resampler (with optional NegBinom fit).

## 1.2.3

### Fixed
- `TypeError` in `DataTransformer` when applying numeric constraints (min, max, dtype) to categorical/string columns.
- Added robust type coercion to ensure constraints are applied correctly to transformed data.

## 1.2.2

### Fixed
- CTGAN embedding cardinality to avoid `IndexError` when using high-cardinality categorical columns.
- Databricks example returns in-memory DataFrames and cleans timestamps/nulls for safer Arrow/pandas conversion.

### Added
- Initial MkDocs site scaffold with guides, demos, and API reference.
