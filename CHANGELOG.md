# Changelog

All notable changes to SynthoHive are documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [1.4.0] - 2026-03-27

### Security
- Fixed unsalted SHA-256 hashing in PII sanitizer - now uses HMAC-SHA256 with per-instance random salt to prevent rainbow table attacks
- Fixed SQL injection vulnerability in `save_to_hive()` - path values are now validated
- Added security warning for `torch.load(weights_only=False)` usage
- Fixed `Synthesizer.load()` to type-check deserialized objects

### Fixed

#### Core Models (CTGAN)
- **CRITICAL**: Embedding layers are now included in the generator optimizer - previously they were never updated during training
- **CRITICAL**: Checkpoint validation no longer crashes when model uses context conditioning
- Fixed optimizer zero_grad ordering to follow standard PyTorch pattern (zero_grad -> forward -> backward -> step)
- `sample()` now restores generator training mode after evaluation
- Discriminator now uses both dimensions from `discriminator_dim` tuple
- Added gradient clipping (max_norm=1.0) to both generator and discriminator optimizers
- Replaced `assert` statements with proper `ValueError` raises
- Added input validation for `num_rows` in `sample()`

#### Data Transformer
- **CRITICAL**: `fit()` now properly resets internal state on re-fit, preventing stale data corruption
- **CRITICAL**: Per-column seeds now use deterministic SHA-256 hash instead of Python's non-deterministic `hash()`
- **CRITICAL**: All-null numeric columns no longer crash BayesianGMM - handled gracefully with constant representation
- Empty DataFrames now raise clear `ValueError` instead of cryptic sklearn errors
- BayesianGMM `n_components` is now clamped to available sample count
- Unknown categories during transform are handled gracefully instead of crashing
- Added epsilon to VGM normalization to prevent division by zero
- Removed dead `"number"` dtype reference in constraint checking

#### Relational Orchestration
- **CRITICAL**: Self-referencing foreign keys (e.g., `manager_id -> employees.id`) no longer raise false cycle errors
- Zero child rows now create empty DataFrames instead of silently skipping (preventing downstream crashes)
- `write_pandas` default mode changed from `"append"` to `"overwrite"` to prevent data corruption
- Added validation that `parent_context_cols` exist in parent table
- PK assignment now uses actual DataFrame length instead of requested row count
- `validate_schema()` is now called at the start of `fit_all()`
- NegBinom linkage model handles edge cases (zero variance, underdispersion) with Poisson/constant fallbacks
- `get_table()` return values are now null-checked with clear error messages
- Replaced all `print()` calls with structured `log.info()` logging

#### Privacy & Sanitization
- Fixed all default PII regex patterns to be anchored (preventing false positives on substrings)
- Added PII detection for names, addresses, and dates of birth
- Added column name alias matching (e.g., "mobile", "cell", "tel" all match "phone")
- `_mask_value` custom fallback now correctly applies per-value instead of per-Series
- Null values are now preserved through masking and hashing operations
- PII detection now uses random sampling instead of biased `head(100)`
- Added input validation for `pii_map` column names

#### Interface & Connectors
- `fit()` now warns when `sampling_strategy` parameter is passed but not yet implemented
- `Synthesizer` serialization now excludes SparkSession (via `__getstate__`/`__setstate__`)
- Fixed incorrect error message in `generate_validation_report` (said "synthetic" when reading "real" data)
- Integer-to-float FK dtype compatibility is now handled correctly
- `PrivacyConfig.epsilon` now validates positive values
- `Metadata.add_table()` now raises `SchemaError` instead of generic `ValueError`
- `RelationalSampler` now cascades sampling through full hierarchy (not just one level)
- Fixed ambiguous column references in relational sampling joins (using semi-join)

#### Exceptions
- Added `GenerationError` exception class for synthesis failures
- Added `PrivacyError` exception class for sanitization failures

### Tests
- Fixed `AssertionError` typo in seed regression test
- Replaced `sys.exit(0)` with `pytest.skip()` in retail test
- Fixed tautological null handling assertions
- Added deterministic seeds to validation and observability tests
- Migrated hardcoded file paths to pytest `tmp_path` fixtures

## [1.3.0]

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

## [1.2.3]

### Fixed
- `TypeError` in `DataTransformer` when applying numeric constraints (min, max, dtype) to categorical/string columns.
- Added robust type coercion to ensure constraints are applied correctly to transformed data.

## [1.2.2]

### Fixed
- CTGAN embedding cardinality to avoid `IndexError` when using high-cardinality categorical columns.
- Databricks example returns in-memory DataFrames and cleans timestamps/nulls for safer Arrow/pandas conversion.

### Added
- Initial MkDocs site scaffold with guides, demos, and API reference.
