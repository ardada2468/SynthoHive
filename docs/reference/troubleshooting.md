---
title: Troubleshooting
---

# Troubleshooting

## General

- **Spark not found**: Ensure `pyspark` is installed and `SPARK_HOME` is set. For local-only runs, some demos may be limited.
- **Delta support**: Install `delta-spark` and use Spark 3.2+.
- **GPU vs CPU**: CTGAN runs on CPU by default; set `device="cuda"` when available.
- **High-cardinality categoricals**: Increase `embedding_threshold` to use embeddings instead of OHE.
- **Validation failures**: Inspect KS/TVD results; large TVD often means categorical imbalance—check sampling strategy or increase training epochs.
- **Doc build errors**: Run `pip install -e .[docs]`; ensure `mkdocs` and `mkdocstrings` are installed.

## Exception Types

SynthoHive uses a typed exception hierarchy. Catch specific exceptions for targeted error handling:

| Exception | When raised |
| :--- | :--- |
| `SynthoHiveError` | Base class for all SynthoHive errors. |
| `SchemaError` | Invalid metadata, missing FK definitions, invalid identifiers. Raised by `Metadata.add_table()` (previously raised `ValueError`). |
| `SchemaValidationError` | FK type mismatches, missing FK columns, or invalid FK references detected during `validate_schema()`. |
| `TrainingError` | NaN loss, training divergence, GPU OOM during `fit()`. |
| `SerializationError` | Save/load failures, corrupt or incompatible checkpoints. |
| `ConstraintViolationError` | Generated data violates numeric constraints when `enforce_constraints=True`. |
| `GenerationError` | Synthesis failures during `sample()` or orchestration. *(New in v1.4.0)* |
| `PrivacyError` | PII sanitization failures (e.g., invalid `pii_map` columns). *(New in v1.4.0)* |

## Common Issues Fixed in v1.4.0

- **Self-referencing FKs cause cycle error**: Self-referencing foreign keys (e.g., `manager_id -> employees.id`) are now handled correctly and no longer raise false cycle detection errors.
- **Re-fitting a DataTransformer produces corrupt output**: `DataTransformer.fit()` now resets internal state on re-fit. If you experienced stale data after calling `fit()` multiple times on different datasets, upgrade to v1.4.0.
- **All-null numeric column crashes BayesianGMM**: Columns that are entirely null are now represented as constants instead of crashing the GMM fitter.
- **PII detection false positives**: Default regex patterns are now fully anchored. If you previously saw columns incorrectly flagged as PII, this should be resolved.
- **`write_pandas` appends instead of overwriting**: The default mode is now `"overwrite"`. If you relied on append behavior, pass `mode="append"` explicitly.
- **`PrivacyConfig.epsilon` accepts zero or negative values**: Epsilon is now validated to be a positive number. Update any configurations that used `epsilon=0`.
