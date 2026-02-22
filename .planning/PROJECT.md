# SynthoHive

## What This Is

SynthoHive is a Python SDK for synthetic data generation targeting data engineers. It trains on real multi-table datasets and generates statistically realistic fake data that preserves referential integrity — so pipelines, queries, and application code built against synthetic data work unchanged on production data. The existing codebase has the right architecture but suffers from unreliable ML training, unverifiable output quality, and broken relational integrity for non-trivial schemas.

## Core Value

A data engineer can give SynthoHive a real multi-table schema, train on it, generate synthetic data, and trust that code written against that data will work on the real thing — without babysitting the training process or manually validating the output.

## Requirements

### Validated

- ✓ CTGAN-based generative model for tabular data — existing (`syntho_hive/core/models/ctgan.py`)
- ✓ DataTransformer: reversible encoding of mixed-type tabular data — existing (`syntho_hive/core/data/transformer.py`)
- ✓ Multi-table schema definition via Pydantic (`Metadata`, `TableConfig`, `Constraint`) — existing (`syntho_hive/interface/config.py`)
- ✓ SchemaGraph: DAG-based dependency ordering for multi-table training — existing (`syntho_hive/relational/`)
- ✓ Spark/Delta Lake connector for data I/O — existing (`syntho_hive/connectors/spark_io.py`)
- ✓ Pandas DataFrame support — existing (throughout codebase)
- ✓ CSV/Parquet file support — existing (`syntho_hive/connectors/`)
- ✓ Statistical validation framework (KS test, TVD, correlation) — existing (`syntho_hive/validation/`)
- ✓ PII detection and sanitization — existing (`syntho_hive/privacy/`)
- ✓ High-level `Synthesizer` API (`fit()`, `sample()`) — existing (`syntho_hive/interface/synthesizer.py`)

### Active

- [ ] Reliable end-to-end training: training completes without crashes, hangs, or silent failures
- [ ] Correct model serialization: save/load full model state (DataTransformer + CTGAN weights) for inference without retraining
- [ ] Correct child table generation: context conditioning uses freshly sampled parent context per child row, not last-batch stale context
- [ ] Memory-safe multi-table generation: large schemas don't OOM when writing to disk
- [ ] Statistical quality gates: training emits measurable quality metrics; generation can be auto-validated against a threshold
- [ ] Pluggable model strategy: CTGAN as default, architecture designed to allow swapping in other models (TVAE, etc.)
- [ ] SQL database connector: connect to Postgres/MySQL/Snowflake/BigQuery and read table data directly
- [ ] Robust error handling: all exceptions surfaced with actionable messages (no bare `except:`, no silent swallowing)
- [ ] End-to-end test suite: automated tests covering single-table, multi-table, and referential integrity scenarios

### Out of Scope

- CLI interface — Python SDK is the interface; CLI adds complexity without value for the target user
- REST API / server mode — not needed for data engineering workflows
- Real-time / streaming data — batch-only for v1
- Mobile / web UI — SDK only
- PII auto-remediation in generated output — PII detection exists; active enforcement deferred until core ML is trustworthy

## Context

The codebase (`syntho_hive/`) is a fully structured Python package with the right architecture. The critical bugs blocking production use are:

1. **Incomplete model serialization** (`ctgan.py:482-503`): Save/load only persists generator/discriminator weights, not DataTransformer. Loaded models can't run inference.
2. **Context data mismatch in child table generation** (`ctgan.py:359-362`): Child rows generated using stale last-batch context instead of freshly sampled parent context — this is the root cause of referential integrity failures on non-trivial schemas.
3. **Memory accumulation in multi-table generation** (`orchestrator.py:224-226`): All generated DataFrames held in memory even when writing to disk. Fails on large schemas.
4. **Silent error handling** (`synthesizer.py:153`, `transformer.py:222-246`): Errors swallowed silently, making debugging impossible.
5. **Unimplemented embedding logic** (`ctgan.py:132-185`): `_apply_embeddings()` is a stub, creating confusion about data flow.

Stack: Python 3.9+, PyTorch, PySpark, Pandas, scikit-learn, SciPy, Pydantic v2, Faker.

## Constraints

- **Tech Stack**: Python 3.9+ — not changing runtime or primary language
- **ML Framework**: PyTorch — CTGAN implementation stays in PyTorch; any new models must also use PyTorch
- **Compatibility**: Existing `Synthesizer` public API (`fit()`, `sample()`, `generate_validation_report()`) must remain stable — data engineers using it shouldn't need to rewrite their code
- **Data Scale**: Must handle datasets large enough to require Spark (100M+ rows) for the Spark path; in-memory Pandas path for smaller datasets

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CTGAN as default model | Well-proven for tabular mixed-type data; existing implementation to build on | — Pending |
| Pluggable model architecture | Avoid lock-in; allows TVAE or other models without rewriting orchestrator | — Pending |
| Fix existing codebase vs rewrite | Architecture is sound; bugs are specific and fixable; rewrite risks introducing new problems | — Pending |
| Spark as optional (not required) dependency | Data engineers without Spark clusters need to use the SDK too | — Pending |

---
*Last updated: 2026-02-22 after initialization*
