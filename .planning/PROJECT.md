# SynthoHive

## What This Is

SynthoHive is a Python SDK for synthetic data generation targeting data engineers. It trains on real multi-table datasets and generates statistically realistic fake data that preserves referential integrity — so pipelines, queries, and application code built against synthetic data work unchanged on production data.

**v1.0 shipped:** The codebase now has a reliable CTGAN pipeline — typed exceptions, full model serialization, deterministic seeding, constraint enforcement, and SQL injection prevention. The core ML pipeline is trustworthy; multi-table relational correctness is next.

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
- ✓ CORE-01: Training completes without crashes/silent failures — typed exceptions surface all errors — v1.0
- ✓ CORE-02: `save()` persists full model state (7 components) to directory checkpoint — v1.0
- ✓ CORE-03: `load()` restores saved model; `sample()` works without retraining — v1.0
- ✓ CORE-04: All bare/silent except blocks eliminated; exceptions typed and logged — v1.0
- ✓ QUAL-04: Numeric constraint violations raise `ConstraintViolationError` with column name and observed value — v1.0
- ✓ QUAL-05: `Synthesizer.fit(seed=N)` produces bit-identical output across independent runs — v1.0
- ✓ CONN-04: `save_to_hive()` validates identifiers via `_SAFE_IDENTIFIER` allowlist before SQL interpolation — v1.0
- ✓ TEST-01: E2E single-table fit → sample → validate test passes — v1.0
- ✓ TEST-03: Serialization round-trip test (fit → save → load → sample) passes — v1.0
- ✓ TEST-05: Seed regression test (bit-identical output with `check_exact=True`) passes — v1.0

### Active

- [ ] REL-01: Child table generation uses freshly sampled parent context (not stale last-batch context)
- [ ] REL-02: FK cardinality uses empirical distribution or Negative Binomial (not Gaussian Mixture)
- [ ] REL-03: FK type mismatches raised at `validate_schema()` time, before training
- [ ] REL-04: Multi-table generation releases DataFrames after disk write — no OOM accumulation
- [ ] REL-05: Generated parent/child tables join with zero orphans and zero missing parents
- [ ] CONN-02: PySpark 4.0+ and delta-spark 4.0+ pins in pyproject.toml match installed venv
- [ ] TEST-02: Multi-table E2E test: 3-table FK join validates zero orphans
- [ ] MODEL-01: `StagedOrchestrator` accepts `model_cls` parameter — no hardcoded CTGAN in orchestration
- [ ] MODEL-02: `Synthesizer` exposes `model` parameter with documented supported classes
- [ ] MODEL-03: Any class implementing `ConditionalGenerativeModel` ABC routes correctly through pipeline
- [ ] CORE-05: Training emits structured progress (epoch, loss, ETA)
- [ ] QUAL-01: `sample(quality_threshold=N)` raises with failing column names and TVD scores
- [ ] QUAL-02: Column-level quality metrics emitted after every `sample()` call
- [ ] QUAL-03: Model checkpoints on validation metric (TVD/KS), not generator loss
- [ ] CONN-01: SQL connector reads from Postgres, MySQL, Snowflake, BigQuery via SQLAlchemy 2.0
- [ ] CONN-03: CSV/Parquet connectors work without Spark session (Pandas-native path)
- [ ] TEST-04: Connector test reads from Postgres via psycopg2 and produces correctly-typed DataFrame

### Out of Scope

- CLI interface — Python SDK is the interface; CLI adds complexity without value for the target user
- REST API / server mode — not needed for data engineering workflows
- Real-time / streaming data — batch-only for v1
- Mobile / web UI — SDK only
- PII auto-remediation in generated output — PII detection exists; active enforcement deferred until core ML is trustworthy
- Differential privacy by default — degrades statistical fidelity; acceptable as explicit opt-in only
- Auto-tuning of hyperparameters — premature optimization before basic training reliability is solved

## Context

**v1.0 shipped 2026-02-22.** The core CTGAN single-table pipeline is now trustworthy:
- Exception hierarchy: `SynthoHiveError` → 5 typed subclasses; zero bare excepts anywhere in `syntho_hive/`
- Serialization: 7-file directory checkpoint; cold `load()` + `sample()` without retraining verified
- Determinism: `fit(seed=42).sample(100, seed=7)` bit-identical across independent runs
- Security: SQL injection prevented in `save_to_hive()` via `_SAFE_IDENTIFIER` allowlist regex
- Test harness: 12 regression tests covering E2E training, serialization, seed, and constraint violation

**Stack:** Python 3.14, PyTorch 2.6+, PySpark 4.0+, delta-spark 4.0+, Pandas, scikit-learn, SciPy, Pydantic v2, joblib, structlog, Faker.

**Known issues / tech debt from v1.0:**
- Stale `.venv` — `pip install -e .` required before Phase 2 (otherwise PYTHONPATH must be set)
- No `tests/` regression test for `SchemaError` on invalid DB name (CONN-04 rejection path — implementation correct, coverage gap)
- Pandas 2.x copy-on-write semantics may affect `transformer.py` `.values` mutations — audit before pinning `pandas>=2.0.0`

## Constraints

- **Tech Stack**: Python 3.9+ — not changing runtime or primary language
- **ML Framework**: PyTorch — CTGAN implementation stays in PyTorch; any new models must also use PyTorch
- **Compatibility**: Existing `Synthesizer` public API (`fit()`, `sample()`, `generate_validation_report()`) must remain stable — data engineers using it shouldn't need to rewrite their code
- **Data Scale**: Must handle datasets large enough to require Spark (100M+ rows) for the Spark path; in-memory Pandas path for smaller datasets

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix existing codebase vs rewrite | Architecture is sound; bugs are specific and fixable; rewrite risks introducing new problems | ✓ Good — Phase 1 fixed 5 targeted bugs without regression |
| CTGAN as default model | Well-proven for tabular mixed-type data; existing implementation to build on | ✓ Good — reliable after Phase 1 |
| Pluggable model architecture | Avoid lock-in; allows TVAE or other models without rewriting orchestrator | — Pending (Phase 3) |
| Spark as optional (not required) dependency | Data engineers without Spark clusters need to use the SDK too | — Pending (Phase 5) |
| `raise...from exc` exception chaining | Callers always see chained tracebacks including root cause | ✓ Good — surfaced PyTorch 2.6 incompatibility immediately |
| joblib for full checkpoint serialization | Handles sklearn objects and nn.ModuleDict; avoids state_dict limitations | ✓ Good — cold load verified across 7 components |
| Directory-based checkpoint (not single .pt) | Allows per-component serialization strategy (torch vs joblib) | ✓ Good — clean separation, extensible |
| Per-column hash-derived seeds | Prevents correlated BayesianGMM fitting across columns | ✓ Good — bit-identical reproducibility confirmed |
| Allowlist regex for SQL identifiers | Closed-by-default; denylist always risks missing new injection vectors | ✓ Good — SchemaError raised before any Spark operation |
| `enforce_constraints=False` default | Preserves backward compatibility; violations were previously silently filtered | ✓ Good — callers opt into raise behavior |
| TVAE architecture research before Phase 3 | Avoid repeating CTGAN embedding stub pattern | ⚠️ Revisit before Phase 3 starts |

---
*Last updated: 2026-02-22 after v1.0 milestone*
