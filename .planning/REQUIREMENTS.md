# Requirements: SynthoHive

**Defined:** 2026-02-22
**Core Value:** A data engineer can train on any multi-table schema, generate synthetic data, and trust that code written against that data will work on the real thing — without babysitting training or manually validating output.

## v1 Requirements

### Core Reliability

- [ ] **CORE-01**: Training completes without crashes, hangs, or silent failures — all exceptions surface with actionable error messages
- [ ] **CORE-02**: `save()` persists full model state (DataTransformer, context_transformer, data_column_info, embedding_layers, generator/discriminator weights) to a single artifact
- [ ] **CORE-03**: `load()` restores a saved model and runs `sample()` successfully without requiring the original training data or retraining
- [ ] **CORE-04**: Bare `except:` and silent `except Exception: pass` blocks are eliminated project-wide; all exceptions are typed and logged with context
- [ ] **CORE-05**: Training emits structured progress feedback (epoch number, loss values, ETA) so engineers know the job is alive

### Relational Integrity

- [ ] **REL-01**: Child table generation uses freshly sampled parent context per generator training step — not the discriminator's stale last-batch context
- [ ] **REL-02**: FK cardinality uses empirical distribution or Negative Binomial — not Gaussian Mixture, which produces negative sample counts
- [ ] **REL-03**: FK type mismatches are detected and raised at `validate_schema()` time, before training begins
- [ ] **REL-04**: Multi-table generation releases DataFrames from memory after writing to disk when `output_path_base` is set — no OOM accumulation
- [ ] **REL-05**: Generated parent/child tables can be joined on FK columns with zero orphaned references and zero missing parents

### Data Quality

- [ ] **QUAL-01**: `sample()` optionally enforces statistical quality gates (configurable TVD threshold per column); generation raises if output fails validation
- [ ] **QUAL-02**: Column-level quality metrics (KS statistic, TVD, correlation delta) are emitted after every `sample()` call, not just on explicit report request
- [ ] **QUAL-03**: Model checkpointing uses validation metrics (TVD or KS) as the criterion — not generator loss, which does not correlate with output quality in WGAN-GP
- [ ] **QUAL-04**: Numeric constraints (min, max, dtype) are enforced on generated output; violations raise with the column name and observed value, not silently pass
- [ ] **QUAL-05**: `Synthesizer.fit()` accepts a `seed` parameter that produces deterministic DataTransformer encoding, training initialization, and sample output

### Connectors

- [ ] **CONN-01**: SQL connector reads tables from Postgres, MySQL, Snowflake, and BigQuery via SQLAlchemy 2.0 + dialect drivers; returns Pandas DataFrame
- [ ] **CONN-02**: Spark/Delta Lake connector works correctly with PySpark 4.0+ and delta-spark 4.0+; `pyproject.toml` pins updated to match actual installed versions
- [ ] **CONN-03**: CSV and Parquet connectors work without a Spark session — Pandas-based path for engineers without a Spark cluster
- [ ] **CONN-04**: `save_to_hive()` database name is validated against an allowlist pattern before SQL interpolation — no injection via unsanitized user input

### Model Architecture

- [ ] **MODEL-01**: `StagedOrchestrator` accepts a `model_cls` parameter (default: `CTGAN`) and uses the pluggable `ConditionalGenerativeModel` ABC — no hardcoded CTGAN import in orchestration logic
- [ ] **MODEL-02**: `Synthesizer` exposes a `model` parameter to users with a documented list of supported model classes
- [ ] **MODEL-03**: Any class implementing `ConditionalGenerativeModel` (fit, sample, save, load) can be passed as `model_cls` and orchestrated correctly through the multi-table pipeline

### Testing

- [ ] **TEST-01**: End-to-end test: single-table fit → sample → validate — passes with a known small dataset (iris, titanic, or equivalent)
- [ ] **TEST-02**: End-to-end test: multi-table fit → sample → FK join validates zero orphans on a 3-table schema (parent → child → grandchild)
- [ ] **TEST-03**: Serialization round-trip test: fit → save → load → sample produces output matching the pre-save sample distribution (within TVD tolerance)
- [ ] **TEST-04**: Connector test: SQL connector reads from at least one dialect (Postgres via pg8000 or psycopg2) and produces a correctly-typed Pandas DataFrame
- [ ] **TEST-05**: Regression test: training with a fixed seed produces bit-identical synthetic output across two independent runs

## v2 Requirements

### Privacy

- **PRIV-01**: PIISanitizer wired into `StagedOrchestrator.fit_all()` when `PrivacyConfig` is provided — PII is sanitized before model training, not after
- **PRIV-02**: SHA256 PII hash replaced with salted HMAC-SHA256 to prevent rainbow-table reversal
- **PRIV-03**: Parent-context PII leakage guard: when parent FK columns are propagated to child rows, sanitizer is applied to those columns in the child context

### Extended Models

- **MODEL-04**: TVAE (Variational Autoencoder for tabular data) implemented as a second `ConditionalGenerativeModel` and registered as a supported `model` option
- **MODEL-05**: Training progress and convergence diagnostics differ per model type; each model exposes a `training_metrics()` method

### Checkpoint Resume

- **CHK-01**: Long training runs can be interrupted and resumed from the last saved checkpoint without data loss or retraining from scratch

### Extended Validation

- **QUAL-06**: `generate_validation_report()` includes mutual information between column pairs (beyond marginal KS/TVD) to catch correlation drift
- **QUAL-07**: Referential integrity check included in validation report: orphan rate, missing parent rate, cardinality distribution comparison

## Out of Scope

| Feature | Reason |
|---------|--------|
| REST API / server mode | Wrong interface for data engineers; orthogonal to ML quality; adds auth/networking complexity |
| CLI interface | Python SDK is the interface; YAML config schemas are harder to maintain than Python objects |
| Differential privacy by default | Degrades statistical fidelity; acceptable as explicit opt-in only (already in PrivacyConfig as a future parameter) |
| Real-time / streaming synthesis | CTGAN requires batch inference; single-row generation has poor statistical properties |
| Auto-tuning of hyperparameters | Premature optimization before basic training reliability is solved |
| GUI / web dashboard | Wrong audience; data engineers use notebooks and scripts |
| Federated / distributed model training | Infrastructure complexity entirely out of scope for v1 |
| Auto-remediation of constraint violations | Silently patching generated data masks model quality issues |

## Traceability

*Updated during roadmap creation — 2026-02-22*

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 1 | Pending |
| CORE-02 | Phase 1 | Pending |
| CORE-03 | Phase 1 | Pending |
| CORE-04 | Phase 1 | Pending |
| CORE-05 | Phase 4 | Pending |
| REL-01 | Phase 2 | Pending |
| REL-02 | Phase 2 | Pending |
| REL-03 | Phase 2 | Pending |
| REL-04 | Phase 2 | Pending |
| REL-05 | Phase 2 | Pending |
| QUAL-01 | Phase 4 | Pending |
| QUAL-02 | Phase 4 | Pending |
| QUAL-03 | Phase 4 | Pending |
| QUAL-04 | Phase 1 | Pending |
| QUAL-05 | Phase 1 | Pending |
| CONN-01 | Phase 5 | Pending |
| CONN-02 | Phase 2 | Pending |
| CONN-03 | Phase 5 | Pending |
| CONN-04 | Phase 1 | Pending |
| MODEL-01 | Phase 3 | Pending |
| MODEL-02 | Phase 3 | Pending |
| MODEL-03 | Phase 3 | Pending |
| TEST-01 | Phase 1 | Pending |
| TEST-02 | Phase 2 | Pending |
| TEST-03 | Phase 1 | Pending |
| TEST-04 | Phase 5 | Pending |
| TEST-05 | Phase 1 | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-22*
*Last updated: 2026-02-22 after roadmap creation*
