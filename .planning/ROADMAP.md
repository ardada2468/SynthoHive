# Roadmap: SynthoHive

## Overview

SynthoHive has a sound architecture with five specific, fixable bugs blocking production use. This roadmap progresses from reliability (making existing behavior trustworthy) through correctness (making multi-table synthesis work) to extensibility (pluggable models) and observability (quality gates) before adding the one major missing capability (SQL connectors). Each phase delivers a coherent, verifiable improvement that unblocks the next.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Core Reliability** - Error handling, full model serialization, deterministic output, and security hardening
- [ ] **Phase 2: Relational Correctness** - Fix FK context conditioning, cardinality distribution, memory safety, and schema validation
- [ ] **Phase 3: Model Pluggability** - Decouple CTGAN from orchestrator; expose pluggable model strategy via ConditionalGenerativeModel ABC
- [ ] **Phase 4: Validation and Quality Gates** - Wire statistical validation into sample(), emit training metrics, checkpoint on quality not generator loss
- [ ] **Phase 5: SQL Connectors** - Read directly from Postgres, MySQL, Snowflake, BigQuery without requiring a Spark session

## Phase Details

### Phase 1: Core Reliability
**Goal**: Engineers can train a model, save it, and load it in a separate process for inference — with typed exceptions surfacing every failure and a deterministic seed producing reproducible output
**Depends on**: Nothing (first phase)
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, QUAL-04, QUAL-05, CONN-04, TEST-01, TEST-03, TEST-05
**Success Criteria** (what must be TRUE):
  1. `synthesizer.fit()` on any valid schema raises a typed, descriptive exception on failure — no bare `except:` blocks remain anywhere in the codebase
  2. `synthesizer.save(path)` followed by `synthesizer.load(path)` in a fresh Python process with no training data available successfully calls `sample()` and produces output
  3. `Synthesizer(seed=42).fit(data).sample(100)` produces bit-identical output across two independent runs in the same environment
  4. Numeric constraints (min, max, dtype) on generated columns raise with the column name and observed value when violated rather than silently passing
  5. `save_to_hive()` with a database name containing characters outside `[a-zA-Z0-9_]` raises a validation error before any SQL is executed
**Plans**: 4 plans

Plans:
- [x] 01-01-PLAN.md — Create exception hierarchy; audit and replace all bare/silent excepts; wrap synthesizer public API boundaries
- [ ] 01-02-PLAN.md — Fix CTGAN.save()/load() to persist full checkpoint via directory-based layout using joblib + torch
- [ ] 01-03-PLAN.md — Add seed parameter to CTGAN.fit()/sample() and DataTransformer; add enforce_constraints opt-in to sample()
- [ ] 01-04-PLAN.md — Patch save_to_hive() SQL injection; write TEST-01, TEST-03, TEST-05 test files

### Phase 2: Relational Correctness
**Goal**: Multi-table synthesis produces correct referential integrity — FK columns join cleanly with zero orphans, cardinality reflects the real parent distribution, and generation stays memory-bounded regardless of schema size
**Depends on**: Phase 1
**Requirements**: REL-01, REL-02, REL-03, REL-04, REL-05, CONN-02, TEST-02
**Success Criteria** (what must be TRUE):
  1. Generated parent and child tables join on FK columns with zero orphaned references and zero missing parents on any schema with 2+ tables
  2. `StagedOrchestrator` uses freshly sampled parent context per generator training step — FK value distributions in child output match the empirical distribution of parent PK values
  3. FK type mismatches between parent PK and child FK (e.g., int vs. string) are raised at `validate_schema()` time before training begins
  4. Multi-table generation with `output_path_base` set keeps peak memory bounded to at most two DataFrames simultaneously — no accumulation of all tables in RAM
  5. PySpark 4.0+ and delta-spark 4.0+ version pins in `pyproject.toml` match the installed venv without conflict
**Plans**: TBD

Plans:
- [ ] 02-01: Fix stale context conditioning in CTGAN generator training loop (ctgan.py:359-362); replace GMM with empirical/NegBinom distribution in LinkageModel
- [ ] 02-02: Add FK type mismatch detection in `validate_schema()`; implement memory-safe generation (release DataFrames after disk write)
- [ ] 02-03: Update pyproject.toml PySpark/delta-spark pins to match venv; write TEST-02 (3-table FK join zero-orphan test)

### Phase 3: Model Pluggability
**Goal**: `StagedOrchestrator` accepts any class implementing `ConditionalGenerativeModel` via dependency injection — CTGAN is the default but is no longer hardcoded, and the pattern is validated by a working second model
**Depends on**: Phase 2
**Requirements**: MODEL-01, MODEL-02, MODEL-03
**Success Criteria** (what must be TRUE):
  1. `StagedOrchestrator` has no hardcoded CTGAN import in its orchestration logic — the model class is injected via `model_cls` parameter
  2. `Synthesizer(model=CustomModel)` with any class implementing the `ConditionalGenerativeModel` ABC (`fit`, `sample`, `save`, `load`) routes correctly through the full multi-table pipeline
  3. The `Synthesizer` public API documents the `model` parameter with the list of supported model classes; existing callers omitting `model` get CTGAN by default with no behavior change
**Plans**: TBD

Plans:
- [ ] 03-01: Remove hardcoded CTGAN from `StagedOrchestrator`; add `model_cls` parameter defaulting to CTGAN; verify `ConditionalGenerativeModel` ABC covers fit/sample/save/load contract
- [ ] 03-02: Expose `model` parameter on `Synthesizer`; document supported classes; add integration test with a minimal stub model to verify the ABC contract end-to-end

### Phase 4: Validation and Quality Gates
**Goal**: Engineers see training progress in real-time, models checkpoint on statistical quality not generator loss, and `sample()` can enforce a minimum quality threshold automatically
**Depends on**: Phase 3
**Requirements**: CORE-05, QUAL-01, QUAL-02, QUAL-03
**Success Criteria** (what must be TRUE):
  1. Training emits structured per-epoch progress (epoch number, loss values, ETA) so an engineer watching the process can confirm the job is alive and converging
  2. Column-level quality metrics (KS statistic, TVD, correlation delta) are emitted after every `sample()` call without requiring an explicit validation report request
  3. `sample(quality_threshold=0.15)` raises with the failing column names and their TVD scores when generated output does not meet the threshold — output is not silently returned
  4. Model checkpointing saves the epoch with the best validation metric (TVD or KS), not the epoch with the lowest generator loss
**Plans**: TBD

Plans:
- [ ] 04-01: Wire structured progress callbacks into CTGAN training loop (epoch, loss, ETA); add tqdm support
- [ ] 04-02: Emit column-level quality metrics from every `sample()` call; wire `StatisticalValidator` into sample pipeline with optional threshold gate
- [ ] 04-03: Replace generator-loss checkpoint criterion with validation-metric-based checkpointing (best TVD/KS epoch)

### Phase 5: SQL Connectors
**Goal**: Engineers without a Spark cluster can read tables directly from Postgres, MySQL, Snowflake, or BigQuery and feed them to `Synthesizer.fit()` using only a connection string
**Depends on**: Phase 4
**Requirements**: CONN-01, CONN-03, TEST-04
**Success Criteria** (what must be TRUE):
  1. `SQLConnector(connection_string).read_table("schema.table")` returns a correctly-typed Pandas DataFrame from at least Postgres, MySQL, Snowflake, and BigQuery without requiring a PySpark session
  2. CSV and Parquet connectors work end-to-end via a Pandas-based path — no Spark session required for engineers without a cluster
  3. `Synthesizer.fit()` accepts a `SQLConnector` input in place of a DataFrame and trains correctly on the result
  4. The SQL connector test reads from Postgres via pg8000 or psycopg2 and produces a correctly-typed Pandas DataFrame with column dtypes matching the source schema
**Plans**: TBD

Plans:
- [ ] 05-01: Implement `SQLConnector` in `syntho_hive/connectors/sql_io.py` using SQLAlchemy 2.0 + dialect drivers (psycopg2, pymysql, snowflake-sqlalchemy, google-cloud-bigquery)
- [ ] 05-02: Decouple CSV/Parquet connectors from Spark session; add Pandas-native read path
- [ ] 05-03: Wire `SQLConnector` into `Synthesizer.fit()` input handling; write TEST-04 (Postgres connector test)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Reliability | 1/4 | In progress | - |
| 2. Relational Correctness | 0/3 | Not started | - |
| 3. Model Pluggability | 0/2 | Not started | - |
| 4. Validation and Quality Gates | 0/3 | Not started | - |
| 5. SQL Connectors | 0/3 | Not started | - |
