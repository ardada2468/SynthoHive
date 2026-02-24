# Roadmap: SynthoHive

## Milestones

- ✅ **v1.0 Core Reliability** — Phase 1 (shipped 2026-02-22)
- ✅ **v1.1 Relational Correctness** — Phases 2, 3, 6, 7 (shipped 2026-02-24)
- 📋 **v1.2 Quality & Connectors** — Phases 8, 9, 10, 11 (planned)

## Phases

<details>
<summary>✅ v1.0 Core Reliability (Phase 1) — SHIPPED 2026-02-22</summary>

- [x] Phase 1: Core Reliability (5/5 plans) — completed 2026-02-22

**Goal**: Engineers can train a model, save it, and load it in a separate process for inference — with typed exceptions surfacing every failure and a deterministic seed producing reproducible output
**Requirements satisfied**: CORE-01, CORE-02, CORE-03, CORE-04, QUAL-04, QUAL-05, CONN-04, TEST-01, TEST-03, TEST-05

Plans:
- [x] 01-01-PLAN.md — Create exception hierarchy; audit and replace all bare/silent excepts; wrap synthesizer public API boundaries
- [x] 01-02-PLAN.md — Fix CTGAN.save()/load() to persist full checkpoint via directory-based layout using joblib + torch
- [x] 01-03-PLAN.md — Add seed parameter to CTGAN.fit()/sample() and DataTransformer; add enforce_constraints opt-in to sample()
- [x] 01-04-PLAN.md — Patch save_to_hive() SQL injection; write TEST-01, TEST-03, TEST-05 test files
- [x] 01-05-PLAN.md — Gap closure: raise ConstraintViolationError on violations; replace silent excepts with logged warnings; add constraint violation test

Archive: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v1.1 Relational Correctness (Phases 2, 3, 6, 7) — SHIPPED 2026-02-24</summary>

- [x] Phase 2: Relational Correctness (5/5 plans) — completed 2026-02-23
- [x] Phase 3: Model Pluggability (3/3 plans) — completed 2026-02-23
- [x] Phase 6: Synthesizer Validation Hardening (1/1 plan) — completed 2026-02-23
- [x] Phase 7: Test Suite Alignment (1/1 plan) — completed 2026-02-23

**Requirements satisfied**: REL-01, REL-02, REL-03, REL-04, REL-05, CONN-02, TEST-02, MODEL-01, MODEL-02, MODEL-03, TEST-SH

### Phase 2: Relational Correctness
**Goal**: Multi-table synthesis produces correct referential integrity — FK columns join cleanly with zero orphans, cardinality reflects the real parent distribution, and generation stays memory-bounded regardless of schema size

Plans:
- [x] 02-01-PLAN.md — Add SchemaValidationError; extend validate_schema() with collect-all FK type/column checks
- [x] 02-02-PLAN.md — Fix stale context conditioning in CTGAN training loop; memory-safe DataFrame release; update Spark version pins
- [x] 02-03-PLAN.md — Replace GaussianMixture in LinkageModel with empirical/NegBinom cardinality distribution
- [x] 02-04-PLAN.md — Write TEST-02: 3-table and 4-table FK chain zero-orphan tests
- [x] 02-05-PLAN.md — Gap closure: fix test_interface.py regression — ValueError → SchemaValidationError assertions

### Phase 3: Model Pluggability
**Goal**: `StagedOrchestrator` accepts any class implementing `ConditionalGenerativeModel` via dependency injection

Plans:
- [x] 03-01-PLAN.md — Add model_cls DI parameter to StagedOrchestrator; issubclass guard; ABC
- [x] 03-02-PLAN.md — Replace backend: str with model: Type[ConditionalGenerativeModel] in Synthesizer; StubModel integration test
- [x] 03-03-PLAN.md — Gap closure: add importmode = "importlib" to pyproject.toml

### Phase 6: Synthesizer Validation Hardening
**Goal**: Synthesizer public façade enforces both structural and data-level FK validation; invalid model class injection fails at `__init__` regardless of Spark session

Plans:
- [x] 06-01-PLAN.md — Add issubclass guard in Synthesizer.__init__() (TD-04); pass real_data to validate_schema() in fit() (TD-01)

### Phase 7: Test Suite Alignment
**Goal**: All tests in `test_interface.py` pass with zero pre-existing failures

Plans:
- [x] 07-01-PLAN.md — Update test_interface.py: ValueError → TrainingError assertions; fix stale call signature checks (14/14 passing)

Archive: `.planning/milestones/v1.1-ROADMAP.md`

</details>

### 📋 v1.2 Quality & Connectors (Phases 8–11)

**Milestone Goal:** Give data engineers visibility into training progress and generated data quality, and let them connect to SQL databases and read files without a Spark session.

- [ ] **Phase 8: Training Observability** - Wire structured progress and quality-driven checkpointing into the fit() loop (**2 plans**)
- [ ] **Phase 9: Sample Quality** - Enforce per-column quality thresholds and expose TVD/KS metrics from sample()
- [ ] **Phase 10: SQL Connectors** - Read from Postgres and MySQL via SQLAlchemy 2.0 with dtype-correct DataFrames; integration test
- [ ] **Phase 11: Pandas-Native File I/O** - CSV and Parquet connectors work without a Spark session via Pandas I/O

## Phase Details

### Phase 8: Training Observability
**Goal**: Engineers can watch `fit()` run in real time — seeing epoch number, loss values, and estimated time remaining as structured log events — and trust that the saved checkpoint represents the epoch with the best statistical quality, not the lowest generator loss
**Depends on**: Phase 7 (v1.1 complete)
**Requirements**: CORE-05, QUAL-03
**Success Criteria** (what must be TRUE):
  1. Running `fit()` emits at least one structured log event per epoch containing epoch number, generator loss, discriminator loss, and a non-zero ETA estimate
  2. An engineer consuming the logs can determine current epoch and estimated completion time without adding any custom instrumentation
  3. After `fit()` completes, the checkpoint on disk corresponds to the epoch that achieved the best validation TVD/KS score — not the final epoch or the epoch with lowest generator loss
  4. A subsequent cold `load()` + `sample()` uses the best-quality checkpoint, producing output with statistical quality at least as good as the best epoch seen during training
**Plans**: TBD

**Plans:** 1/2 plans executed

Plans:
- [ ] 08-01-PLAN.md — Add tqdm + structlog events to CTGAN.fit(); thread progress_bar and checkpoint_interval through Synthesizer and StagedOrchestrator
- [ ] 08-02-PLAN.md — Replace generator-loss checkpointing with val-metric checkpointing (best_checkpoint/ + final_checkpoint/); write test_training_observability.py and update test_checkpointing.py

### Phase 9: Sample Quality
**Goal**: Engineers can call `sample()` and immediately see per-column statistical quality metrics in logs, and optionally fail fast with a `QualityError` when any column's TVD exceeds an explicit threshold
**Depends on**: Phase 8
**Requirements**: QUAL-01, QUAL-02
**Success Criteria** (what must be TRUE):
  1. Every call to `sample()` emits structured log events containing TVD and KS scores for each column in the generated DataFrame — no additional method call required
  2. `sample(quality_threshold=0.15)` raises `QualityError` when any column's TVD exceeds 0.15; the exception includes the failing column names and their TVD scores; no DataFrame is returned silently
  3. `sample(return_metrics=True)` returns a `(DataFrame, metrics_dict)` tuple where `metrics_dict` contains per-column TVD and KS values; the default `sample()` call continues to return only a DataFrame (no breaking API change)
  4. `QualityError` is a typed subclass within the `SynthoHiveError` hierarchy and includes structured fields for failing columns and scores (not a free-form string)
**Plans**: TBD

Plans:
- [ ] 09-01: Add QualityError to exception hierarchy; wire StatisticalValidator into sample() pipeline — emit column-level TVD/KS via structlog after every call
- [ ] 09-02: Add quality_threshold gate to sample() — raise QualityError with column names and TVD scores on threshold breach; add return_metrics=True path returning (DataFrame, metrics_dict)

### Phase 10: SQL Connectors
**Goal**: Engineers can point `SQLConnector` at a Postgres or MySQL database and receive a correctly-typed Pandas DataFrame without configuring a Spark session, then verify the connector with an integration test against a live Postgres instance
**Depends on**: Phase 9
**Requirements**: CONN-01, TEST-04
**Success Criteria** (what must be TRUE):
  1. `SQLConnector(url="postgresql+psycopg2://...")` instantiates and `read_table("schema.table")` returns a Pandas DataFrame with column dtypes that match the source schema (int columns are int64, string columns are object, etc.)
  2. The same `SQLConnector` API works with a MySQL connection URL using pymysql — no code changes required beyond the URL
  3. No PySpark session is created or required at any point during `SQLConnector` construction or `read_table()` execution
  4. The integration test `test_sql_connector_postgres` reads a known table from a Postgres instance and asserts correct column dtypes on the returned DataFrame; the test passes in the local dev environment
**Plans**: TBD

Plans:
- [ ] 10-01: Implement SQLConnector in syntho_hive/connectors/sql_io.py using SQLAlchemy 2.0 — Postgres (psycopg2) and MySQL (pymysql) dialects; correct dtype mapping; no Spark dependency
- [ ] 10-02: Write TEST-04 integration test — read from Postgres via psycopg2/SQLAlchemy; assert column dtypes on returned DataFrame

### Phase 11: Pandas-Native File I/O
**Goal**: Engineers can use `CSVConnector` and `ParquetConnector` to read files without an active Spark session — all file I/O routes through Pandas when Spark is unavailable
**Depends on**: Phase 10
**Requirements**: CONN-03
**Success Criteria** (what must be TRUE):
  1. `CSVConnector(path).read()` returns a Pandas DataFrame when no Spark session is present — no `SparkSession` import error or fallback failure
  2. `ParquetConnector(path).read()` returns a Pandas DataFrame when no Spark session is present — pyarrow or fastparquet used for Parquet decoding
  3. When a Spark session is available, both connectors continue to use the Spark-based path — the Pandas path is additive, not a replacement
  4. Existing tests that use CSV/Parquet connectors with a mocked Spark session continue to pass without modification
**Plans**: TBD

Plans:
- [ ] 11-01: Decouple CSVConnector and ParquetConnector from mandatory Spark session — add Pandas-native read path with Spark-first, Pandas-fallback routing; update tests

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Core Reliability | v1.0 | 5/5 | Complete | 2026-02-22 |
| 2. Relational Correctness | v1.1 | 5/5 | Complete | 2026-02-23 |
| 3. Model Pluggability | v1.1 | 3/3 | Complete | 2026-02-23 |
| 6. Synthesizer Validation Hardening | v1.1 | 1/1 | Complete | 2026-02-23 |
| 7. Test Suite Alignment | v1.1 | 1/1 | Complete | 2026-02-23 |
| 8. Training Observability | 1/2 | In Progress|  | - |
| 9. Sample Quality | v1.2 | 0/2 | Not started | - |
| 10. SQL Connectors | v1.2 | 0/2 | Not started | - |
| 11. Pandas-Native File I/O | v1.2 | 0/1 | Not started | - |
