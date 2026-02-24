# Roadmap: SynthoHive

## Milestones

- ✅ **v1.0 Core Reliability** — Phase 1 (shipped 2026-02-22)
- ✅ **v1.1 Relational Correctness** — Phases 2, 3, 6, 7 (shipped 2026-02-24)
- 📋 **v1.2 Quality & Connectors** — Phases 4-5 (planned)

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

### 📋 v1.2 Quality & Connectors (Phases 4-5)

- [ ] **Phase 4: Validation and Quality Gates** - Wire statistical validation into sample(), emit training metrics, checkpoint on quality not generator loss
- [ ] **Phase 5: SQL Connectors** - Read directly from Postgres, MySQL, Snowflake, BigQuery without requiring a Spark session

## Phase Details

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

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Core Reliability | v1.0 | 5/5 | ✅ Complete | 2026-02-22 |
| 2. Relational Correctness | v1.1 | 5/5 | ✅ Complete | 2026-02-23 |
| 3. Model Pluggability | v1.1 | 3/3 | ✅ Complete | 2026-02-23 |
| 6. Synthesizer Validation Hardening | v1.1 | 1/1 | ✅ Complete | 2026-02-23 |
| 7. Test Suite Alignment | v1.1 | 1/1 | ✅ Complete | 2026-02-23 |
| 4. Validation and Quality Gates | v1.2 | 0/3 | Not started | - |
| 5. SQL Connectors | v1.2 | 0/3 | Not started | - |
