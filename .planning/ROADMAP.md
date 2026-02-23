# Roadmap: SynthoHive

## Milestones

- ✅ **v1.0 Core Reliability** — Phase 1 (shipped 2026-02-22)
- 🚧 **v1.1 Relational Correctness** — Phases 2-3 (planned)
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

### 🚧 v1.1 Relational Correctness (Phases 2-3)

- [ ] **Phase 2: Relational Correctness** - Fix FK context conditioning, cardinality distribution, memory safety, and schema validation
- [ ] **Phase 3: Model Pluggability** - Decouple CTGAN from orchestrator; expose pluggable model strategy via ConditionalGenerativeModel ABC

### 📋 v1.2 Quality & Connectors (Phases 4-5)

- [ ] **Phase 4: Validation and Quality Gates** - Wire statistical validation into sample(), emit training metrics, checkpoint on quality not generator loss
- [ ] **Phase 5: SQL Connectors** - Read directly from Postgres, MySQL, Snowflake, BigQuery without requiring a Spark session

## Phase Details

### Phase 2: Relational Correctness
**Goal**: Multi-table synthesis produces correct referential integrity — FK columns join cleanly with zero orphans, cardinality reflects the real parent distribution, and generation stays memory-bounded regardless of schema size
**Depends on**: Phase 1 (v1.0)
**Requirements**: REL-01, REL-02, REL-03, REL-04, REL-05, CONN-02, TEST-02
**Success Criteria** (what must be TRUE):
  1. Generated parent and child tables join on FK columns with zero orphaned references and zero missing parents on any schema with 2+ tables
  2. `StagedOrchestrator` uses freshly sampled parent context per generator training step — FK value distributions in child output match the empirical distribution of parent PK values
  3. FK type mismatches between parent PK and child FK (e.g., int vs. string) are raised at `validate_schema()` time before training begins
  4. Multi-table generation with `output_path_base` set keeps peak memory bounded to at most two DataFrames simultaneously — no accumulation of all tables in RAM
  5. PySpark 4.0+ and delta-spark 4.0+ version pins in `pyproject.toml` match the installed venv without conflict
**Plans**: 5 plans

Plans:
- [ ] 02-01-PLAN.md — Add SchemaValidationError to exception hierarchy; extend validate_schema() with collect-all FK type/column checks; add linkage_method to TableConfig
- [ ] 02-02-PLAN.md — Fix stale context conditioning in CTGAN generator training loop; memory-safe DataFrame release in orchestrator; update Spark version pins
- [ ] 02-03-PLAN.md — Replace GaussianMixture in LinkageModel with empirical/NegBinom cardinality distribution
- [ ] 02-04-PLAN.md — Write TEST-02: 3-table and 4-table FK chain zero-orphan tests, schema validation error tests, cardinality accuracy test
- [ ] 02-05-PLAN.md — Gap closure: fix test_interface.py regression — update ValueError assertions to SchemaValidationError

### Phase 3: Model Pluggability
**Goal**: `StagedOrchestrator` accepts any class implementing `ConditionalGenerativeModel` via dependency injection — CTGAN is the default but is no longer hardcoded, and the pattern is validated by a working second model
**Depends on**: Phase 2
**Requirements**: MODEL-01, MODEL-02, MODEL-03
**Success Criteria** (what must be TRUE):
  1. `StagedOrchestrator` has no hardcoded CTGAN import in its orchestration logic — the model class is injected via `model_cls` parameter
  2. `Synthesizer(model=CustomModel)` with any class implementing the `ConditionalGenerativeModel` ABC (`fit`, `sample`, `save`, `load`) routes correctly through the full multi-table pipeline
  3. The `Synthesizer` public API documents the `model` parameter with the list of supported model classes; existing callers omitting `model` get CTGAN by default with no behavior change
**Plans**: 2 plans

Plans:
- [ ] 03-01-PLAN.md — Add model_cls DI parameter to StagedOrchestrator; replace both CTGAN() call sites with self.model_cls(); update Dict type annotation; add issubclass guard; document constructor convention in ABC
- [ ] 03-02-PLAN.md — Replace backend: str with model: Type[ConditionalGenerativeModel] = CTGAN in Synthesizer; forward model_cls to StagedOrchestrator; write StubModel integration test proving MODEL-03 ABC contract end-to-end

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
| 2. Relational Correctness | 4/5 | In Progress | - | - |
| 3. Model Pluggability | v1.1 | 0/2 | Planned | - |
| 4. Validation and Quality Gates | v1.2 | 0/3 | Not started | - |
| 5. SQL Connectors | v1.2 | 0/3 | Not started | - |
