# Requirements: SynthoHive

**Defined:** 2026-02-23
**Milestone:** v1.2 Quality & Connectors
**Core Value:** A data engineer can train on any multi-table schema, generate synthetic data, and trust that code written against that data will work on the real thing — without babysitting training or manually validating output.

## v1.2 Requirements

Requirements for v1.2 release. Each maps to exactly one roadmap phase.

### Observability

- [x] **CORE-05**: Data engineer can observe training progress — epoch number, generator loss, discriminator loss, and ETA — emitted as structured log events during `fit()`

### Quality Enforcement

- [ ] **QUAL-01**: Data engineer can call `sample(quality_threshold=N)` and receive a `QualityError` (including failing column names and TVD scores) when any column's TVD exceeds the threshold
- [ ] **QUAL-02**: Data engineer can call `sample()` and see column-level TVD and KS metrics in structured logs; `sample(return_metrics=True)` returns a `(DataFrame, metrics_dict)` tuple without breaking the default API
- [x] **QUAL-03**: Data engineer can trust that the saved checkpoint from `fit()` corresponds to the epoch with the best validation TVD/KS score — not generator loss

### Connectors

- [ ] **CONN-01**: Data engineer can read from a Postgres or MySQL database using `SQLConnector(url=...)` and receive a correctly-typed Pandas DataFrame, with no Spark session required
- [ ] **CONN-03**: Data engineer can read CSV and Parquet files using `CSVConnector` and `ParquetConnector` without an active Spark session (Pandas-native I/O path)
- [ ] **TEST-04**: Postgres integration test reads a table via SQLAlchemy + psycopg2 and asserts correct column dtypes on the returned DataFrame

## v2 Requirements

Deferred to v1.3 or later.

### SQL Connectors (Cloud)

- **CONN-01b**: Data engineer can read from Snowflake via SQLAlchemy 2.0 + snowflake-sqlalchemy
- **CONN-01c**: Data engineer can read from BigQuery via SQLAlchemy 2.0 + sqlalchemy-bigquery

### Production-scale Validation

- **TEST-02b**: Zero-orphan guarantee verified at production scale (10k+ rows) — outstanding from v1.1 tech debt

## Out of Scope

| Feature | Reason |
|---------|--------|
| Snowflake / BigQuery connectors | Need cloud credentials for integration testing; defer to v1.3 |
| Write connectors (insert/upsert to SQL) | Read-first; write path adds auth/permission complexity; use case less common in synthetic data workflows |
| `sample(return_metrics=True)` as default (breaking) | Must preserve existing `sample() -> DataFrame` signature for existing callers |
| Auto-tune hyperparameters based on QUAL metrics | Premature; engineers need visibility first before automated tuning |
| Differential privacy enforcement | Degrades statistical fidelity; remains explicit opt-in only |

## Traceability

Which phases cover which requirements. Updated 2026-02-24 during v1.2 roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-05 | Phase 8 | Complete |
| QUAL-03 | Phase 8 | Complete |
| QUAL-01 | Phase 9 | Pending |
| QUAL-02 | Phase 9 | Pending |
| CONN-01 | Phase 10 | Pending |
| TEST-04 | Phase 10 | Pending |
| CONN-03 | Phase 11 | Pending |

**Coverage:**
- v1.2 requirements: 7 total
- Mapped to phases: 7 (100%)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-23*
*Last updated: 2026-02-24 after v1.2 roadmap creation — all 7 requirements mapped to phases 8–11*
