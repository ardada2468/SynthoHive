# Stack Research

**Domain:** Production-grade synthetic tabular data Python SDK
**Researched:** 2026-02-22
**Confidence:** HIGH (versions verified against installed venv artifacts; rationale from codebase analysis + training knowledge with explicit LOW flags where unverifiable)

---

## Context: This is a Fix-and-Extend Milestone, Not a Rebuild

The existing SynthoHive codebase has sound architecture. The goal is to identify:
1. Which existing dependencies to upgrade and to what version
2. What new dependencies to add (SQL connectors, better serialization, expanded validation)
3. What to remove or stop using
4. What to explicitly avoid

All version numbers below were verified against the actual installed `.venv/lib/python3.14/site-packages/` dist-info files unless explicitly flagged LOW confidence.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10+ (currently using 3.14) | SDK runtime | torch 2.9.x requires Python >=3.10. The codebase declares `requires-python = ">=3.9"` in pyproject.toml but the venv runs 3.14. Update pyproject.toml to `>=3.10` to match the actual torch constraint. [HIGH confidence — verified from torch-2.9.1 METADATA] |
| PyTorch | 2.9.1 (installed) | CTGAN generator/discriminator neural networks | Already at latest stable. The CTGAN uses `torch.autograd.grad()`, `state_dict()`, `nn.ModuleDict` — all stable APIs in 2.x. Upgrade from pinned `>=1.10.0` to `>=2.0.0,<3.0` in pyproject.toml. [HIGH confidence — verified] |
| Pandas | 2.3.3 (installed) | Primary data structure throughout all modules | Already at latest. Upgrade pyproject.toml from `>=1.3.0` to `>=2.0.0`. Pandas 2.x has copy-on-write semantics — review any `.values` mutations in transformer.py. [HIGH confidence — verified] |
| PySpark | 4.0.1 (installed) | Distributed data I/O, Delta Lake, multi-table orchestration | Upgrade from `>=3.2.0` to `>=4.0.0`. delta-spark 4.0.0 requires pyspark>=4.0.0 (verified from delta_spark-4.0.0 METADATA). This is a non-trivial major version jump — Spark 4.0 removes some UDF APIs. Audit `pandas_udf` usage in orchestrator.py line 4. [HIGH confidence — version verified; API stability is MEDIUM — needs testing] |
| Pydantic | 2.12.5 (installed) | Schema validation for Metadata, TableConfig, PrivacyConfig | Already at latest v2. Upgrade pyproject.toml from `>=2.0.0` to `>=2.10.0`. Pydantic v2 is substantially faster than v1. No v3 exists yet. [HIGH confidence — verified] |

### Tabular Generative Models

| Library | Version | Purpose | Recommendation |
|---------|---------|---------|----------------|
| Custom CTGAN (PyTorch) | existing | Core synthesis model | **Keep as primary.** The existing implementation in `syntho_hive/core/models/ctgan.py` is architecturally sound — the bugs are specific (serialization, context mismatch, embedding stub) not structural. Fix those bugs rather than replacing the implementation. [HIGH confidence] |
| SDV (Synthetic Data Vault) | Do NOT add as dependency | Reference implementation | **Do not import SDV as a dependency.** SDV bundles its own CTGAN/TVAE which would create a duplicate dependency chain and add ~300MB to install size. SynthoHive implements CTGAN natively. Use SDV source code only as a reference when debugging the embedding/transformer logic. [MEDIUM confidence — based on known SDV package weight] |
| TVAE (future pluggable model) | PyTorch-native, implement via base.py | Variational autoencoder alternative for tabular data | **Do not add yet.** The `ConditionalGenerativeModel` base class in `base.py` already defines the plug-in contract (`fit`, `sample`, `save`, `load`). When adding TVAE, implement it to satisfy that interface. TVAE trains faster than CTGAN and is better for low-cardinality data but produces less sharp categorical distributions. [MEDIUM confidence — training knowledge] |

### Data Connectors (SQL Databases)

The SQL connector is currently **missing entirely** — it is listed in PROJECT.md as an active requirement but no connector exists. This is the most significant new dependency addition needed.

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| SQLAlchemy | >=2.0.0 | Universal SQL database abstraction layer | SQLAlchemy 2.0 unified sync/async APIs and added strict type checking. Use `create_engine()` + `pandas.read_sql_table()` for connector reads. Supports PostgreSQL, MySQL, SQLite, Oracle, SQL Server natively via dialect drivers. The engine object serializes cleanly and can be passed as config. Do NOT use SQLAlchemy for distributed/warehouse reads — those go through Spark. [MEDIUM confidence — training knowledge, widely verified by ecosystem] |
| psycopg2-binary | >=2.9.0 | PostgreSQL dialect driver for SQLAlchemy | `psycopg2-binary` includes compiled binary, removing the need for libpq-dev at install time. For production use `psycopg2` (non-binary) to avoid LGPL binary bundling concerns. [MEDIUM confidence] |
| pymysql | >=1.1.0 | MySQL/MariaDB dialect driver | Pure Python, no system dependency. Use `mysql+pymysql://` as SQLAlchemy dialect prefix. [MEDIUM confidence] |
| snowflake-sqlalchemy | >=1.6.0 | Snowflake connector via SQLAlchemy | Official Snowflake-maintained package. Use `snowflake+snowflake://` dialect prefix. Pairs with the Snowflake Connector for Python. **Make optional** — users without Snowflake should not be forced to install it. [MEDIUM confidence] |
| google-cloud-bigquery | >=3.0.0 | BigQuery connector | BigQuery does not use SQLAlchemy. Use `bigquery.Client().query()` + `to_dataframe()` directly, or use `pandas_gbq` as a thin wrapper. The Spark path (via BigQuery Spark connector JAR) is better for large BigQuery reads. **Make optional.** [MEDIUM confidence] |

**Connector Architecture Recommendation:** Model the SQL connector as a separate `syntho_hive/connectors/sql_connector.py` that implements the same read interface as `SparkIO.read_dataset()` but returns a Pandas DataFrame directly. This avoids requiring Spark for small-to-medium SQL reads and aligns with the PROJECT.md constraint that Spark is optional.

### Model Serialization

The existing `save()`/`load()` in `ctgan.py` (lines 482–503) is broken — it saves only `state_dict()` for generator/discriminator, omitting DataTransformer state. The fix requires full object serialization.

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| joblib | 1.5.2 (installed) | Full CTGAN object serialization including DataTransformer | joblib is already installed as a scikit-learn transitive dependency. Use `joblib.dump(self, path)` / `joblib.load(path)` for full object persistence. joblib handles numpy arrays and sklearn objects (GaussianMixture, OneHotEncoder) more efficiently than pickle. Replaces the broken partial `torch.save()` approach. [HIGH confidence — verified as installed] |
| torch.save / torch.load | Built into PyTorch | Neural network weight checkpoints within the full serialization | Keep `state_dict()` saves for intermediate training checkpoints only — these are fine for resumable training. The full inference-ready checkpoint must use joblib or pickle wrapping the entire CTGAN object including transformer. [HIGH confidence] |

**Do NOT use cloudpickle or dill** — neither is installed, and joblib is a more debuggable choice that handles the specific types in use (numpy arrays, sklearn mixtures, PyTorch modules via their own `__getstate__`).

**Serialization Pattern:** The recommended approach is:
```python
# save()
import joblib
state = {
    "transformer": self.transformer,           # sklearn objects, numpy
    "context_transformer": self.context_transformer,
    "generator_state": self.generator.state_dict(),
    "discriminator_state": self.discriminator.state_dict(),
    "embedding_layers_state": self.embedding_layers.state_dict(),
    "metadata": self.metadata,
    "config": { ... hyperparameters ... }
}
joblib.dump(state, path)

# load()
state = joblib.load(path)
self.transformer = state["transformer"]
self.generator.load_state_dict(state["generator_state"])
# etc.
```

### Statistical Validation

The existing `StatisticalValidator` in `validation/statistical.py` uses KS test and TVD. These are correct but incomplete for production use.

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| SciPy | 1.16.3 (installed) | KS test, Chi-square, statistical tests | Already installed and used correctly. Upgrade pyproject.toml from `>=1.7.0` to `>=1.10.0`. [HIGH confidence — verified] |
| NumPy | 2.3.5 (installed) | Numerical array ops for correlation, Frobenius norm | Already at latest. The `check_correlations()` method in `statistical.py` uses `np.linalg.norm()` correctly. Upgrade pyproject.toml from `>=1.21.0` to `>=2.0.0`. [HIGH confidence — verified] |
| scikit-learn | 1.8.0 (installed) | `mutual_info_score` for categorical correlation, `GaussianMixture` for mode detection | Already installed. Add `from sklearn.metrics import mutual_info_score` to the validator for categorical column pair correlation measurement — this is missing from the current implementation. Upgrade pyproject.toml from `>=1.0.0` to `>=1.5.0`. [HIGH confidence — verified] |

**What to add for production-grade statistical validation:**

The current validator only does column-wise KS and TVD. Missing:
1. **Pairwise column correlation preservation** — Compare correlation matrices between real and synthetic (partially implemented in `check_correlations()` but not gated)
2. **Column coverage** — Are all categories from real data represented in synthetic?
3. **Referential integrity validation** — Do all foreign keys in synthetic child tables reference valid parent PKs?

These should be implemented using the already-installed scipy + sklearn, not new packages.

**Do NOT add** the `table_evaluator` or `sdmetrics` packages — they are heavy, version-opinionated, and duplicate work already done in the codebase. The existing statistical.py framework is the right foundation to extend.

### Testing

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| pytest | 9.0.2 (installed) | Test runner | Upgrade pyproject.toml from `>=7.0.0` to `>=8.0.0`. Add `pytest-cov` for coverage reports. [HIGH confidence — verified] |
| pytest-cov | >=4.0.0 | Coverage measurement | Not currently installed. Required for the quality gate: "CTGAN embedding roundtrip" and "constraint roundtrip" tests have HIGH priority per CONCERNS.md. [MEDIUM confidence] |
| pytest-mock | >=3.12.0 | Mock Spark sessions and IO in unit tests | Not currently installed. The current tests in `test_models.py` and `test_relational.py` likely do full integration without mocking Spark. Mocking enables fast CI without a Spark cluster. [MEDIUM confidence] |
| hypothesis | >=6.100.0 | Property-based testing for DataTransformer constraint roundtrips | Not currently installed. The `constraint roundtrip` test coverage gap in CONCERNS.md is ideal for property-based testing: generate arbitrary constraint configs, verify `transform → inverse_transform` preserves them. [MEDIUM confidence] |

### Supporting Libraries

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| structlog | 25.5.0 (installed) | Structured logging | Already installed, upgrade pyproject.toml from `>=21.1.0` to `>=25.0.0`. Replace all bare `print()` calls in `synthesizer.py`, `orchestrator.py` with structured log events. Critical for actionable error messages (CONCERNS.md: silent error handling). [HIGH confidence — verified] |
| Faker | 38.2.0 (installed) | PII replacement in privacy sanitizer | Already installed, upgrade pyproject.toml from `>=13.0.0` to `>=30.0.0`. [HIGH confidence — verified] |
| PyArrow | 22.0.0 (installed) | Parquet I/O for CSV/Parquet connectors | Already installed. Upgrade pyproject.toml from `>=8.0.0` to `>=15.0.0`. Note: PySpark 4.0 requires `pyarrow>=11.0.0` for the sql extra. [HIGH confidence — verified] |
| networkx | 3.6.1 (installed) | DAG traversal for SchemaGraph | Already installed as a transitive dependency. The `SchemaGraph` in `relational/graph.py` uses it for topological sort. Pin explicitly in pyproject.toml as `networkx>=3.0`. [HIGH confidence — verified] |
| delta-spark | 4.0.0 (installed) | Delta Lake ACID transactions for Spark tables | Already installed. Upgrade pyproject.toml from `>=2.0.0` to `>=4.0.0`. Critical: delta-spark 4.0.0 requires pyspark>=4.0.0 — these must be bumped together. [HIGH confidence — verified from delta_spark METADATA] |

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| joblib (serialization) | cloudpickle | If the model ever needs to serialize lambdas or locally-defined classes (e.g. for Spark UDF pickling). Not needed here. |
| joblib (serialization) | dill | If joblib fails on a specific sklearn/PyTorch version combination. Both handle the same types; joblib is better tested with sklearn objects. |
| SQLAlchemy 2.0 (SQL connector) | connectorx | connectorx is faster for bulk reads via Rust backend, but does not support Snowflake/BigQuery. Use connectorx as an optional performance backend for Postgres/MySQL if SQLAlchemy proves slow at scale. [LOW confidence — not verified] |
| SQLAlchemy 2.0 (SQL connector) | pandas.read_sql directly | `read_sql` uses SQLAlchemy under the hood since Pandas 2.0. Do not bypass SQLAlchemy — the abstraction layer is the point. |
| pytest-hypothesis (testing) | manual parametrize | Hypothesis finds edge cases (null values, extreme floats, empty DataFrames) that manual parametrize misses. Worth the setup cost given the constraint roundtrip bugs in CONCERNS.md. |
| Custom CTGAN (model) | SDV/CTGAN package | Only if the team decides to stop maintaining the custom implementation. SDV's CTGAN is production-tested but removes control over the conditioning logic and embedding architecture. |
| psycopg2-binary | asyncpg | asyncpg is faster but async-only. SynthoHive is synchronous (Pandas/Spark batch). psycopg2-binary is correct here. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `torch>=1.10.0` pin | Too wide — allows ancient versions where `nn.ModuleDict.state_dict()` behavior differs. Also, torch 2.x dropped Python 3.8 support and changed `torch.load()` default behavior (requires `weights_only=True` in 2.6+). | Pin to `>=2.0.0,<3.0` |
| `pyspark>=3.2.0` pin | Allows Spark 3.x which is incompatible with delta-spark 4.0.0 (delta-spark 4.0 requires pyspark>=4.0.0). Mixed versions will silently break. | Pin to `>=4.0.0` |
| `pandas>=1.3.0` pin | Pandas 2.0 introduced copy-on-write (CoW) and removed `.swaplevel()` behaviors. The existing code has `.values` mutations that may behave differently. | Pin to `>=2.0.0`, audit CoW impact |
| Bare `pickle` for serialization | pickle has no versioning or validation. If DataTransformer's sklearn internals change across scikit-learn versions, unpickling fails with opaque errors. | Use joblib which adds checksum validation |
| `torch.load(path)` without `weights_only` | PyTorch 2.6+ changed the default of `torch.load()` to `weights_only=True`, which breaks loading the current checkpoint format (a dict with nested objects). The current `load()` in ctgan.py line 501 will raise a warning or error on PyTorch 2.6+. | Use `torch.load(path, weights_only=False)` explicitly, or switch to the joblib full-object approach |
| SDV as a library dependency | Adds ~300MB of transitive dependencies including its own CTGAN/TVAE copies that conflict with SynthoHive's native PyTorch models. | Use SDV's GitHub source as reference only |
| `sdmetrics` package | Requires SDV ecosystem; redundant with existing SciPy-based validators. Heavy install (~150MB). | Extend existing `StatisticalValidator` with sklearn metrics |
| `table_evaluator` | Abandoned, last updated 2022, incompatible with Pandas 2.x. | Use existing SciPy validator |
| PySpark for small-to-medium SQL reads | Spark startup overhead (~10 seconds) is unacceptable for tables under 1M rows. SQL connector should bypass Spark for small reads. | SQLAlchemy + pandas.read_sql for non-Spark path |
| `flake8` (linting) | flake8 has no type-awareness and misses many issues that mypy catches. The project already has mypy. | Keep mypy, consider adding `ruff` as a fast linter to replace flake8 [LOW confidence — community preference, not verified] |

---

## Stack Patterns by Variant

**If the user has a Spark cluster (large-scale, 100M+ rows):**
- Read via `SparkIO.read_dataset()` → `toPandas()` for CTGAN training
- Write synthetic output via `SparkIO.write_pandas()` → Parquet/Delta
- Use delta-spark 4.0.0 for transactional writes

**If the user has no Spark cluster (small-to-medium, <10M rows):**
- Read via new `SQLConnector` (SQLAlchemy) or CSV/Parquet file directly
- CTGAN trains on Pandas DataFrame directly
- Output written as Parquet via PyArrow or CSV

**If adding TVAE as a second model:**
- Implement `TVAE(ConditionalGenerativeModel)` in `syntho_hive/core/models/tvae.py`
- Expose via `Synthesizer(backend="TVAE")` — the `backend` param already exists in `synthesizer.py`
- Use same DataTransformer (shared, not forked)

**If model serialization needs to be backward-compatible across major sklearn upgrades:**
- Supplement joblib with a version manifest file saved alongside the model: `{"sklearn": sklearn.__version__, "torch": torch.__version__, "synthohive": __version__}`
- Validate at load time, raise `IncompatibleVersionError` instead of letting joblib fail opaquely

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| delta-spark 4.0.0 | pyspark>=4.0.0 | Hard requirement — verified from delta_spark-4.0.0 METADATA: `Requires-Dist: pyspark (>=4.0.0)`. Do not mix delta-spark 4.x with pyspark 3.x. |
| pyspark 4.0.1 | pyarrow>=11.0.0 (for sql extra) | Verified from pyspark-4.0.1 METADATA: `Requires-Dist: pyarrow>=11.0.0; extra == "sql"`. Current pyarrow 22.0.0 satisfies this. |
| torch 2.9.1 | Python >=3.10 | Verified from torch-2.9.1 METADATA: `Requires-Python: >=3.10`. Current pyproject.toml says `>=3.9` — must be updated. |
| pandas 2.3.3 | numpy>=1.23 | Pandas 2.x requires numpy >=1.23. Current numpy 2.3.5 satisfies this. |
| scikit-learn 1.8.0 | scipy>=1.3.2 | Current scipy 1.16.3 satisfies. The `sparse_output` parameter in OneHotEncoder was renamed to `sparse` in 1.2 — verify which name the transformer.py uses. |
| SQLAlchemy 2.0 (new) | psycopg2>=2.9 | SQLAlchemy 2.0 requires psycopg2 >=2.9 for PostgreSQL. |
| SQLAlchemy 2.0 (new) | pymysql>=1.0 | Compatible with current pymysql >=1.0. |

---

## pyproject.toml Changes Required

```toml
# Update these existing pins:
requires-python = ">=3.10"           # was >=3.9; torch 2.9.1 requires 3.10+
"numpy>=2.0.0"                       # was >=1.21.0
"pandas>=2.0.0"                      # was >=1.3.0
"torch>=2.0.0,<3.0"                  # was >=1.10.0
"pyspark>=4.0.0"                     # was >=3.2.0
"scipy>=1.10.0"                      # was >=1.7.0
"scikit-learn>=1.5.0"                # was >=1.0.0
"faker>=30.0.0"                      # was >=13.0.0
"delta-spark>=4.0.0"                 # was >=2.0.0
"pyarrow>=15.0.0"                    # was >=8.0.0
"structlog>=25.0.0"                  # was >=21.1.0
"networkx>=3.0"                      # add explicit (was transitive only)

# Add these new dependencies:
"sqlalchemy>=2.0.0"                  # SQL database connector
"psycopg2-binary>=2.9.0"            # PostgreSQL driver (or psycopg2 for production)
"pymysql>=1.1.0"                     # MySQL driver

# Optional extras (add to [project.optional-dependencies]):
[project.optional-dependencies]
snowflake = ["snowflake-sqlalchemy>=1.6.0"]
bigquery = ["google-cloud-bigquery>=3.0.0"]

# Update dev dependencies:
"pytest>=8.0.0"                      # was >=7.0.0
"pytest-cov>=4.0.0"                  # add new
"pytest-mock>=3.12.0"               # add new
"hypothesis>=6.100.0"               # add new
```

---

## Installation (Updated)

```bash
# Core (production)
pip install "torch>=2.0.0,<3.0" pyspark>=4.0.0 "delta-spark>=4.0.0" \
    pandas>=2.0.0 numpy>=2.0.0 scipy>=1.10.0 scikit-learn>=1.5.0 \
    pydantic>=2.10.0 pyarrow>=15.0.0 sqlalchemy>=2.0.0 \
    psycopg2-binary>=2.9.0 pymysql>=1.1.0 \
    faker>=30.0.0 structlog>=25.0.0 networkx>=3.0

# Optional connectors
pip install "syntho_hive[snowflake]"   # adds snowflake-sqlalchemy
pip install "syntho_hive[bigquery]"    # adds google-cloud-bigquery

# Dev dependencies
pip install pytest>=8.0.0 pytest-cov>=4.0.0 pytest-mock>=3.12.0 \
    hypothesis>=6.100.0 black mypy flake8
```

---

## Sources

- **HIGH confidence (verified from dist-info METADATA in .venv/lib/python3.14/site-packages/):**
  - `torch-2.9.1.dist-info/METADATA` — PyTorch version, Python requirement (>=3.10)
  - `pyspark-4.0.1.dist-info/METADATA` — PySpark version, pyarrow requirement for sql extra
  - `delta_spark-4.0.0.dist-info/METADATA` — delta-spark version, pyspark>=4.0.0 requirement
  - `pandas-2.3.3.dist-info/METADATA` — Pandas version
  - `numpy-2.3.5.dist-info/METADATA` — NumPy version
  - `scipy-1.16.3.dist-info/METADATA` — SciPy version
  - `scikit_learn-1.8.0.dist-info/METADATA` — scikit-learn version
  - `pydantic-2.12.5.dist-info/METADATA` — Pydantic version
  - `pyarrow-22.0.0.dist-info/METADATA` — PyArrow version
  - `joblib-1.5.2.dist-info/METADATA` — joblib version (transitive via scikit-learn)
  - `faker-38.2.0.dist-info/METADATA` — Faker version
  - `structlog-25.5.0.dist-info/METADATA` — structlog version
  - `pytest-9.0.2.dist-info/METADATA` — pytest version
  - `networkx-3.6.1.dist-info/METADATA` — networkx version
  - `delta-spark-4.0.0.dist-info/METADATA` — delta-spark version
- **HIGH confidence (codebase analysis):**
  - `syntho_hive/core/models/ctgan.py` lines 482–503 — confirmed broken serialization, pickle/joblib comment
  - `syntho_hive/connectors/spark_io.py` — confirmed no SQL connector exists
  - `syntho_hive/validation/statistical.py` — confirmed KS+TVD approach, missing mutual info
  - `.planning/codebase/CONCERNS.md` — priority bug list, dependency risk section
  - `.planning/PROJECT.md` — SQL connector requirement, constraint list
- **MEDIUM confidence (training knowledge, plausible but unverified via official docs):**
  - SQLAlchemy 2.0 API and dialect compatibility
  - Snowflake-sqlalchemy, google-cloud-bigquery package names and version guidance
  - pytest-mock, hypothesis version recommendations
  - SDV package weight estimates (~300MB)

---

*Stack research for: SynthoHive — production-grade synthetic tabular data Python SDK*
*Researched: 2026-02-22*
