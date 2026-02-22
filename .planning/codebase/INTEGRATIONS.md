# External Integrations

**Analysis Date:** 2026-02-22

## APIs & External Services

**None detected** - SynthoHive does not integrate with external REST APIs or third-party services. All functionality is self-contained within the package.

## Data Storage

**Databases:**
- Apache Spark Hive Tables - Primary data storage integration
  - Connection: Via `SparkSession` object passed at runtime
  - Client: `SparkIO` utility in `syntho_hive/connectors/spark_io.py`
  - Format support: CSV, Parquet, Delta Lake format via Spark's native readers
  - Usage: `read_dataset()` and `write_dataset()` methods handle all database I/O operations

**File Storage:**
- Local filesystem - Primary storage mechanism
- Parquet files - Columnar storage format for efficient data persistence
- CSV files - Supported input format for data loading
- Delta Lake - Transactional table format via delta-spark 2.0.0+ for data versioning

**Caching:**
- None - No external caching layer (Redis, Memcached) detected
- In-memory caching handled by pandas DataFrames and PyTorch tensors during model training

## Authentication & Identity

**Auth Provider:**
- None - Not applicable. SynthoHive is a library, not a service. Authentication is managed by the consumer at the SparkSession level when connecting to Hadoop, Hive, or cloud storage systems.

**Spark Integration:**
- Relies on underlying Spark cluster authentication (Kerberos, cloud IAM, or local mode)
- No built-in credential management; delegated to Spark configuration

## Monitoring & Observability

**Error Tracking:**
- None - No integrated error tracking service detected

**Logs:**
- Python logging module via standard `logging` library
- Structured logging via `structlog 21.1.0+` (declared dependency, framework in place)
- Logger usage detected in `syntho_hive/privacy/faker_contextual.py` (line 27: `self.logger = logging.getLogger(__name__)`)
- Error messages logged on PII generation failures with fallback to "REDACTED" value
- Console output via print statements in user-facing APIs (e.g., `syntho_hive/interface/synthesizer.py`)

## CI/CD & Deployment

**Hosting:**
- Not detected - SynthoHive is a library, not a deployed service
- Distributed as Python package via pip

**CI Pipeline:**
- GitHub Actions - CI workflow reference in `pyproject.toml` and repo badges in `README.md`
- Badge URL: `https://github.com/ardada2468/SynthoHive/actions/workflows/ci.yml`

## Environment Configuration

**Required env vars:**
- None required at runtime - All configuration is code-based via Pydantic classes
- SparkSession initialization may require environment variables depending on cluster setup (e.g., SPARK_HOME, HADOOP_CONF_DIR for Spark clusters)

**Secrets location:**
- Not applicable - No secrets management required for library usage
- User applications must handle database credentials when creating SparkSession instances

## Webhooks & Callbacks

**Incoming:**
- None - Not applicable

**Outgoing:**
- None - Not applicable

## Data Format Support

**Input Formats:**
- CSV files - Supported via Spark CSV reader with auto schema inference in `syntho_hive/connectors/spark_io.py`
- Parquet files - Columnar format support via Spark
- Hive tables - Direct table reference via `spark.table(table_name)` in SparkIO

**Output Formats:**
- Parquet (default) - Primary output format for synthetic data in `SparkIO.write_dataset()`
- CSV - Optionally supported via format parameter
- Delta Lake - Supported for transactional writes via delta-spark
- Pandas DataFrames - In-memory representation for small datasets
- JSON - Validation reports output as structured JSON in `syntho_hive/validation/report_generator.py`
- HTML - Human-readable validation reports generated in `report_generator.py`

## Data Pipeline Integration Points

**Input:**
- `Metadata` class defines schema: table names, column types, primary/foreign keys in `syntho_hive/interface/config.py`
- Real data loaded via `SparkIO.read_dataset()` from filesystem paths or Hive table names
- Optional privacy sanitization before model training via `PIISanitizer` in `syntho_hive/privacy/sanitizer.py`

**Processing:**
- Data transformation handled by `DataTransformer` in `syntho_hive/core/data/transformer.py`
- Multi-table orchestration via `StagedOrchestrator` in `syntho_hive/relational/orchestrator.py`
- CTGAN model training using PyTorch (GPU or CPU)

**Output:**
- Synthetic data written via `SparkIO.write_dataset()` or `write_pandas()`
- Validation reports generated via `ValidationReport.generate()` in `syntho_hive/validation/report_generator.py`
- Reports output as JSON or HTML to disk

## Dependencies on External Systems at Runtime

**Spark Cluster:**
- Mandatory - SparkSession required for `StagedOrchestrator` operations
- Can be local standalone mode or distributed cluster
- Spark version 3.2.0 or later assumed based on dependency specification

**File System:**
- Local filesystem - For reading/writing CSV, Parquet, Delta tables
- Cloud storage (S3, GCS, ADLS) - Supported via Spark's native connector if Spark cluster has credentials configured

---

*Integration audit: 2026-02-22*
