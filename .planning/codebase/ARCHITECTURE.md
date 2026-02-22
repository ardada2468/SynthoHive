# Architecture

**Analysis Date:** 2026-02-22

## Pattern Overview

**Overall:** Layered pipeline with graph-based relational orchestration

**Key Characteristics:**
- Multi-table relational synthesis with parent-child dependency ordering
- Privacy-first design with automatic PII detection before model training
- Deep generative models (CTGAN + WGAN-GP) for tabular data generation
- Apache Spark integration for distributed I/O and potential future scaling
- Modular separation: config → privacy → models → relational orchestration → validation

## Layers

**Interface Layer (User-Facing):**
- Purpose: High-level API for end users
- Location: `syntho_hive/interface/`
- Contains: `Synthesizer` class, `Metadata` and `PrivacyConfig` configuration classes
- Depends on: Orchestrator, ValidationReport, SparkIO
- Used by: User code (via `Synthesizer`)

**Configuration & Metadata Layer:**
- Purpose: Define schema, constraints, privacy rules, and relational structure
- Location: `syntho_hive/interface/config.py`
- Contains: `Metadata`, `TableConfig`, `Constraint`, `PrivacyConfig` Pydantic models
- Depends on: pydantic
- Used by: All downstream layers for schema validation and constraint enforcement

**Privacy & Sanitization Layer:**
- Purpose: Detect and sanitize PII before model training
- Location: `syntho_hive/privacy/`
- Contains: `PIISanitizer` (column detection, regex/pattern matching), `ContextualFaker` (context-aware replacement), `PiiRule` configuration
- Depends on: faker, regex patterns
- Used by: Orchestrator during data ingestion

**Core Data Processing Layer:**
- Purpose: Transform raw tabular data into model-ready tensor representations
- Location: `syntho_hive/core/data/`
- Contains: `DataTransformer` (reversible encoding), `ClusterBasedNormalizer` (continuous columns via Bayesian GMM), categorical processors
- Depends on: scikit-learn (OneHotEncoder, BayesianGaussianMixture)
- Used by: CTGAN during fit/sample operations

**Generative Model Layer:**
- Purpose: Deep learning architectures for tabular synthesis
- Location: `syntho_hive/core/models/`
- Contains: `GenerativeModel` (abstract base), `ConditionalGenerativeModel` (base with context), `CTGAN` (Conditional Tabular GAN with WGAN-GP)
- Depends on: PyTorch, DataTransformer
- Used by: Orchestrator for single-table training

**Relational Orchestration Layer:**
- Purpose: Manage multi-table synthesis with foreign key integrity
- Location: `syntho_hive/relational/`
- Contains: `StagedOrchestrator` (master coordinator), `SchemaGraph` (dependency DAG), `LinkageModel` (cardinality learning)
- Depends on: CTGAN, DataTransformer, SparkIO
- Used by: Synthesizer.fit() and Synthesizer.sample()

**Connectors & I/O Layer:**
- Purpose: Abstract data reading/writing across storage formats
- Location: `syntho_hive/connectors/`
- Contains: `SparkIO` (parquet, delta, CSV, Hive tables), `sampling` utilities
- Depends on: PySpark, PyArrow
- Used by: Orchestrator for data ingestion and output writing

**Validation & Reporting Layer:**
- Purpose: Quality assurance and comparison metrics
- Location: `syntho_hive/validation/`
- Contains: `StatisticalValidator` (KS test, TVD, correlation), `ValidationReport` (HTML/JSON generation)
- Depends on: scipy, pandas
- Used by: Synthesizer.generate_validation_report()

## Data Flow

**Training Flow (fit):**

1. User calls `Synthesizer.fit(data, ...)`
2. Synthesizer delegates to `StagedOrchestrator.fit_all(real_data_paths, ...)`
3. Orchestrator builds `SchemaGraph` from metadata to determine table dependency order
4. For each table (topological order):
   - `SparkIO.read_dataset()` loads data from path or Hive table
   - Convert Spark DataFrame to Pandas (current limitation)
   - **Privacy check**: Apply `PIISanitizer` to detect PII columns
   - **Data transformation**: `DataTransformer.fit()` profiles columns and builds reversible encoders
   - **Linkage training** (if child table with foreign keys): `LinkageModel.fit()` learns cardinality distribution from parent-child counts
   - **Model training**: `CTGAN.fit()` trains generator/discriminator on transformed data
   - Store trained CTGAN model in `orchestrator.models[table_name]`
   - Store linkage model in `orchestrator.linkage_models[table_name]`

**Generation Flow (sample):**

1. User calls `Synthesizer.sample(num_rows, output_path, ...)`
2. Synthesizer delegates to `StagedOrchestrator.generate(num_rows, output_path_base)`
3. Orchestrator processes tables in topological generation order:
   - **Root tables** (no foreign keys): `CTGAN.sample(num_rows)` generates data unconditionally
   - **Child tables** (with foreign keys):
     - Get parent synthetic data from previous generation step
     - `LinkageModel.sample_counts()` determines how many child records per parent
     - `CTGAN.sample(num_rows, context=parent_context)` generates child data conditioned on parent attributes
4. All generated tables written to disk via `SparkIO.write_dataset()` (parquet or delta format)
5. Return mapping of table name to output paths

**Validation Flow:**

1. User calls `Synthesizer.generate_validation_report(real_data, synthetic_data, output_path)`
2. Load both real and synthetic datasets using SparkIO
3. Convert to Pandas for statistical analysis
4. `StatisticalValidator.compare_columns()` runs per-column tests:
   - Numeric: Kolmogorov-Smirnov test
   - Categorical: Total Variation Distance (TVD)
5. `StatisticalValidator.check_correlations()` compares Frobenius norm of correlation matrices
6. `ValidationReport.generate()` creates HTML/JSON visualization with test results

**State Management:**

- **Fitted model state**: Persisted in `StagedOrchestrator.models` (dict of table → CTGAN) and `StagedOrchestrator.linkage_models`
- **Metadata state**: Immutable `Metadata` object passed to Orchestrator at init; not modified during fitting
- **Configuration state**: Privacy rules in `PrivacyConfig` applied during orchestration; no state changes
- **Transformer state**: Each `DataTransformer` instance captures fit parameters (normalizer components, OneHotEncoder state) for later inverse transformation

## Key Abstractions

**Synthesizer (Façade):**
- Purpose: Hide complexity of orchestration, privacy, validation behind a single API
- Examples: `syntho_hive/interface/synthesizer.py` class `Synthesizer`
- Pattern: Façade pattern; delegates to specialized components

**GenerativeModel / ConditionalGenerativeModel (Contract):**
- Purpose: Define contract for any tabular generative model
- Examples: `syntho_hive/core/models/base.py` abstract classes
- Pattern: Abstract base classes; allows future model swapping (e.g., TVAE, VGAN)

**SchemaGraph (Dependency Manager):**
- Purpose: Model relational structure as a DAG for deterministic generation order
- Examples: `syntho_hive/relational/graph.py` class `SchemaGraph`
- Pattern: Directed acyclic graph with topological sort; ensures parents generated before children

**DataTransformer (Reversible Encoder):**
- Purpose: Convert raw tabular data ↔ tensor representations suitable for neural networks
- Examples: `syntho_hive/core/data/transformer.py` class `DataTransformer`
- Pattern: Transformer pattern (fit/transform/inverse_transform); preserves reversibility

**LinkageModel (Cardinality Learner):**
- Purpose: Capture parent-child count distribution to maintain referential integrity
- Examples: `syntho_hive/relational/linkage.py` class `LinkageModel`
- Pattern: Probabilistic model (GMM); samples child counts aligned to parent context

**PIISanitizer (Privacy Guardian):**
- Purpose: Detect sensitive columns and replace with realistic fake data
- Examples: `syntho_hive/privacy/sanitizer.py` class `PIISanitizer`
- Pattern: Pattern matching + rule engine + contextual generation

## Entry Points

**Synthesizer.fit():**
- Location: `syntho_hive/interface/synthesizer.py` method `fit`
- Triggers: User calls `synth.fit(data, epochs, batch_size, ...)`
- Responsibilities: Validate SparkSession, delegate to orchestrator, coordinate training workflow

**Synthesizer.sample():**
- Location: `syntho_hive/interface/synthesizer.py` method `sample`
- Triggers: User calls `synth.sample(num_rows, output_path, ...)`
- Responsibilities: Generate synthetic data from trained models, write to disk or return in-memory

**Synthesizer.generate_validation_report():**
- Location: `syntho_hive/interface/synthesizer.py` method `generate_validation_report`
- Triggers: User calls after sample generation to compare real vs. synthetic
- Responsibilities: Load datasets, run statistical tests, produce HTML/JSON report

**Synthesizer.save_to_hive():**
- Location: `syntho_hive/interface/synthesizer.py` method `save_to_hive`
- Triggers: User calls to register generated tables as Hive external tables
- Responsibilities: Create database, register tables via Spark SQL

## Error Handling

**Strategy:** Explicit validation + exception propagation

**Patterns:**

- **Configuration validation**: `Metadata.validate_schema()` checks FK references point to valid parent tables before training starts
- **Data validation**: `DataTransformer.fit()` raises `ValueError` if metadata is missing table configs
- **Model state validation**: `Synthesizer.fit()` raises `ValueError` if SparkSession unavailable or sample_size invalid
- **SparkIO robustness**: `SparkIO.read_dataset()` attempts table read first, falls back to path-based read; supports CSV, Parquet, Delta formats transparently
- **Privacy analysis safety**: `PIISanitizer.analyze()` samples first 100 rows to avoid scanning massive datasets during detection
- **Validation error recovery**: `StatisticalValidator` returns error dict per column (not exception-based) so partial failures don't block report generation

## Cross-Cutting Concerns

**Logging:** Console-based via `print()` statements in orchestrator and synthesizer; examples: "Fitting model for table: {table_name}", "Generating data with {backend} backend"

**Validation:**
- Schema-level: FK reference validation in `Metadata.validate_schema()`
- Data-level: Type mismatch detection in `StatisticalValidator.compare_columns()`
- Constraint-level: Numeric bounds enforcement in `DataTransformer` during column fitting

**Authentication:** Implicit via SparkSession configuration; users must provide authenticated Spark session to Synthesizer

**Multi-table consistency:**
- `SchemaGraph` ensures deterministic generation order (parents before children)
- `LinkageModel` maintains cardinality fidelity (child count distribution)
- Foreign key columns excluded from transformation (preserved as-is via `DataTransformer._excluded_columns`)

---

*Architecture analysis: 2026-02-22*
