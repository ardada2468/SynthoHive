# Codebase Structure

**Analysis Date:** 2026-02-22

## Directory Layout

```
SynthoHive/
├── syntho_hive/                    # Main package (Python 3.9+)
│   ├── __init__.py                 # Exports Metadata, PrivacyConfig, version
│   ├── interface/                  # User-facing API
│   │   ├── synthesizer.py          # Synthesizer class (main entry point)
│   │   ├── config.py               # Metadata, TableConfig, PrivacyConfig
│   │   └── __init__.py
│   ├── core/                       # Deep learning models & data processing
│   │   ├── models/                 # Generative architectures
│   │   │   ├── base.py             # GenerativeModel, ConditionalGenerativeModel (ABC)
│   │   │   ├── ctgan.py            # CTGAN with WGAN-GP
│   │   │   ├── layers.py           # ResidualLayer, Discriminator, EntityEmbeddingLayer
│   │   │   └── __init__.py
│   │   ├── data/                   # Data transformation & encoding
│   │   │   ├── transformer.py      # DataTransformer (reversible encoding)
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── relational/                 # Multi-table orchestration
│   │   ├── orchestrator.py         # StagedOrchestrator (training & generation coordinator)
│   │   ├── graph.py                # SchemaGraph (FK dependency DAG)
│   │   ├── linkage.py              # LinkageModel (cardinality distribution)
│   │   └── __init__.py
│   ├── privacy/                    # PII detection & sanitization
│   │   ├── sanitizer.py            # PIISanitizer, PiiRule
│   │   ├── faker_contextual.py     # ContextualFaker (locale-aware generation)
│   │   └── __init__.py
│   ├── connectors/                 # I/O adapters
│   │   ├── spark_io.py             # SparkIO (Spark DataFrame read/write)
│   │   ├── sampling.py             # Sampling utilities
│   │   └── __init__.py
│   ├── validation/                 # Quality assurance
│   │   ├── statistical.py          # StatisticalValidator (KS test, TVD, correlation)
│   │   ├── report_generator.py     # ValidationReport (HTML/JSON output)
│   │   └── __init__.py
│   ├── tests/                      # Unit tests (within package)
│   │   ├── test_validation.py
│   │   ├── test_relational.py
│   │   ├── test_interface.py
│   │   └── __init__.py
│   └── __pycache__/                # Compiled Python cache (gitignored)
├── tests/                          # Integration & end-to-end tests
│   ├── test_privacy.py             # Privacy feature tests
│   ├── test_models.py              # Model training & sampling tests
│   ├── test_transformer.py         # DataTransformer tests
│   ├── test_null_handling.py       # Missing value handling
│   ├── test_checkpointing.py       # Model persistence
│   ├── core/                       # Submodule tests
│   │   ├── test_embeddings.py      # Entity embedding tests
│   │   └── test_constraints.py     # Constraint enforcement tests
│   ├── e2e_scenarios/              # End-to-end workflows
│   │   └── retail_test.py          # Multi-table retail example
│   ├── repro_quickstart.py         # Reproduction of quickstart.py
│   ├── verify_sample_api_update.py # API verification
│   ├── manual_privacy_test.py      # Manual privacy checks
│   └── test_project_setup.py       # Setup verification
├── examples/                       # Demonstration notebooks & scripts
│   └── demos/                      # Runnable demos
│       ├── 01_single_table_ctgan/  # Single-table generation
│       ├── 02_privacy_sanitization/  # PII detection & removal
│       ├── 03_validation_report/   # Validation reporting
│       ├── 04_relational_linkage_ctgan/  # Multi-table with linkage
│       └── 05_transformer_embeddings/    # Entity embeddings
├── docs/                           # Documentation (MkDocs)
│   ├── index.md                    # Overview
│   ├── installation.md
│   ├── api/                        # API reference
│   └── guides/                     # User guides
├── quickstart.py                   # Simple single-table example
├── verify_quickstart.py            # Validation script
├── pyproject.toml                  # Python package config (setuptools)
├── mkdocs.yml                      # Documentation config
├── README.md                       # Project overview
└── LICENSE                         # MIT license
```

## Directory Purposes

**syntho_hive/ (Main Package)**
- Purpose: Production library code
- Contains: Interface, models, orchestration, privacy, validation modules
- Key files: `__init__.py` exports public API

**syntho_hive/interface/**
- Purpose: User-facing API boundary
- Contains: `Synthesizer` class, configuration classes, README
- Key files: `synthesizer.py` (entry point), `config.py` (data classes)

**syntho_hive/core/**
- Purpose: Deep learning and data processing internals
- Contains: Model architectures (CTGAN, discriminator, embeddings) and reversible data transformation
- Key files: `models/ctgan.py` (GAN training), `data/transformer.py` (encoding)

**syntho_hive/relational/**
- Purpose: Multi-table orchestration logic
- Contains: DAG-based scheduling, cardinality modeling, staged synthesis
- Key files: `orchestrator.py` (master coordinator), `graph.py` (dependency ordering)

**syntho_hive/privacy/**
- Purpose: Privacy enforcement before modeling
- Contains: PII detection (regex/heuristics), contextual fake data generation
- Key files: `sanitizer.py` (detection & replacement), `faker_contextual.py` (locale-aware faking)

**syntho_hive/connectors/**
- Purpose: Storage abstraction layer
- Contains: Spark I/O, Delta Lake support, format-agnostic reading
- Key files: `spark_io.py` (PySpark wrapper)

**syntho_hive/validation/**
- Purpose: Quality assurance and reporting
- Contains: Statistical tests (KS, TVD), correlation analysis, HTML/JSON report generation
- Key files: `statistical.py` (test logic), `report_generator.py` (visualization)

**tests/ (Integration & E2E)**
- Purpose: Test suite separate from package
- Contains: Feature tests, scenario tests, reproduction scripts
- Key files: Test files organized by feature (privacy, models, null handling)

**examples/demos/**
- Purpose: Runnable demonstrations
- Contains: Five complete workflows (single-table, privacy, validation, relational, embeddings)
- Key files: Each demo directory contains input scripts, output data, documentation

**docs/ (Documentation)**
- Purpose: User-facing documentation
- Contains: Installation, API reference, guides
- Generated from source via MkDocs

## Key File Locations

**Entry Points:**
- `syntho_hive/interface/synthesizer.py` - `Synthesizer` class; main API
- `quickstart.py` - Simple single-table example
- `examples/demos/01_single_table_ctgan/` - Complete working demo

**Configuration:**
- `syntho_hive/interface/config.py` - Metadata, TableConfig, Constraint, PrivacyConfig classes
- `pyproject.toml` - Package build config (version 1.2.3, dependencies)
- `mkdocs.yml` - Documentation site config

**Core Logic:**
- `syntho_hive/core/models/ctgan.py` - CTGAN GAN architecture + training loop
- `syntho_hive/core/data/transformer.py` - Column-wise encoding (continuous, categorical, embeddings)
- `syntho_hive/relational/orchestrator.py` - Multi-table training & generation orchestration
- `syntho_hive/relational/graph.py` - Topological sort of FK dependencies
- `syntho_hive/relational/linkage.py` - Cardinality distribution learning

**Privacy & Validation:**
- `syntho_hive/privacy/sanitizer.py` - PII detection engine
- `syntho_hive/privacy/faker_contextual.py` - Locale-aware fake data generation
- `syntho_hive/validation/statistical.py` - Statistical test implementations
- `syntho_hive/validation/report_generator.py` - HTML/JSON report rendering

**Testing:**
- `tests/test_privacy.py` - Privacy feature tests
- `tests/test_models.py` - CTGAN training/sampling tests
- `tests/core/test_embeddings.py` - Entity embedding tests
- `tests/e2e_scenarios/retail_test.py` - Multi-table end-to-end test

## Naming Conventions

**Files:**
- `*.py` - All Python modules
- `*_test.py` - Test files (picked up by pytest `python_files = test_*.py`)
- `config.py` - Configuration/schema definition files
- `transformer.py` - Data encoding/transformation
- `orchestrator.py` - Coordination logic

**Directories:**
- `core/` - Core algorithms (models, data processing)
- `interface/` - User-facing API
- `relational/` - Multi-table orchestration
- `connectors/` - External system adapters
- `validation/` - Quality assurance
- `privacy/` - Privacy enforcement

**Classes:**
- `*Synthesizer` - User-facing façade
- `*Model` - Model base classes
- `*Orchestrator` - Coordination classes
- `*Validator` - Validation/testing classes
- `*Config` - Configuration data classes (Pydantic BaseModel)
- `Data*` / `*Transformer` - Data processing classes
- `*Graph` - Graph data structures
- `*IO` - I/O adapters

**Functions/Methods:**
- `fit()` - Train/fit a model
- `sample()` - Generate samples
- `save()` / `load()` - Model persistence
- `fit_all()` - Coordinate fitting across multiple entities
- `generate()` - Generate synthetic data (multi-table)
- `validate_*()` - Validation methods
- `read_*()` / `write_*()` - I/O operations

## Where to Add New Code

**New Feature (Generative Model Variant):**
- Primary code: Create new class in `syntho_hive/core/models/` inheriting from `GenerativeModel` or `ConditionalGenerativeModel`
- Example: `class TVAE(ConditionalGenerativeModel): ...` in `syntho_hive/core/models/tvae.py`
- Register in: `syntho_hive/interface/synthesizer.py` backend parameter handling
- Tests: `tests/test_models.py` or new `tests/core/test_tvae.py`

**New Privacy Rule or Sanitizer:**
- Primary code: Extend `syntho_hive/privacy/sanitizer.py` with new `PiiRule` or custom handler
- Example: Add new rule type to `PrivacyConfig.default()` or create subclass
- Tests: `tests/test_privacy.py`

**New Validation Check:**
- Primary code: Add method to `syntho_hive/validation/statistical.py` (StatisticalValidator)
- Integration: Call from `syntho_hive/validation/report_generator.py` (ValidationReport.generate())
- Tests: `tests/test_validation.py`

**New I/O Connector:**
- Primary code: Create new adapter in `syntho_hive/connectors/`
- Example: `class ElasticsearchIO: ...` in `syntho_hive/connectors/elasticsearch_io.py`
- Pattern: Inherit or match `SparkIO` interface (read_dataset, write_dataset)
- Tests: New test file `tests/connectors/test_elasticsearch_io.py`

**New Data Transformation Strategy:**
- Primary code: Extend `syntho_hive/core/data/transformer.py` (DataTransformer)
- Example: Add new column processing mode in `_prepare_categorical()` or new transformer class
- Tests: `tests/test_transformer.py`

**New Demonstration:**
- Primary code: Create new subdirectory in `examples/demos/06_your_feature_name/`
- Structure: Input scripts, output directory, README
- Pattern: Follow pattern of existing demos (01-05)

## Special Directories

**syntho_hive/tests/ (Internal):**
- Purpose: Unit tests included in package
- Generated: No
- Committed: Yes
- Note: `pyproject.toml` excludes `tests*` from setuptools package discovery; tests run via `pytest tests/`

**test_output/ and output/**
- Purpose: Artifact directories for test/demo outputs
- Generated: Yes (created during test runs)
- Committed: No (in `.gitignore`)

**site/ (Generated Documentation)**
- Purpose: Built HTML documentation
- Generated: Yes (via `mkdocs build`)
- Committed: Yes (for GitHub Pages deployment)

**build/ and .egg-info/ and *.egg-info/ (Build Artifacts)**
- Purpose: Package build artifacts
- Generated: Yes (via `pip install -e .`)
- Committed: No (in `.gitignore`)

**.planning/ (Analysis & Planning)**
- Purpose: GSD codebase analysis and planning documents
- Generated: Yes (created by analyzer agents)
- Committed: Yes
- Structure: `codebase/` contains ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md, STACK.md, INTEGRATIONS.md

**.github/workflows/**
- Purpose: CI/CD pipeline definitions
- Generated: No (manual)
- Committed: Yes
- Key file: `ci.yml` (runs pytest, linting, builds docs)

---

*Structure analysis: 2026-02-22*
