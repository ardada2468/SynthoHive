# Technology Stack

**Analysis Date:** 2026-02-22

## Languages

**Primary:**
- Python 3.9+ - Main language for entire codebase. Required version specified in `pyproject.toml`
- PyTorch (Python) - Deep learning framework for CTGAN model implementation in `syntho_hive/core/models/ctgan.py`

## Runtime

**Environment:**
- CPython 3.9 or higher
- Virtual environment (.venv) used for isolated dependency management

**Package Manager:**
- pip (via setuptools) - Standard Python package manager
- Lockfile: Not detected (using setuptools with pyproject.toml)

## Frameworks

**Core:**
- PySpark 3.2.0+ - Distributed data processing and SQL execution for large-scale dataset handling
- Pydantic 2.0.0+ - Data validation and configuration management for `PrivacyConfig`, `Metadata`, `TableConfig` classes in `syntho_hive/interface/config.py`

**Machine Learning:**
- PyTorch 1.10.0+ - Neural network framework for CTGAN generator and discriminator networks
- scikit-learn 1.0.0+ - For preprocessing (OneHotEncoder) and mixture models (GaussianMixture, BayesianGaussianMixture) in `syntho_hive/core/data/transformer.py` and `syntho_hive/relational/linkage.py`
- NumPy 1.21.0+ - Numerical computations and array operations
- SciPy 1.7.0+ - Statistical functions (ks_2samp, chisquare for validation)

**Data Processing:**
- Pandas 1.3.0+ - Tabular data manipulation and analysis across all modules
- PyArrow 8.0.0+ - Efficient Parquet file handling for Spark IO operations

**Development & Documentation:**
- pytest 7.0.0+ - Unit testing framework (configured in `pyproject.toml` with testpaths=["tests"])
- black - Code formatting (in dev dependencies)
- flake8 - Linting (in dev dependencies)
- mypy - Static type checking (in dev dependencies)
- mkdocs 1.6.0+ - Documentation generation
- mkdocs-material 9.5.0+ - Material theme for documentation
- mkdocstrings[python] 0.25.0+ - API documentation from docstrings (Google style)

**Utilities:**
- Faker 13.0.0+ - Realistic fake data generation for PII replacement in `syntho_hive/privacy/faker_contextual.py`
- delta-spark 2.0.0+ - Delta Lake integration for Spark
- structlog 21.1.0+ - Structured logging framework (declared as dependency, used in logging patterns)

## Key Dependencies

**Critical:**
- torch 1.10.0+ - Core to generative model training and inference. Required for CTGAN architecture in `syntho_hive/core/models/ctgan.py` and `syntho_hive/core/models/layers.py`
- pyspark 3.2.0+ - Essential for distributed data I/O, SQL operations, and multi-table orchestration in `syntho_hive/connectors/spark_io.py` and `syntho_hive/relational/orchestrator.py`
- pandas 1.3.0+ - Primary data structure for table manipulation across all validation, privacy, and core modules
- pydantic 2.0.0+ - Ensures schema integrity and configuration validation before model training

**Infrastructure:**
- pyarrow 8.0.0+ - Handles columnar data format (Parquet) for efficient storage and retrieval
- delta-spark 2.0.0+ - Provides transactional guarantees and time-travel for Spark tables
- scipy 1.7.0+ - Statistical testing (KS test, Chi-square) for validation reports in `syntho_hive/validation/statistical.py`
- scikit-learn 1.0.0+ - Machine learning preprocessing and clustering for data transformation and linkage modeling

## Configuration

**Environment:**
- Configuration via Python classes: `Metadata`, `PrivacyConfig`, `TableConfig` in `syntho_hive/interface/config.py`
- Schema defined programmatically using Pydantic models
- No .env file usage detected - configuration is code-based via Pydantic classes

**Build:**
- `pyproject.toml` - Single source of truth for project metadata, dependencies, and tool configuration
- Setuptools with `[build-system]` configuration using setuptools >= 61.0
- Package discovery via setuptools.packages.find with include pattern `["syntho_hive*"]`

**Development:**
- Pytest configuration in `pyproject.toml` with testpaths=["tests"] and python_files="test_*.py"
- mkdocs.yml for documentation site configuration with Material theme

## Platform Requirements

**Development:**
- Python 3.9+ runtime
- pip for dependency installation
- Virtual environment (venv or similar)
- Optional: CUDA toolkit if GPU acceleration desired (PyTorch will use GPU if available)

**Production:**
- Python 3.9+ runtime
- Apache Spark cluster (local or distributed) for data processing
- Sufficient memory for model training (depends on dataset size; batch_size defaults to 500)
- Optional: CUDA for GPU-accelerated model training

---

*Stack analysis: 2026-02-22*
