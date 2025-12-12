# SynthoHive 2.0

[![CI](https://github.com/Start-End/SynthoHive/actions/workflows/ci.yml/badge.svg)](https://github.com/Start-End/SynthoHive/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/syntho-hive.svg)](https://badge.fury.io/py/syntho-hive)

**SynthoHive** is a comprehensive, production-grade synthetic data engine designed for enterprise environments. It specializes in generating high-utility, privacy-preserving synthetic data for complex relational databases.

Beyond simple single-table generation, SynthoHive excels at maintaining referential integrity across multi-table schemas, preserving statistical correlations, and ensuring strict privacy compliance through automated PII handling.

## 🚀 Key Features

*   **Multi-Table Relational Generation**: Maintains parent-child relationships and foreign key integrity using intelligent graph-based orchestration.
*   **Privacy-First Design**: Automated PII detection and sanitization logic to ensure no sensitive data leaks into the model training process.
*   **Deep Generative Models**: State-of-the-art CTGAN (Conditional Tabular GAN) and WGAN-GP implementations for robust statistical fidelity.
*   **Enterprise Scale**: Built with Spark integration to handle large-scale datasets efficiently.
*   **Comprehensive Validation**: Automated statistical reports (HTML/JSON) comparing real vs. synthetic data utility (KS Test, TVD, Correlation analysis).

*   **Comprehensive Validation**: Automated statistical reports (HTML/JSON) comparing real vs. synthetic data utility (KS Test, TVD, Correlation analysis).

---

## 📦 Installation & Requirements

Ensure you have the following dependencies installed:

*   **Python 3.9+**
*   **PySpark**: For distributed data processing.
*   **PyArrow**: Required for efficient Parquet file handling (`pip install pyarrow`).
*   **Torch**: For training deep generative models.

---

## 📂 Project Modules

The system is organized into several modular components:

### 1. [Core Data & Metadata](syntho_hive/core/data/README.md)
*   **Path**: `syntho_hive/core/data`
*   **Function**: Manages data ingestion, schema definitions, and validation.
*   **Key Components**:
    *   `Metadata`: Defines tables, columns, data types, and primary/foreign keys.
    *   `SparkIO`: Scalable data reading/writing using PySpark.
    *   `SchemaValidator`: Ensures data integrity constraints.

### 2. [Generative Models](syntho_hive/core/models/README.md)
*   **Path**: `syntho_hive/core/models`
*   **Function**: Contains the deep learning architectures for data synthesis.
*   **Key Components**:
    *   `CTGAN`: Conditional Tabular GAN for mixed numeric/categorical data.
    *   `DataTransformer`: Encodes tabular data into vector representations suitable for Neural Networks (One-hot encoding, Entity Embeddings, Variational Gaussian Mixture Models).

### 3. [Relational Orchestration](syntho_hive/relational/README.md)
*   **Path**: `syntho_hive/relational`
*   **Function**: The brain of the multi-table generation process.
*   **Key Components**:
    *   `Graph`: Constructs a DAG of table dependencies to determine generation order (Parents -> Children).
    *   `LinkageModel`: Learns the cardinality (1:N) relationship to determine how many child records to generate for each parent.
    *   `Orchestrator`: Manages the end-to-end flow of training and generation across all tables.

### 4. [Privacy & Sanitization](syntho_hive/privacy/README.md)
*   **Path**: `syntho_hive/privacy`
*   **Function**: Ensures privacy compliance *before* modeling begins.
*   **Key Components**:
    *   `PIISanitizer`: Automatically detects and removes sensitive information.
    *   `ContextualFaker`: Replaces sensitive data with realistic fake data based on context (e.g., generating US phone numbers for US addresses).
    *   **Rules Engine**: Configurable Regex and Named Entity Recognition (NER) rules.

### 5. [Validation & Reporting](syntho_hive/validation/README.md)
*   **Path**: `syntho_hive/validation`
*   **Function**: Quality Assurance for the synthetic data.
*   **Key Components**:
    *   `StatisticalValidator`: Runs KS Tests (numeric), TVD (categorical), and correlation checks.
    *   `ValidationReport`: Generates detailed HTML/JSON reports comparing Real vs. Synthetic distributions.

### 6. [Interface](syntho_hive/interface/README.md)
*   **Path**: `syntho_hive/interface`
*   **Function**: The entry point for users.
*   **Key Components**:
    *   CLI tools and high-level Python APIs to abstract complexity.

---

## 🛠️ Usage Examples

### A. Privacy Sanitization
Clean your raw data before training:
```python
from syntho_hive.privacy.sanitizer import PIISanitizer
import pandas as pd

df = pd.read_csv("raw_users.csv")
sanitizer = PIISanitizer()

# Detect and Scrub
clean_df = sanitizer.sanitize(df)
print(sanitizer.analyze(df))  # See what was found
```

### B. Relational Data Generation
Generate a full database schema:
```python
from syntho_hive.relational.orchestrator import Orchestrator
from syntho_hive.core.data.metadata import Metadata

# 1. Define Schema
metadata = Metadata.load_from_json("schema.json")

# 2. Train and Generate
orchestrator = Orchestrator(metadata)
orchestrator.fit_all(real_data_paths={"users": "data/users.csv", "orders": "data/orders.csv"})

# 3. Generate 1000 root users (and associated orders)
orchestrator.generate(num_rows_root=1000, output_path="synthetic_output/")
```

### C. Validation
Check the quality of your output:
```python
from syntho_hive.validation.report_generator import ValidationReport

report = ValidationReport()
report.generate(
    real_data=real_dfs,
    synth_data=synthetic_dfs,
    output_path="quality_report.html"
)
```

## 🤝 Contributing
1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
