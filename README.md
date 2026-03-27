# SynthoHive

[![CI](https://github.com/ardada2468/SynthoHive/actions/workflows/ci.yml/badge.svg)](https://github.com/ardada2468/SynthoHive/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/synthohive.svg)](https://badge.fury.io/py/synthohive)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SynthoHive** is a comprehensive, production-grade synthetic data engine designed for enterprise environments. It specializes in generating high-utility, privacy-preserving synthetic data for complex relational databases.

Beyond simple single-table generation, SynthoHive excels at maintaining referential integrity across multi-table schemas, preserving statistical correlations, and ensuring strict privacy compliance through automated PII handling.

## Key Features

*   **Multi-Table Relational Generation**: Maintains parent-child relationships and foreign key integrity using intelligent graph-based orchestration via `StagedOrchestrator`.
*   **Privacy-First Design**: Automated PII detection and sanitization with regex-based rules and context-aware faking to ensure no sensitive data leaks into the model training process.
*   **Deep Generative Models**: CTGAN (Conditional Tabular GAN) with WGAN-GP training for robust statistical fidelity, plus Entity Embeddings for high-cardinality columns.
*   **Null Value Support**: Native handling of missing data in both continuous (via null indicators) and categorical (via sentinel tokens) columns.
*   **Enterprise Scale**: Built with PySpark and Delta Lake integration to handle large-scale datasets efficiently.
*   **Comprehensive Validation**: Automated statistical reports (HTML/JSON) comparing real vs. synthetic data utility (KS Test, TVD, Correlation analysis).

---

## Installation

```bash
pip install synthohive
```

### Requirements

*   **Python 3.9+**
*   **PySpark 4.0+**: For distributed data processing (required for the `Synthesizer` facade; individual components like `CTGAN` work without Spark).
*   **PyTorch**: For training deep generative models.
*   **PyArrow**: Required for efficient Parquet file handling.

Install all core dependencies at once:

```bash
pip install synthohive pyspark pandas pyarrow
```

---

## Project Modules

The system is organized into several modular components:

### 1. Interface (`syntho_hive.interface`)
*   **Key Components**:
    *   `Synthesizer`: High-level facade that wires metadata, privacy, and orchestration into a single API.
    *   `Metadata`: Schema definition with `add_table()` for registering tables, primary keys, foreign keys, constraints, and context columns.
    *   `PrivacyConfig`: Configuration for privacy guardrails (PII strategy, differential privacy settings).
    *   `TableConfig`: Per-table configuration including keys, PII columns, and constraints.

### 2. Core Models (`syntho_hive.core`)
*   **Key Components**:
    *   `CTGAN`: Conditional Tabular GAN for mixed numeric/categorical data with WGAN-GP training.
    *   `DataTransformer`: Encodes tabular data into vector representations (One-Hot Encoding, Entity Embeddings, Variational Gaussian Mixture Models).
    *   `EntityEmbeddingLayer`: Learnable embedding layer for high-cardinality categorical variables.

### 3. Relational Orchestration (`syntho_hive.relational`)
*   **Key Components**:
    *   `SchemaGraph`: Constructs a DAG of table dependencies and performs topological sort for generation order.
    *   `LinkageModel`: Learns the cardinality (1:N) relationship to determine how many child records to generate per parent.
    *   `StagedOrchestrator`: Manages the end-to-end flow of training and generation across all tables.

### 4. Privacy & Sanitization (`syntho_hive.privacy`)
*   **Key Components**:
    *   `PIISanitizer`: Automatically detects and sanitizes sensitive information using configurable regex-based rules.
    *   `ContextualFaker`: Replaces sensitive data with realistic fake data based on locale context (e.g., generating US phone numbers for US addresses).
    *   `PiiRule`: Configurable rules with actions: `drop`, `mask`, `hash`, `fake`, `keep`, `custom`.

### 5. Validation & Reporting (`syntho_hive.validation`)
*   **Key Components**:
    *   `StatisticalValidator`: Runs KS Tests (numeric), TVD (categorical), and correlation checks (Frobenius norm).
    *   `ValidationReport`: Generates detailed HTML/JSON reports comparing real vs. synthetic distributions.

### 6. Connectors (`syntho_hive.connectors`)
*   **Key Components**:
    *   `SparkIO`: Scalable data reading/writing using PySpark and Delta Lake.
    *   `RelationalSampler`: Stratified sampling for parent-child table hierarchies.

---

## Usage Examples

### A. Privacy Sanitization
Clean your raw data before training:
```python
from syntho_hive.privacy.sanitizer import PIISanitizer
import pandas as pd

df = pd.read_csv("raw_users.csv")
sanitizer = PIISanitizer()

# Detect PII columns automatically
detected = sanitizer.analyze(df)
print("Detected PII:", detected)

# Sanitize the data
clean_df = sanitizer.sanitize(df, pii_map=detected)
```

### B. Single-Table CTGAN (No Spark Required)
Train and generate from a single table:
```python
from syntho_hive.interface.config import Metadata
from syntho_hive.core.models.ctgan import CTGAN

metadata = Metadata()
metadata.add_table("customers", pk="customer_id")

model = CTGAN(metadata, batch_size=128, epochs=300)
model.fit(train_df, table_name="customers")

synthetic_df = model.sample(1000)
```

### C. Relational Data Generation (With Spark)
Generate a full database schema using the `Synthesizer` facade:
```python
from syntho_hive.interface.config import Metadata, PrivacyConfig
from syntho_hive.interface.synthesizer import Synthesizer

# 1. Define Schema
metadata = Metadata()
metadata.add_table("users", pk="user_id")
metadata.add_table("orders", pk="order_id",
                   fk={"user_id": "users.user_id"},
                   parent_context_cols=["region"])

# 2. Train
synth = Synthesizer(metadata=metadata, privacy_config=PrivacyConfig(), spark_session=spark)
synth.fit(data={"users": "data/users.parquet", "orders": "data/orders.parquet"}, epochs=300)

# 3. Generate (child counts determined automatically by LinkageModel)
output_paths = synth.sample(num_rows={"users": 1000}, output_format="parquet", output_path="/tmp/output")
```

### D. Validation
Check the quality of your output:
```python
from syntho_hive.validation.report_generator import ValidationReport

report = ValidationReport()
report.generate(
    real_data={"users": real_df},
    synth_data={"users": synthetic_df},
    output_path="quality_report.html"
)
```

---

## Documentation

See [CHANGELOG.md](CHANGELOG.md) for release history and upgrade notes.

Full documentation is available at the [MkDocs site](https://ardada2468.github.io/SynthoHive) or can be built locally:

```bash
pip install .[docs]
mkdocs serve
```

---

## Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Acknowledgments

SynthoHive builds on the foundational work of the CTGAN team:

```bibtex
@inproceedings{ctgan,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
