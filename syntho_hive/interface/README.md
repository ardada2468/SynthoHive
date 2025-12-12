# SynthoHive Interface

The `syntho_hive/interface` module provides the high-level API for interacting with the SynthoHive synthetic data generation engine. It serves as the primary entry point for users to define schemas, configure privacy guardrails, train models, and generate synthetic datasets.

## Overview

This module is designed to abstract away the complexities of the underlying relational orchestration and generative modeling layers. It exposes a user-friendly `Synthesizer` class and configuration objects using `Pydantic` for validation.

## Key Components

### 1. Synthesizer (`synthesizer.py`)

The `Synthesizer` class is the main coordinator. It manages the lifecycle of the synthetic data generation process.

**Key Features:**
*   **Spark Integration**: Seamlessly integrates with PySpark for scalable data processing.
*   **Model Management**: Orchestrates the training and generation of models across multiple tables while maintaining referential integrity.
*   **Privacy Handling**: Applies privacy configurations during the generation process.
*   **Validation**: Includes built-in tools to generate statistical validation reports comparing real and synthetic data.
*   **Hive Integration**: Directly registers generated datasets as Hive tables.

**Usage:**

```python
from syntho_hive.interface.synthesizer import Synthesizer
from syntho_hive.interface.config import Metadata, PrivacyConfig
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# config objects
metadata = Metadata(...) 
privacy = PrivacyConfig(...)

# Initialize Synthesizer
synthesizer = Synthesizer(
    metadata=metadata,
    privacy_config=privacy,
    spark_session=spark,
    backend="CTGAN"
)

# Train
synthesizer.fit(
    database="real_db_name",
    epochs=300,        # Configurable training epochs
    batch_size=500     # Configurable batch size
)

# Generate
paths = synthesizer.sample(num_rows={"users": 1000})

# Save results
synthesizer.save_to_hive(paths, target_db="synthetic_db")
```

### 2. Configuration (`config.py`)

This file defines the data structures used to configure the synthesis process.

#### `Metadata`
Defines the schema of the relational database.
*   `tables`: A dictionary of `TableConfig` objects.
*   `add_table()`: Helper method to add tables with validation.
*   `validate_schema()`: Validates logical integrity (e.g., that foreign keys point to existing parents).

#### Table Configuration

Define `TableConfig` for each table:
- `name`: Table name (must match source data file name excluding extension).
- `pk`: Primary Key column name.
- `fk`: Dictionary of foreign keys mapping `local_col` to `parent_table.parent_col`.
- `constraints`: Dictionary mapping `column_name` to `Constraint` definitions.

#### Constraints
You can enforce data types and value ranges using `Constraint`:
- `dtype`: Force output type ("int" or "float").
- `min`: Minimum value (inclusive).
- `max`: Maximum value (inclusive).

**Example:**
```python
from syntho_hive.interface.config import Constraint

meta.add_table(
    "users", 
    pk="user_id",
    constraints={
        "age": Constraint(dtype="int", min=18, max=90),
        "score": Constraint(dtype="float", min=0.0, max=1.0)
    }
)
```

### Relational Modeling.
*   `parent_context_cols`: Parent columns to use as context for conditional generation.

#### `PrivacyConfig`
Global privacy settings.
*   `enable_differential_privacy`: Toggle for DP.
*   `epsilon`: Privacy budget.
*   `pii_strategy`: Strategy for PII handling (`"mask"`, `"faker"`, `"context_aware_faker"`).

## Workflow

1.  **Define Schema**: Create a `Metadata` object and populate it with `TableConfig`s describing your source database.
2.  **Configure Privacy**: Set up `PrivacyConfig` according to compliance requirements.
3.  **Initialize**: Instantiate the `Synthesizer`.
4.  **Fit**: Call `fit()` to learn distributions from the real data. This stage handles dependency resolution and trains models in topological order.
5.  **Sample**: Call `sample()` to generate new data. The engine respects foreign keys and ensures referential integrity.
6.  **Validate**: Use `generate_validation_report()` to assess quality.
7.  **Deploy**: Use `save_to_hive()` to make the data available for downstream consumers.

## Requirements

*   **PySpark**: A simplified or full Spark environment is required for `fit`, `sample`, and `save_to_hive`.
*   **Pandas**: Used for internal data handling in some connectors and reporting.
