# Core Data: Transformation & Preprocessing

This directory contains the data transformation logic responsible for converting raw tabular data into numerical formats suitable for deep learning models (GANs), and reversibly converting the model outputs back to the original tabular format.

## Overview

Generative models like GANs work best with normalized, continuous numerical data. Real-world tabular data, however, is complex:
-   **Multi-modal Continuous Data**: Columns like "Amount" or "Age" often have multiple peaks (modes) or long tails. Simple min-max scaling destroys this structure.
-   **Categorical Data**: Columns like "City" or "Category" are discrete and symbolic.
-   **Relational Keys**: Primary Keys (PK) and Foreign Keys (FK) are structural identifiers, not content features.

The `DataTransformer` solves these challenges to ensure the generative model learns the *content* distribution effectively.

## Components

### 1. DataTransformer (`transformer.py`)

The main orchestrator class. It automatically detects column types and applies the appropriate sub-transformer.

-   **Input**: `pandas.DataFrame` + `TableConfig` (Metadata).
-   **Responsibility**:
    1.  **Relational Filtering**: Uses metadata to **identify and exclude** Primary Keys (PK) and Foreign Keys (FK) from the transformation. These columns are handled by the Relational Orchestrator/Sampler logic, not the statistical GAN.
    2.  **Type Detection**: Automatically classifies columns as Continuous (Numeric) or Categorical (Object/String).
    3.  **Type Detection**: Automatically classifies columns as Continuous (Numeric) or Categorical (Object/String).
    4.  **Cardinality Check**: Checks categorical columns against `embedding_threshold` (default 50).
        -   **High Cardinality (> Threshold)**: Uses `LabelEncoder` to convert to integer indices. Metadata tagged as `categorical_embedding`. Handled via Entity Embeddings in the model.
        -   **Low Cardinality (<= Threshold)**: Uses `OneHotEncoder` to flatten into multiple binary columns. Metadata tagged as `categorical`.
    5.  **Transformation**: Applies `ClusterBasedNormalizer` to continuous columns, `LabelEncoder` or `OneHotEncoder` to categorical ones.
    6.  **Concatenation**: Flattens all transformed outputs into a single matrix. (Note: Embedding columns remain as single index columns in this matrix).

### 2. ClusterBasedNormalizer (`transformer.py`)

Changes the game for continuous data modeling. Instead of simple scaling, it uses a **Variational Gaussian Mixture Model (VGM)**.

#### The Problem with Min-Max
If a column has values clustered around 0 and 100 (bimodal), min-max scaling maps them to ~0.0 and ~1.0. A generic GAN generating "0.5" produces a value (50) that might be extremely unlikely in the real data.

#### The Solution (VGM)
The `ClusterBasedNormalizer`:
1.  **Fits a GMM**: Learns $k$ Gaussian modes (clusters) that describe the data density.
2.  **Transforms**:
    -   Predicts the **Cluster Probability** (which mode does this point belong to?).
    -   Calculates a **Normalized Scalar** (where is this point relative to that mode's mean/std?).
3.  **Output encoding**: A value $x$ becomes a vector:
    $$ [\underbrace{0, 1, 0, \dots}_{\text{One-Hot Cluster}}, \underbrace{0.45}_{\text{Normalized Scalar}}] $$

This allows the GAN to learn: "This point is in Cluster 2, and it's slightly above the cluster mean."

## Usage

```python
from syntho_hive.core.data.transformer import DataTransformer
from syntho_hive.interface.config import Metadata
import pandas as pd

# Setup Metadata
meta = Metadata()
meta.add_table("users", pk="user_id")

data = pd.DataFrame({
    "user_id": [1, 2, 3],       # Will be excluded
    "age": [25, 30, 65],        # Will be VGM normalized
    "city": ["NY", "SF", "NY"]  # Will be One-Hot encoded
})

# Initialize & Fit
transformer = DataTransformer(metadata=meta)
transformer.fit(data, table_name="users")

# Transform (To Matrix)
# Result shape: (N, age_modes + 1 + city_categories)
matrix = transformer.transform(data)

# Inverse Transform (To DataFrame)
# Result: Original dataframe structure (minus PKs)
df_recovered = transformer.inverse_transform(matrix)
```

## Key Features

-   **Reversibility**: Essential for synthetic data generation. Transformation $\rightarrow$ Generation $\rightarrow$ Inverse Transformation.
-   **Relational Integrity**: Respects schema definitions to avoid corrupting structural keys.
-   **Robustness**: Handles mixed types and multi-modal distributions automatically.
