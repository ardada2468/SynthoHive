# Core Models: Conditional Tabular GAN (CTGAN)

This directory contains the implementation of the core generative models used in SynthoHive, primarily the **Conditional Tabular GAN (CTGAN)**.

## Overview

SynthoHive uses a deep learning approach to generate synthetic tabular data that preserves the statistical properties, correlations, and relational structure of the original dataset.

The core model is a **CTGAN** adapted for relational schemas. It creates high-fidelity synthetic data by learning the underlying distribution of a table, conditioned on its parent table's attributes.

## Architecture

The model follows a Generative Adversarial Network (GAN) architecture with two competing neural networks:

### 1. Generator ($G$)
-   **Input**: A random noise vector ($\mathbf{z}$) + Conditional Vector ($\mathbf{c}$).
-   **Structure**: A sequence of **Residual Layers**. Deep residual networks allow the model to learn complex, non-linear dependencies without suffering from vanishing gradients.
-   **Output**: A synthetic row vector in the transformed feature space.

### 2. Discriminator ($D$)
-   **Input**: A row vector (Real or Fake) + Conditional Vector ($\mathbf{c}$).
-   **Structure**: A Multi-Layer Perceptron (MLP) with LeakyReLU activations.
-   **Output**: A scalar score (critic value) indicating realism.

### 3. WGAN-GP (Wasserstein GAN with Gradient Penalty)
Standard GANs are notoriously unstable for tabular data. We implement **WGAN-GP** to improve stability:
-   **Wasserstein Loss**: Measures the distance between the real and model distributions (Earth Mover's Distance).
-   **Gradient Penalty**: Enforces 1-Lipschitz continuity on the discriminator (critic), preventing mode collapse and ensuring smoother convergence.

## Data Processing Pipeline

The model does not train on raw data. It relies on the `DataTransformer` (in `../data/transformer.py`) to preprocess inputs into a format suitable for neural networks.

1.  **Continuous Columns**:
    -   Modeled using **Variational Gaussian Mixture (VGM)**.
    -   Each value `x` is represented as a concatenation of:
        -   **Mode Vector**: A one-hot vector indicating which cluster/mode the value belongs to.
        -   **Scalar**: The normalized value within that cluster (mean-centered and scaled).
        -   **Null Indicator**: (Optional) 1 if the value was missing, 0 otherwise.
    -   This allows the model to capture multi-modal distributions (e.g., income distribution with distinct peaks for different brackets) and missing data patterns.

2.  **Categorical Columns**:
    -   Converted using **One-Hot Encoding**.
    -   High-cardinality columns are handled via **Entity Embeddings** (mapped to dense vectors), while low-cardinality columns use One-Hot Encoding. This is configurable via `embedding_threshold`.

3.  **Relational Context**:
    -   Unlike standard CTGAN, our model accepts an optional `context` DataFrame (attributes from a parent table).
    -   This context is transformed using a dedicated `DataTransformer` and concatenated with the input noise/data.
    -   This enables **Conditional Generation**: $P(\text{Child} | \text{Parent})$.

## Usage

### Initialization
```python
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata

# Define Metadata
meta = Metadata()
meta.add_table("transactions", pk="id", fk={"user_id": "users.id"})

# Initialize Model
model = CTGAN(
    metadata=meta,
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    batch_size=500,
    epochs=300
)
```

### Training
```python
# data: DataFrame of child table (target)
# context: DataFrame of parent attributes for each row (optional)
# checkpoint_dir: Directory to save best model and metrics (optional)
model.fit(
    data, 
    context=context, 
    table_name="transactions",
    checkpoint_dir="./checkpoints/transactions",
    log_metrics=True
)
```

### Checkpointing & Monitoring
During training, the model can automatically save artifacts to a `checkpoint_dir`:
-   **`best_model.pt`**: The state dictionary of the model with the lowest Generator Loss.
-   **`last_model.pt`**: The model state at the end of the last epoch.
-   **`training_metrics.csv`**: A CSV log containing `epoch`, `loss_g`, and `loss_d` for every epoch.

### Sampling
```python
# Generate synthetic data conditioned on specific parent rows
synthetic_data = model.sample(num_rows=100, context=context_df)
```

## Implementation Details

-   **`ctgan.py`**: Main class managing the model lifecycle, training loop, and transformer integration.
-   **`layers.py`**: PyTorch modules for `ResidualLayer` (ResNet blocks) and `Discriminator`.
-   **`base.py`**: Abstract base classes defining the `GenerativeModel` interface.
