---
title: Fitting Models
---

# Fitting Models

The `fit` process is the core of SynthoHive's generative engine. It takes your real relational database and trains a set of machine learning models that learn both the **structure** (relationships between tables) and the **content** (statistical distribution of data within tables).

## Concepts

SynthoHive uses a **Hybrid Relational Approach**. Instead of trying to train one massive model for the entire database (which is often computationally infeasible), it breaks the problem down:

1.  **Linkage Models**: Learn *how many* child records typically exist for a given parent record. For example, "How many `Orders` does a typical `User` have?"
2.  **Generative Models (CTGAN)**: Learn the content of each table individually. To preserve relational integrity dev-to-dev (e.g., ensuring an Order's `city` matches the User's `region`), child models are **conditioned** on context from their parent tables.

### The Role of Metadata

Your `Metadata` object is the blueprint for this process. It tells the synthesizer:
*   **Hierarchy**: Which tables are parents and which are children (defined by Foreign Keys).
*   **Context**: Which columns from a parent should influence the generation of a child (e.g., `region` affecting `shipping_speed`).
*   **Constraints**: Data types and logical rules that must be preserved.

## The Fitting Workflow

When you call `synthesizer.fit()`, the following steps occur for each table in your metadata:

### 1. Data Ingestion
The system reads your real data using Spark.
*   **Note**: Currently, the system converts Spark DataFrames to **Pandas** for checking into the CTGAN backend. This means the working set for a single table must fit in memory.

### 2. Preprocessing & Transformation
Each column is analyzed and transformed:
*   **Continuous Columns**: Modeled using **Variational Gaussian Mixture Models (VGM)** to handle multi-modal distributions (e.g., a salary distribution with peaks at $40k and $120k).
*   **Categorical Columns**:
    *   **Low Cardinality**: Converted using One-Hot Encoding.
    *   **High Cardinality**: Converted using **Entity Embeddings** (see [Embeddings Guide](embeddings.md)).
*   **Primary/Foreign Keys**: Excluded from the content model (CTGAN) because they are structural identifiers, not semantic content.

### 3. Linkage Learning
For child tables, a `LinkageModel` describes the relationship with its "Driver" parent (the primary foreign key table).
*   It calculates the probability distribution of child counts given parent attributes.
*   *Example*: A `User` in the "Enterprise" segment might have 50-100 `logs`, while a "Free" user has 0-5.

### 4. Conditional Training (The "Magic")
To ensure relational consistency, child tables are trained with **Context**.
1.  The system performs a left join of the Child table with selected columns from the Parent table.
2.  The CTGAN model is trained on this joined dataset.
3.  **Result**: When generating a new `Order`, the model receives the specific `User`'s context (e.g., `Region=US`) and generates an invoice meant for that region.

## Configuration

You can tune the fitting process via the `fit()` arguments and global config.

```python
synth.fit(
    data="metrics_db",
    epochs=300,
    batch_size=500,
    embedding_threshold=50
)
```

### Key Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `epochs` | `int` | `300` | Number of training iterations. Higher means better quality but longer training. <br>• **Testing/Dev**: Use `10-50` to verify the pipeline runs.<br>• **Production**: Use `300+` for high-fidelity data. |
| `batch_size` | `int` | `500` | Number of samples processed at once. <br>• Increase for speed (if GPU memory allows).<br>• Decrease if you hit Out-Of-Memory (OOM) errors. |
| `embedding_threshold` | `int` | `50` | Columns with unique values > this number will use Embeddings instead of One-Hot. |
| `validate` | `bool` | `False` | If `True`, automatically runs a validation report after training (requires `validate` extra dependencies). |

!!! warning "Unused Parameters"
    The parameters `sample_size` and `sampling_strategy` currently appear in the `fit()` signature but are **not yet implemented**. The system currently uses the full dataset provided in the `data` argument. Future versions will support downsampling large datasets before training.

### Hardware & Performance

*   **Memory**: The most critical resource. Since data is converted to Pandas/Numpy for training, you generally need **2-3x the size of your largest single table** in RAM.
*   **CPU vs GPU**:
    *   **CPU**: Default. Works fine for smaller datasets (< 100k rows/table).
    *   **GPU**: Highly recommended for production training. CTGAN is a neural network; training on a GPU is 10-50x faster.
    *   To use GPU, ensure you have a CUDA-compatible PyTorch version installed. The synthesizer automatically detects and uses CUDA if available.

## Troubleshooting

### Out of Memory (OOM) Errors
*   **Symptom**: Process crashes with `MemoryError` or `CUDA out of memory`.
*   **Fix 1**: Reduce `batch_size` (e.g., to 100 or 50).
*   **Fix 2**: Reduce the number of context columns in `Metadata`. Joining parent context increases the width of the training data.
*   **Fix 3**: Increase `embedding_threshold` carefully (embeddings use less memory than massive One-Hot vectors, but the transformation step itself needs memory).

### Mode Collapse (Output is constant or repetitive)
*   **Symptom**: The generated data looks very repetitive (e.g., every User is named "John").
*   **Cause**: The Generator found a "safe" output that fools the Discriminator, or the Discriminator is too weak.
*   **Fix**:
    *   Increase `epochs`. In early epochs, GANs often produce noise or mode-collapsed data before converging.
    *   Check for data skew. If 99% of your real data is "John", the model is actually correct!

### Training is too slow
*   **Fix**: Enable GPU acceleration.
*   **Fix**: Explicitly exclude irrelevant text columns (e.g., "Description" fields that are unique per row) from `Metadata` or mark them as PII to be faked rather than modeled. Learning free-text fields with a tabular GAN is inefficient.
