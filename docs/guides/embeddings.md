---
title: Embeddings & High Cardinality
---

# Feature Embeddings

SynthoHive provides specialized handling for **High Cardinality** categorical columns (e.g., Use ID, Zip Code, Product ID) using **Entity Embeddings**. This avoids the computational explosion of One-Hot Encoding and allows the model to learn semantic relationships between categories.

## When to use Embeddings vs One-Hot?

| Feature | One-Hot Encoding | Entity Embeddings |
| :--- | :--- | :--- |
| **Technique** | Creates a binary column for every unique category. | Maps each category to a dense vector of floating point numbers. |
| **Cardinality** | Low (< 50 unique values). | High (> 50 unique values). |
| **Memory Usage** | High (Sparse but wide). | Low (Compact dense vectors). |
| **Relationships** | Independent. No relationship between 'A' and 'B'. | Learned. Similar categories end up close in vector space. |
| **Example** | `Gender`, `MaritalStatus`. | `ZipCode`, `UserID`, `ICD9_Code`. |

## How it Works

The transformation pipeline automatically detects high-cardinality columns based on a threshold.

### 1. Detection
During `DataTransformer.fit()`, the system checks the number of unique values in each categorical column.
If `num_unique > embedding_threshold` (default: 50), the column is flagged for embedding.

### 2. Transformation
*   **One-Hot**: Converts string `"A"` -> `[1, 0, 0]`.
*   **Embedding**: Converts string `"A"` -> `Integer Index (42)`.

### 3. Model Training (CTGAN)
Inside the neural network model:
*   **Generator**: Outputs a probability distribution (logits) over all possible categories.
*   **Discriminator**: Feeds the index (for real data) or probability-weighted vector (for fake data) into a learnable **Embedding Layer**.
*   **Learning**: The model learns to place similar entities near each other. For example, if Zip Codes `10001` and `10002` have similar correlations with `Income`, their embedding vectors will become similar during training.

## Configuration

You can control the threshold for switching to embeddings globally or per-model.

### Global Configuration
Set the `embedding_threshold` when initializing the synthesizer or calling `fit`.

```python
synth.fit(
    data=df,
    embedding_threshold=100  # Only use embeddings if > 100 unique values
)
```

Lowering this value forces more columns to use embeddings, which saves memory but might reduce precision for small categorical sets. increasing it uses One-Hot for more columns, which is more precise but memory-intensive.

## Use Cases

### 1. Geographical Data
Zip codes, Cities, or State abbreviations often have hundreds of values. Embeddings allow the model to learn valid geography (e.g., that "NY" and "NJ" are related) rather than treating them as unrelated tokens.

### 2. ID Columns
While primary keys like `UserID` are usually excluded, you might have **Foreign Keys** or distinct identifiers like `ProductCode` that you want to synthesize while preserving their statistical properties.

### 3. Medical Codes
ICD-9 or CPT codes have thousands of distinct values. Embeddings are essential for synthesizing electronic health records (EHR) effectively.
