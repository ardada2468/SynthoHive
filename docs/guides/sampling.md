---
title: Sampling & Relational
---

# Sampling & Relational Generation

Once your models are trained, you can generate synthetic data. SynthoHive uses a **Top-Down Logic** for generation, ensuring that parent records are created before their children, preserving referential integrity.

## The Generation Logic

### 1. Root Tables
Generation starts with tables that have no foreign keys (Roots).
*   **Method**: The CTGAN model samples `n` rows directly from the learned latent space.
*   **Identity**: Primary Keys (e.g., `user_id`) are assigned sequentially (1, 2, 3...) to these new rows.

### 2. Child Tables (The Cascade)
Once a parent table exists, its children are generated using the **Linkage Model**.
*   **Step A: Sample Counts**: For every row in the Parent table, the Linkage Model predicts *how many* children it should have.
    *   *Real World Analogy*: For User #1, the model predicts 2 orders. For User #2, it predicts 0.
*   **Step B: Expand Context**: The Parent's "Context Columns" (e.g., `Region`) are repeated `k` times, where `k` is the predicted count.
*   **Step C: Conditional Sampling**: The Child CTGAN model generates the specific details for these rows, *conditioned* on the repeated parent context.
*   **Step D: Assign FKs**: The Parent's ID is assigned to the foreign key column (`user_id`).

### 3. Multi-Parent Tables (Secondary FKs)
If a table has multiple parents (e.g., `OrderItems` has `order_id` AND `product_id`):
1.  One parent is chosen as the **Driver** (usually the first FK alphabetically or by configuration). This driver controls the *volume* (how many items per order).
2.  The other parent (`product_id`) is assigned via **Random Sampling**.
    *   The system creates the rows based on the Order driver.
    *   For the `product_id` column, it randomly picks valid IDs from the already-generated `Products` table.
    *   *Note*: This random assignment preserves the validity of the FK (it points to a real Product), but does not currently enforce strict joint distributions between two parents.

## Scaling & Volume

You control the volume of the generated database by setting the row counts for **Root Tables**.

```python
num_rows = {
    "users": 100_000,    # Root table: Exact count
    "products": 500      # Root table: Exact count
}

# Child tables (e.g., "orders") are NOT specified here. 
# Their volume is determined by the Linkage Model ratios (e.g., avg 5 orders/user).
```

!!! tip "Scaling Factor"
    If you want a 2x larger database, simply double the root counts. The child tables will naturally scale up by ~2x because the linkage ratios (children per parent) remain constant.

## API Usage

### `synthesizer.sample()`

```python
output = synth.sample(
    num_rows={"users": 1000},
    output_format="delta",
    output_path="/tmp/synthetic_db"
)
```

Arguments:
-   `num_rows` (Dict[str, int]): Map of root table names to desired row counts.
-   `output_format` (str): Format for writing files. Default is `"delta"`.
-   `output_path` (Optional[str]):
    -   If provided (str): Writes files to disk at this path. Returns a dictionary of table paths.
    -   If `None`: Returns a dictionary of **Pandas DataFrames** in memory.

### Example: In-Memory Generation
Useful for smaller datasets or unit tests.

```python
dfs = synth.sample(
    num_rows={"users": 100}, 
    output_path=None
)

users_df = dfs["users"]
orders_df = dfs["orders"]
print(users_df.head())
```

## Performance & Limitations

### Memory Usage
The current implementation generates data in **batches per table**.
-   **Limitation**: The entire generated table must fit in memory before writing to disk.
-   **Workaround**: If you need 100M rows, do not generate them in one generic call. Write a loop to generate 1M rows 100 times, saving each batch to disk.

### Referential Integrity
-   **Primary Keys**: Guaranteed unique for the generated batch.
-   **Foreign Keys**: Guaranteed valid (always point to an existing parent in the current batch).

!!! warning "Partitioning"
    Generation is not currently distributed across Spark workers. It runs on the driver logic. Future versions will support distributed generation for massive scale.
