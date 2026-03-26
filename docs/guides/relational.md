---
title: Relational Concepts
---

# Relational Data Generation

SynthoHive specializes in maintaining referential integrity and statistical correlations across multiple tables. This guide explains the core concepts behind our orchestration engine.

## The "Driver Parent" Concept

In a complex schema, a child table might refer to multiple parent tables. For example, an `Orders` table might refer to both `Users` and `Products`.

When generating a synthetic `Order`:
1.  **Which parent dictates existence?** We treat one foreign key relationship as the **Driver**. The driver is selected as the first FK alphabetically, or can be configured.
2.  **How many records?** We use a `LinkageModel` to learn the distribution of child records per driver parent (e.g., "Users typically have 0-5 orders").

### Secondary Parents
Other foreign keys (e.g., `Product`) are treated as **Secondary**. These are assigned via random sampling from the already-generated parent table to ensure referential integrity, but they do not drive the count of generated records.

## Contextual Conditioning

To preserve correlations across tables (e.g., "Users in NY order Winter Coats"), we use **Conditional Generation**.

1.  **Fit Phase**: We join relevant columns from the Driver Parent (e.g., `User.City`) to the Child Table.
2.  **Training**: The CTGAN model learns not just `P(Order)`, but `P(Order | User.City)`.
3.  **Generation Phase**:
    *   We generate a synthetic User: `{ID: 1, City: "NY"}`.
    *   The `LinkageModel` says "Generate 3 orders for User 1".
    *   We pass `City="NY"` as context to the Order Generator.
    *   The generator produces orders statistically likely for a NY user.

## The Orchestration Flow

1.  **Schema Analysis**: `SchemaGraph` constructs a Directed Acyclic Graph (DAG) of the schema from FK definitions.
2.  **Topological Sort**: `get_generation_order()` determines generation order (Parents -> Children).
3.  **Root Generation**: Generate independent root tables using standard CTGAN.
4.  **Child Loop**:
    *   Load synthetic parent data.
    *   Sample child counts for each parent row via `LinkageModel`.
    *   Repeat parent IDs and Context attributes.
    *   Generate child rows conditioned on repeated context.
    *   Sample valid FKs for secondary parents.
    *   Assign sequential Primary Keys.

## Code Example

### Defining a Relational Schema

```python
from syntho_hive.interface.config import Metadata

metadata = Metadata()

# Root table (no FK)
metadata.add_table("users", pk="user_id")

# Child table with FK and context conditioning
metadata.add_table(
    "orders",
    pk="order_id",
    fk={"user_id": "users.user_id"},
    parent_context_cols=["region"],
    constraints={"basket_value": {"dtype": "float", "min": 1.0}},
)
```

### Training and Generating (Without Spark)

```python
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.relational.linkage import LinkageModel

# Train parent model
users_model = CTGAN(metadata, batch_size=128, epochs=300)
users_model.fit(users_df, table_name="users")

# Train linkage model
linkage = LinkageModel()
linkage.fit(users_df, orders_df, fk_col="user_id", pk_col="user_id")

# Train child model with parent context
context_df = orders_df.merge(users_df[["user_id", "region"]], on="user_id")[["region"]]
orders_model = CTGAN(metadata, batch_size=128, epochs=300)
orders_model.fit(orders_df, context=context_df, table_name="orders")

# Generate
synth_users = users_model.sample(1000)
synth_users.insert(0, "user_id", range(1, 1001))

counts = linkage.sample_counts(synth_users)
# ... build context and generate children (see Demo 04 for full example)
```

## Next Steps

- [**Sampling Guide**](sampling.md): Detailed API usage for `synthesizer.sample()` and volume control.
- [**Demo 04**](../demos/04_relational_linkage_ctgan.md): Full runnable example of relational generation.
- [**Fitting Guide**](fitting.md): Deep dive into the training process and parameter tuning.
