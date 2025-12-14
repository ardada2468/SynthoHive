---
title: Relational Concepts
---

# Relational Data Generation

SynthoHive specializes in maintaining referential integrity and statistical correlations across multiple tables. This guide explains the core concepts behind our orchestration engine.

## The "Driver Parent" Concept

In a complex schema, a child table might refer to multiple parent tables. For example, an `Orders` table might refer to both `Users` and `Products`.

When generating a synthetic `Order`:
1.  **Which parent dictates existence?** We treat one foreign key relationship as the **Driver**. Usually, this is the entity that "owns" the record (e.g., `User`).
2.  **How many records?** We use a `LinkageModel` to learn the distribution of child records per driver parent (e.g., "Users typically have 0-5 orders").

### Secondary Parents
Other foreign keys (e.g., `Product`) are treated as **Secondary**. These are assigned to ensure referential integrity, but they do not drive the count of generated records.

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

1.  **Schema Analysis**: Construct a Directed Acyclic Graph (DAG) of the schema.
2.  **Topological Sort**: Determine generation order (Parents -> Children).
3.  **Root Generation**: Generate independent root tables using standard CTGAN.
4.  **Child Loop**:
    *   Load synthetic parent data.
    *   Sample child counts for each parent row.
    *   Repeat parent IDs and Context attributes.
    *   Generate child rows conditioned on repeated context.
    *   Sample valid FKs for secondary parents.
    *   Assign Primary Keys.
