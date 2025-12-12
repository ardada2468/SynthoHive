# Relational Module

This directory contains the logic for modeling and generating multi-table relational datasets.

## Components

### 1. Schema Graph (`graph.py`)
Builds a Direct Acyclic Graph (DAG) of table dependencies based on Foreign Keys defined in `Metadata`.
*   Determines topological order for "Parent-First" generation.
*   Handles cycle detection (currently supports acyclic schemas only).

### 2. Linkage Model (`linkage.py`)
Models the **Cardinality Relationship** between Parent and Child tables (1:N relationships).
*   Learns the distribution of $N$ (number of children) for each parent.
*   Uses a Gaussian Mixture Model (GMM) to approximate this distribution.
*   During generation, samples a child count $k_i$ for each parent $i$, then repeats the parent's context row $k_i$ times.

### 3. Staged Orchestrator (`orchestrator.py`)
The central engine that manages the full pipeline. It coordinates the `SparkIO`, `CTGAN` models, and `LinkageModel`s.

#### `fit_all(real_data_paths, epochs=300, batch_size=500, **model_kwargs)`
Trains models for every table in the schema. Accepts `epochs`, `batch_size`, and other model arguments to control the training process.
1.  Iterates through all tables.
2.  For **Child Tables**:
    *   Joins Child data with Parent data to extract "Parent Context" attributes (defined in `TableConfig.parent_context_cols`).
    *   Trains a `LinkageModel` to learn the distribution of child records per parent.
    *   Trains a Conditional `CTGAN` on Child data, conditioned on Parent Context.
3.  For **Root Tables**:
    *   Trains a standard `CTGAN` (unconditional).

#### `generate(num_rows_root, output_path)`
Generates synthetic data ensuring referential integrity:
1.  Determines generation order (Parents before Children).
2.  **Step 1: Generate Root**:
    *   Samples $N$ rows from the Root table's `CTGAN`.
    *   Assigns new Primary Keys.
3.  **Step 2: Generate Child**:
    *   Reads the *recently generated* Parent data.
    *   Uses `LinkageModel` to sample how many children each generated parent should have.
    *   Repeats Parent Context rows to match these counts.
    *   Feeds this repeated context into the Child's Conditional `CTGAN` to generate the child rows.
    *   Assigns Foreign Keys (linking back to the specific generated Parent IDs) and new Primary Keys.

## Key Features
*   **Referential Integrity**: Guarantees valid Foreign Keys by construction (generating children *from* specific parent instances).
*   **Statistical Correlation**: Preserves correlations between Parent attributes (e.g., User Region) and Child attributes (e.g., Order Amount) via conditional generation.
*   **Context Isolation**: Handles column name collisions between parent and child tables during context merging.
