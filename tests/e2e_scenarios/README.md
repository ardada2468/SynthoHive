# Retail End-to-End Test Scenario

This directory contains the `retail_test.py` script, which serves as a comprehensive end-to-end (E2E) test for the SynthoHive synthetic data pipeline. It simulates a real-world scenario involving a relational database for a retail application.

## Overview

The test demonstrates the full lifecycle of synthetic data generation:
1.  **Generation of "Real" Ground Truth Data**: Creating a fake "original" dataset.
2.  **Privacy Sanitization**: Masking sensitive PII (Personally Identifiable Information).
3.  **Model Training & Synthesis**: Learning the statistical patterns of the data and generating new synthetic records.
4.  **Validation**: Comparing the statistical properties of the original vs. synthetic data.

## Workflow Phases

The script processes data in four distinct phases:

### Phase 1: Ground Truth Generation
Valid "Raw" data is generated using the `Faker` library to simulate a relational database schema.
*   **Tables Generated**:
    *   `regions` (Root): Geographic regions.
    *   `products` (Root): Catalog of items with prices and categories.
    *   `users` (Child of Regions): Customers with personal info (email, phone).
    *   `orders` (Child of Users): Transactional records linked to users.
    *   `order_items` (Child of Orders): Line items for each order.
    *   `shipments` (Child of Orders): Shipping details for completed orders.
*   **Output**: CSV files in `output/test_retail/raw/`.

### Phase 2: Privacy Sanitization
Sensitive information is removed or obfuscated before the model creates synthetic data, ensuring privacy preservation.
*   **Tool**: `PIISanitizer`
*   **Transformations**:
    *   **Email**: Faked (Replaced with realistic but fake emails).
    *   **Phone**: Faked.
    *   **Address**: Faked.
    *   **Tracking Number**: Masked (e.g., `****1234`).
*   **Output**: Sanitized CSV files in `output/test_retail/clean/`. This is the dataset used for training.

### Phase 3: Relational Modeling (Training & Generation)
The core SynthoHive engine trains on the *Clean* data to learn the schema and statistical distributions.
*   **Tool**: `StagedOrchestrator` (uses Spark & CTGAN).
*   **Process**:
    1.  **Metadata Definition**: Defines PKs, FKs, and parent context columns (e.g., spending depends on user age).
    2.  **Training**: Fits probabilistic models (CTGAN) for each table, respecting relational constraints.
    3.  **Generation**: Generates new, synthetic records. It starts with root tables and cascades down to children to maintain referential integrity.
*   **Output**: Synthetic parquet/CSV files in `output/test_retail/synthetic/`.

### Phase 4: Validation & Reporting
The final phase verifies that the synthetic data accurately reflects the statistical properties of the original data.
*   **Tool**: `ValidationReport`
*   **Metrics**:
    *   **KS Test**: Compares distributions of numerical columns.
    *   **TVD**: Compares distributions of categorical columns.
    *   **Correlation Distance**: Checks if pairwise correlations are preserved.
*   **Output**: An HTML report at `output/test_retail/report.html` containing:
    *   Pass/Fail status for each column.
    *   Detailed statistics (Mean, Std, Min, Max).
    *   Side-by-side data previews.

## How to Run

Execute the script from the project root:

```bash
python3 tests/e2e_scenarios/retail_test.py
```

Check the results in `output/test_retail/report.html`.
