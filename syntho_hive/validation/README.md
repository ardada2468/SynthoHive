# SynthoHive Validation Module

The `syntho_hive/validation` module provides tools to assess the quality and utility of synthetic data by comparing it against the original real data. It includes statistical tests for distribution similarity and correlation retention, along with a report generator to visualize the results.

## Key Components

### 1. StatisticalValidator (`statistical.py`)

The `StatisticalValidator` class is the core engine for performing statistical checks. It compares the real and synthetic datasets column by column to determine how well the synthetic data preserves the statistical properties of the original data.

#### Features:
*   **Robust Input Handling**: Safely handles empty DataFrames, missing columns, and type mismatches between real and synthetic data.
*   **Numeric Column Validation (KS Test)**: Uses the **Kolmogorov-Smirnov (KS) Test** to compare the distributions of continuous numeric variables.
*   **Categorical Column Validation (TVD)**: Uses **Total Variation Distance (TVD)** to compare the frequency distributions of categorical variables.
*   **Correlation & Structure**:
    *   **Correlation Matrix Comparison**: Calculates the **Frobenius Norm** of the difference between the correlation matrices of the real and synthetic data. A lower value indicates better preservation of linear relationships between variables.

#### Metrics Explained:
*   **KS Statistic**: A value between 0 and 1. Lower is better (0 = identical distributions).
*   **P-Value**: Indicates statistical significance. We generally look for a p-value > 0.05 to fail to reject the null hypothesis (meaning the distributions are essentially the same).
*   **TVD (Total Variation Distance)**: A value between 0 and 1. It represents the sum of absolute differences in probabilities for all categories, halved. Lower is better (0 = identical frequencies).
*   **Correlation Distance**: The Euclidean distance between the two correlation matrices. Closer to 0 means better structural preservation.

### 2. ValidationReport (`report_generator.py`)

The `ValidationReport` class orchestrates the validation process for one or multiple tables and generates a readable report.

#### Supported Formats:
*   **HTML**: Generates a self-contained HTML file with styled tables, pass/fail indicators, and color-coded status.
*   **JSON**: Exports raw metrics to a JSON file for programmatic consumption.

#### Usage:

```python
from syntho_hive.validation.report_generator import ValidationReport
import pandas as pd

# 1. Prepare your data (Real and Synthetic dictionaries)
real_data = {
    "users": pd.read_csv("real_users.csv"),
    "transactions": pd.read_csv("real_transactions.csv")
}
synth_data = {
    "users": pd.read_csv("synthetic_users.csv"),
    "transactions": pd.read_csv("synthetic_transactions.csv")
}

# 2. Initialize and Generate
report_gen = ValidationReport()

# Generate HTML Report
report_gen.generate(
    real_data=real_data, 
    synth_data=synth_data, 
    output_path="validation_report.html"
)

# Generate JSON Report
report_gen.generate(
    real_data=real_data, 
    synth_data=synth_data, 
    output_path="metrics.json"
)
```

## Running Tests

Unit tests are provided to ensure the reliability of the validation logic. You can run them using:

```bash
python3 syntho_hive/tests/test_validation.py
```

This will output a test HTML report to `syntho_hive/test_output/report.html` for you to inspect.
