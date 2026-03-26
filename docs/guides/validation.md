---
title: Validation & Reporting
---

# Quality Validation

How do you know your synthetic data is good? SynthoHive provides a comprehensive `ValidationReport` to quantify fidelity.

## Metrics

### 1. Kolmogorov-Smirnov (KS) Test
*   **Target**: Continuous Columns (Float/Int)
*   **Measure**: The maximum distance between the cumulative distribution functions (CDFs) of real and synthetic data.
*   **Interpretation**:
    *   `0.0` = Perfect fit (Distributions are identical).
    *   `1.0` = Totally different.
    *   Typically, < 0.1 is considered excellent quality.

### 2. Total Variation Distance (TVD)
*   **Target**: Categorical Columns
*   **Measure**: Half the sum of absolute differences between category probabilities.
*   **Interpretation**:
    *   `0.0` = Perfect fit (Category frequencies match exactly).
    *   `1.0` = Totally different.

### 3. Correlation Distance
*   **Target**: Column Pairs
*   **Measure**: We compute correlation matrices (Pearson for continuous columns) for both Real and Synthetic datasets. The score is the Frobenius norm (L2 norm) of the difference matrix.
*   **Goal**: Measures how well the model captured relationships *between* columns (e.g., Age vs. Income).

## Output Formats

### HTML Report
The `ValidationReport.generate()` method produces a self-contained HTML file containing:

1.  **Column Validation Metrics**: KS test and TVD results per column with pass/fail indicators.
2.  **Correlation Distance**: Frobenius norm comparing real vs. synthetic correlation matrices.
3.  **Detailed Statistics**: Side-by-side descriptive statistics (mean, std, min, max for numeric; unique count, top value for categorical).
4.  **Row Previews**: Snippets of raw data to verify formatting.

### JSON Report
When the output path ends with `.json`, the report generates a structured JSON file containing all computed metrics, suitable for programmatic consumption or CI/CD pipelines.

## Usage

### HTML Report
```python
from syntho_hive.validation.report_generator import ValidationReport

report = ValidationReport()
report.generate(
    real_data={"users": real_df},      # Dict[str, pd.DataFrame]
    synth_data={"users": synth_df},    # Dict[str, pd.DataFrame]
    output_path="report.html"
)
```

### JSON Report
```python
report.generate(
    real_data={"users": real_df},
    synth_data={"users": synth_df},
    output_path="metrics.json"
)
```

## Next Steps

- [**Demo 03**](../demos/03_validation_report.md): Full runnable example generating both HTML and JSON reports.
- [**Fitting Guide**](fitting.md): Tune training parameters to improve validation scores.
