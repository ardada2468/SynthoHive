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
    *   Typically, $< 0.1$ is considered excellent quality.

### 2. Total Variation Distance (TVD)
*   **Target**: Categorical Columns
*   **Measure**: Half the sum of absolute differences between category probabilities.
*   **Interpretation**:
    *   `0.0` = Perfect fit (Category frequencies match exactly).
    *   `1.0` = Totally different.

### 3. Correlation Distance
*   **Target**: Column Pairs
*   **Measure**: We compute correlation matrices (Pearson for continuous, Theil's U for categorical) for both Real and Synthetic datasets. The score is the L2 norm of the difference matrix.
*   **Goal**: Measures how well the model captured relationships *between* columns (e.g., Age vs. Income).

## The HTML Report

The `ValidationReport.generate()` method produces a self-contained HTML file containing:

1.  **Summary Score**: Aggregate utility score (0-100%).
2.  **Column Distributions**: Overlay plots (Histogram/KDE) for every column.
3.  **Correlation Heatmaps**: Side-by-side comparison of Real vs. Synthetic associations.
4.  **Row Previews**: Snippets of raw data to verify formatting.

## Usage

```python
from syntho_hive.validation.report_generator import ValidationReport

report = ValidationReport()
report.generate(
    real_data=real_dfs,      # Dict[str, pd.DataFrame]
    synth_data=synth_dfs,    # Dict[str, pd.DataFrame]
    output_path="report.html"
)
```
