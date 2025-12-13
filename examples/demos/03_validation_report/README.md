## Validation Report (HTML + JSON)

- **What it shows:** Running `ValidationReport` on side-by-side real vs. synthetic data to quantify KS/TVD/correlation gaps.
- **Key APIs:** `ValidationReport.generate`.
- **Outputs:** `outputs/validation_report.html` and `outputs/validation_metrics.json`.

### Run

```bash
python examples/demos/03_validation_report/run.py
```

Flags:
- `--rows`: Number of records per table to generate for the comparison (default 300).
- `--output-dir`: Where to place the report artifacts.

### What to look at
- The HTML report contains pass/fail statuses per column and correlation distance.
- The JSON metrics make it easy to feed into CI or dashboards.

