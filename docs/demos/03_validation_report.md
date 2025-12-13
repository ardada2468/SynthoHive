---
title: Validation Report Demo
---

# Validation Report Demo

Path: `examples/demos/03_validation_report`

## Goal
Compare real vs synthetic tables, compute KS/TVD metrics, correlation distance, and render an HTML report.

## Run
```bash
python examples/demos/03_validation_report/run.py
```

## Outputs
- `outputs/validation_metrics.json`
- `outputs/validation_report.html`
- Sample inputs: `outputs/real_users.csv`, `outputs/synthetic_users.csv`

## Notes
- Uses `ValidationReport.generate` for both JSON and HTML formats.
- Open the HTML in a browser to inspect column-level results and previews.

## Source Code
```python
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from syntho_hive.validation.report_generator import ValidationReport


def make_real_data(num_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    df = pd.DataFrame(
        {
            "user_id": np.arange(1, num_rows + 1),
            "age": rng.integers(18, 70, size=num_rows),
            "region": rng.choice(["NE", "SE", "MW", "W"], size=num_rows, p=[0.25, 0.25, 0.25, 0.25]),
            "monthly_spend": rng.normal(120, 35, size=num_rows).round(2),
            "active": rng.choice([0, 1], size=num_rows, p=[0.35, 0.65]),
        }
    )
    return df


def make_synthetic_variant(real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a synthetic dataset with slight distribution shifts to highlight
    how the validator surfaces differences.
    """
    rng = np.random.default_rng(22)
    synth = real_df.copy()
    synth["monthly_spend"] = (synth["monthly_spend"] * rng.normal(1.05, 0.05, size=len(real_df))).round(2)
    synth["active"] = rng.choice([0, 1], size=len(real_df), p=[0.4, 0.6])
    synth["region"] = rng.choice(["NE", "SE", "MW", "W"], size=len(real_df), p=[0.35, 0.2, 0.25, 0.2])
    return synth


def main():
    parser = argparse.ArgumentParser(description="Generate a validation report for real vs synthetic data.")
    parser.add_argument("--rows", type=int, default=300, help="Rows per dataset to generate.")
    parser.add_argument(
        "--output-dir",
        default="examples/demos/03_validation_report/outputs",
        help="Directory where the report files will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_df = make_real_data(args.rows)
    synthetic_df = make_synthetic_variant(real_df)

    real_df.to_csv(output_dir / "real_users.csv", index=False)
    synthetic_df.to_csv(output_dir / "synthetic_users.csv", index=False)

    report = ValidationReport()
    html_path = output_dir / "validation_report.html"
    json_path = output_dir / "validation_metrics.json"

    report.generate(real_data={"users": real_df}, synth_data={"users": synthetic_df}, output_path=str(html_path))
    report.generate(real_data={"users": real_df}, synth_data={"users": synthetic_df}, output_path=str(json_path))

    print(f"Wrote HTML report to {html_path}")
    print(f"Wrote JSON metrics to {json_path}")


if __name__ == "__main__":
    main()
```
 
