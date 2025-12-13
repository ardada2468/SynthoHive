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
