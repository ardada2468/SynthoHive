---
title: Demo Overview
---

# Demo Overview

The `examples/demos` folder contains runnable scenarios that demonstrate SynthoHive's core capabilities. Each demo is self-contained and generates its own sample data.

## Setup

Install the package from the repository root:

```bash
pip install -e .
```

Each demo has a `run.py` script and an `outputs/` folder with sample artifacts.

## Available Demos

| Demo | Description | Spark Required? |
| :--- | :--- | :--- |
| [01 - Single Table CTGAN](01_single_table_ctgan.md) | Train CTGAN on a single table with mixed data types. | No |
| [02 - Privacy Sanitization](02_privacy_sanitization.md) | Detect and sanitize PII, compare raw vs sanitized outputs. | No |
| [03 - Validation Report](03_validation_report.md) | Generate validation metrics and HTML/JSON reports comparing real vs synthetic. | No |
| [04 - Relational Linkage CTGAN](04_relational_linkage_ctgan.md) | Train relational CTGAN with linkage, synthesize users/orders with FK integrity. | No |
| [05 - Transformer Embeddings](05_transformer_embeddings.md) | Demonstrate DataTransformer encoding behavior and round-trip recovery. | No |

## Running a Demo

```bash
# From the repository root
python examples/demos/01_single_table_ctgan/run.py
python examples/demos/02_privacy_sanitization/run.py
python examples/demos/03_validation_report/run.py
python examples/demos/04_relational_linkage_ctgan/run.py
python examples/demos/05_transformer_embeddings/run.py
```

## Jupyter Notebooks

Interactive notebook versions of each demo are available in the `notebooks/` directory. These provide step-by-step walkthroughs with inline output.
