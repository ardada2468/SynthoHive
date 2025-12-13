---
title: Demo Overview
---

# Demo Overview

The `examples/demos` folder contains runnable scenarios. Ensure dependencies (Spark/Delta for relational pieces) are installed, then run from repo root:

```bash
pip install -e .[docs]
```

Each demo has a `run.py` script and an `outputs/` folder with sample artifacts.

## Available demos
- **01_single_table_ctgan**: Train CTGAN on a single table with mixed data types.
- **02_privacy_sanitization**: Detect and sanitize PII, compare raw vs sanitized outputs.
- **03_validation_report**: Generate validation metrics and HTML report comparing real vs synthetic.
- **04_relational_linkage_ctgan**: Train relational CTGAN with linkage, synthesize users/orders.
- **05_transformer_embeddings**: Demonstrate transformer-based embeddings and recovery. 
