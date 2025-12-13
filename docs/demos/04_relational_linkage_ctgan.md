---
title: Relational Linkage CTGAN Demo
---

# Relational Linkage CTGAN Demo

Path: `examples/demos/04_relational_linkage_ctgan`

## Goal
Train relational CTGAN on users/orders with parent-child linkage and generate synthetic tables preserving FKs.

## Run
```bash
python examples/demos/04_relational_linkage_ctgan/run.py
```

## Outputs
- `outputs/users.csv`
- `outputs/orders.csv`

## Notes
- Demonstrates `StagedOrchestrator`, `LinkageModel`, and conditional CTGAN.
- Requires Spark/Delta for IO; ensure SparkSession is available. 
