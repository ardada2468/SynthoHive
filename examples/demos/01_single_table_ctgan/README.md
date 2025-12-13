## Single-Table CTGAN (No Spark)

- **What it shows:** Minimal end-to-end CTGAN training on one table, handling mixed numeric/categorical data while ignoring primary keys.
- **Key APIs:** `Metadata.add_table`, `CTGAN.fit`, `CTGAN.sample`.
- **Outputs:** `outputs/synthetic_customers.csv` plus training log.

### Run

```bash
python examples/demos/01_single_table_ctgan/run.py --epochs 5 --rows 200
```

Flags:
- `--epochs`: GAN training epochs (small defaults for quick runs).
- `--rows`: Number of synthetic records to sample after training.
- `--output-dir`: Where to place outputs (default inside this demo).

### What to look at
- Console logs showing CTGAN training progress.
- `synthetic_customers.csv` schema: PK is regenerated, distributions mimic the training data.

