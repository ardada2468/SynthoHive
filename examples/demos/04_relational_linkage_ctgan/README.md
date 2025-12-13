## Relational Pipeline without Spark (Linkage + Conditional CTGAN)

- **What it shows:** The core relational logic—linkage modeling, parent-context conditioning, and referential integrity—run locally with pandas and CTGAN (no Spark required).
- **Key APIs:** `Metadata.add_table`, `LinkageModel.fit/sample_counts`, `CTGAN.fit/sample` with context.
- **Outputs:** `outputs/users.csv` and `outputs/orders.csv` synthetic tables.

### Run

```bash
python examples/demos/04_relational_linkage_ctgan/run.py --parents 150 --epochs 3
```

Flags:
- `--parents`: Number of synthetic parent rows (users) to generate.
- `--epochs`: GAN training epochs (kept small for speed).
- `--output-dir`: Where to place generated tables.

### What to look at
- Console logs showing linkage model training and conditional GAN training.
- Generated `orders` rows should always reference valid `user_id` values from the generated `users` table.

