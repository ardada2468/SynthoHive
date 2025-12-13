---
title: Single Table CTGAN Demo
---

# Single Table CTGAN Demo

Path: `examples/demos/01_single_table_ctgan`

## Goal
Train a CTGAN on a single table with mixed types (numerical, categorical) and generate synthetic data.

## Run
```bash
python examples/demos/01_single_table_ctgan/run.py
```

## Source Code
```python
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from syntho_hive.interface.config import Metadata
from syntho_hive.core.models.ctgan import CTGAN


def build_training_data(num_rows: int = 600) -> pd.DataFrame:
    """Create a small, mixed-type table to keep the demo self contained."""
    rng = np.random.default_rng(42)
    cities = ["NYC", "SF", "SEA", "CHI", "DAL", "MIA"]
    channels = ["web", "retail", "partner"]

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, num_rows + 1),
            "age": rng.integers(18, 75, size=num_rows),
            "annual_spend": rng.normal(55000, 15000, size=num_rows).clip(5000, 150000).round(2),
            "city": rng.choice(cities, p=[0.28, 0.22, 0.18, 0.14, 0.1, 0.08], size=num_rows),
            "signup_channel": rng.choice(channels, p=[0.6, 0.3, 0.1], size=num_rows),
            "loyalty_score": rng.uniform(0, 1, size=num_rows).round(4),
        }
    )
    return df


def configure_metadata() -> Metadata:
    """Tell SynthoHive which columns are structural (PK) vs. modeled."""
    meta = Metadata()
    meta.add_table(
        name="customers",
        pk="customer_id",
        pii_cols=["customer_id"],
        high_cardinality_cols=["city"],
    )
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train a CTGAN on a single table.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for the GAN.")
    parser.add_argument("--rows", type=int, default=200, help="Number of synthetic rows to generate.")
    parser.add_argument(
        "--output-dir",
        default="examples/demos/01_single_table_ctgan/outputs",
        help="Directory where outputs will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building training data...")
    train_df = build_training_data()
    meta = configure_metadata()

    print("Fitting CTGAN (small config for demo speed)...")
    model = CTGAN(
        meta,
        batch_size=128,
        epochs=args.epochs,
        embedding_dim=64,
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
        device="cpu",
    )
    model.fit(train_df, table_name="customers")

    print(f"Sampling {args.rows} synthetic rows...")
    synthetic_df = model.sample(args.rows)
    synthetic_df.insert(0, "customer_id", range(1, len(synthetic_df) + 1))

    output_path = output_dir / "synthetic_customers.csv"
    synthetic_df.to_csv(output_path, index=False)
    print(f"Wrote synthetic data to {output_path}")
    print(synthetic_df.head())


if __name__ == "__main__":
    main()
```

## Outputs
- `outputs/synthetic_customers.csv`

## Notes
- Demonstrates basic usage of `CTGAN` and `Metadata`.
