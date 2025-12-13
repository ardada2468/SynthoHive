---
title: Transformer Embeddings Demo
---

# Transformer Embeddings Demo

Path: `examples/demos/05_transformer_embeddings`

## Goal
Show transformer-based embeddings and recovery of transformed features.

## Run
```bash
python examples/demos/05_transformer_embeddings/run.py
```

## Outputs
- `outputs/transformed.npy`
- `outputs/recovered.csv`

## Notes
- Highlights `DataTransformer` behavior with embedding thresholds.
- Useful for inspecting how categorical embeddings are produced and inverted.

## Source Code
```python
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from syntho_hive.core.data.transformer import DataTransformer
from syntho_hive.interface.config import Metadata


def make_data(num_rows: int, high_card: int) -> pd.DataFrame:
    rng = np.random.default_rng(100)
    df = pd.DataFrame(
        {
            "product_id": np.arange(1, num_rows + 1),
            "category": rng.choice(["electronics", "apparel", "home", "toys"], size=num_rows, p=[0.35, 0.25, 0.25, 0.15]),
            "brand": rng.choice([f"brand_{i}" for i in range(high_card)], size=num_rows),
            "price": rng.normal(80, 25, size=num_rows).clip(5, 300).round(2),
            "inventory": rng.integers(0, 500, size=num_rows),
        }
    )
    return df


def build_metadata(embedding_threshold: int) -> Metadata:
    meta = Metadata()
    meta.add_table(
        name="products",
        pk="product_id",
        high_cardinality_cols=["brand"],
    )
    # DataTransformer reads embedding_threshold from the class initialization
    # rather than the metadata field directly, but we keep metadata accurate
    # for PK/FK stripping.
    return meta


def main():
    parser = argparse.ArgumentParser(description="Inspect DataTransformer encoding behavior.")
    parser.add_argument("--rows", type=int, default=120, help="Number of rows to fabricate.")
    parser.add_argument(
        "--embedding-threshold",
        type=int,
        default=20,
        help="Switch to embeddings when categories exceed this threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/demos/05_transformer_embeddings/outputs",
        help="Directory to write transformed artifacts.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = make_data(args.rows, high_card=args.embedding_threshold + 5)
    meta = build_metadata(args.embedding_threshold)

    transformer = DataTransformer(metadata=meta, embedding_threshold=args.embedding_threshold)
    print("Fitting transformer (drops PK/FK automatically)...")
    transformer.fit(df, table_name="products")
    transformed = transformer.transform(df)

    print(f"Original shape: {df.shape}")
    print(f"Transformed matrix shape: {transformed.shape}")
    print(f"First 5 rows (dense matrix):\n{transformed[:5]}")

    recovered = transformer.inverse_transform(transformed)
    recovered.insert(0, "product_id", range(1, len(recovered) + 1))

    np.save(output_dir / "transformed.npy", transformed)
    recovered.to_csv(output_dir / "recovered.csv", index=False)

    print(f"Wrote dense matrix to {output_dir/'transformed.npy'}")
    print(f"Wrote recovered table to {output_dir/'recovered.csv'}")
    print(recovered.head())


if __name__ == "__main__":
    main()
```
 
