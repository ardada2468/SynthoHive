## Transformer & Embeddings Inspection

- **What it shows:** How `DataTransformer` drops PK/FK columns, applies one-hot vs. entity embeddings based on cardinality, and performs reversible transforms.
- **Key APIs:** `DataTransformer.fit/transform/inverse_transform`.
- **Outputs:** Printed shapes and CSVs under `outputs/` to visualize encoded data.

### Run

```bash
python examples/demos/05_transformer_embeddings/run.py
```

Flags:
- `--rows`: Number of rows to fabricate for the inspection (default 120).
- `--embedding-threshold`: Cardinality threshold for switching from one-hot to embeddings.

### What to look at
- Console logs showing detected column types and the transformed matrix shape.
- `transformed.npy` (compressed) and `recovered.csv` confirming round-trip fidelity.

