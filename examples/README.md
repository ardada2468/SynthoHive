## Demo Gallery

- `01_single_table_ctgan`: Train a lightweight CTGAN on a single table with mixed numeric/categorical fields and sample new rows without Spark.
- `02_privacy_sanitization`: Detect and sanitize PII using the configurable `PIISanitizer`, including custom rules.
- `03_validation_report`: Compare real vs. synthetic data with `ValidationReport` and emit both HTML and JSON outputs.
- `04_relational_linkage_ctgan`: Reproduce the relational pipeline manually (LinkageModel + conditional CTGAN) to generate parent/child tables with referential integrity—no Spark required.
- `05_transformer_embeddings`: Inspect how `DataTransformer` handles PK/FK removal, one-hot vs. entity embeddings, and reversible transforms.

All demos are self contained and create their own synthetic training data on the fly. Run any demo with `python examples/demos/<demo_name>/run.py`. Output artifacts land under the demo's `outputs/` directory.

