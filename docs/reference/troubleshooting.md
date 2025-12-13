---
title: Troubleshooting
---

# Troubleshooting

- **Spark not found**: Ensure `pyspark` is installed and `SPARK_HOME` is set. For local-only runs, some demos may be limited.
- **Delta support**: Install `delta-spark` and use Spark 3.2+.
- **GPU vs CPU**: CTGAN runs on CPU by default; set `device="cuda"` when available.
- **High-cardinality categoricals**: Increase `embedding_threshold` to use embeddings instead of OHE.
- **Validation failures**: Inspect KS/TVD results; large TVD often means categorical imbalance—check sampling strategy or increase training epochs.
- **Doc build errors**: Run `pip install -e .[docs]`; ensure `mkdocs` and `mkdocstrings` are installed. 
