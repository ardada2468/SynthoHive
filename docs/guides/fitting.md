---
title: Fitting Models
---

# Fitting Models

## Inputs
- `Metadata` with tables, PK/FK, constraints, and parent context columns.
- Optional `PrivacyConfig` for PII handling.
- SparkSession (required for full relational flows in current prototype).

## Minimal relational fit
```python
from syntho_hive.interface.config import Metadata, PrivacyConfig
from syntho_hive.interface.synthesizer import Synthesizer

metadata = Metadata()
metadata.add_table(
    name="users",
    pk="user_id",
    fk={},  # root
    parent_context_cols=[],
)
metadata.add_table(
    name="orders",
    pk="order_id",
    fk={"user_id": "users.user_id"},
    parent_context_cols=["region"],
)

privacy = PrivacyConfig()
synth = Synthesizer(metadata=metadata, privacy_config=privacy, spark_session=spark)

synth.fit(
    database="my_db",
    sampling_strategy="relational_stratified",
    sample_size=5_000_000,
    epochs=300,
    batch_size=500,
    embedding_threshold=50,
)
```

### What happens during fit
1. Each table is read via `SparkIO`.
2. `DataTransformer.fit` profiles columns (continuous VGM, categorical OHE/embedding).
3. `LinkageModel.fit` learns child counts (per driver parent).
4. `CTGAN.fit` trains generator/discriminator; context columns are merged for conditional learning.

### Tips
- Ensure PK/FK are correctly set in `Metadata` so transformers exclude them from modeling.
- For high-cardinality categoricals, tune `embedding_threshold`.
- Start with smaller `epochs`/`batch_size` to validate the flow, then scale up. 
