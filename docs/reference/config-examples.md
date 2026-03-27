---
title: Config Examples
---

# Config Examples

## Simple single-table metadata
```python
from syntho_hive.interface.config import Metadata

metadata = Metadata()
metadata.add_table(
    name="customers",
    pk="customer_id",
    fk={},
    parent_context_cols=[],
    constraints={}
)
```

## Relational metadata with constraints
```python
metadata.add_table(
    name="users",
    pk="user_id",
    fk={},
    constraints={
        "age": {"dtype": "int", "min": 18, "max": 99},
    },
    parent_context_cols=[]
)

metadata.add_table(
    name="orders",
    pk="order_id",
    fk={"user_id": "users.user_id"},
    parent_context_cols=["region"],
    constraints={"amount": {"dtype": "float", "min": 0.0}}
)
```

## Privacy config
```python
from syntho_hive.interface.config import PrivacyConfig

privacy = PrivacyConfig(
    enable_differential_privacy=False,
    epsilon=1.0,       # Must be a positive number (validated in v1.4.0)
    pii_strategy="context_aware_faker",
    k_anonymity_threshold=5,
    pii_columns=["email", "phone"]
)
```

## Exception handling
```python
from syntho_hive.exceptions import (
    SchemaError,
    TrainingError,
    GenerationError,
    PrivacyError,
)

try:
    metadata.add_table("orders", pk="order_id", fk={"user_id": "users.user_id"})
except SchemaError as e:
    print(f"Schema definition error: {e}")

try:
    synth.fit(data={"orders": "data/orders.parquet"}, epochs=300)
except TrainingError as e:
    print(f"Training failed: {e}")

try:
    synth.sample(num_rows={"users": 1000}, output_path="/tmp/output")
except GenerationError as e:
    print(f"Generation failed: {e}")
```
