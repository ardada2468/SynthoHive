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
    epsilon=1.0,
    pii_strategy="context_aware_faker",
    k_anonymity_threshold=5,
    pii_columns=["email", "phone"]
)
```
