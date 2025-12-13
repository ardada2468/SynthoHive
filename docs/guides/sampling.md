---
title: Sampling & Relational
---

# Sampling & Relational Generation

## Root tables
- `CTGAN.sample` generates `num_rows` rows.
- PKs are assigned sequentially after generation.

## Child tables
1. `LinkageModel.sample_counts` predicts child counts per parent.
2. Parent PKs are repeated based on sampled counts to form FKs.
3. Optional parent context columns are repeated for conditional sampling.
4. Secondary FKs (multiple parents) are filled by random selection from respective parent PKs.

## Code sketch (after fit)
```python
num_rows = {"users": 1000, "orders": 4000}
output_paths = synth.sample(num_rows=num_rows, output_format="delta")
# Returns mapping of table -> output path
```

## Stratified sampling from source
- `RelationalSampler` can downsample roots and cascade to children while keeping distribution via semijoins.
- Configure `sample_size` and optional `stratify_by` to balance classes. 
