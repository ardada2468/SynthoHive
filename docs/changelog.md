---
title: Changelog
---

# Changelog

- 1.2.3
  - Fix `TypeError` in `DataTransformer` when applying numeric constraints (min, max, dtype) to categorical/string columns.
  - Added robust type coercion to ensure constraints are applied correctly to transformed data.

- 1.2.2
  - Fix CTGAN embedding cardinality to avoid IndexError when using high-cardinality categorical columns.
  - Databricks example returns in-memory DataFrames and cleans timestamps/nulls for safer Arrow/pandas conversion.
- Initial MkDocs site scaffold with guides, demos, and API reference.
