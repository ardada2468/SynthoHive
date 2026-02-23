---
phase: quick/1-fix-arrowstringarray-reshape-notimplemen
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - syntho_hive/core/data/transformer.py
  - syntho_hive/relational/linkage.py
  - syntho_hive/relational/orchestrator.py
autonomous: true
requirements:
  - BUGFIX-ArrowStringArray-reshape

must_haves:
  truths:
    - "All 13 failing tests pass on Python 3.11 with pandas 2.x + PyArrow backend"
    - "No NotImplementedError raised when calling .values.reshape(-1, 1) on ArrowStringArray-backed columns"
    - "Numeric columns in ClusterBasedNormalizer still transform and inverse_transform correctly"
    - "Categorical OneHot and LabelEncoder paths still produce correct shapes"
    - "Relational linkage and orchestrator FK/PK repeat operations work with string ID columns"
  artifacts:
    - path: "syntho_hive/core/data/transformer.py"
      provides: "ArrowStringArray-safe reshape via .to_numpy()"
      contains: "to_numpy"
    - path: "syntho_hive/relational/linkage.py"
      provides: "ArrowStringArray-safe reshape in LinkageModel.fit"
      contains: "to_numpy"
    - path: "syntho_hive/relational/orchestrator.py"
      provides: "ArrowStringArray-safe np.repeat on PK/FK columns"
      contains: "to_numpy"
  key_links:
    - from: "syntho_hive/core/data/transformer.py (DataTransformer.fit, categorical branch)"
      to: "OneHotEncoder.fit()"
      via: "col_data_filled.to_numpy(dtype=str).reshape(-1, 1)"
      pattern: "to_numpy.*reshape"
    - from: "syntho_hive/core/data/transformer.py (DataTransformer.transform, categorical branch)"
      to: "OneHotEncoder.transform()"
      via: "col_data_filled.to_numpy(dtype=str).reshape(-1, 1)"
      pattern: "to_numpy.*reshape"
    - from: "syntho_hive/core/data/transformer.py (ClusterBasedNormalizer.fit/transform)"
      to: "BayesianGaussianMixture.fit()"
      via: "data.to_numpy(dtype=float).reshape(-1, 1)"
      pattern: "to_numpy.*reshape"
    - from: "syntho_hive/relational/orchestrator.py"
      to: "np.repeat()"
      via: "parent_df[col].to_numpy()"
      pattern: "to_numpy"
---

<objective>
Fix NotImplementedError caused by calling .values.reshape(-1, 1) on ArrowStringArray-backed pandas Series in Python 3.11 with pandas 2.x + PyArrow.

Purpose: Pandas 2.0+ uses ArrowStringArray as the default string dtype when PyArrow is installed. ArrowStringArray does not support .reshape() because it is backed by a 1D pyarrow.ChunkedArray, not a numpy array. The .values property on an ArrowStringArray returns an ArrowExtensionArray, not a numpy ndarray. Calling .reshape(-1, 1) on it raises NotImplementedError. The fix is to replace .values.reshape(-1, 1) with .to_numpy(...).reshape(-1, 1) throughout the codebase, since pd.Series.to_numpy() always returns a plain numpy ndarray.

Output: Three patched source files. All 13 currently-failing tests pass.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Replace .values.reshape with .to_numpy().reshape in transformer.py</name>
  <files>syntho_hive/core/data/transformer.py</files>
  <action>
In `DataTransformer.fit()` (categorical OneHot branch, line ~125):
  Change: `values = col_data_filled.values.reshape(-1, 1)`
  To:     `values = col_data_filled.to_numpy(dtype=str).reshape(-1, 1)`

In `DataTransformer.transform()` (categorical branch, line ~173):
  Change: `values = col_data_filled.values.reshape(-1, 1)`
  To:     `values = col_data_filled.to_numpy(dtype=str).reshape(-1, 1)`

In `ClusterBasedNormalizer.fit()` (lines ~323 and ~326):
  Change: `values = data.fillna(self.fill_value).values.reshape(-1, 1)`
  To:     `values = data.fillna(self.fill_value).to_numpy(dtype=float).reshape(-1, 1)`
  Change: `values = data.values.reshape(-1, 1)`
  To:     `values = data.to_numpy(dtype=float).reshape(-1, 1)`

In `ClusterBasedNormalizer.transform()` (lines ~342, ~347, ~350):
  Change: `values_raw = data.values.reshape(-1, 1)`
  To:     `values_raw = data.to_numpy(dtype=float).reshape(-1, 1)`
  Change: `null_indicator = pd.isnull(data).values.astype(float).reshape(-1, 1)`
  To:     `null_indicator = pd.isnull(data).to_numpy(dtype=float).reshape(-1, 1)`
  Change: `values_clean = data.fillna(self.fill_value).values.reshape(-1, 1)`
  To:     `values_clean = data.fillna(self.fill_value).to_numpy(dtype=float).reshape(-1, 1)`

Note: `_prepare_categorical()` already calls `.astype(object)` and `.astype(str)`, so the categorical Series passed to `.to_numpy(dtype=str)` is safe. For numeric Series in ClusterBasedNormalizer, `dtype=float` ensures a plain float64 numpy array regardless of ArrowFloatArray backing.

Do NOT change the `_prepare_categorical` method itself — it is correct as-is.
Do NOT change `pd.isnull(data)` — that call produces a regular bool Series and `.to_numpy(dtype=float)` converts cleanly.
  </action>
  <verify>
Run: `python -m pytest tests/test_transformer.py -x -q 2>&1 | tail -20`
Expected: All transformer tests pass, no NotImplementedError.
  </verify>
  <done>All tests in tests/test_transformer.py pass without NotImplementedError.</done>
</task>

<task type="auto">
  <name>Task 2: Replace .values.reshape and .values in linkage.py and orchestrator.py</name>
  <files>
    syntho_hive/relational/linkage.py
    syntho_hive/relational/orchestrator.py
  </files>
  <action>
In `syntho_hive/relational/linkage.py`, `LinkageModel.fit()` (line ~55):
  Change: `X = count_df["child_count"].values.reshape(-1, 1)`
  To:     `X = count_df["child_count"].to_numpy(dtype=float).reshape(-1, 1)`
  Reason: `count_df["child_count"]` is produced by a merge + fillna(0), could be float64 or Arrow-backed. `.to_numpy(dtype=float)` guarantees a plain ndarray for GaussianMixture.

In `syntho_hive/relational/orchestrator.py`, relational generation block (~lines 179, 185):
  Change: `parent_ids_repeated = np.repeat(parent_df[driver_parent_pk].values, counts)`
  To:     `parent_ids_repeated = np.repeat(parent_df[driver_parent_pk].to_numpy(), counts)`

  Change (inside the `if context_cols:` loop):
    `context_repeated_vals[col] = np.repeat(parent_df[col].values, counts)`
  To:
    `context_repeated_vals[col] = np.repeat(parent_df[col].to_numpy(), counts)`

  Reason: PK columns are often string IDs. When pandas uses ArrowStringArray for strings, `.values` returns an ArrowExtensionArray that `np.repeat` cannot handle. `.to_numpy()` with no dtype arg lets pandas choose the best numpy representation (object for strings, float64 for numerics).

Also check line ~211 in orchestrator.py where `valid_pks = p_df[p_pk].values` appears:
  Change: `valid_pks = p_df[p_pk].values`
  To:     `valid_pks = p_df[p_pk].to_numpy()`
  (This feeds into np.random.choice or similar — same fix pattern.)
  </action>
  <verify>
Run the full failing test suite:
  `python -m pytest tests/test_e2e_single_table.py tests/test_models.py tests/test_null_handling.py tests/test_seed_regression.py tests/test_serialization.py tests/test_transformer.py tests/test_constraint_violation.py tests/core/test_embeddings.py -x -q 2>&1 | tail -30`
Expected: 0 errors, 0 NotImplementedError.
  </verify>
  <done>
All 13 previously-failing tests pass. No NotImplementedError from ArrowStringArray reshape.
Full suite run: `python -m pytest tests/ -q 2>&1 | tail -5` shows 0 errors related to reshape.
  </done>
</task>

</tasks>

<verification>
After both tasks complete:
1. `python -m pytest tests/ -q 2>&1 | tail -10` — zero failures from NotImplementedError.
2. Spot-check synthetic data generation still produces correct output shapes:
   `python -c "import pandas as pd; pd.options.future.infer_string = True; from tests.test_transformer import *"` or run the test suite with PANDAS_FUTURE_INFER_STRING=1 if available.
</verification>

<success_criteria>
- Zero instances of `NotImplementedError: ... does not support reshape` in test output.
- All tests in the 8 affected test files pass on Python 3.11.
- No regressions in previously-passing tests.
- Every `.values.reshape(-1, 1)` in transformer.py, linkage.py replaced with `.to_numpy(...).reshape(-1, 1)`.
- Every `.values` fed to `np.repeat()` in orchestrator.py replaced with `.to_numpy()`.
</success_criteria>

<output>
After completion, create `.planning/quick/1-fix-arrowstringarray-reshape-notimplemen/1-SUMMARY.md`
</output>
