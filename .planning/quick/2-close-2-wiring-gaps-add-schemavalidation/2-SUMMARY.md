---
phase: quick
plan: 2
subsystem: public-api, synthesizer
tags: [wiring, export, schema-validation, fix]
dependency_graph:
  requires: []
  provides: [SchemaValidationError public export, validate_schema wiring in fit()]
  affects: [syntho_hive/__init__.py, syntho_hive/interface/synthesizer.py]
tech_stack:
  added: []
  patterns: [exception hierarchy, pre-flight validation gate]
key_files:
  created: []
  modified:
    - syntho_hive/__init__.py
    - syntho_hive/interface/synthesizer.py
decisions:
  - validate_schema called before print/data-loading so schema errors surface before any I/O begins
  - validate=False default unchanged — zero regression for existing callers
metrics:
  duration: 8 min
  completed: 2026-02-22
---

# Quick Task 2: Close Two Wiring Gaps (SchemaValidationError export + validate_schema in fit())

**One-liner:** Exported SchemaValidationError from the package public API and wired the dead validate=True code path in Synthesizer.fit() to actually call self.metadata.validate_schema().

## Tasks Completed

| Task | Name | Commit | Files Modified |
|------|------|--------|----------------|
| 1 | Export SchemaValidationError from syntho_hive/__init__.py | 51aa75e | syntho_hive/__init__.py |
| 2 | Wire validate_schema into Synthesizer.fit() | 7a699f5 | syntho_hive/interface/synthesizer.py |

## What Was Done

### Task 1: Export SchemaValidationError

Added `SchemaValidationError` to the exceptions import block in `syntho_hive/__init__.py`, placed directly after `SchemaError` to make the subclass hierarchy visible in import ordering. `from syntho_hive import SchemaValidationError` now succeeds without ImportError.

### Task 2: Wire validate_schema into fit()

Added the validate gate inside `Synthesizer.fit()` immediately after the `sample_size` guard and before the print statement:

```python
if validate:
    self.metadata.validate_schema()
```

This runs structural FK checks before any data is loaded or paths are resolved. `SchemaValidationError` (a `SynthoHiveError` subclass) propagates unchanged through the existing `except SynthoHiveError: raise` block. The no-arg call to `validate_schema()` performs structural FK checks only — table existence and FK format — which is correct for a pre-flight check before training.

## Verification Results

1. `from syntho_hive import SchemaValidationError; print(SchemaValidationError.__bases__)` outputs `(<class 'syntho_hive.exceptions.SchemaError'>,)` — confirms SchemaError parentage.

2. `grep "SchemaValidationError" syntho_hive/__init__.py` matches.

3. `grep -n "validate_schema" syntho_hive/interface/synthesizer.py` shows line 92 inside fit().

4. `pytest syntho_hive/tests/test_interface.py syntho_hive/tests/test_relational.py` — all 7 relational tests pass; `test_metadata_validation` and `test_metadata_invalid_fk_format` pass. 4 pre-existing failures in test_interface.py are unrelated to this plan's changes (confirmed by baseline check before any edits).

## Deviations from Plan

### Pre-existing Test Failures (Out of Scope)

4 tests in `test_interface.py` were already failing before any changes in this task:
- `test_synthesizer_fit_requires_spark` — ValueError wrapping mismatch
- `test_synthesizer_sample_requires_spark` — same wrapping issue
- `test_synthesizer_fit_call` — missing epochs/batch_size kwargs in assert_called_with
- `test_synthesizer_sample_call` — generate() called with keyword args, test expects positional

Baseline check confirmed these failures existed before task execution. Out of scope per deviation scope boundary. Logged for deferred fix.

## Self-Check: PASSED

- syntho_hive/__init__.py - FOUND (SchemaValidationError in import block)
- syntho_hive/interface/synthesizer.py - FOUND (validate_schema at line 92)
- Commit 51aa75e - FOUND
- Commit 7a699f5 - FOUND
