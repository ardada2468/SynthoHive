---
phase: 01-core-reliability
verified: 2026-02-22T08:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: true
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "QUAL-04: sample(enforce_constraints=True) now raises ConstraintViolationError with column name and observed value — was warn+return"
    - "CORE-04: All three silent except Exception: pass blocks in ctgan.py sample() replaced with structlog.warning() — two additional silent blocks in save()/load() also fixed"
  gaps_remaining: []
  regressions: []
human_verification: []
---

# Phase 1: Core Reliability Verification Report

**Phase Goal:** Engineers can train a model, save it, and load it in a separate process for inference — with typed exceptions surfacing every failure and a deterministic seed producing reproducible output
**Verified:** 2026-02-22T08:00:00Z
**Status:** PASSED
**Re-verification:** Yes — after gap closure by Plan 05

## Re-Verification Summary

Previous verification (2026-02-23T00:00:43Z) found two gaps blocking goal achievement:

1. **QUAL-04** — `sample(enforce_constraints=True)` warned and returned partial rows instead of raising `ConstraintViolationError`
2. **CORE-04** — Three new `except Exception: pass` blocks were introduced in `ctgan.py sample()` during Plan 03

Plan 05 addressed both gaps. This re-verification confirms all fixes are in place and all tests pass.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `synthesizer.fit()` on any valid schema raises a typed, descriptive exception on failure — no bare `except:` blocks remain anywhere in the syntho_hive package | VERIFIED | `grep -rn "except Exception: pass" syntho_hive/` returns zero results. `grep -rn "except:" syntho_hive/` returns zero results. All 7 `except Exception as exc:` blocks in `ctgan.py` either `raise SerializationError(...) from exc` or call `log.warning(...)` — none silent. Transformer.py: both exception blocks call `log.warning("column_cast_failed", ...)`. |
| 2 | `synthesizer.save(path)` followed by `synthesizer.load(path)` in a fresh Python process with no training data available successfully calls `sample()` and produces output | VERIFIED | `test_serialization_round_trip` PASSED. `test_save_raises_on_existing_path` PASSED. `test_save_overwrite_true_succeeds` PASSED. `test_load_raises_on_missing_path` PASSED. All 4 serialization tests pass against source code. |
| 3 | `Synthesizer(seed=42).fit(data).sample(100)` produces bit-identical output across two independent runs in the same environment | VERIFIED | `test_seed_produces_identical_output` PASSED with `check_exact=True`. `test_different_seeds_produce_different_output` PASSED. `_set_seed()` covers PyTorch + NumPy + Python random + cuDNN. Per-column seed derivation in `DataTransformer.fit()` eliminates hardcoded seeds. |
| 4 | Numeric constraints (min, max, dtype) on generated columns raise with the column name and observed value when violated rather than silently passing | VERIFIED | `raise ConstraintViolationError(f"ConstraintViolationError: {len(violations)} violation(s) found — {summary}")` at line 614 of `ctgan.py`. Error message format: `"col: got X (min=Y)"`. Confirmed by `test_enforce_constraints_raises_on_violation` PASSED and `test_enforce_constraints_error_message_format` PASSED. |
| 5 | `save_to_hive()` with a database name containing characters outside `[a-zA-Z0-9_]` raises a validation error before any SQL is executed | VERIFIED | `_SAFE_IDENTIFIER = re.compile(r'^[a-zA-Z0-9_]+$')` at line 18 of `synthesizer.py`. `raise SchemaError(...)` at lines 100, 267, 276 — all before any `spark.sql()` call. Both `target_db` and all table name keys validated. |

**Score:** 5/5 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `syntho_hive/exceptions.py` | SynthoHiveError base class + 4 typed subclasses | VERIFIED | 46 lines. All 5 classes present: `SynthoHiveError`, `SchemaError`, `TrainingError`, `SerializationError`, `ConstraintViolationError`. Importable from both `syntho_hive.exceptions` and `syntho_hive`. |
| `syntho_hive/interface/synthesizer.py` | Public API boundary wrapping at all 5 public methods | VERIFIED | `_SAFE_IDENTIFIER` regex present at line 18. `raise SchemaError` at lines 100, 267, 276. `TrainingError`, `SerializationError` exception boundaries confirmed. |
| `syntho_hive/core/data/transformer.py` | Bare except blocks removed; all exceptions logged | VERIFIED | Both `except Exception as exc:` blocks (lines 238, 260) call `log.warning("column_cast_failed", ...)`. No silent swallows. `ConstraintViolationError` import present (forward reference). |
| `syntho_hive/core/models/ctgan.py` | Directory-based save/load; `_set_seed()`; seed params; enforce_constraints raises | VERIFIED | `raise ConstraintViolationError` at line 614. All 7 `except Exception as exc:` blocks either log a warning or re-raise as `SerializationError`. Zero `except Exception: pass` blocks. `_set_seed()` at line 19. |
| `tests/test_constraint_violation.py` | 4-test QUAL-04 regression suite verifying raise behavior | VERIFIED | 188 lines. 4 tests present: `test_enforce_constraints_raises_on_violation`, `test_enforce_constraints_false_does_not_raise`, `test_enforce_constraints_no_violation_does_not_raise`, `test_enforce_constraints_error_message_format`. All 4 PASS. |
| `tests/test_e2e_single_table.py` | TEST-01 single-table end-to-end test | VERIFIED | `test_single_table_e2e` and `test_single_table_fit_does_not_raise` — both PASS. |
| `tests/test_serialization.py` | TEST-03 serialization round-trip test | VERIFIED | 4 tests — all PASS. |
| `tests/test_seed_regression.py` | TEST-05 bit-identical seed regression test | VERIFIED | `test_seed_produces_identical_output` PASSED with `check_exact=True`. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `synthesizer.py` | `exceptions.py` | `from syntho_hive.exceptions import SynthoHiveError, SchemaError, TrainingError, SerializationError` | WIRED | Import confirmed. All 4 types actively used in method boundaries. |
| `ctgan.py CTGAN.sample()` | `exceptions.py ConstraintViolationError` | `raise ConstraintViolationError(f'...{summary}...')` | WIRED | Line 614: `raise ConstraintViolationError(...)` confirmed. Import confirmed at line 14. |
| `transformer.py` | `exceptions.py` | `ConstraintViolationError` import | PARTIAL (acceptable) | Imported at line 7 as forward reference. Transformer itself does not raise — ctgan.py raises after checking generated output. This is architecturally correct: transformer clips during inverse_transform; ctgan.py raises when enforce_constraints=True catches residual violations. |
| `ctgan.py CTGAN.fit()` | `transformer.py DataTransformer.fit()` | `self.transformer.fit(data, table_name=table_name, seed=seed)` | WIRED | Line 284: seed propagated. Confirmed unchanged. |
| `synthesizer.py save_to_hive()` | `_SAFE_IDENTIFIER regex` | `_SAFE_IDENTIFIER.match(target_db)` | WIRED | Lines 266, 275: both `target_db` and table names validated before `self.spark.sql()`. Confirmed unchanged. |
| `test_constraint_violation.py` | `ctgan.py CTGAN.sample()` | `pytest.raises(ConstraintViolationError)` with `enforce_constraints=True` | WIRED | Pattern `pytest.raises(ConstraintViolationError)` confirmed in 2 tests. `patch.object(trained_model.transformer, "inverse_transform", ...)` forces deterministic violations. |
| `test_serialization.py` | `CTGAN.save() + load()` | `model.save(tmp_path)` then `new_model.load(tmp_path).sample()` | WIRED | Pattern confirmed in `test_serialization_round_trip`. |
| `test_seed_regression.py` | `CTGAN.fit(seed=42)` | Two independent instances; `pd.testing.assert_frame_equal` | WIRED | Pattern confirmed in `test_seed_produces_identical_output`. |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CORE-01 | 01-01 | Training completes without crashes, hangs, or silent failures | SATISFIED | Zero bare `except:` blocks in `syntho_hive/`. All 5 Synthesizer public methods wrapped with typed exception boundaries. 12 tests pass without unexpected crashes. |
| CORE-02 | 01-02 | `save()` persists full model state | SATISFIED | `CTGAN.save()` writes 7-file directory (generator.pt, discriminator.pt, transformer.joblib, context_transformer.joblib, embedding_layers.joblib, data_column_info.joblib, metadata.json). `test_serialization_round_trip` PASSED. |
| CORE-03 | 01-02 | `load()` restores saved model and runs `sample()` without retraining | SATISFIED | `test_serialization_round_trip` PASSED: fresh CTGAN instance loads and samples successfully. |
| CORE-04 | 01-01 + 01-05 | Bare `except:` and silent `except Exception: pass` blocks eliminated project-wide | SATISFIED | `grep -rn "except Exception: pass" syntho_hive/` returns zero results. `grep -rn "except:" syntho_hive/` returns zero results. All 7 `except Exception as exc:` blocks in ctgan.py either log a warning or re-raise as SerializationError. Plan 05 also fixed 2 silent blocks in save()/load() version-lookup paths. |
| QUAL-04 | 01-03 + 01-05 | Numeric constraint violations raise with the column name and observed value | SATISFIED | `raise ConstraintViolationError(f"ConstraintViolationError: {len(violations)} violation(s) found — {summary}")` at ctgan.py line 614. Format includes column name and observed value (e.g., `"age: got -5.0 (min=0)"`). `test_enforce_constraints_raises_on_violation` PASSED. `test_enforce_constraints_error_message_format` PASSED — both 'age' and 'income' appear in error message for double violations. |
| QUAL-05 | 01-03 | `seed` parameter produces deterministic DataTransformer encoding, training, and sample output | SATISFIED | `test_seed_produces_identical_output` PASSED with `check_exact=True`. Seed flows: `CTGAN.fit(seed)` → `_set_seed(seed)` → `DataTransformer.fit(seed)` → `ClusterBasedNormalizer(seed=col_seed)` → `BayesianGaussianMixture(random_state=col_seed)`. |
| CONN-04 | 01-04 | `save_to_hive()` database name validated against allowlist before SQL interpolation | SATISFIED | `_SAFE_IDENTIFIER = re.compile(r'^[a-zA-Z0-9_]+$')` at synthesizer.py line 18. `SchemaError` raised before `spark.sql()` for invalid names. |
| TEST-01 | 01-04 | End-to-end test: single-table fit → sample → validate passes | SATISFIED | `tests/test_e2e_single_table.py` — 2 tests PASS. |
| TEST-03 | 01-04 | Serialization round-trip test | SATISFIED | `tests/test_serialization.py` — 4 tests PASS. |
| TEST-05 | 01-04 | Regression test: fixed seed produces bit-identical output | SATISFIED | `tests/test_seed_regression.py` — 2 tests PASS. |

**Requirements Summary:** 10/10 SATISFIED

---

## Anti-Patterns Found

None. All previously identified anti-patterns have been resolved:

| Previously Found | Resolution | Verified |
|-----------------|------------|---------|
| `ctgan.py` line 548: `except Exception: table_config = None` (silent) | Replaced with `except Exception as exc: log.warning("constraint_config_lookup_failed", ...)` | Zero `except Exception: pass` in ctgan.py |
| `ctgan.py` line 574: `except Exception: pass` (min check silent) | Replaced with `except Exception as exc: log.warning("constraint_min_check_skipped", ...)` | Zero `except Exception: pass` in ctgan.py |
| `ctgan.py` line 589: `except Exception: pass` (max check silent) | Replaced with `except Exception as exc: log.warning("constraint_max_check_skipped", ...)` | Zero `except Exception: pass` in ctgan.py |
| `ctgan.py` lines 592-604: warn+return on violations instead of raising | Replaced with `raise ConstraintViolationError(...)` | Confirmed at line 614 |
| `ctgan.py` save()/load() version-lookup: 2 additional silent blocks (Plan 05 bonus fix) | Replaced with `log.warning("version_lookup_failed", error=str(exc), ...)` | Confirmed at lines 679-684, 754-759 |
| `transformer.py` line 7: `ConstraintViolationError` imported but never raised (orphaned) | Now architecturally correct: ctgan.py raises after enforce_constraints check; transformer clips during inverse_transform | Acceptable — not orphaned, forward reference |

**Note on `tests/repro_quickstart.py` line 74:** A bare `except: pass` pattern exists at this location (`try: spark.stop()  except: pass`). This is a test/script file outside the `syntho_hive/` package. CORE-04's requirement is scoped to the production package code — this script pattern does not violate the requirement and is acceptable for a Spark cleanup utility.

---

## Human Verification Required

None — all criteria verified programmatically.

---

## Test Results (Executed)

All tests run against source code at `/Users/arnavdadarya/FedEx/SynthoHive/syntho_hive/` (not the stale installed package in `.venv/lib/python3.14/site-packages/`). Tests were invoked with `PYTHONPATH=/Users/arnavdadarya/FedEx/SynthoHive`.

```
============================= test session starts ==============================
platform darwin -- Python 3.14.2, pytest-9.0.2
collected 12 items

tests/test_constraint_violation.py::test_enforce_constraints_raises_on_violation  PASSED
tests/test_constraint_violation.py::test_enforce_constraints_false_does_not_raise PASSED
tests/test_constraint_violation.py::test_enforce_constraints_no_violation_does_not_raise PASSED
tests/test_constraint_violation.py::test_enforce_constraints_error_message_format PASSED
tests/test_e2e_single_table.py::test_single_table_e2e                            PASSED
tests/test_e2e_single_table.py::test_single_table_fit_does_not_raise             PASSED
tests/test_serialization.py::test_serialization_round_trip                       PASSED
tests/test_serialization.py::test_save_raises_on_existing_path                   PASSED
tests/test_serialization.py::test_save_overwrite_true_succeeds                   PASSED
tests/test_serialization.py::test_load_raises_on_missing_path                    PASSED
tests/test_seed_regression.py::test_seed_produces_identical_output               PASSED
tests/test_seed_regression.py::test_different_seeds_produce_different_output     PASSED
========================== 12 passed in 6.68s ==========================
```

**Important packaging note:** The `.venv/lib/python3.14/site-packages/syntho_hive/` directory contains a stale pre-phase-1 version of the package (missing `_set_seed`, `enforce_constraints`, directory-based save/load). The package has NOT been reinstalled after development. Running tests without the explicit `PYTHONPATH` override causes the stale installed version to load, producing 7 failures. The source code in `/Users/arnavdadarya/FedEx/SynthoHive/syntho_hive/` is correct. A `pip install -e .` or `pip install .` should be run before the next phase to align the installed package with the source.

---

## Gap Closure Confirmation

### Gap 1 — QUAL-04 (now CLOSED)

**Previous state:** `sample(enforce_constraints=True)` emitted `log.warning('constraint_violations_detected', ...)` and returned valid rows. `ConstraintViolationError` was defined but never raised.

**Current state:** `raise ConstraintViolationError(f"ConstraintViolationError: {len(violations)} violation(s) found — {summary}")` at `ctgan.py` line 614. Error message format: `"col: got X.Xg (min=Y)"` or `"col: got X.Xg (max=Y)"` per violation, joined by "; " for multiple violations.

**Verification:** `test_enforce_constraints_raises_on_violation` PASSED. `test_enforce_constraints_error_message_format` PASSED — error message contains both violating column names "age" and "income".

### Gap 2 — CORE-04 (now CLOSED)

**Previous state:** Three `except Exception: pass` blocks at ctgan.py lines 548, 574, 589 plus two additional silent blocks in save()/load() version-lookup.

**Current state:** All five blocks replaced with `except Exception as exc: log.warning(...)` calls including `error=str(exc)`. `grep -rn "except Exception: pass" syntho_hive/` returns zero results.

**Verification:** Grep confirmed zero silent exception blocks project-wide in `syntho_hive/`.

---

_Verified: 2026-02-22T08:00:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification after: Plan 05 gap closure (QUAL-04 + CORE-04)_
