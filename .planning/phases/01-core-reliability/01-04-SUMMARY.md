---
phase: 01-core-reliability
plan: "04"
subsystem: testing
tags: [sql-injection, security, ctgan, pytest, serialization, seed, regression-tests, allowlist]

# Dependency graph
requires:
  - phase: 01-core-reliability
    plan: "02"
    provides: "CTGAN.save()/load() directory-based serialization (7-file checkpoint bundle)"
  - phase: 01-core-reliability
    plan: "03"
    provides: "CTGAN.fit(seed=N) and CTGAN.sample(seed=N) deterministic seeding"
provides:
  - "SQL injection prevention in save_to_hive() via _SAFE_IDENTIFIER allowlist regex"
  - "TEST-01: test_e2e_single_table.py — fit+sample end-to-end on 200-row dataset"
  - "TEST-03: test_serialization.py — round-trip save/load/sample, overwrite guard, missing path guard"
  - "TEST-05: test_seed_regression.py — bit-identical output with same seeds"
affects: [02-any, 03-any, ci-gating]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Allowlist regex pattern: _SAFE_IDENTIFIER = re.compile(r'^[a-zA-Z0-9_]+$') at module level — validate before SQL interpolation"
    - "SQL injection prevention: validate user-supplied identifiers before any spark.sql() call; raise SchemaError immediately"
    - "Regression test pattern: two independent CTGAN.fit(seed=42).sample(100, seed=7) runs verified via pd.testing.assert_frame_equal(check_exact=True)"

key-files:
  created:
    - tests/test_e2e_single_table.py
    - tests/test_serialization.py
    - tests/test_seed_regression.py
  modified:
    - syntho_hive/interface/synthesizer.py

key-decisions:
  - "Allowlist (allowlist-only) approach for SQL identifier validation — any character not in [a-zA-Z0-9_] is rejected before Spark is touched"
  - "Validate both database name (target_db) and all table names (synthetic_data keys) in save_to_hive() — both are interpolated into SQL strings"
  - "Empty string fails the allowlist regex (^ requires at least one character after match) — zero-length identifiers are invalid SQL"
  - "Fixtures defined locally in each test file (no shared conftest) — no pre-existing conftest.py in tests/"

patterns-established:
  - "Security validation pattern: module-level compiled regex, validate before any side effects, raise domain-specific error (SchemaError)"
  - "Test isolation pattern: each test file defines its own small_dataset and meta fixtures with np.random.seed(0) for reproducibility"

requirements-completed: [CONN-04, TEST-01, TEST-03, TEST-05]

# Metrics
duration: 4min
completed: 2026-02-22
---

# Phase 1 Plan 04: SQL Injection Prevention and Regression Test Suite Summary

**SQL injection allowlist in save_to_hive() via _SAFE_IDENTIFIER regex and permanent regression harness (TEST-01, TEST-03, TEST-05) covering end-to-end training, serialization round-trip, and bit-identical seed reproducibility**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-22T23:49:10Z
- **Completed:** 2026-02-22T23:53:51Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Patched `save_to_hive()` to validate both `target_db` and all `synthetic_data` table name keys against `^[a-zA-Z0-9_]+$` before any `spark.sql()` call — closes SQL injection vulnerability (CONN-04)
- TEST-01 (`test_e2e_single_table.py`): 2 tests validate that CTGAN fits on a 200-row dataset in 3 epochs, samples 50 rows with correct non-PK column names and no all-null columns
- TEST-03 (`test_serialization.py`): 4 tests validate directory round-trip save/load, `overwrite=False` guard raises `SerializationError`, `overwrite=True` succeeds, and missing path raises `SerializationError`
- TEST-05 (`test_seed_regression.py`): 2 tests confirm two independent `CTGAN.fit(seed=42).sample(100, seed=7)` runs produce bit-identical DataFrames; different seeds produce different output

## Task Commits

Each task was committed atomically:

1. **Task 1: Patch save_to_hive() with SQL injection allowlist validation** - `fd8e863` (feat)
2. **Task 2: Write TEST-01, TEST-03, TEST-05 test files** - `5c4e570` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified
- `syntho_hive/interface/synthesizer.py` — Added `import re`, `_SAFE_IDENTIFIER` module-level regex, and 18-line validation block at start of `save_to_hive()` checking `target_db` and all `synthetic_data` keys
- `tests/test_e2e_single_table.py` — TEST-01: 2 tests (end-to-end fit/sample, fit-does-not-raise)
- `tests/test_serialization.py` — TEST-03: 4 tests (round-trip, overwrite-False guard, overwrite-True success, missing-path guard)
- `tests/test_seed_regression.py` — TEST-05: 2 tests (bit-identical same seeds, different-seeds produce different output)

## Decisions Made
- Allowlist approach chosen over denylist — allowlist is closed by default (everything forbidden unless explicitly allowed); denylist always risks missing a new injection vector
- Both `target_db` and all `synthetic_data` table name keys validated — both are interpolated into SQL strings (`CREATE DATABASE IF NOT EXISTS {target_db}`, `DROP TABLE IF EXISTS {target_db}.{table}`, `CREATE TABLE {target_db}.{table}`)
- `SchemaError` raised immediately before any Spark operation — validation is a pure Python check with no side effects
- Fixtures defined locally in each test file (no shared conftest.py) — no pre-existing conftest.py exists, and local fixtures keep each test file self-contained

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None — all 8 new tests passed on first run. Full test suite: 28 passed, 0 failed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 1 is complete: all 4 plans executed, all 10 requirements addressed (CONN-01/02/03/04, QUAL-01/02/03/04/05, TEST-01/03/05)
- Permanent regression harness in place: future regressions in serialization, reproducibility, or basic training will immediately fail these tests
- Full test suite at 28 passing tests with no pre-existing failures from Phase 1 changes
- Phase 2 can build on a stable, tested CTGAN pipeline with SQL injection prevention and deterministic reproducibility

## Self-Check: PASSED

- FOUND: syntho_hive/interface/synthesizer.py
- FOUND: tests/test_e2e_single_table.py
- FOUND: tests/test_serialization.py
- FOUND: tests/test_seed_regression.py
- FOUND: .planning/phases/01-core-reliability/01-04-SUMMARY.md
- Commit fd8e863 (Task 1): present
- Commit 5c4e570 (Task 2): present

---
*Phase: 01-core-reliability*
*Completed: 2026-02-22*
