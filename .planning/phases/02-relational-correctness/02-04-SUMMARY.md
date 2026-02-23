---
phase: 02-relational-correctness
plan: "04"
subsystem: testing
tags: [unittest, pytest, fk-integrity, cardinality, schema-validation, mock, ctgan, orchestrator]

# Dependency graph
requires:
  - phase: 02-01
    provides: SchemaValidationError, validate_schema(real_data=) with FK type/column checks
  - phase: 02-02
    provides: StagedOrchestrator with io= injection, memory-safe write-and-release generation
  - phase: 02-03
    provides: LinkageModel empirical/NegBinom cardinality model with non-negative integer counts
provides:
  - TEST-02: TestFKChainIntegrity class with 5 test methods in syntho_hive/tests/test_relational.py
  - Empirical proof that REL-05 is satisfied — generated tables join with zero orphans
  - SchemaValidationError raised for FK type mismatch and missing FK column edge cases
  - Cardinality accuracy assertion (within 20% relative tolerance) for empirical LinkageModel
affects: [02-relational-correctness, end-to-end-validation, fk-integrity, cardinality]

# Tech tracking
tech-stack:
  added: [tempfile (stdlib)]
  patterns:
    - "MockSparkDF + read_side_effect: routes disk reads (generated output) through CSV, training reads through in-memory DataFrame"
    - "write_side_effect: saves pandas DataFrame to CSV in temp dir so child tables can read-back parent output during generation"
    - "_make_mock_io helper: encapsulates full mock IO setup for reuse across FK chain tests"

key-files:
  created: []
  modified:
    - syntho_hive/tests/test_relational.py

key-decisions:
  - "test_relational.py is the canonical location for relational tests (inside package, not top-level tests/)"
  - "SchemaValidationError imports done inline inside test methods to keep import graph clear"
  - "FK chain tests use epochs=2, batch_size=20 to complete fast while still exercising the full fit+generate pipeline"
  - "Zero-orphan check uses inner join row count == child table row count (not set inclusion) to catch exact counts"

patterns-established:
  - "_make_mock_io helper: encapsulate MockSparkDF class + read/write side effects into reusable setup method"
  - "Temp dir + tearDown shutil.rmtree: ensures test isolation with no leftover disk state"

requirements-completed: [REL-05, TEST-02]

# Metrics
duration: 7min
completed: 2026-02-23
---

# Phase 02 Plan 04: TEST-02 FK Chain Integrity Test Suite Summary

**TestFKChainIntegrity class with 5 Spark-free tests proving zero orphan FK joins, SchemaValidationError on type mismatch and missing column, and empirical cardinality accuracy within 20% tolerance**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-02-23T01:47:54Z
- **Completed:** 2026-02-23T01:55:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Added TestFKChainIntegrity class to syntho_hive/tests/test_relational.py with 5 test methods
- test_3_table_chain_zero_orphans: users → orders → items, 10 generated users, inner join count asserted equal to child table length at both join levels
- test_4_table_chain_zero_orphans: extends to users → orders → items → reviews, 5 generated users, all 3 join levels verified zero orphans
- test_fk_type_mismatch_raises_schema_validation_error: int PK / str FK → SchemaValidationError with 'mismatch' or 'type' in message
- test_fk_missing_column_raises_schema_validation_error: missing user_id column → SchemaValidationError with 'missing' or 'user_id' in message
- test_cardinality_within_tolerance: empirical LinkageModel on 50 parents × exactly 3 children each; sampled mean within 20% relative error of 3.0
- All tests run without a real Spark session (MockSparkDF + mocked IO pattern)
- Added tempfile import and reorganized stdlib imports at top of test file

## Task Commits

Each task was committed atomically:

1. **Task 1: Write and verify TEST-02 (TestFKChainIntegrity)** - `e37bddd` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `syntho_hive/tests/test_relational.py` - Appended TestFKChainIntegrity class (254 lines); added tempfile import; reorganized stdlib import order

## Decisions Made

- Test file location is `syntho_hive/tests/test_relational.py` (inside package) — the plan front matter says `tests/test_relational.py` but the actual file discovered on disk is inside the package; matched the existing file location
- `_make_mock_io` helper extracts the read/write mock wiring to avoid repeating the MockSparkDF boilerplate in both FK chain tests
- The read_side_effect detects whether the requested path is in the temp output dir (disk read of generated data) or a training data path (in-memory DataFrame) by matching the table_out_path
- SchemaValidationError imports are done inline inside test methods (matching plan guidance) to keep the module-level import list clean
- epochs=2, batch_size=20 chosen to keep full-pipeline tests fast (under 5 seconds each) while still exercising the real fit+generate code path

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None. All 5 tests passed on first run. Pre-existing failures in syntho_hive/tests/test_interface.py (6 tests) were confirmed to be pre-existing before this plan's changes and are out of scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- TEST-02 suite (TestFKChainIntegrity) is complete and passing — provides empirical proof that Plans 01-03 work together to produce FK-correct multi-table output
- REL-05 and TEST-02 requirements fulfilled
- All relational correctness plans (01-04) for phase 02 are now complete
- Ready for the next phase or further relational correctness plans

## Self-Check: PASSED

- syntho_hive/tests/test_relational.py: FOUND
- .planning/phases/02-relational-correctness/02-04-SUMMARY.md: FOUND
- Commit e37bddd: FOUND (test(02-04): add TestFKChainIntegrity)

---
*Phase: 02-relational-correctness*
*Completed: 2026-02-23*
