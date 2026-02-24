---
phase: 02-relational-correctness
plan: "05"
subsystem: testing
tags: [pytest, exceptions, SchemaValidationError, test_interface, gap-closure]

# Dependency graph
requires:
  - phase: 02-relational-correctness/02-01
    provides: "SchemaValidationError exception class and validate_schema() raising it"
provides:
  - "test_metadata_validation passes with SchemaValidationError assertion"
  - "test_metadata_invalid_fk_format passes with SchemaValidationError assertion"
  - "Phase 02 gap closure: all planned tests now passing"
affects: [phase-02-verification, test-interface]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - syntho_hive/tests/test_interface.py

key-decisions:
  - "No new decisions — gap-closure fix only; plan executed exactly as specified"

patterns-established: []

requirements-completed: [REL-03, TEST-02]

# Metrics
duration: 2min
completed: 2026-02-23
---

# Phase 02 Plan 05: Gap Closure — SchemaValidationError Test Fix Summary

**Corrected two test assertions in test_interface.py that expected ValueError but validate_schema() now raises SchemaValidationError, closing the only remaining Phase 02 test gap**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-23T02:06:33Z
- **Completed:** 2026-02-23T02:08:17Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Added `from syntho_hive.exceptions import SchemaValidationError` import to test_interface.py
- Changed `pytest.raises(ValueError)` to `pytest.raises(SchemaValidationError)` in `test_metadata_validation`
- Changed `pytest.raises(ValueError)` to `pytest.raises(SchemaValidationError)` in `test_metadata_invalid_fk_format`
- Both tests now pass; Phase 02 full suite: 15 passed, 4 pre-existing failures unchanged

## Regression Root Cause

Plan 01 (02-01-PLAN.md) correctly changed `validate_schema()` to raise `SchemaValidationError` (a subclass of `SchemaError`, not `ValueError`). The two tests in `test_interface.py` that verified this behavior were written before the exception hierarchy change and asserted `pytest.raises(ValueError)`. Since `SchemaValidationError` does not inherit from `ValueError`, the `pytest.raises(ValueError)` context managers never caught the exception — the exception propagated and caused a test error rather than a pass.

## The Three-Line Fix

1. **Line 6** (new): `from syntho_hive.exceptions import SchemaValidationError`
2. **Line 35**: `pytest.raises(ValueError, ...)` → `pytest.raises(SchemaValidationError, ...)`
3. **Line 39**: `pytest.raises(ValueError, ...)` → `pytest.raises(SchemaValidationError, ...)`

## Final Phase 02 Test Status

Full suite result (`python -m pytest syntho_hive/tests/ -v`): **15 passed, 4 failed**

| Test | Status | Notes |
|------|--------|-------|
| test_metadata_validation | PASSED | Fixed in this plan |
| test_metadata_invalid_fk_format | PASSED | Fixed in this plan |
| test_synthesizer_init_no_spark | PASSED | Pre-existing pass |
| test_save_to_hive | PASSED | Pre-existing pass |
| test_synthesizer_fit_requires_spark | FAILED | Pre-existing failure (out of scope) |
| test_synthesizer_sample_requires_spark | FAILED | Pre-existing failure (out of scope) |
| test_synthesizer_fit_call | FAILED | Pre-existing failure (out of scope) |
| test_synthesizer_sample_call | FAILED | Pre-existing failure (out of scope) |
| TestLinkageModel::test_fit_sample | PASSED | |
| TestOrchestrator::test_orchestrator_flow | PASSED | |
| TestFKChainIntegrity::test_3_table_chain_zero_orphans | PASSED | |
| TestFKChainIntegrity::test_4_table_chain_zero_orphans | PASSED | |
| TestFKChainIntegrity::test_cardinality_within_tolerance | PASSED | |
| TestFKChainIntegrity::test_fk_missing_column_raises_schema_validation_error | PASSED | |
| TestFKChainIntegrity::test_fk_type_mismatch_raises_schema_validation_error | PASSED | |
| TestValidation::test_empty_dataframe | PASSED | |
| TestValidation::test_html_report_generation | PASSED | |
| TestValidation::test_perfect_match | PASSED | |
| TestValidation::test_type_mismatch | PASSED | |

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix SchemaValidationError assertions in test_interface.py** - `c01056a` (fix)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified

- `syntho_hive/tests/test_interface.py` - Added SchemaValidationError import; updated two pytest.raises assertions from ValueError to SchemaValidationError

## Decisions Made

None - gap-closure fix applied exactly as specified in plan.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None — the `python` command was not available in the shell path; used `.venv/bin/python` instead. This is a pre-existing environment configuration (no `.venv` activation in shell profile).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 02 (relational-correctness) is fully complete: all 5 plans executed
- Requirements REL-03 and TEST-02 are fulfilled
- The 4 remaining failures in test_interface.py are pre-existing and require separate planning (Synthesizer Spark integration test fixes)
- Phase 03 (TVAE or next milestone) can proceed

---
*Phase: 02-relational-correctness*
*Completed: 2026-02-23*

## Self-Check: PASSED

- FOUND: syntho_hive/tests/test_interface.py
- FOUND: .planning/phases/02-relational-correctness/02-05-SUMMARY.md
- FOUND: commit c01056a (fix(02-05): update test_interface.py to expect SchemaValidationError)
