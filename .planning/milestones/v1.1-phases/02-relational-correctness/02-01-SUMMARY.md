---
phase: 02-relational-correctness
plan: "01"
subsystem: schema-validation
tags: [pydantic, exceptions, schema, fk, validation, numpy, pandas]

# Dependency graph
requires:
  - phase: 01-core-reliability
    provides: exceptions.py base hierarchy (SynthoHiveError, SchemaError); config.py Metadata/TableConfig models
provides:
  - SchemaValidationError exception class subclassing SchemaError
  - validate_schema(real_data=None) with collect-all FK type/column checks
  - _dtypes_compatible() module-level helper using numpy kind codes
  - TableConfig.linkage_method field (empirical/negbinom) for per-table cardinality config
affects:
  - 02-02 (stale FK context conditioning — uses Metadata/TableConfig from config.py)
  - 02-03 (memory-safe generation — uses Metadata from config.py)
  - Any caller of validate_schema() (now raises SchemaValidationError not ValueError)

# Tech tracking
tech-stack:
  added: [numpy (dtype kind codes for type compatibility), pandas (DataFrame type hints)]
  patterns:
    - "Collect-all validation: accumulate all errors into list, raise once with newline-joined message"
    - "Exception subclassing for backward compatibility: SchemaValidationError(SchemaError) preserves existing except SchemaError handlers"
    - "Module-level helper functions for reusable logic outside class scope"

key-files:
  created: []
  modified:
    - syntho_hive/exceptions.py
    - syntho_hive/interface/config.py

key-decisions:
  - "SchemaValidationError subclasses SchemaError (not SynthoHiveError directly) so existing except SchemaError handlers catch both"
  - "collect-all pattern: gather all FK errors before raise so engineers can fix entire schema in one pass"
  - "validate_schema() is backward compatible: callers passing no arguments get original table-existence and FK-format checks"
  - "linkage_method defaults to 'empirical' — always matches training data distribution exactly; 'negbinom' available for statistical fit"
  - "_dtypes_compatible uses numpy kind codes (i/u/f/U/O/S) and returns True for pandas extension types to avoid false positives"

patterns-established:
  - "Collect-all validation: errors list + single raise at end, not fail-fast"
  - "Optional real_data parameter pattern for additive behavior on existing methods"

requirements-completed: [REL-03]

# Metrics
duration: 3min
completed: 2026-02-23
---

# Phase 2 Plan 01: Schema Validation and Exception Hierarchy Summary

**SchemaValidationError class and collect-all FK validation added to exceptions.py and config.py, with linkage_method per-table cardinality field on TableConfig**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-23T01:36:55Z
- **Completed:** 2026-02-23T01:39:28Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added SchemaValidationError as a subclass of SchemaError, preserving existing `except SchemaError` handler compatibility
- Extended validate_schema() to accept optional real DataFrames and perform collect-all FK type mismatch and missing-column detection
- Added linkage_method: Literal["empirical", "negbinom"] = "empirical" to TableConfig for per-table cardinality method configuration
- Added _dtypes_compatible() module-level helper using numpy kind codes to detect int vs string FK type mismatches
- All 32 existing tests continue to pass (zero regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add SchemaValidationError to exceptions.py** - `4048f9e` (feat)
2. **Task 2: Extend validate_schema() and add linkage_method to TableConfig** - `62c09ef` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `syntho_hive/exceptions.py` - Added SchemaValidationError(SchemaError) class with collect-all docstring
- `syntho_hive/interface/config.py` - Added numpy/pandas/SchemaValidationError imports; linkage_method field on TableConfig; _dtypes_compatible() module-level helper; rewritten validate_schema(real_data=None) with collect-all error pattern

## Decisions Made

- SchemaValidationError subclasses SchemaError (not SynthoHiveError directly) — preserves existing `except SchemaError` handlers catching both
- validate_schema() uses collect-all pattern (not fail-fast) so engineers see all schema problems at once
- Optional real_data=None parameter keeps backward compatibility; callers without data still get structural checks
- _dtypes_compatible returns True for pandas extension types (StringDtype, Int64Dtype) to avoid false positives from edge cases

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

The plan's Test 5 verification script used `TableConfig(pk='id')` which omits the required `name` field (pre-existing required field from v1.0). The `linkage_method` field itself works correctly. The test was manually corrected to `TableConfig(name='test', pk='id')` for verification — no code change was needed.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- SchemaValidationError and extended validate_schema() are ready for use in test suite (02-02 TEST plan)
- TableConfig.linkage_method field is ready for cardinality distribution implementation (02-02 or later plan)
- No blockers for Phase 2 Plan 02

---
*Phase: 02-relational-correctness*
*Completed: 2026-02-23*

## Self-Check: PASSED

- FOUND: syntho_hive/exceptions.py
- FOUND: syntho_hive/interface/config.py
- FOUND: .planning/phases/02-relational-correctness/02-01-SUMMARY.md
- FOUND commit 4048f9e (Task 1: SchemaValidationError)
- FOUND commit 62c09ef (Task 2: validate_schema + linkage_method)
