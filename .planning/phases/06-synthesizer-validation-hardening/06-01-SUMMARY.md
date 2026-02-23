---
phase: 06-synthesizer-validation-hardening
plan: 01
subsystem: interface
tags: [synthesizer, validation, model-guard, schema-validation, fk-type-check]

# Dependency graph
requires:
  - phase: 02-relational-correctness
    provides: validate_schema(real_data=) FK dtype mismatch detection in config.py
  - phase: 03-model-pluggability
    provides: ConditionalGenerativeModel ABC, issubclass guard pattern in StagedOrchestrator
provides:
  - TD-04 fix: issubclass guard in Synthesizer.__init__() fires regardless of spark_session presence
  - TD-01 fix: fit(validate=True, data=DataFrames) passes real_data= to validate_schema() for FK type checks
  - Regression test for TD-04 (test_synthesizer_rejects_invalid_model_cls_without_spark)
  - Regression test for TD-01 (test_synthesizer_fit_validate_catches_fk_type_mismatch)
affects:
  - phase: 07-db-connector-gap-closure
  - any caller of Synthesizer.__init__ or Synthesizer.fit with validate=True

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Guard fires at __init__ time unconditionally — not gated behind spark_session presence
    - validate block reordered before orchestrator guard — schema validation does not require Spark
    - real_data passthrough: isinstance(data, dict) and isinstance(next(iter(data.values())), pd.DataFrame)

key-files:
  created: []
  modified:
    - syntho_hive/interface/synthesizer.py
    - syntho_hive/tests/test_interface.py

key-decisions:
  - "TD-04: issubclass guard added in Synthesizer.__init__() before self.metadata assignment — fires for both spark_session=None and spark_session=<session> paths, closing the gap where StagedOrchestrator was the only guard site"
  - "TD-01: validate block moved before orchestrator check so fit(validate=True, data=DataFrames) works without Spark; real_data=data passed only when data is a dict of DataFrames, preserving backward compat for string/path-dict callers"
  - "isinstance(model, type) added before issubclass() — safe-guards against non-class inputs (instances, strings) that would crash raw issubclass() with TypeError"

patterns-established:
  - "Facade guard pattern: public facade validates inputs at __init__ time, not deferred to inner components"
  - "Conditional real_data passthrough: isinstance check on dict values before passing to validate_schema()"

requirements-completed: [REL-03, MODEL-02]

# Metrics
duration: 2min
completed: 2026-02-23
---

# Phase 6 Plan 01: Synthesizer Validation Hardening Summary

**TD-04 issubclass guard and TD-01 real_data passthrough wired directly in Synthesizer facade, closing two silent E2E breakpoints with no new dependencies**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-23T22:00:19Z
- **Completed:** 2026-02-23T22:02:52Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Fixed TD-04: `Synthesizer.__init__()` now raises `TypeError` with message matching "ConditionalGenerativeModel" for any non-subclass model argument, unconditionally regardless of `spark_session` presence
- Fixed TD-01: `Synthesizer.fit(validate=True, data=DataFrames)` now calls `validate_schema(real_data=data)` enabling FK dtype mismatch detection through the public facade without requiring Spark
- Added two regression tests that lock in both fixes and verify the no-spark path works for both operations

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix TD-04 issubclass guard and TD-01 real_data passthrough in synthesizer.py** - `bb5304b` (fix)
2. **Task 2: Add TD-04 and TD-01 regression tests to test_interface.py** - `9b5e1fb` (test)

**Plan metadata:** (docs commit — see final commit)

## Files Created/Modified

- `syntho_hive/interface/synthesizer.py` — Added issubclass guard block in `__init__()` before `self.metadata` assignment; reordered validate block in `fit()` to before orchestrator check with real_data conditional passthrough
- `syntho_hive/tests/test_interface.py` — Added `test_synthesizer_rejects_invalid_model_cls_without_spark` and `test_synthesizer_fit_validate_catches_fk_type_mismatch` at end of file

## Decisions Made

- `isinstance(model, type)` added before `issubclass()` to safely handle non-class inputs (instances, strings) that would crash raw `issubclass()` with a `TypeError` — mirrors the pattern from `StagedOrchestrator` but with this additional safety
- Validate block reordered to BEFORE the orchestrator guard — `validate_schema()` operates on in-memory metadata, no Spark required; this allows `fit(validate=True, data=DataFrames)` to surface errors in the no-Spark path
- `real_data` only passed when `data` is a `dict` with `pd.DataFrame` values — preserves backward compatibility for string (DB name) and dict-of-paths callers who receive structural-only validation as before

## Deviations from Plan

None — plan executed exactly as written. Both edits were surgical as specified, no extra changes needed.

## Issues Encountered

None — `synthesizer.py` had `pd` and `ConditionalGenerativeModel` already imported. No dependency gaps. 4 pre-existing test failures (TD-02 scope, excluded per plan's verification section) were confirmed to pre-date these changes via git stash verification.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- REL-03 and MODEL-02 E2E flows no longer have the identified break points
- Phase 7 (DB connector gap closure / TD-02 fixes) can proceed independently — the 4 pre-existing test failures (`test_synthesizer_fit_requires_spark`, `test_synthesizer_sample_requires_spark`, `test_synthesizer_fit_call`, `test_synthesizer_sample_call`) are TD-02 scope
- All 21 non-TD-02 tests pass cleanly

---
*Phase: 06-synthesizer-validation-hardening*
*Completed: 2026-02-23*

## Self-Check: PASSED

- synthesizer.py: FOUND
- test_interface.py: FOUND
- 06-01-SUMMARY.md: FOUND
- commit bb5304b (fix TD-04 + TD-01 synthesizer.py): FOUND
- commit 9b5e1fb (test regression tests): FOUND
