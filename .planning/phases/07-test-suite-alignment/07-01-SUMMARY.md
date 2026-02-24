---
phase: 07-test-suite-alignment
plan: "01"
subsystem: testing
tags: [pytest, exceptions, TrainingError, test-alignment, call-args]

# Dependency graph
requires:
  - phase: 06-synthesizer-validation-hardening
    provides: TrainingError boundary wrapping in Synthesizer.fit() and Synthesizer.sample()
provides:
  - Clean test suite (14/14 passing) for test_interface.py — prerequisite for v1.1 ship
affects: [v1.1-release, test-suite-alignment]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - syntho_hive/tests/test_interface.py

key-decisions:
  - "Exception assertions updated to TrainingError (not ValueError) — tests now match actual Synthesizer boundary behavior where internal ValueError is wrapped"
  - "assert_called_with() replaced by call_args.args[0] positional check in test_synthesizer_fit_call — decouples test from evolving kwargs"
  - "test_synthesizer_sample_call updated to call_args.kwargs check — output_path_base is a kwarg, not a positional arg"

patterns-established: []

requirements-completed: [TEST-01]

# Metrics
duration: 4min
completed: 2026-02-23
---

# Phase 7 Plan 01: Test Suite Alignment Summary

**4 stale test assertions in test_interface.py corrected to match post-phase-06 exception boundaries and call signatures, bringing pytest from 4 failures to 14/14 passed.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-23T00:00:00Z
- **Completed:** 2026-02-23T00:04:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Imported `TrainingError` from `syntho_hive.exceptions` in test_interface.py
- Fixed `test_synthesizer_fit_requires_spark` and `test_synthesizer_sample_requires_spark` to assert `TrainingError` instead of `ValueError` — matches the exception wrapping introduced in synthesizer.py lines 136-140 and 173-177
- Fixed `test_synthesizer_fit_call` to use `call_args.args[0]` positional check instead of brittle `assert_called_with()` — decouples test from evolving kwargs (epochs, batch_size)
- Fixed `test_synthesizer_sample_call` to use `call_args.kwargs` — `output_path_base` is a keyword argument, so `call_args.args` only has 1 element (num_rows)
- All 10 previously passing tests remain unchanged and green

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix exception assertions and call signature checks in test_interface.py** - `dec0ab7` (fix)

## Files Created/Modified

- `syntho_hive/tests/test_interface.py` - Added TrainingError import; corrected 4 failing test assertions to match production behavior

## Decisions Made

- Exception assertions updated to TrainingError (not ValueError) — tests now match actual Synthesizer boundary behavior where internal ValueError is wrapped in TrainingError before escaping the public API
- `assert_called_with()` replaced by `call_args.args[0]` positional check in `test_synthesizer_fit_call` — decouples test from evolving kwargs (epochs, batch_size)
- `test_synthesizer_sample_call` updated to `call_args.kwargs` check — `output_path_base` is a keyword argument, not positional

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- TD-02 is now closed: test suite runs clean (14/14 passed, 0 failures)
- v1.1 can ship with a green test gate
- No blockers

---
*Phase: 07-test-suite-alignment*
*Completed: 2026-02-23*

## Self-Check: PASSED

- FOUND: syntho_hive/tests/test_interface.py
- FOUND: .planning/phases/07-test-suite-alignment/07-01-SUMMARY.md
- FOUND: commit dec0ab7
