---
phase: 01-core-reliability
plan: "05"
subsystem: testing
tags: [ctgan, exceptions, constraints, structlog, pytest, mocking]

# Dependency graph
requires:
  - phase: 01-core-reliability
    provides: CTGAN sample() with enforce_constraints parameter (Plan 03) and SQL injection patch (Plan 04)
provides:
  - QUAL-04 satisfied: ConstraintViolationError raised (not warn+return) when enforce_constraints=True and violations exist
  - CORE-04 satisfied: zero silent exception swallows in syntho_hive/core/models/ctgan.py
  - 4-test regression suite (tests/test_constraint_violation.py) verifying raise behavior
affects: [02-data-quality, 03-model-variety, 05-sql-connectors]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Constraint violations raise ConstraintViolationError with column name + observed value; callers use enforce_constraints=False for old warn+return behavior"
    - "All except Exception blocks use 'as exc' and log via structlog — no bare silent swallows anywhere in ctgan.py"

key-files:
  created:
    - tests/test_constraint_violation.py
  modified:
    - syntho_hive/core/models/ctgan.py

key-decisions:
  - "ConstraintViolationError raised (not warn+return) in sample() when enforce_constraints=True — satisfies QUAL-04 and ROADMAP success criterion 4"
  - "enforce_constraints=False (the default) preserves pre-existing warn-and-return behavior for backward compatibility"
  - "Test uses MagicMock/patch.object injection approach to guarantee violations without depending on probabilistic model output"
  - "[Rule 2 deviation] Also fixed 2 silent except Exception blocks in save()/load() version-lookup paths (CORE-04 completeness)"

patterns-established:
  - "Constraint violation path: build violations list -> raise ConstraintViolationError with full summary string including all violating column names and observed vs expected values"
  - "Silent except blocks replaced with structlog WARNING including error=str(exc) and context note"

requirements-completed: [QUAL-04, CORE-04]

# Metrics
duration: 5min
completed: 2026-02-22
---

# Phase 1 Plan 05: Constraint Violation Raise and Silent Except Elimination Summary

**ConstraintViolationError now raised (not warn+return) on violations in CTGAN.sample(enforce_constraints=True), with all five silent except blocks in ctgan.py replaced by structlog warnings**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-22T00:00:00Z
- **Completed:** 2026-02-22T00:05:00Z
- **Tasks:** 2
- **Files modified:** 2 (ctgan.py, tests/test_constraint_violation.py)

## Accomplishments
- Changed violations block in sample() to raise ConstraintViolationError instead of logging + returning partial data — closes QUAL-04
- Replaced all 3 silent `except Exception: pass` blocks in sample() with structlog warnings (as planned) — closes CORE-04
- Also replaced 2 silent exception blocks in save()/load() version-lookup paths (CORE-04 completeness, Rule 2 auto-fix)
- Added `ConstraintViolationError` to ctgan.py import line
- Wrote 4-test regression suite in tests/test_constraint_violation.py — all 4 pass
- Full test suite: 32 tests pass, 0 failed

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix ctgan.py sample() — raise ConstraintViolationError and replace silent excepts** - `25cd9ce` (fix)
2. **Task 2: Write test_constraint_violation.py verifying raise behavior and run full test suite** - `2345dc3` (feat)

## Files Created/Modified
- `syntho_hive/core/models/ctgan.py` — Changed violations block to raise; replaced 3 silent excepts in sample() and 2 in save()/load() with structlog warnings; added ConstraintViolationError import
- `tests/test_constraint_violation.py` — 4-test QUAL-04 regression suite: raise on violation, no raise on enforce_constraints=False, no raise on clean data, error message names all violating columns

## Decisions Made
- ConstraintViolationError is raised instead of warn+return — satisfies QUAL-04 and ROADMAP success criterion 4
- enforce_constraints=False (default) preserved for backward compatibility — callers who relied on warn+return can opt out
- Tests use MagicMock/patch.object injection so constraint violations are deterministic regardless of model output

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Fixed 2 additional silent except blocks in save()/load() version-lookup**
- **Found during:** Task 1 (ctgan.py modifications)
- **Issue:** grep -n "except Exception:" revealed 2 bare silent blocks in save() and load() (lines ~679, ~754) that fetch __version__ and silently fall back to "unknown" without logging — contradicts CORE-04 requirement for zero silent swallows
- **Fix:** Added structlog.warning("version_lookup_failed", ...) with error=str(exc) to both blocks before setting current_version = "unknown"
- **Files modified:** syntho_hive/core/models/ctgan.py
- **Verification:** grep -n "except Exception:" returns zero results; 32 tests pass
- **Committed in:** 25cd9ce (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 — missing critical logging for CORE-04 completeness)
**Impact on plan:** Auto-fix required to satisfy the plan's own must_haves truth ("grep -n 'except Exception:' shows only logged warnings — no silent swallows"). No scope creep.

## Issues Encountered
None — plan executed smoothly. All 4 new tests passed on first run. No mocking adjustments needed.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- QUAL-04 SATISFIED: ConstraintViolationError raised with column name and observed value
- CORE-04 SATISFIED: zero silent exception swallows anywhere in syntho_hive/core/models/ctgan.py
- Phase 1 all 5 plans complete (01 through 05)
- All 32 tests pass; ready for Phase 2

---
*Phase: 01-core-reliability*
*Completed: 2026-02-22*

## Self-Check: PASSED

- FOUND: syntho_hive/core/models/ctgan.py
- FOUND: tests/test_constraint_violation.py
- FOUND: .planning/phases/01-core-reliability/01-05-SUMMARY.md
- FOUND commit: 25cd9ce (fix ctgan.py)
- FOUND commit: 2345dc3 (test_constraint_violation.py)
