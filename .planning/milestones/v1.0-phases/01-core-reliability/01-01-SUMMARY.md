---
phase: 01-core-reliability
plan: "01"
subsystem: core
tags: [exceptions, error-handling, structlog, typed-exceptions, ctgan, transformer, synthesizer]

# Dependency graph
requires: []
provides:
  - syntho_hive/exceptions.py with SynthoHiveError base class and 4 typed subclasses
  - Typed exception boundary wrapping on all 5 Synthesizer public API methods
  - Zero bare except blocks across syntho_hive/ package
  - Structured warning logging in transformer.py column cast operations
  - CTGAN save/load with SerializationError wrapping and PyTorch 2.6 weights_only fix
affects:
  - 01-02 (schema validation plan will use SchemaError)
  - 01-03 (constraint violation plan will use ConstraintViolationError)
  - 01-04 (serialization plan will use SerializationError)
  - All future phases (exception hierarchy is the foundation every other plan builds on)

# Tech tracking
tech-stack:
  added: [structlog (already in dependencies; now actively used)]
  patterns: [typed-exception-hierarchy, raise-from-exc-chaining, API-boundary-wrapping]

key-files:
  created:
    - syntho_hive/exceptions.py
  modified:
    - syntho_hive/__init__.py
    - syntho_hive/interface/synthesizer.py
    - syntho_hive/core/data/transformer.py
    - syntho_hive/core/models/ctgan.py

key-decisions:
  - "Use raise...from exc throughout so callers always see chained tracebacks including root cause"
  - "generate_validation_report() outer boundary uses generic SynthoHiveError (can fail for multiple reasons)"
  - "transformer.py column cast failures log at WARNING level rather than raising (single column failure should not abort whole batch)"
  - "Added save()/load() to Synthesizer using joblib for full-object persistence (not just state_dict)"
  - "Fixed torch.load() to pass weights_only=True for PyTorch 2.6+ compatibility (blocking concern from STATE.md)"

patterns-established:
  - "Public API boundary: try block wraps method body; except SynthoHiveError: raise; except Exception as exc: raise TypedError(...) from exc"
  - "Internal logging: structlog module-level log = structlog.get_logger() at top of each module"
  - "Exception import: from syntho_hive.exceptions import <relevant classes> at top of each file"

requirements-completed: [CORE-01, CORE-04]

# Metrics
duration: 5min
completed: 2026-02-22
---

# Phase 1 Plan 01: Exception Hierarchy and Bare Except Elimination Summary

**SynthoHiveError hierarchy (5 typed classes) established; all bare/silent excepts eliminated; synthesizer public API boundary wrapped with typed re-raises using raise...from exc**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-22T23:32:36Z
- **Completed:** 2026-02-22T23:37:10Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created `syntho_hive/exceptions.py` with `SynthoHiveError` base and 4 typed subclasses (`SchemaError`, `TrainingError`, `SerializationError`, `ConstraintViolationError`)
- Exported all 5 exception classes from `syntho_hive/__init__.py` for top-level imports
- Eliminated the bare `except:` at synthesizer.py:153 (was catching `KeyboardInterrupt` and `SystemExit`)
- Eliminated 2 silent `except Exception: pass` blocks in transformer.py (lines 228, 245) replacing with structured `structlog` warnings
- Wrapped all 5 Synthesizer public API methods (`fit`, `sample`, `save`, `load`, `generate_validation_report`) with typed exception boundaries
- Added `save()`/`load()` methods to `Synthesizer` using `joblib` for full-object persistence
- Added `structlog` logger + typed exception imports to `ctgan.py` and `transformer.py`
- Fixed `torch.load()` to pass `weights_only=True` (PyTorch 2.6+ breaking change flagged in STATE.md)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create exception hierarchy** - `6a4dd10` (feat)
2. **Task 2: Eliminate bare excepts; wrap API boundaries** - `ab197cc` (feat)
3. **Task 2 deviation: Add typed exceptions to CTGAN** - `cf87152` (feat)

**Plan metadata:** (final docs commit - see below)

## Files Created/Modified
- `syntho_hive/exceptions.py` - New: SynthoHiveError base + SchemaError, TrainingError, SerializationError, ConstraintViolationError (45 lines)
- `syntho_hive/__init__.py` - Modified: exports all 5 exception classes
- `syntho_hive/interface/synthesizer.py` - Modified: structlog + typed exception imports; all 5 public methods wrapped; save()/load() added
- `syntho_hive/core/data/transformer.py` - Modified: structlog + ConstraintViolationError import; 2 silent excepts replaced with logged warnings
- `syntho_hive/core/models/ctgan.py` - Modified: structlog + SerializationError/TrainingError imports; save()/load() wrapped with SerializationError; torch.load() weights_only=True fix

## Decisions Made
- `generate_validation_report()` outer boundary uses generic `SynthoHiveError` since it can fail for multiple reasons (data loading, report generation, Spark connectivity)
- `transformer.py` column cast failures log at `WARNING` level rather than raising — single column failure should not abort the entire batch inverse transform
- Added `save()`/`load()` to `Synthesizer` using `joblib` (not state_dict) for full-object persistence including transformer state
- Exception chaining via `raise TypedError("...") from exc` used everywhere to preserve root cause in tracebacks

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added typed exception wrapping to CTGAN.save() and CTGAN.load()**
- **Found during:** Task 2 (audit phase)
- **Issue:** Plan's `must_haves.key_links` required `from syntho_hive.exceptions import SerializationError, TrainingError` in `ctgan.py`, but `save()`/`load()` in ctgan.py had no exception handling at all
- **Fix:** Added structlog + exception imports; wrapped `save()` and `load()` with `SerializationError` boundaries
- **Files modified:** `syntho_hive/core/models/ctgan.py`
- **Verification:** `from syntho_hive.core.models.ctgan import CTGAN` succeeds; tests still pass
- **Committed in:** `cf87152`

**2. [Rule 1 - Bug] Fixed torch.load() default weights_only argument for PyTorch 2.6+**
- **Found during:** Task 2 (fixing ctgan.py load())
- **Issue:** `torch.load(path)` default changed to `weights_only=True` in PyTorch 2.6+, meaning existing call without the argument triggers a deprecation warning and will break in future PyTorch releases; this was explicitly flagged in STATE.md blockers
- **Fix:** Changed `torch.load(path)` to `torch.load(path, weights_only=True)` in `CTGAN.load()`
- **Files modified:** `syntho_hive/core/models/ctgan.py`
- **Verification:** Import succeeds; no test regressions
- **Committed in:** `cf87152`

---

**Total deviations:** 2 auto-fixed (1 missing critical [Rule 2], 1 bug fix [Rule 1])
**Impact on plan:** Both auto-fixes directly required for correctness and spec compliance. No scope creep.

## Issues Encountered
None - both issues were caught during audit and fixed inline.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Exception hierarchy is complete and stable — all Phase 1 plans (01-02 through 01-04) can import from `syntho_hive.exceptions`
- `SchemaError` ready for Plan 02 (schema validation)
- `ConstraintViolationError` import already placed in transformer.py, ready for Plan 03
- `SerializationError` foundation laid in both synthesizer.py and ctgan.py, ready for Plan 04
- Pre-existing test failures (test_null_handling, test_transformer_categorical_constraint_fix) are unchanged and expected to be addressed in later plans

---
*Phase: 01-core-reliability*
*Completed: 2026-02-22*
