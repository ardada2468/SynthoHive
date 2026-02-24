---
phase: 03-model-pluggability
plan: "03"
subsystem: testing
tags: [pytest, importlib, pyproject, gap-closure]

# Dependency graph
requires:
  - phase: 03-02
    provides: MODEL-03 StubModel integration tests in test_interface.py
provides:
  - pytest addopts --import-mode=importlib forcing editable-install source resolution
  - MODEL-03 requirement fully satisfied under default pytest invocation
affects: [any future plan that adds tests to test_interface.py]

# Tech tracking
tech-stack:
  added: []
  patterns: [pytest addopts for import mode control in pyproject.toml]

key-files:
  created: []
  modified:
    - pyproject.toml

key-decisions:
  - "Used addopts = '--import-mode=importlib' (not importmode = importlib) — pytest 9.0.2 does not recognize importmode as a valid ini_options key; addopts is the correct approach"
  - "importlib mode activates editable install finder, ensuring source tree is loaded instead of stale site-packages copy"

patterns-established:
  - "pytest import resolution: addopts = '--import-mode=importlib' prevents stale site-packages shadowing in editable install projects"

requirements-completed: [MODEL-03]

# Metrics
duration: 2min
completed: 2026-02-23
---

# Phase 03 Plan 03: Model Pluggability Gap Closure Summary

**pytest import resolution fixed via addopts = --import-mode=importlib, unblocking all 4 MODEL-phase tests under default pytest invocation without CLI flags**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-23T03:35:01Z
- **Completed:** 2026-02-23T03:37:05Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- All 4 MODEL-phase tests (MODEL-01, MODEL-02, MODEL-03) now pass under default `pytest` invocation with no CLI flags required
- Pre-existing failure count remains exactly 4 (Spark-environment tests only — no regressions)
- test_relational.py: 7/7 pass unchanged
- MODEL-03 requirement fully satisfied end-to-end: Phase 03 score advances to 7/7 must-haves

## Task Commits

Each task was committed atomically:

1. **Task 1: Add importmode = "importlib" to [tool.pytest.ini_options]** - `87c6215` (chore)
2. **Task 2: Fix to addopts; all 4 MODEL-phase tests confirmed passing** - `a4ecfad` (fix)

**Plan metadata:** committed with docs commit

## Files Created/Modified
- `pyproject.toml` - Added `addopts = "--import-mode=importlib"` under `[tool.pytest.ini_options]`; `testpaths` and `python_files` unchanged

## Decisions Made
- Used `addopts = "--import-mode=importlib"` rather than `importmode = "importlib"` — pytest 9.0.2 emits `PytestConfigWarning: Unknown config option: importmode` for the latter; `addopts` is the correct and recognized ini_options key for passing command-line flags by default.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Wrong pyproject.toml key for pytest import mode in pytest 9.0.2**
- **Found during:** Task 2 (verification run)
- **Issue:** The plan specified `importmode = "importlib"` as a `[tool.pytest.ini_options]` key, but pytest 9.0.2 does not recognize `importmode` as a valid ini option — it emitted `PytestConfigWarning: Unknown config option: importmode` and ignored the setting, causing the 4 MODEL-phase tests to still fail.
- **Fix:** Changed `importmode = "importlib"` to `addopts = "--import-mode=importlib"` in `[tool.pytest.ini_options]`. This passes the `--import-mode=importlib` flag automatically on every pytest invocation.
- **Files modified:** `pyproject.toml`
- **Verification:** Re-ran `pytest syntho_hive/tests/test_interface.py -v` with no CLI flags — all 4 MODEL-phase tests PASSED; no warning about unknown config option.
- **Committed in:** `a4ecfad` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Auto-fix was required for correctness — the invalid key was silently ignored by pytest 9.0.2. Using `addopts` achieves identical end-user behavior (default invocation, no flags). No scope creep.

## Issues Encountered

- The `importmode` ini option is not valid in pytest 9.0.2. The `--import-mode` flag must be set via `addopts`. Task 1 initially committed with the wrong key; Task 2 corrected it before completing verification.

## Full pytest Output (test_interface.py -v, default invocation)

```
platform darwin -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0
configfile: pyproject.toml

syntho_hive/tests/test_interface.py::test_metadata_validation PASSED
syntho_hive/tests/test_interface.py::test_metadata_invalid_fk_format PASSED
syntho_hive/tests/test_interface.py::test_synthesizer_init_no_spark PASSED
syntho_hive/tests/test_interface.py::test_synthesizer_fit_requires_spark FAILED   (pre-existing Spark)
syntho_hive/tests/test_interface.py::test_synthesizer_sample_requires_spark FAILED (pre-existing Spark)
syntho_hive/tests/test_interface.py::test_synthesizer_fit_call FAILED             (pre-existing Spark)
syntho_hive/tests/test_interface.py::test_synthesizer_sample_call FAILED          (pre-existing Spark)
syntho_hive/tests/test_interface.py::test_save_to_hive PASSED
syntho_hive/tests/test_interface.py::test_stub_model_routes_through_pipeline PASSED
syntho_hive/tests/test_interface.py::test_synthesizer_accepts_model_parameter PASSED
syntho_hive/tests/test_interface.py::test_synthesizer_default_model_is_ctgan PASSED
syntho_hive/tests/test_interface.py::test_issubclass_guard_rejects_invalid_model_cls PASSED

4 failed, 8 passed in 2.37s
```

Resolution path: `addopts` change alone resolved the gap. `pip install -e .` was NOT required.

## Gap Closure Confirmation

| Must-Have | Status |
|-----------|--------|
| test_stub_model_routes_through_pipeline PASSED under default pytest | VERIFIED |
| test_synthesizer_accepts_model_parameter PASSED under default pytest | VERIFIED |
| test_synthesizer_default_model_is_ctgan PASSED under default pytest | VERIFIED |
| test_issubclass_guard_rejects_invalid_model_cls PASSED under default pytest | VERIFIED |
| Pre-existing failure count remains at 4 (no regressions) | VERIFIED |
| pyproject.toml contains import mode configuration | VERIFIED |

**Phase 03 score: 7/7 must-haves satisfied. MODEL-03 fully closed.**

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 03 model pluggability is complete (MODEL-01, MODEL-02, MODEL-03 all satisfied)
- Any ConditionalGenerativeModel subclass can be plugged in via `model=` parameter at Synthesizer construction
- No blockers for next phase

---
*Phase: 03-model-pluggability*
*Completed: 2026-02-23*

## Self-Check: PASSED

- FOUND: `.planning/phases/03-model-pluggability/03-03-SUMMARY.md`
- FOUND: `pyproject.toml` with `addopts = "--import-mode=importlib"`
- FOUND: commit `87c6215` (Task 1)
- FOUND: commit `a4ecfad` (Task 2)
