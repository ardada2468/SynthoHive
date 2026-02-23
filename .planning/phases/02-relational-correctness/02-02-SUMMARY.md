---
phase: 02-relational-correctness
plan: "02"
subsystem: ml-training
tags: [ctgan, gan, pytorch, pyspark, delta-spark, orchestrator, memory-management]

# Dependency graph
requires:
  - phase: 02-relational-correctness/02-01
    provides: Schema validation foundation for FK cardinality context

provides:
  - CTGAN generator context independent resampling (eliminates FK cardinality drift)
  - legacy_context_conditioning flag for backwards-compatible behavior
  - StagedOrchestrator memory-safe generation with on_write_failure policy
  - pyproject.toml Spark 4.x version pins with upper bounds

affects:
  - 02-relational-correctness (remaining plans)
  - any plan using StagedOrchestrator.generate() with output_path_base
  - any plan using CTGAN.fit() with context data

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Generator context independent resample: np.random.randint for fresh idx each generator step"
    - "Write-and-release pattern: write DataFrame to disk, skip generated_tables accumulation when output_path_base set"
    - "Failure policy helper: module-level function with raise/cleanup/retry semantics"
    - "IO injection: optional io= parameter on orchestrator for testability without Spark"

key-files:
  created: []
  modified:
    - syntho_hive/core/models/ctgan.py
    - syntho_hive/relational/orchestrator.py
    - pyproject.toml

key-decisions:
  - "legacy_context_conditioning defaults to False (fix is opt-in for regression) — old behavior requires explicit True"
  - "on_write_failure defaults to 'raise' to preserve existing error semantics"
  - "StagedOrchestrator.__init__ accepts optional io= parameter directly so tests don't require a live SparkSession"
  - "gen_context_batch used throughout generator step (not real_context_batch) to correctly decouple discriminator and generator context"
  - "pyproject.toml upper bounds <5.0.0 added to prevent accidental delta-spark Spark major version mismatch"

patterns-established:
  - "Write-and-release: when output_path_base is set, DataFrames are NOT stored in generated_tables — child tables read parent data from disk"
  - "Failure policy: _write_with_failure_policy() encapsulates raise/cleanup/retry outside the generation loop"

requirements-completed: [REL-01, REL-04, CONN-02]

# Metrics
duration: 4min
completed: 2026-02-22
---

# Phase 02 Plan 02: Relational Correctness Bug Fixes Summary

**Fixed CTGAN stale context bug (FK cardinality drift), added memory-safe write-and-release in StagedOrchestrator with on_write_failure policy, and pinned PySpark/delta-spark to 4.x range in pyproject.toml**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-23T01:36:57Z
- **Completed:** 2026-02-23T01:41:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- CTGAN generator training block now independently resamples context for each generator step via `np.random.randint`, eliminating the FK cardinality drift caused by reusing the stale discriminator batch context
- `legacy_context_conditioning: bool = False` constructor parameter added with full save/load round-trip persistence for backwards compatibility
- `StagedOrchestrator` generates with memory-safe write-and-release when `output_path_base` is set — DataFrames are written to disk and not stored in `generated_tables`, preventing OOM on large schemas
- `on_write_failure` parameter (`'raise'` | `'cleanup'` | `'retry'`) added with `_write_with_failure_policy()` helper
- pyproject.toml updated: `pyspark>=4.0.0,<5.0.0` and `delta-spark>=4.0.0,<5.0.0`

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix stale context conditioning in CTGAN generator training (REL-01)** - `60f40c2` (fix)
2. **Task 2: Memory-safe generation in StagedOrchestrator + pyproject.toml Spark pins (REL-04, CONN-02)** - `26fb22d` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `syntho_hive/core/models/ctgan.py` - Added `legacy_context_conditioning` parameter, fixed generator training block to use independent context resample, updated save/load for persistence
- `syntho_hive/relational/orchestrator.py` - Added `on_write_failure` parameter, optional `io=` injection, `_write_with_failure_policy()` module helper, write-and-release in generation loop
- `pyproject.toml` - Updated pyspark and delta-spark version pins to 4.x range with upper bounds

## Decisions Made

- `legacy_context_conditioning` defaults to `False` (new correct behavior) — existing code using the old behavior must explicitly pass `True` for backwards compatibility. The default ensures the fix is applied automatically to all new CTGAN instances.
- `on_write_failure` defaults to `'raise'` to preserve existing error semantics — no silent data loss on write failures.
- `StagedOrchestrator.__init__` accepts an optional `io=` parameter so unit tests can mock the IO layer without spinning up a SparkSession. When `io` is provided, `spark` is ignored.
- Generator step uses `gen_context_batch` variable throughout (not `real_context_batch`) to correctly decouple discriminator and generator context sampling.
- Upper bounds `<5.0.0` on Spark pins prevent accidental upgrades to future major versions where delta-spark compatibility may break (delta-spark has strict Spark major version coupling).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed gen_context_batch reference in generator's fake_input construction**
- **Found during:** Task 1 (Fix stale context conditioning in CTGAN)
- **Issue:** After introducing `gen_context_batch` for independent context sampling, the `fake_input` construction for the generator's discriminator forward pass still referenced `real_context_batch` (the old variable). This would cause a NameError when `context_data is not None` and `legacy_context_conditioning=False`.
- **Fix:** Changed `fake_input = torch.cat([fake_data_batch, real_context_batch], ...)` to use `gen_context_batch` and updated the condition from `if real_context_batch is not None` to `if context_data is not None`.
- **Files modified:** `syntho_hive/core/models/ctgan.py`
- **Verification:** All 32 tests pass; CTGAN instantiation with both default and legacy flags verified.
- **Committed in:** `60f40c2` (Task 1 commit)

**2. [Rule 2 - Missing Critical] Added optional io= parameter to StagedOrchestrator for testability**
- **Found during:** Task 2 (StagedOrchestrator changes)
- **Issue:** The plan's verification command calls `StagedOrchestrator(metadata=meta, io=io, on_write_failure='raise')` with a mocked `io` object. The existing `__init__` only accepted `spark: SparkSession` and called `SparkIO(spark)` internally, making unit testing impossible without a live Spark cluster.
- **Fix:** Added `io: Optional[Any] = None` parameter. When provided, it is used directly; when absent, `SparkIO(spark)` is created from `spark` as before.
- **Files modified:** `syntho_hive/relational/orchestrator.py`
- **Verification:** Verification assertion passes with `MagicMock()` io; existing test_models.py multi-table test continues to pass.
- **Committed in:** `26fb22d` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both auto-fixes essential for correctness and testability. No scope creep.

## Issues Encountered

- `test_checkpointing` showed order-dependent flakiness when run with `-x` flag if a prior test run left behind `./test_checkpoints/best_model.pt`. This is a pre-existing test isolation issue (not caused by our changes) — running the full suite without `-x` passes all 32 tests reliably.

## User Setup Required

None - no external service configuration required.

## Self-Check: PASSED

All files exist and commits verified:
- `syntho_hive/core/models/ctgan.py` - FOUND
- `syntho_hive/relational/orchestrator.py` - FOUND
- `pyproject.toml` - FOUND
- `.planning/phases/02-relational-correctness/02-02-SUMMARY.md` - FOUND
- Commit `60f40c2` - FOUND
- Commit `26fb22d` - FOUND

## Next Phase Readiness

- CTGAN generator context fix and orchestrator memory safety are complete and tested
- Plan 02 requirements REL-01, REL-04, and CONN-02 are fulfilled
- Ready for remaining plans in phase 02-relational-correctness
- No known blockers

---
*Phase: 02-relational-correctness*
*Completed: 2026-02-22*
