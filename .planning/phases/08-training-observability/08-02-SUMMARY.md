---
phase: 08-training-observability
plan: "02"
subsystem: training
tags: [ctgan, structlog, checkpointing, statistical-validation, tvd, ks-test, qual-03]

# Dependency graph
requires:
  - phase: 08-training-observability/08-01
    provides: training_start/epoch_end/training_complete structlog events; tqdm progress bar; checkpoint_interval parameter wired end-to-end

provides:
  - Validation-metric checkpointing (mean TVD/KS via StatisticalValidator) replacing generator-loss criterion
  - best_checkpoint/ directory saved on epoch with lowest mean TVD/KS statistic
  - final_checkpoint/ directory saved at end of training (replaces last_model.pt)
  - training_complete event with real best_epoch, best_val_metric, checkpoint_path values
  - epoch_end events include val_metric field only on checkpoint epochs
  - 7 new tests covering CORE-05 and QUAL-03 in test_training_observability.py

affects:
  - Any downstream phase that cold-loads a CTGAN checkpoint (must use best_checkpoint/ path)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Validation-metric checkpointing: StatisticalValidator.compare_columns() on min(len(data), 500) synthetic rows every checkpoint_interval epochs"
    - "epoch_end log event includes val_metric only on checkpoint epochs — field absent on non-checkpoint epochs"
    - "self.save(path, overwrite=True) for best_checkpoint — required on every improvement to avoid SerializationError on second call"
    - "Generator/discriminator set to eval() before self.sample(), restored to train() after"

key-files:
  created:
    - tests/test_training_observability.py
  modified:
    - syntho_hive/core/models/ctgan.py
    - tests/test_checkpointing.py

key-decisions:
  - "Validation sample capped at min(len(data), 500) rows to limit checkpoint overhead without full-epoch generation"
  - "val_metric added directly to epoch_log_fields dict before log.info('epoch_end') — avoids separate epoch_checkpoint event"
  - "Fallback to final_checkpoint as best when no checkpoint epoch ran (e.g. epochs < checkpoint_interval)"
  - "best_model.pt and last_model.pt fully removed — no coexistence with new naming"

patterns-established:
  - "Checkpoint directories (not .pt files) are the canonical save format — both best_checkpoint/ and final_checkpoint/ are directories"
  - "Cold load pattern: fresh CTGAN instance + .load(best_checkpoint_path) + .sample() works without original data"

requirements-completed: [QUAL-03]

# Metrics
duration: 5min
completed: 2026-02-24
---

# Phase 8 Plan 02: Training Observability Summary

**Validation-metric checkpointing (mean TVD/KS via StatisticalValidator) replacing generator-loss criterion, saving best_checkpoint/ and final_checkpoint/ directories instead of .pt files**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-24T05:57:43Z
- **Completed:** 2026-02-24T06:02:30Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- CTGAN.fit() now checkpoints based on lowest mean TVD/KS statistic from StatisticalValidator — the saved checkpoint corresponds to statistically optimal epoch, not lowest generator loss
- best_checkpoint/ (directory) saved on every improvement; final_checkpoint/ (directory) saved at end of training; old best_model.pt and last_model.pt fully removed
- training_complete event now carries real best_epoch, best_val_metric, checkpoint_path — sentinels from Plan 01 replaced with actual values
- epoch_end events include val_metric field on checkpoint epochs only; absent on non-checkpoint epochs
- 7 tests in test_training_observability.py covering CORE-05 (structured log events) and QUAL-03 (validation-metric checkpointing); all 39 suite tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace generator-loss checkpointing with validation-metric checkpointing in CTGAN.fit()** - `b529ee1` (feat)
2. **Task 2: Update test_checkpointing.py and write test_training_observability.py** - `6797736` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `syntho_hive/core/models/ctgan.py` - Replaced best_loss/generator-loss checkpoint with StatisticalValidator-based validation-metric checkpoint; renamed outputs to best_checkpoint/ and final_checkpoint/; updated training_complete event; val_metric in epoch_end on checkpoint epochs only
- `tests/test_checkpointing.py` - Updated to assert best_checkpoint/ and final_checkpoint/ directories; added checkpoint_interval=1 and progress_bar=False
- `tests/test_training_observability.py` - New file with 7 tests: test_epoch_log_events, test_training_start_event, test_training_complete_event, test_eta_seconds_non_zero_after_first_epoch, test_progress_bar_false_does_not_suppress_log_events, test_best_checkpoint_is_best_val_epoch, test_cold_load_uses_best_checkpoint

## Decisions Made
- Validation sample capped at `min(len(data), 500)` rows to limit per-checkpoint overhead; per RESEARCH.md guidance
- val_metric added to `epoch_log_fields` dict before `log.info("epoch_end")` rather than emitting a separate `epoch_checkpoint` event — cleaner approach preserving single event per epoch
- Fallback: when no checkpoint epoch ran (epochs < checkpoint_interval), treat final_checkpoint as best with best_val_metric=inf and best_epoch=epochs-1

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 8 (Training Observability) is now complete: CORE-05 (structured log events) and QUAL-03 (validation-metric checkpointing) both satisfied
- All 39 tests pass
- Ready for Phase 9 (next v1.2 phase)

## Self-Check: PASSED

- FOUND: syntho_hive/core/models/ctgan.py (best_val_metric: present, best_checkpoint: present)
- FOUND: tests/test_checkpointing.py (best_checkpoint: present)
- FOUND: tests/test_training_observability.py (test_best_checkpoint_is_best_val_epoch: present)
- FOUND commit: b529ee1 (Task 1)
- FOUND commit: 6797736 (Task 2)
- All 39 pytest tests passing

---
*Phase: 08-training-observability*
*Completed: 2026-02-24*
