---
phase: 01-core-reliability
plan: "02"
subsystem: model-serialization
tags: [ctgan, joblib, pytorch, serialization, checkpoint, round-trip]

# Dependency graph
requires:
  - phase: 01-01
    provides: SerializationError exception class and typed exception wrapping in CTGAN
provides:
  - CTGAN.save(path) writes a 7-file directory checkpoint (generator.pt, discriminator.pt, transformer.joblib, context_transformer.joblib, embedding_layers.joblib, data_column_info.joblib, metadata.json)
  - CTGAN.load(path) reconstructs the full model from a directory checkpoint without training data
  - Cold load-and-sample: fresh CTGAN().load(path).sample(N) works without any prior fit()
  - SerializationError on path-exists without overwrite=True
  - Structlog WARNING on version mismatch (not raise)
affects:
  - 01-03 (sampling/validation uses same CTGAN instance)
  - Phase 2 (multi-table Synthesizer will call CTGAN.save/load)
  - Phase 3 (TVAE serialization should follow same 7-file directory pattern)

# Tech tracking
tech-stack:
  added: [joblib (used for sklearn/numpy-heavy object serialization)]
  patterns:
    - Directory-based model checkpoint (not single .pt file) for complete model persistence
    - joblib.dump for sklearn transformers and nn.ModuleDict
    - torch.load(..., weights_only=False) for PyTorch 2.6+ compatibility
    - Restore saved data_column_info and embedding_layers after _build_model() overwrites them

key-files:
  created: []
  modified:
    - syntho_hive/core/models/ctgan.py
    - tests/test_models.py

key-decisions:
  - "joblib used for DataTransformer, context_transformer, embedding_layers, data_column_info — handles sklearn objects and nn.ModuleDict efficiently via pickle"
  - "weights_only=False required for torch.load on PyTorch 2.6+ (default changed to True, breaks custom objects)"
  - "_build_model() calls _compile_layout() which overwrites data_column_info and embedding_layers with fresh (untrained) layers — load() must restore from joblib after _build_model() to preserve trained weights"
  - "Version mismatch logs structlog WARNING but does not raise — allows cross-version loads with user awareness"
  - "overwrite parameter is keyword-only (* separator) to prevent positional misuse"

patterns-established:
  - "Directory checkpoint pattern: model components split by type (torch.save for nets, joblib for sklearn)"
  - "Post-_build_model restore: load saved data_column_info and embedding_layers after architecture reconstruction to preserve trained state"

requirements-completed: [CORE-02, CORE-03]

# Metrics
duration: 6min
completed: 2026-02-22
---

# Phase 01 Plan 02: CTGAN Directory-Based Serialization Summary

**CTGAN.save()/load() replaced with 7-file directory checkpoint enabling cold load-and-sample without retraining, using joblib for sklearn objects and torch.save for network weights**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-22T23:40:01Z
- **Completed:** 2026-02-22T23:46:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- CTGAN.save(path) writes a complete 7-file directory checkpoint: generator.pt, discriminator.pt, transformer.joblib, context_transformer.joblib, embedding_layers.joblib, data_column_info.joblib, metadata.json
- CTGAN.load(path) reconstructs the full model (transformer, context_transformer, embedding_layers, column layout, network weights) from directory checkpoint without any training data
- Cold load-and-sample verified: `CTGAN(meta).load(path).sample(50)` returns 50 rows without error
- SerializationError raised on path-exists without overwrite=True; overwrite=True succeeds on existing path
- Structlog WARNING logged on version mismatch (not raise); load continues and succeeds

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement directory-based CTGAN.save() with full checkpoint** - `5c7324f` (feat)
2. **Task 2: Implement CTGAN.load() to reconstruct full model from directory checkpoint** - `7349c07` (feat, committed by linter alongside additional 01-03 additions)
3. **[Rule 1 - Bug] Fix test cleanup for directory-based save** - `57aa3d5` (fix)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `syntho_hive/core/models/ctgan.py` - CTGAN.save() and load() replaced with full directory checkpoint implementation
- `tests/test_models.py` - Updated test_ctgan_full_cycle to handle directory checkpoint (shutil.rmtree, cold load)

## Decisions Made
- joblib chosen for DataTransformer, context_transformer, embedding_layers, data_column_info serialization — handles sklearn objects and nn.ModuleDict via pickle with good NumPy efficiency
- weights_only=False required for torch.load() on PyTorch 2.6+ (default changed to True, breaks custom object deserialization)
- _build_model() internally calls _compile_layout(self.transformer) which resets self.data_column_info and self.embedding_layers with freshly-initialized (untrained) layers; load() must restore from joblib after _build_model() completes to preserve trained embedding weights
- overwrite parameter is keyword-only (using * separator) per plan specification to prevent positional misuse

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_ctgan_full_cycle cleanup and cold load logic**
- **Found during:** Task 2 verification (running tests/test_models.py)
- **Issue:** Pre-existing test used `os.remove(tmp_path)` and `new_model.fit()` before `load()` — both assumptions invalid with directory-based save and cold load capability
- **Fix:** Updated test cleanup to `shutil.rmtree()` for directory; removed the redundant `fit()` before `load()` since cold load now works
- **Files modified:** tests/test_models.py
- **Verification:** All 3 tests in test_models.py pass (3 passed, 0 failed)
- **Committed in:** 57aa3d5

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test cleanup)
**Impact on plan:** Auto-fix required for test suite to pass with correct directory-based implementation. No scope creep.

## Issues Encountered
- The linter auto-committed load() implementation alongside unrelated 01-03 additions (seed parameter, enforce_constraints) in commit 7349c07. The load() implementation is correct and all functionality verified.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CTGAN serialization is now complete and correct — save/load round-trip verified
- DataTransformer, context_transformer, embedding_layers, and column layout survive the joblib round-trip
- Ready for Plan 03 (constraint enforcement, sample validation)
- Future phases using CTGAN can call save()/load() for cold persistence

---
*Phase: 01-core-reliability*
*Completed: 2026-02-22*

## Self-Check: PASSED

- syntho_hive/core/models/ctgan.py: FOUND
- tests/test_models.py: FOUND
- .planning/phases/01-core-reliability/01-02-SUMMARY.md: FOUND
- Commit 5c7324f (feat: save() implementation): FOUND
- Commit 7349c07 (feat: load() implementation): FOUND
- Commit 57aa3d5 (fix: test cleanup): FOUND
