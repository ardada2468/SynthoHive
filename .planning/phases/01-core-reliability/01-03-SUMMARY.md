---
phase: 01-core-reliability
plan: "03"
subsystem: model-training
tags: [ctgan, seed, reproducibility, determinism, constraints, structlog, bayesian-gmm, pytorch, numpy]

# Dependency graph
requires:
  - phase: 01-core-reliability
    plan: "01"
    provides: "exception hierarchy (SerializationError, ConstraintViolationError), transformer.py column-cast warnings"
provides:
  - "_set_seed() helper in ctgan.py for deterministic PyTorch/NumPy/random/cuDNN seeding"
  - "CTGAN.fit(seed=N) with auto-generation and structlog INFO logging when seed=None"
  - "CTGAN.sample(seed=N) with optional deterministic sampling"
  - "DataTransformer.fit(seed=N) with per-column derived seed propagated to ClusterBasedNormalizer"
  - "ClusterBasedNormalizer(seed=N) — no longer hardcodes random_state=42"
  - "CTGAN.sample(enforce_constraints=True) with structlog WARNING on violations, returns only valid rows"
affects: [01-04, 02-any, testing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-column seed derivation: col_seed = (seed + abs(hash(col)) % 100_000) avoids correlated RNG sequences"
    - "Seed auto-generation pattern: randint(0, 2**31-1) logged at INFO so runs are reproducible without explicit seed"
    - "enforce_constraints=False default preserves backward compatibility; opt-in True for post-generation auditing"
    - "Constraint checking is a no-op when no constraints are configured — never raises on unconstrained tables"

key-files:
  created: []
  modified:
    - syntho_hive/core/models/ctgan.py
    - syntho_hive/core/data/transformer.py

key-decisions:
  - "Separate seeds for fit() and sample() — each accepts its own seed; no coupling between training and inference RNG state"
  - "Auto-generate seed in fit() when None is provided and log it at INFO so data engineers can reproduce any run"
  - "No auto-generation in sample() — only apply seed when explicitly provided (per CONTEXT.md)"
  - "Per-column seed derivation via hash prevents correlated BayesianGMM fitting across columns"
  - "enforce_constraints is a post-hoc audit layer — inverse_transform() already clips values; this catches residual violations"
  - "Constraint violations return partial data (valid rows only) with WARNING log — caller decides if violation rate is acceptable"

patterns-established:
  - "Seed propagation: CTGAN.fit(seed) -> DataTransformer.fit(seed) -> ClusterBasedNormalizer(seed=col_seed)"
  - "Violation logging: structlog.warning('constraint_violations_detected', violation_count, violations, valid_rows, total_rows)"

requirements-completed: [QUAL-04, QUAL-05]

# Metrics
duration: 4min
completed: 2026-02-22
---

# Phase 1 Plan 03: Seed Control and Constraint Violation Checking Summary

**Deterministic seed control added to CTGAN.fit() and CTGAN.sample() via _set_seed() (torch+numpy+random+cuDNN), propagated through DataTransformer to ClusterBasedNormalizer's BayesianGMM; enforce_constraints=True opt-in warns and returns only constraint-valid rows**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-22T23:40:05Z
- **Completed:** 2026-02-22T23:44:06Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `_set_seed()` module-level helper to `ctgan.py` covering PyTorch, NumPy, Python random, and cuDNN determinism flags
- `CTGAN.fit(seed=N)` auto-generates seed when None, logs via structlog INFO so every run is reproducible; propagates seed to `DataTransformer.fit(seed=N)`
- `DataTransformer.fit(seed=N)` derives per-column deterministic seeds and passes them to `ClusterBasedNormalizer` — eliminating the hardcoded `random_state=42`
- `CTGAN.sample(enforce_constraints=True)` scans generated rows against `TableConfig.constraints`, emits a structlog WARNING listing all violations, returns only valid rows without crashing

## Task Commits

Each task was committed atomically:

1. **Task 1: Add seed parameter to CTGAN.fit() and propagate to DataTransformer** - `50c2684` (feat)
2. **Task 2: Add enforce_constraints opt-in to CTGAN.sample() with violation logging** - `7349c07` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified
- `syntho_hive/core/models/ctgan.py` — Added `_set_seed()`, seed params in `fit()`/`sample()`, `enforce_constraints` param in `sample()` with violation filtering and structlog WARNING
- `syntho_hive/core/data/transformer.py` — Added `seed` param to `DataTransformer.fit()` and `ClusterBasedNormalizer.__init__()`; replaced hardcoded `random_state=42` with parameterized `random_state`

## Decisions Made
- Separate seeds for `fit()` and `sample()` — each can be independently controlled (training reproducibility vs inference reproducibility are distinct concerns)
- Auto-generate seed in `fit()` when `None` provided; always log it at INFO so engineers can reproduce any run without pre-planning
- No auto-generation in `sample()` — only apply seed when explicitly provided (asymmetric design preserves flexibility)
- Per-column seed derivation via `(seed + abs(hash(col)) % 100_000)` prevents correlated BayesianGMM fitting where all columns share one value
- `enforce_constraints=False` default preserves backward compatibility; `inverse_transform()` already clips values within ranges, so the enforcement layer is a post-hoc audit
- Violations cause partial data return (valid rows only) with WARNING — never crash; caller decides if violation rate is acceptable

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Pre-existing test failure in `tests/test_models.py::test_ctgan_full_cycle`: the test tries to `os.remove('test_ctgan_model.pth')` but plan 01-02 changed `save()` to write a directory. This is unrelated to plan 01-03 and was confirmed to exist before any changes in this plan. No action taken — out of scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CTGAN reproducibility (QUAL-05) complete: `fit(seed=42)` + `sample(seed=7)` produce deterministic output across identical calls
- Constraint violation auditing (QUAL-04) complete: `sample(enforce_constraints=True)` warns and filters without crashing
- Plan 01-04 (the next plan in this phase) can build on reliable seed-controlled training
- Pre-existing test failure in `test_ctgan_full_cycle` (save/load API mismatch from 01-02) should be addressed before the test suite is relied upon for CI gating

---
*Phase: 01-core-reliability*
*Completed: 2026-02-22*
