---
phase: 02-relational-correctness
plan: "03"
subsystem: relational
tags: [numpy, scipy, nbinom, empirical, cardinality, linkage, structlog]

# Dependency graph
requires:
  - phase: 02-01
    provides: linkage_method field on TableConfig (empirical/negbinom literal)
provides:
  - Empirical histogram resampler as default cardinality model in LinkageModel
  - Optional NegBinom cardinality fit via method='negbinom'
  - Non-negative integer child counts guaranteed from sample_counts()
  - StagedOrchestrator wires linkage_method from TableConfig into LinkageModel
affects: [02-relational-correctness, relational-generation, fk-integrity]

# Tech tracking
tech-stack:
  added: [scipy.stats.nbinom (lazy import)]
  patterns: [empirical-resampling, method-of-moments-negbinom, structlog-fallback-warning]

key-files:
  created: []
  modified:
    - syntho_hive/relational/linkage.py
    - syntho_hive/relational/orchestrator.py

key-decisions:
  - "Empirical resampler (numpy.random.choice) is the default method — draws directly from observed child counts, guaranteed non-negative, exact distributional match"
  - "NegBinom uses method-of-moments fitting (mu/var) and falls back to empirical with structlog WARNING when variance <= mean (Poisson regime or insufficient overdispersion)"
  - "scipy.stats imported lazily inside sample_counts() to avoid hard dependency when using empirical mode only"
  - "Silent except/fallback block that cast FK types to string removed — FK type validation is now handled upstream by validate_schema()"

patterns-established:
  - "Runtime method fallback: method attribute mutated in-place from 'negbinom' to 'empirical' when fallback triggers, so sample_counts() path selection stays simple"
  - "Lazy import inside method body: from scipy import stats only when negbinom path is actually used"

requirements-completed: [REL-02]

# Metrics
duration: 5min
completed: 2026-02-22
---

# Phase 02 Plan 03: Relational Correctness — LinkageModel Cardinality Rewrite Summary

**Replaced GaussianMixture cardinality model with empirical histogram resampler (numpy) and optional NegBinom distribution (scipy), eliminating negative sample counts that caused FK integrity failures**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-23T01:43:47Z
- **Completed:** 2026-02-23T01:48:00Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Removed GaussianMixture from linkage.py entirely — import and usage both eliminated
- Rewrote LinkageModel with empirical histogram resampler as default (numpy.random.choice over _observed_counts)
- Added optional NegBinom fit via method='negbinom' using method-of-moments; auto-falls back to empirical with structlog WARNING when variance <= mean
- Deleted silent except/fallback block that cast FK types to string (masked real type mismatches now caught by validate_schema())
- StagedOrchestrator now reads linkage_method from TableConfig and passes it to LinkageModel constructor

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace GaussianMixture in LinkageModel with empirical/NegBinom distribution** - `f857ff0` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `syntho_hive/relational/linkage.py` - Rewritten: GaussianMixture removed, empirical resampler + NegBinom with fallback
- `syntho_hive/relational/orchestrator.py` - Updated LinkageModel instantiation to pass linkage_method from TableConfig

## Decisions Made

- Empirical resampler chosen as default because it guarantees non-negative integers and exactly matches the observed child count distribution without any distributional assumptions
- NegBinom fit uses method-of-moments (mu, var -> p, n) — standard, closed-form, no optimization needed
- NegBinom falls back to empirical (not raises) because non-overdispersed data is a valid dataset property, not an error condition
- scipy imported lazily inside sample_counts() body so the empirical path has zero scipy dependency
- The silent except/fallback that converted types to string was removed — this masked FK type mismatches that are now properly caught and reported by validate_schema() (added in Plan 01)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- LinkageModel now guarantees non-negative integer counts, resolving the core source of FK integrity failures in relational generation
- The empirical/negbinom method is configurable per-table via the linkage_method field on TableConfig (set in Plan 01)
- All 32 existing tests pass after the rewrite
- Ready for the next relational correctness plan

## Self-Check: PASSED

- syntho_hive/relational/linkage.py: FOUND
- syntho_hive/relational/orchestrator.py: FOUND
- .planning/phases/02-relational-correctness/02-03-SUMMARY.md: FOUND
- Commit f857ff0: FOUND

---
*Phase: 02-relational-correctness*
*Completed: 2026-02-22*
