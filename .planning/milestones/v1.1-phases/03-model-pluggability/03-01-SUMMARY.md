---
phase: 03-model-pluggability
plan: 01
subsystem: api
tags: [dependency-injection, abc, ctgan, orchestrator, model-pluggability]

# Dependency graph
requires:
  - phase: 02-relational-correctness
    provides: StagedOrchestrator with SparkIO, on_write_failure, legacy_context_conditioning — the class being refactored here
provides:
  - StagedOrchestrator with model_cls dependency injection (Type[ConditionalGenerativeModel] = CTGAN)
  - issubclass() guard at __init__ time raises TypeError for invalid model_cls values
  - ConditionalGenerativeModel ABC docstring documenting constructor convention
affects: [04-tvae-model, 05-multi-dialect-connectors, any plan that instantiates StagedOrchestrator]

# Tech tracking
tech-stack:
  added: []
  patterns: [dependency-injection, abstract-base-class-convention-docstring]

key-files:
  created: []
  modified:
    - syntho_hive/relational/orchestrator.py
    - syntho_hive/core/models/base.py

key-decisions:
  - "model_cls defaults to CTGAN — zero breaking changes for existing callers"
  - "issubclass() guard fires at __init__ time, not fit_all() — fail-fast pattern"
  - "CTGAN import stays in orchestrator.py as the default value; ConditionalGenerativeModel import added alongside it"

patterns-established:
  - "Dependency injection pattern: Type[ABC] parameter with default concrete class — no hardcoded constructor calls in orchestration logic"
  - "ABC constructor convention: documented in docstring because Python cannot enforce __init__ signatures statically"

requirements-completed: [MODEL-01]

# Metrics
duration: 2min
completed: 2026-02-23
---

# Phase 03 Plan 01: Model Pluggability — StagedOrchestrator Dependency Injection Summary

**StagedOrchestrator decoupled from hardcoded CTGAN via model_cls: Type[ConditionalGenerativeModel] = CTGAN parameter with issubclass() guard and ABC constructor convention docstring**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-23T03:12:24Z
- **Completed:** 2026-02-23T03:14:38Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- StagedOrchestrator.__init__ now accepts any ConditionalGenerativeModel subclass via model_cls parameter; defaults to CTGAN for zero breaking changes
- Both CTGAN() constructor call sites in fit_all() replaced with self.model_cls() — zero hardcoded CTGAN instantiation in orchestration logic
- issubclass() guard raises descriptive TypeError at __init__ time for invalid model_cls values
- self.models type annotation updated from Dict[str, CTGAN] to Dict[str, ConditionalGenerativeModel]
- ConditionalGenerativeModel ABC docstring documents constructor convention (metadata, batch_size, epochs, **kwargs)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add model_cls parameter to StagedOrchestrator and replace all CTGAN call sites** - `22d102e` (feat)
2. **Task 2: Add constructor convention docstring to ConditionalGenerativeModel ABC** - `4c843cb` (feat)

## Files Created/Modified
- `syntho_hive/relational/orchestrator.py` — Added Type import, ConditionalGenerativeModel import, model_cls parameter with issubclass() guard, self.model_cls storage, updated Dict annotation, replaced 2 CTGAN() calls with self.model_cls(), updated docstring
- `syntho_hive/core/models/base.py` — Added Constructor convention docstring to ConditionalGenerativeModel class

## Decisions Made
- model_cls defaults to CTGAN — existing callers that don't pass model_cls see identical behavior, no breaking change
- issubclass() guard fires at __init__ time, not fit_all() — fail-fast pattern surfaces misconfiguration before any data is loaded
- CTGAN import stays in orchestrator.py as the default value for model_cls; it is no longer used as a constructor call site

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

Two pre-existing test failures (test_synthesizer_fit_requires_spark, test_synthesizer_sample_requires_spark) were present before and after our changes — confirmed by git stash verification. These are Spark-environment failures unrelated to this plan's changes.

## Next Phase Readiness
- MODEL-01 satisfied: StagedOrchestrator has zero hardcoded CTGAN constructors in orchestration logic
- Any ConditionalGenerativeModel subclass (e.g., TVAE from Phase 04) can now be plugged in at construction time
- The constructor convention docstring in base.py guides implementors of future model classes

---
*Phase: 03-model-pluggability*
*Completed: 2026-02-23*
