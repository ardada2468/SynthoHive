---
phase: 03-model-pluggability
plan: 02
subsystem: api
tags: [dependency-injection, abc, ctgan, synthesizer, model-pluggability, integration-test, stub-model]

# Dependency graph
requires:
  - phase: 03-model-pluggability plan 01
    provides: StagedOrchestrator with model_cls DI and ConditionalGenerativeModel ABC constructor convention
  - phase: 02-relational-correctness
    provides: StagedOrchestrator with SparkIO, on_write_failure — class being exposed through Synthesizer API
provides:
  - Synthesizer.model parameter (Type[ConditionalGenerativeModel] = CTGAN) replacing backend: str
  - StubModel integration test proving end-to-end MODEL-03 contract
  - 4 new tests: stub routing, model= acceptance, CTGAN default, issubclass guard
affects: [04-tvae-model, 05-multi-dialect-connectors, any caller that instantiates Synthesizer with a custom model]

# Tech tracking
tech-stack:
  added: []
  patterns: [public-api-model-injection, stub-model-integration-test-pattern]

key-files:
  created: []
  modified:
    - syntho_hive/interface/synthesizer.py
    - syntho_hive/tests/test_interface.py

key-decisions:
  - "Type import added to synthesizer.py alongside ConditionalGenerativeModel and CTGAN imports — self.model_cls = model stored in __init__"
  - "StagedOrchestrator constructed with model_cls=self.model_cls — end-to-end injection confirmed"
  - "sample() uses self.model_cls.__name__ instead of hardcoded string, so progress prints are accurate for any model"
  - "StubModel uses constructor convention (metadata, batch_size, epochs, **kwargs) exactly matching ABC docstring from Plan 01"

patterns-established:
  - "Stub model pattern: minimal ConditionalGenerativeModel subclass with fit() storing columns and sample() returning zero-filled DataFrame of correct shape — enables orchestrator integration tests without real training"
  - "Model class injection pattern: Synthesizer(metadata, privacy, model=CustomModel) routes model_cls through to StagedOrchestrator unchanged"

requirements-completed: [MODEL-02, MODEL-03]

# Metrics
duration: 3min
completed: 2026-02-23
---

# Phase 03 Plan 02: Synthesizer model= API and MODEL-03 StubModel Integration Test Summary

**Synthesizer.model: Type[ConditionalGenerativeModel] = CTGAN replaces backend: str, forwarded to StagedOrchestrator; StubModel integration test proves any ConditionalGenerativeModel subclass routes end-to-end through the multi-table pipeline**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-23T03:16:59Z
- **Completed:** 2026-02-23T03:19:55Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Removed `backend: str = "CTGAN"` from `Synthesizer.__init__` — replaced with `model: Type[ConditionalGenerativeModel] = CTGAN`
- `self.model_cls = model` stored; `StagedOrchestrator` constructed with `model_cls=self.model_cls`
- `sample()` print statement now uses `self.model_cls.__name__` — accurate for any injected model
- `StubModel` integration test (MODEL-03) proves StagedOrchestrator routes fit/generate through arbitrary ConditionalGenerativeModel subclass
- 4 new tests all pass: stub routing, model= acceptance, CTGAN default, issubclass guard

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace backend: str with model parameter in Synthesizer and forward to StagedOrchestrator** - `5a4b2cb` (feat)
2. **Task 2: Write StubModel integration test for MODEL-03 ABC contract verification** - `1bc18d1` (feat)

## Files Created/Modified
- `syntho_hive/interface/synthesizer.py` — Added `Type` to typing imports; added `ConditionalGenerativeModel` and `CTGAN` imports; replaced `backend: str` with `model: Type[ConditionalGenerativeModel] = CTGAN` in signature; updated docstring with supported classes and constructor convention; replaced `self.backend = backend` with `self.model_cls = model`; updated `StagedOrchestrator` construction to pass `model_cls=self.model_cls`; updated `sample()` print to use `self.model_cls.__name__`
- `syntho_hive/tests/test_interface.py` — Appended `StubModel` class, `_MockSparkDF` helper, and 4 new test functions: `test_stub_model_routes_through_pipeline`, `test_synthesizer_accepts_model_parameter`, `test_synthesizer_default_model_is_ctgan`, `test_issubclass_guard_rejects_invalid_model_cls`

## Decisions Made
- Type import added to synthesizer.py alongside ConditionalGenerativeModel and CTGAN imports — keeps all model-injection imports co-located
- StagedOrchestrator constructed with `model_cls=self.model_cls` — end-to-end injection confirmed by StubModel integration test
- `sample()` uses `self.model_cls.__name__` instead of hardcoded string so progress prints remain accurate for any plugged-in model
- StubModel uses constructor convention `(metadata, batch_size, epochs, **kwargs)` exactly matching ABC docstring from Plan 01 — validates the docstring is actionable

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

4 pre-existing test failures (`test_synthesizer_fit_requires_spark`, `test_synthesizer_sample_requires_spark`, `test_synthesizer_fit_call`, `test_synthesizer_sample_call`) were present before and after changes — confirmed by `git stash` verification. These failures pre-date Plan 03-01 and are out of scope. They are documented as pre-existing in the 03-01 SUMMARY.

## Next Phase Readiness
- MODEL-01, MODEL-02, MODEL-03 all satisfied and verified by passing tests
- Any ConditionalGenerativeModel subclass (e.g., TVAE from Phase 04) can now be injected into Synthesizer via `model=TVAEModel`
- The StubModel pattern established in test_interface.py serves as a reference implementation for future model integration tests

## Self-Check: PASSED

- FOUND: syntho_hive/interface/synthesizer.py
- FOUND: syntho_hive/tests/test_interface.py
- FOUND: .planning/phases/03-model-pluggability/03-02-SUMMARY.md
- FOUND commit: 5a4b2cb (Task 1 - Synthesizer model= parameter)
- FOUND commit: 1bc18d1 (Task 2 - StubModel integration tests)

---
*Phase: 03-model-pluggability*
*Completed: 2026-02-23*
