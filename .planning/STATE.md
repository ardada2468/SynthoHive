# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22 after v1.0 milestone)

**Core value:** A data engineer can train on any multi-table schema, generate synthetic data, and trust that code written against that data will work on the real thing — without babysitting training or manually validating output.
**Current focus:** v1.0 archived — planning next milestone (v1.1 Relational Correctness)

## Current Position

Phase: 06-synthesizer-validation-hardening — Plan 01 complete (1/1 plans)
Status: Phase 06 plan 01 complete. TD-04 issubclass guard and TD-01 real_data passthrough wired in Synthesizer facade. REL-03 and MODEL-02 E2E flows fixed.
Last activity: 2026-02-23 - Completed plan 06-01: TD-04 and TD-01 synthesizer wiring gaps closed with regression tests

Progress: [██████████] v1.0 shipped · v1.1 phase 02 complete · phase 03 complete (MODEL-01, MODEL-02, MODEL-03 all verified) · phase 06 plan 01 complete (REL-03, MODEL-02 E2E fixed)

## Performance Metrics

**v1.0 Velocity:**
- Total plans completed: 5
- Average duration: 4.6 min
- Total execution time: 0.38 hours (23 min)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-core-reliability | 5/5 | 23 min | 4.6 min |
| 02-relational-correctness | 5/5 | 21 min | 4.2 min |
| 03-model-pluggability | 3/3 | 7 min | 2.3 min |
| 06-synthesizer-validation-hardening | 1/1 | 2 min | 2 min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

**02-01 decisions:**
- SchemaValidationError subclasses SchemaError for backward compat with existing except SchemaError handlers
- validate_schema() uses collect-all pattern (not fail-fast) — engineers see all schema problems at once
- validate_schema(real_data=None) stays backward compatible; no-arg callers get original structural checks
- linkage_method defaults to 'empirical' per-table; 'negbinom' available for statistical cardinality fit
- _dtypes_compatible returns True for pandas extension types to avoid false positives
- [Phase 02-relational-correctness]: legacy_context_conditioning defaults to False; old behavior requires explicit True for backwards compatibility
- [Phase 02-relational-correctness]: StagedOrchestrator accepts optional io= for unit testing without live Spark; on_write_failure defaults to raise
- [Phase 02-relational-correctness]: pyproject.toml Spark pins capped at <5.0.0 to prevent delta-spark major version incompatibility
- [Phase 02-relational-correctness]: Empirical resampler (numpy.random.choice) is the default LinkageModel method — draws directly from observed child counts, guaranteed non-negative
- [Phase 02-relational-correctness]: NegBinom falls back to empirical with structlog WARNING when variance <= mean; silent except/fallback block removed from linkage.py
- [Phase 02-relational-correctness]: test_relational.py is the canonical location for relational tests (inside package, not top-level tests/)
- [Phase 02-relational-correctness]: Zero-orphan check uses inner join row count == child table row count to catch exact counts (not set inclusion)
- [Phase 02-relational-correctness]: No new decisions — gap-closure fix only; test assertions updated to match SchemaValidationError hierarchy introduced in Plan 01

**03-01 decisions:**
- model_cls defaults to CTGAN — existing callers that don't pass model_cls see identical behavior, no breaking change
- issubclass() guard fires at __init__ time, not fit_all() — fail-fast pattern surfaces misconfiguration before any data is loaded
- CTGAN import stays in orchestrator.py as the default value for model_cls; it is no longer used as a constructor call site

**03-02 decisions:**
- Type import added to synthesizer.py alongside ConditionalGenerativeModel and CTGAN imports — self.model_cls = model stored in __init__
- StagedOrchestrator constructed with model_cls=self.model_cls — end-to-end injection confirmed
- sample() uses self.model_cls.__name__ — progress prints accurate for any plugged-in model
- StubModel constructor matches ABC convention (metadata, batch_size, epochs, **kwargs) — validates docstring from Plan 01 is actionable

**03-03 decisions:**
- addopts = "--import-mode=importlib" (not importmode = "importlib") — pytest 9.0.2 does not recognize importmode as a valid ini_options key; addopts is the correct approach
- importlib mode activates editable install finder, ensuring source tree is loaded instead of stale site-packages copy shadowing Phase 03 changes

**06-01 decisions:**
- issubclass guard added in Synthesizer.__init__() before self.metadata — fires unconditionally regardless of spark_session presence; isinstance(model,type) added before issubclass() to safely handle non-class inputs
- TD-01: validate block moved before orchestrator check in fit(); real_data=data passed only when data is dict of DataFrames — preserves backward compat for string/path-dict callers

### Pending Todos

None.

### Blockers/Concerns (carry to v1.1)

- [Phase 2 prereq]: `pip install -e .` required before Phase 2 — stale `.venv` produces test failures when PYTHONPATH is not set
- [Phase 2]: Pandas 2.x copy-on-write semantics may affect `transformer.py` `.values` mutations — RESOLVED by quick task 1 (all .values replaced with .to_numpy())
- [Phase 3]: TVAE architecture (encoder/decoder, KL-divergence, reparameterization) warrants `/gsd:research-phase` before implementation
- [Phase 5]: SQLAlchemy dialect-specific behavior for Snowflake and BigQuery warrants `/gsd:research-phase` before implementation

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Fix ArrowStringArray reshape NotImplementedError on Python 3.11 | 2026-02-23 | 217e67b | [1-fix-arrowstringarray-reshape-notimplemen](./quick/1-fix-arrowstringarray-reshape-notimplemen/) |
| 2 | Close two wiring gaps: export SchemaValidationError + wire validate_schema in fit() | 2026-02-22 | 7a699f5 | [2-close-2-wiring-gaps-add-schemavalidation](./quick/2-close-2-wiring-gaps-add-schemavalidation/) |

## Session Continuity

Last session: 2026-02-23
Stopped at: Completed plan 06-01 (TD-04 issubclass guard + TD-01 real_data passthrough; REL-03 and MODEL-02 E2E flows fixed)
Resume file: None
