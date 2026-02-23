# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22 after v1.0 milestone)

**Core value:** A data engineer can train on any multi-table schema, generate synthetic data, and trust that code written against that data will work on the real thing — without babysitting training or manually validating output.
**Current focus:** v1.0 archived — planning next milestone (v1.1 Relational Correctness)

## Current Position

Phase: 02-relational-correctness — Plan 04 complete (all 4 plans complete)
Status: Phase 02 complete. Plan 02-04 (TEST-02 FK chain integrity suite, REL-05 and TEST-02 fulfilled) complete.
Last activity: 2026-02-23 - Completed 02-04: TestFKChainIntegrity — zero-orphan FK chain tests, schema validation tests, cardinality accuracy test

Progress: [████████░░] v1.0 shipped · v1.1 phase 02 complete (4/4 plans)

## Performance Metrics

**v1.0 Velocity:**
- Total plans completed: 5
- Average duration: 4.6 min
- Total execution time: 0.38 hours (23 min)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-core-reliability | 5/5 | 23 min | 4.6 min |
| 02-relational-correctness | 4/4 | 19 min | 4.75 min |

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

## Session Continuity

Last session: 2026-02-23
Stopped at: Completed 02-04-PLAN.md (TEST-02 FK chain integrity suite, REL-05 and TEST-02 fulfilled, phase 02 complete)
Resume file: None
