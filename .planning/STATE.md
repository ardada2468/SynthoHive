# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22 after v1.0 milestone)

**Core value:** A data engineer can train on any multi-table schema, generate synthetic data, and trust that code written against that data will work on the real thing — without babysitting training or manually validating output.
**Current focus:** v1.0 archived — planning next milestone (v1.1 Relational Correctness)

## Current Position

Phase: 02-relational-correctness — Plan 01 complete
Status: Executing v1.1 Relational Correctness. Plan 02-01 (SchemaValidationError + validate_schema) complete.
Last activity: 2026-02-23 - Completed 02-01: SchemaValidationError exception hierarchy and collect-all FK validation

Progress: [████░░░░░░] v1.0 shipped · v1.1 plan 01/N complete

## Performance Metrics

**v1.0 Velocity:**
- Total plans completed: 5
- Average duration: 4.6 min
- Total execution time: 0.38 hours (23 min)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-core-reliability | 5/5 | 23 min | 4.6 min |
| 02-relational-correctness | 1/N | 3 min | 3 min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

**02-01 decisions:**
- SchemaValidationError subclasses SchemaError for backward compat with existing except SchemaError handlers
- validate_schema() uses collect-all pattern (not fail-fast) — engineers see all schema problems at once
- validate_schema(real_data=None) stays backward compatible; no-arg callers get original structural checks
- linkage_method defaults to 'empirical' per-table; 'negbinom' available for statistical cardinality fit
- _dtypes_compatible returns True for pandas extension types to avoid false positives

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
Stopped at: Completed 02-01-PLAN.md (SchemaValidationError + validate_schema collect-all + linkage_method)
Resume file: None
