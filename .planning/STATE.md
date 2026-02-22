# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** A data engineer can train on any multi-table schema, generate synthetic data, and trust that code written against that data will work on the real thing — without babysitting training or manually validating output.
**Current focus:** Phase 1 — Core Reliability

## Current Position

Phase: 1 of 5 (Core Reliability)
Plan: 0 of 4 in current phase
Status: Ready to plan
Last activity: 2026-02-22 — Roadmap created; phases derived from 27 v1 requirements

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Fix existing codebase rather than rewrite — architecture is sound, bugs are specific and locatable
- [Roadmap]: CTGAN as default model; pluggable strategy deferred to Phase 3 after CTGAN is reliable
- [Roadmap]: SQL connectors deferred to Phase 5 — pure addition; needs stable pipeline to be meaningful
- [Phase 1 prereq]: joblib (already installed transitively) chosen for full CTGAN checkpoint serialization over cloudpickle/dill

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1]: `torch.load()` default changed to `weights_only=True` in PyTorch 2.6+ — must be addressed in serialization fix
- [Phase 1]: Pandas 2.x copy-on-write semantics may affect `transformer.py` `.values` mutations — audit needed before pinning `pandas>=2.0.0`
- [Phase 3]: TVAE architecture (encoder/decoder, KL-divergence, reparameterization) warrants `/gsd:research-phase` before implementation to avoid repeating the CTGAN embedding stub pattern
- [Phase 5]: SQLAlchemy dialect-specific behavior for Snowflake and BigQuery warrants `/gsd:research-phase` before implementation

## Session Continuity

Last session: 2026-02-22
Stopped at: Roadmap written; REQUIREMENTS.md traceability confirmed; ready to plan Phase 1
Resume file: None
