# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-23 after v1.2 milestone started)

**Core value:** A data engineer can train on any multi-table schema, generate synthetic data, and trust that code written against that data will work on the real thing — without babysitting training or manually validating output.
**Current focus:** v1.2 Quality & Connectors — Phase 8 Training Observability

## Current Position

Phase: 8 of 11 (Training Observability) — v1.2 first phase
Plan: 1 of 2 (plan 01 complete)
Status: In progress
Last activity: 2026-02-24 — Phase 8 Plan 01 complete (tqdm + structlog training observability)

Progress: [██████████] v1.0 shipped · v1.1 shipped (phases 2, 3, 6, 7 complete, 14/14 tests passing) · v1.2 roadmap ready

## Performance Metrics

**Velocity:**
- Total plans completed: 16
- Average duration: 3.8 min
- Total execution time: ~61 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-core-reliability | 5/5 | 23 min | 4.6 min |
| 02-relational-correctness | 5/5 | 21 min | 4.2 min |
| 03-model-pluggability | 3/3 | 7 min | 2.3 min |
| 06-synthesizer-validation-hardening | 1/1 | 2 min | 2 min |
| 07-test-suite-alignment | 1/1 | 4 min | 4 min |
| 08-training-observability | 1/2 | 4 min | 4 min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Key decisions relevant to v1.2:
- [Phase 07-test-suite-alignment]: TrainingError is the exception boundary for synthesizer public API — internal ValueError is wrapped at fit()/sample() boundary
- [Phase 03-model-pluggability]: issubclass guard fires at Synthesizer.__init__() regardless of Spark session presence
- [Phase 02-relational-correctness]: structlog is already wired for structured logging throughout syntho_hive/ — use it for CORE-05 progress events
- [v1.2 scope]: Snowflake/BigQuery deferred to v1.3 — SQL connectors target Postgres + MySQL only
- [Phase 08-01]: training_complete emits sentinel values (best_epoch=-1, checkpoint_path=None) to satisfy CORE-05 independently of Plan 02 validation checkpointing
- [Phase 08-01]: progress_bar flag controls only tqdm visual bar (disable=not progress_bar); structlog events always fire unconditionally — independent observability channels
- [Phase 08-01]: Three params (progress_bar, checkpoint_interval, checkpoint_dir) threaded explicitly at each API boundary rather than via **kwargs per CONTEXT.md recommendation

### Pending Todos

None.

### Blockers/Concerns

- [Phase 10 prereq]: Postgres integration test (TEST-04) requires a live Postgres instance — dev environment setup needed before Phase 10 plan execution
- [v1.1 carry-forward]: REL-03 partial wiring: validate_schema() data-level checks require explicit data= argument — documented, not enforced at API boundary
- [v1.1 carry-forward]: Production-scale FK chain test (10k+ rows) outstanding — zero-orphan guarantee unverified at realistic dataset sizes

## Session Continuity

Last session: 2026-02-24
Stopped at: Completed 08-01-PLAN.md — tqdm + structlog training observability wired into CTGAN.fit(), parameters threaded through Synthesizer and StagedOrchestrator
Resume file: None
