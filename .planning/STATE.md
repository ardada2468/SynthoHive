# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** A data engineer can train on any multi-table schema, generate synthetic data, and trust that code written against that data will work on the real thing — without babysitting training or manually validating output.
**Current focus:** Phase 1 — Core Reliability

## Current Position

Phase: 1 of 5 (Core Reliability)
Plan: 3 of 4 in current phase
Status: In progress
Last activity: 2026-02-22 — Completed 01-03 (seed control + enforce_constraints in CTGAN)

Progress: [███░░░░░░░] 15%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 5 min
- Total execution time: 0.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-core-reliability | 3/4 | 14 min | 5 min |

**Recent Trend:**
- Last 5 plans: 01-01 (5 min), 01-02 (5 min), 01-03 (4 min)
- Trend: stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Fix existing codebase rather than rewrite — architecture is sound, bugs are specific and locatable
- [Roadmap]: CTGAN as default model; pluggable strategy deferred to Phase 3 after CTGAN is reliable
- [Roadmap]: SQL connectors deferred to Phase 5 — pure addition; needs stable pipeline to be meaningful
- [Phase 1 prereq]: joblib (already installed transitively) chosen for full CTGAN checkpoint serialization over cloudpickle/dill
- [01-01]: raise...from exc used throughout so callers always see chained tracebacks with root cause
- [01-01]: transformer.py column cast failures log at WARNING (not raise) — single column failure should not abort whole batch
- [01-01]: Synthesizer.save()/load() use joblib for full-object persistence (not just state_dict)
- [01-01]: torch.load() now passes weights_only=True — addressed PyTorch 2.6+ blocker
- [01-02]: joblib chosen for DataTransformer, context_transformer, embedding_layers, data_column_info serialization — handles sklearn objects and nn.ModuleDict via pickle
- [01-02]: weights_only=False required for torch.load() in CTGAN.load() — loading custom objects from joblib-serialized nn.ModuleDict requires this flag on PyTorch 2.6+
- [01-02]: _build_model() overwrites data_column_info and embedding_layers; load() must restore from joblib after _build_model() to preserve trained embedding weights
- [01-02]: Version mismatch logs structlog WARNING but does not raise — allows cross-version loads with user awareness
- [01-03]: Separate seeds for fit() and sample() — each independently controlled; no coupling between training and inference RNG
- [01-03]: Auto-generate seed in fit() when None, log at INFO so any run can be reproduced without pre-planning
- [01-03]: Per-column seed derived via hash to prevent correlated BayesianGMM fitting across columns
- [01-03]: enforce_constraints=False default preserves backward compatibility; inverse_transform() already clips values
- [01-03]: Constraint violations return partial data (valid rows) with WARNING log — caller decides if violation rate is acceptable

### Pending Todos

None.

### Blockers/Concerns

- [Phase 1]: Pandas 2.x copy-on-write semantics may affect `transformer.py` `.values` mutations — audit needed before pinning `pandas>=2.0.0`
- [Phase 3]: TVAE architecture (encoder/decoder, KL-divergence, reparameterization) warrants `/gsd:research-phase` before implementation to avoid repeating the CTGAN embedding stub pattern
- [Phase 5]: SQLAlchemy dialect-specific behavior for Snowflake and BigQuery warrants `/gsd:research-phase` before implementation

**RESOLVED:**
- ~~[Phase 1]: `torch.load()` default changed to `weights_only=True` in PyTorch 2.6+~~ — Fixed in 01-01 (cf87152)
- ~~[Phase 1]: Pre-existing test failure `test_ctgan_full_cycle` — test tried `os.remove()` on directory checkpoint~~ — Fixed in 01-02 (57aa3d5)

## Session Continuity

Last session: 2026-02-22
Stopped at: Completed 01-02-PLAN.md — directory-based CTGAN save/load serialization complete
Resume file: None
