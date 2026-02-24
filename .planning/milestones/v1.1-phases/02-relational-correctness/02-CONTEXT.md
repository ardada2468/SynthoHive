# Phase 2: Relational Correctness - Context

**Gathered:** 2026-02-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix multi-table synthesis correctness across three dimensions: (1) stale FK context conditioning during CTGAN training, (2) FK type mismatch and missing-column detection at schema validation time, and (3) memory accumulation during multi-table generation. The result is that FK columns join with zero orphans, cardinality reflects the real parent distribution, and generation stays memory-bounded. Model pluggability, quality gating, and SQL connectors are separate phases.

</domain>

<decisions>
## Implementation Decisions

### FK type mismatch errors
- Raise a new `SchemaValidationError` exception class (not reuse existing SynthoHiveError)
- `validate_schema()` catches both FK type mismatches (int vs string) AND missing FK columns in the child table
- Collect all schema problems before raising — do not fail on the first mismatch — so engineers can fix the whole schema at once
- Error message includes: the mismatched table/column pair, both detected types, and a concrete fix hint (e.g., "cast parent PK to string" or "add column X to child table Y")

### Cardinality distribution
- Replace GMM with a configurable distribution: empirical or NegBinom, selectable per-table in schema config
- Default (when not specified): empirical — always matches training data distribution exactly
- Configuration is per-table (in the schema config), not a global switch
- Stale context fix (freshly drawn parent context per training step) is always-on but includes an opt-out flag for backwards compatibility with existing workflows

### Memory-safe generation
- When a generated table is written to disk and released from memory: log at DEBUG level only (not INFO — doesn't clutter normal runs)
- On disk write failure: configurable via `on_write_failure` parameter with three modes:
  - `'raise'` — fail immediately, leave partial files on disk (default)
  - `'cleanup'` — delete all files written so far, then raise
  - `'retry'` — retry the write once before failing
- Default: `'raise'` — matches fail-fast behavior and avoids surprising cleanup

### TEST-02 coverage
- Cover a 3-table FK chain AND a 4-table chain (deeper hierarchy to catch cascade orphan issues)
- Include FK type mismatch test case (int vs string) — verifies `SchemaValidationError` is raised correctly
- Include missing FK column test case — verifies missing-column detection in `validate_schema()`
- Verify FK referential integrity (zero orphans on join) AND cardinality distribution accuracy (child counts per parent within tolerance of empirical distribution)

### Claude's Discretion
- Whether to collect-all vs fail-fast internally — decided: collect-all for better UX (user left this to Claude)
- Exact cardinality tolerance threshold for the TEST-02 cardinality check
- Retry count and delay for `on_write_failure='retry'` mode
- Exact error message wording beyond the required components

</decisions>

<specifics>
## Specific Ideas

- The opt-out flag for stale context fix should allow old workflows to keep the prior behavior while they migrate — the name and exact API surface is Claude's discretion
- `on_write_failure` parameter likely lives on `StagedOrchestrator` or wherever `output_path_base` is consumed

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-relational-correctness*
*Context gathered: 2026-02-22*
