# Phase 1: Core Reliability - Context

**Gathered:** 2026-02-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the four foundational reliability gaps that make SynthoHive unusable in production: silent error swallowing, broken model save/load, missing seed control, and constraint violations that pass silently. Also patch the SQL injection in `save_to_hive()`. Phase goal: an engineer can train, save, cold-load a model, and get predictable, debuggable behavior at every step.

No new ML capabilities, no new connectors, no relational changes — this phase is entirely about making what exists trustworthy.

</domain>

<decisions>
## Implementation Decisions

### Exception Hierarchy

- Define a custom exception hierarchy rooted at `SynthoHiveError`
- Four typed subclasses: `SchemaError` (bad metadata, missing FK, unsupported column type), `TrainingError` (NaN loss, divergence, GPU OOM), `SerializationError` (save/load failure, corrupt checkpoint, version mismatch), `ConstraintViolationError` (constraint violations, invalid output shape)
- Each exception carries: a human-readable message explaining what went wrong and why, plus the original exception chained via `__cause__` (Python's `raise ... from ...` pattern)
- Wrap at **public API boundaries only** — `Synthesizer.fit()`, `Synthesizer.sample()`, `Synthesizer.save()`, `Synthesizer.load()`, `Synthesizer.generate_validation_report()`. Internal modules (`ctgan.py`, `orchestrator.py`, `transformer.py`, etc.) raise their own typed errors freely; the Synthesizer boundary wraps them into the appropriate top-level SynthoHive type

### Serialization Contract

- Save artifact is a **directory** (e.g., `./syntho_models/customers/`), not a single file
- Directory contains separate files per component: network weights, DataTransformer state, context_transformer state, data_column_info, embedding_layers, metadata (version, schema, timestamp)
- **Default save path**: `./syntho_models/{table_name}/` — works out of the box without specifying a path; caller can override
- `save(path)` **raises `SerializationError` if the path already exists** — engineer must explicitly pass `overwrite=True` to avoid accidental loss of a trained model
- If `load()` detects a version mismatch (saved with different SynthoHive version): **warn via structlog** and attempt load — do not fail; document that schema changes between versions may cause failures
- `torch.load()` must use `weights_only=False` explicitly (required for PyTorch 2.6+)

### Constraint Violation Behavior

- Constraint checking is **opt-in**: `sample(enforce_constraints=True)` — off by default for speed
- When enabled: validate the **full generated batch**, collect all violations, then raise `ConstraintViolationError` with a summary
- Error message format: `"ConstraintViolationError: 3 violations — age: got -3 (min=0); price: got -12.50 (min=0.01); quantity: got 1001 (max=1000)"`
- **Return valid rows, warn about violations** rather than failing the entire call — partial data is returned, the warning lists the violating columns with counts so the engineer can decide whether the violation rate is acceptable
- Constraint application that previously swallowed exceptions silently (`transformer.py:222-246`) must raise `ConstraintViolationError` with column name and observed value

### Seed & Reproducibility

- **Separate seeds**: `fit(seed=42)` controls training randomness; `sample(seed=7)` controls generation randomness — independent control
- When **no seed is provided** for fit: auto-generate a random seed, log it via structlog at INFO level with the message `"Training seed: {seed}"` so the run can be reproduced if needed
- Seed is **not exposed** on the synthesizer object (no `synthesizer.seed_` attribute) — log is the single source of truth
- Seed scope: **PyTorch only** — covers `torch.manual_seed()`, `numpy.random.seed()`, and Python's `random.seed()`. Spark code paths are explicitly excluded (document this in docstrings as Spark's distributed shuffle is inherently non-deterministic)

### Claude's Discretion

- Exact structlog log level and format for training seed and constraint warning messages
- Internal error message wording for each exception type
- Whether `overwrite=True` on `save()` is a positional argument or keyword-only
- File naming convention for components within the save directory (e.g., `weights.pt`, `transformer.joblib`)

</decisions>

<specifics>
## Specific Ideas

- Exception messages should read like a data engineer wrote them — name the failing column, the table, the violated constraint value. Not `"Error in model training"` but `"TrainingError: NaN loss detected at epoch 12. Check for constant columns or extreme outliers in your training data."`
- The save directory should be human-inspectable — an engineer should be able to open the directory and understand what each file is without reading source code

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-core-reliability*
*Context gathered: 2026-02-22*
