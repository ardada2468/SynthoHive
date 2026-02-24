# Phase 6: Synthesizer Validation Hardening - Research

**Researched:** 2026-02-23
**Domain:** Python class validation, public facade wiring, fail-fast guard patterns
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REL-03 | FK type mismatches raised at `validate_schema()` time, before training begins | `validate_schema(real_data=dfs)` already collects all FK type mismatch errors — gap is purely the call site in `Synthesizer.fit()` passing no `real_data` argument (line 105); one-line fix |
| MODEL-02 | `Synthesizer` exposes `model` parameter with documented supported classes | `Synthesizer.__init__()` accepts `model=` and stores `self.model_cls`, but the issubclass guard only fires when `spark_session` is provided (via `StagedOrchestrator.__init__`); when `spark_session=None` the orchestrator is never constructed and the guard never runs; fix is one guard block before the Spark check |
</phase_requirements>

---

## Summary

Phase 6 closes two "partial wiring" gaps in the `Synthesizer` public facade that were catalogued as TD-01 and TD-04 in the v1.1 milestone audit. Both gaps were identified precisely: the audit named the file, the line, and the exact fix required. No new libraries, no new abstractions, and no architectural decisions are needed — this phase is a surgical two-line addition to a single file.

**TD-01 (REL-03):** `Synthesizer.fit()` at line 105 calls `self.metadata.validate_schema()` with no arguments. The method already accepts `real_data: Optional[Dict[str, pd.DataFrame]]` and performs data-level FK type mismatch detection when provided. The fix is to load the DataFrames that `fit()` already resolves from the `data` argument and pass them as `real_data=dfs` before calling `fit_all()`. The path-to-DataFrame loading must happen before the `validate_schema()` call, which requires restructuring the path resolution block (lines 111–118) to run before the validate block (lines 104–105), or passing the already-resolved dict.

**TD-04 (MODEL-02):** `Synthesizer.__init__()` at lines 67–70 only constructs `StagedOrchestrator` when `spark_session` is provided. The issubclass guard lives inside `StagedOrchestrator.__init__()` (lines 81–86 of `orchestrator.py`). When `spark_session=None`, `self.orchestrator = None` is set and the guard never fires. The fix is one guard block at the top of `Synthesizer.__init__()`, before the Spark check, mirroring the exact message and check already in `StagedOrchestrator`.

**Primary recommendation:** Add the issubclass guard to `Synthesizer.__init__()` first (simpler, no data loading), then fix the `validate_schema` call site to pass loaded DataFrames as `real_data`.

---

## Standard Stack

No new libraries required. All tools are already in the project.

### Core (already in use)
| Component | Version | Purpose | Where Used |
|-----------|---------|---------|------------|
| `syntho_hive.core.models.base.ConditionalGenerativeModel` | in-repo | ABC for model type checking | `orchestrator.py:81`, `synthesizer.py:14` |
| `syntho_hive.exceptions.SchemaValidationError` | in-repo | Exception raised on FK mismatch | `config.py:162`, exported from `__init__.py` |
| `syntho_hive.interface.config.Metadata.validate_schema` | in-repo | FK structural + data-level check | called at `synthesizer.py:105` |
| `pytest` | project test suite | Test framework for verification | `test_interface.py` |

### No New Dependencies
This phase adds zero new packages. All required abstractions exist.

---

## Architecture Patterns

### Current State: What Is Broken and Why

#### TD-01 Call Site (`synthesizer.py:104–105`)

```python
# CURRENT (broken) — structural-only, no data-level FK checks
if validate:
    self.metadata.validate_schema()   # ← no real_data argument
```

The `validate_schema` method signature (from `config.py:97`):
```python
def validate_schema(self, real_data: Optional[Dict[str, "pd.DataFrame"]] = None) -> None:
```

When `real_data=None` (the default), only structural checks run (FK format validity, parent table existence). The data-level block at `config.py:133` is gated on `if real_data is not None:`, so FK type mismatch detection — the core of REL-03 — is silently skipped through the Synthesizer facade.

The DataFrames ARE available in `fit()`: after line 111–118, `real_paths` is a `Dict[str, str]` (path strings, not DataFrames). The fix requires either:
- Option A: Load DataFrames from paths before calling `validate_schema`, then pass `real_data=loaded_dfs`
- Option B: Call `validate_schema(real_data=data)` when `data` is already a dict of DataFrames

Looking at the `fit()` signature: `data: Any  # Str (database name) or Dict[str, str] (table paths)`. The `data` parameter is paths, not DataFrames. To pass real DataFrames to `validate_schema`, the code must either load them from Spark (expensive, requires orchestrator) or accept that `validate=True` only works after data is already loaded.

**Correct approach:** The simplest correct fix that doesn't require a Spark read is to pass `data` itself as `real_data` only when `data` is already a dict (user-provided DataFrames). However, looking at the type signature, `data` is always paths, not DataFrames.

**Re-reading the audit description:** "Fix: Add `self.metadata.validate_schema(real_data)` call in `Synthesizer.fit()`, passing the loaded DataFrames before `self.orchestrator.fit_all()`." — the audit assumes DataFrames will be loaded. Given the Spark context is required anyway (line 98 checks `self.orchestrator`), loading them via `self.orchestrator.io.read_dataset()` is feasible. But this is a heavy pre-validation load.

**Pragmatic fix:** The correct interpretation for `validate=True` is: when the user passes a `Dict[str, pd.DataFrame]` directly as `data`, validate with those DataFrames. When `data` is a string (database name), validate schema structurally only (no DataFrames in scope without a read). This avoids loading all tables twice. The fix should handle both cases:

```python
if validate:
    if isinstance(data, dict) and all(isinstance(v, pd.DataFrame) for v in data.values()):
        self.metadata.validate_schema(real_data=data)
    else:
        self.metadata.validate_schema()
```

However, looking at existing tests in `test_interface.py`, `syn.fit("test_db", sample_size=100)` passes a string — not DataFrames. For the test suite to exercise the data-level path, `fit()` would need to accept DataFrames directly.

**Cleanest fix that satisfies the audit criterion exactly:** Accept `data` as `Dict[str, pd.DataFrame]` when `validate=True` and the user wants data-level checking. The `fit()` method already handles `data: Any` — it checks for `str` or `dict`. Extend to detect `dict` of DataFrames vs dict of path strings. When DataFrames detected, pass them as `real_data`. This preserves backward compatibility.

#### TD-04 Guard Location (`synthesizer.py:62–70`)

```python
# CURRENT (broken) — guard fires only when spark_session provided
self.model_cls = model
self.embedding_threshold = embedding_threshold

if self.spark:
    self.orchestrator = StagedOrchestrator(metadata, self.spark, model_cls=self.model_cls)
    # ↑ issubclass guard fires here (inside StagedOrchestrator.__init__)
else:
    self.orchestrator = None  # guard never fires
```

#### Target State After Fix

```python
# FIXED TD-04: guard fires unconditionally at __init__ time
if not (isinstance(model, type) and issubclass(model, ConditionalGenerativeModel)):
    raise TypeError(
        f"model_cls must be a subclass of ConditionalGenerativeModel, "
        f"got {model!r}. Implement fit(), sample(), save(), load() "
        f"and subclass ConditionalGenerativeModel."
    )

self.model_cls = model
self.embedding_threshold = embedding_threshold

if self.spark:
    self.orchestrator = StagedOrchestrator(metadata, self.spark, model_cls=self.model_cls)
else:
    self.orchestrator = None
```

```python
# FIXED TD-01: pass real_data when available
if validate:
    if isinstance(data, dict) and data and isinstance(next(iter(data.values())), pd.DataFrame):
        self.metadata.validate_schema(real_data=data)
    else:
        self.metadata.validate_schema()
```

### Exact issubclass Guard Pattern (from `orchestrator.py:81–86`)

```python
# Source: syntho_hive/relational/orchestrator.py lines 81-86
if not issubclass(model_cls, ConditionalGenerativeModel):
    raise TypeError(
        f"model_cls must be a subclass of ConditionalGenerativeModel, "
        f"got {model_cls!r}. Implement fit(), sample(), save(), load() "
        f"and subclass ConditionalGenerativeModel."
    )
```

The Synthesizer version must wrap in `isinstance(model, type)` first because unlike `StagedOrchestrator` (which gets `model_cls: Type[ConditionalGenerativeModel]` typed), the Synthesizer `model` parameter might receive a non-class (e.g., an instance). `issubclass()` raises `TypeError` itself if passed a non-class — catching that is correct behavior but the guard should be explicit.

### validate_schema Full Signature (from `config.py:97–162`)

```python
def validate_schema(self, real_data: Optional[Dict[str, "pd.DataFrame"]] = None) -> None:
    """
    When real_data is None: structural checks only (FK format, parent table existence)
    When real_data is provided: also checks column existence and dtype compatibility
    Raises SchemaValidationError collecting ALL errors before raising
    """
```

The data-level block (`config.py:133–159`) runs only when `real_data is not None` AND both the child and parent table are present in the dict. Partial data dicts (not all tables present) skip those FK pairs silently.

### Exception Hierarchy (from `exceptions.py`)

```
SynthoHiveError
├── SchemaError
│   └── SchemaValidationError   ← raised by validate_schema()
├── TrainingError               ← wraps all fit() exceptions
├── SerializationError
└── ConstraintViolationError
```

**Critical interaction:** `Synthesizer.fit()` has a catch-all at lines 122–128:
```python
except SynthoHiveError:
    raise   # ← SchemaValidationError passes through unchanged
except Exception as exc:
    raise TrainingError(...)
```

`SchemaValidationError` is a `SynthoHiveError` subclass, so it re-raises directly without being wrapped in `TrainingError`. The planner does NOT need to change the exception wrapping logic.

**For TD-04:** `TypeError` is NOT a `SynthoHiveError`, so if it escapes `__init__()` it will propagate as a raw `TypeError`. The guard in `__init__()` should raise `TypeError` directly (not a `SynthoHiveError` subclass) — this matches the existing `StagedOrchestrator` behavior and is the correct signal for type contract violations.

### Test Contract: What Needs Updating

Existing test at `test_interface.py:200–212`:
```python
def test_issubclass_guard_rejects_invalid_model_cls():
    """MODEL-01: StagedOrchestrator raises TypeError for non-ConditionalGenerativeModel class."""
    with pytest.raises(TypeError, match="ConditionalGenerativeModel"):
        StagedOrchestrator(metadata=meta, model_cls=NotAModel)
```

This tests the orchestrator-level guard. A NEW test is needed for the Synthesizer-level guard (TD-04):
```python
def test_synthesizer_rejects_invalid_model_cls_without_spark():
    """TD-04: Synthesizer raises TypeError at __init__ even when spark_session=None."""
    class NotAModel:
        pass
    with pytest.raises(TypeError, match="ConditionalGenerativeModel"):
        Synthesizer(metadata, privacy_config, spark_session=None, model=NotAModel)
```

For TD-01, a test needs to verify that `fit(validate=True, data=dfs)` where `dfs` contains type-mismatched FK columns raises `SchemaValidationError`:
```python
def test_synthesizer_fit_validate_catches_fk_type_mismatch():
    """TD-01: fit(validate=True, data=DataFrames) raises SchemaValidationError on FK mismatch."""
    ...
    with pytest.raises(SchemaValidationError):
        syn.fit(data={"users": users_df, "orders": orders_df}, validate=True)
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FK type compatibility checking | Custom dtype comparison logic | `_dtypes_compatible()` in `config.py:38–64` | Already handles numpy kind codes, pandas extension types, float/int/string compatibility |
| Collecting multiple validation errors | Manual error list accumulation | `validate_schema()` collect-all pattern already implemented | Method already collects all errors into a list before raising SchemaValidationError |
| Model class type checking | Custom `hasattr` duck-type checks | `issubclass(model, ConditionalGenerativeModel)` | Direct ABC subclass check; mirrors existing guard in orchestrator exactly |
| Error message for invalid model | New message string | Copy exact message from `orchestrator.py:82–85` | Consistent user experience; same error from both paths |

**Key insight:** Every mechanism needed already exists. The phase is wiring existing components, not building new ones.

---

## Common Pitfalls

### Pitfall 1: issubclass() Crash on Non-Class Input

**What goes wrong:** `issubclass(42, ConditionalGenerativeModel)` raises `TypeError: issubclass() arg 1 must be a class`. If the user passes a model instance instead of a class, the guard itself crashes with a confusing error rather than a clear validation message.

**Why it happens:** `issubclass` requires both arguments to be types. The `StagedOrchestrator` signature uses `Type[ConditionalGenerativeModel]` which type checkers enforce, but runtime has no such enforcement.

**How to avoid:** Use `isinstance(model, type) and issubclass(model, ConditionalGenerativeModel)` — the `isinstance(model, type)` check short-circuits before `issubclass` is called if `model` is not a class at all.

**Warning signs:** If the guard is written as `not issubclass(model, ...)` without the type check, passing `model=CTGAN()` (an instance) will produce `TypeError: issubclass() arg 1 must be a class` instead of the intended validation message.

### Pitfall 2: validate_schema() Called Before Path Resolution

**What goes wrong:** If `validate_schema(real_data=data)` is called before the `isinstance(data, str)` / `isinstance(data, dict)` block, the `data` variable may be a string (database name), and passing a string as `real_data` would cause `validate_schema` to fail when accessing `real_data[table_name]`.

**Why it happens:** The `fit()` method resolves `data` into `real_paths` (always a dict of path strings) in lines 111–118. The validate block is currently at lines 104–105 — before path resolution.

**How to avoid:** Either (a) move the validate call to after path resolution and pass `real_paths` (but these are strings, not DataFrames), or (b) detect DataFrames in `data` before the existing validate block. Option (b) is cleaner: check `isinstance(data, dict) and isinstance(next(iter(data.values())), pd.DataFrame)` to determine if the user passed actual DataFrames.

**Warning signs:** A `KeyError` or `AttributeError` inside `validate_schema` after passing a string-keyed dict of paths.

### Pitfall 3: SchemaValidationError Gets Wrapped as TrainingError

**What goes wrong:** `SchemaValidationError` raised inside the `validate` block (lines 104–105) could be caught by the outer `except Exception as exc:` and re-raised as `TrainingError`.

**Why it doesn't actually happen here:** The existing `except SynthoHiveError: raise` block (line 122) re-raises `SynthoHiveError` subclasses unchanged. `SchemaValidationError` inherits from `SchemaError → SynthoHiveError`, so it passes through the first except clause.

**Verification:** Confirm that the `SchemaValidationError` raised in `validate_schema()` is not wrapped. The exception hierarchy confirms: `SchemaValidationError → SchemaError → SynthoHiveError` — caught by `except SynthoHiveError: raise` at line 122. No change needed.

### Pitfall 4: Duplicate Guard (Synthesizer + Orchestrator) Double-Fires

**What goes wrong:** When `spark_session` is provided, `Synthesizer.__init__()` will check the guard AND then `StagedOrchestrator.__init__()` will check it again. This is harmless (same result, same error) but wastes a tiny amount of work.

**Why it happens:** Both guards are now in place. The Synthesizer guard fires unconditionally; the Orchestrator guard fires when Spark is present.

**How to handle:** This is acceptable behavior — duplicate validation is not a bug. The guard is cheap (a single `issubclass` call). Do not remove the Orchestrator guard; it protects against direct `StagedOrchestrator` construction without going through `Synthesizer`.

### Pitfall 5: Test for ValueError Still Present (TD-02, Out of Scope)

**What goes wrong:** The test at `test_interface.py:49` expects `pytest.raises(ValueError, match="SparkSession required")` but `Synthesizer.fit()` wraps all non-`SynthoHiveError` exceptions as `TrainingError`. The `ValueError("SparkSession required")` at line 99 is caught by the outer `except Exception as exc:` and re-raised as `TrainingError`.

**Why it matters for Phase 6:** Adding the issubclass guard at `__init__` time means the test at line 49 will still fail for a different reason (it's about the Spark requirement, not the model class). Phase 6 does NOT fix TD-02 (that's Phase 7). Be careful not to accidentally change the exception wrapping behavior for `ValueError("SparkSession required")`.

**How to avoid:** The new guard fires at `__init__` time, not inside `fit()`. The existing `fit()` ValueError for missing Spark will remain wrapped as `TrainingError` — that is Phase 7's problem. Phase 6 only adds the guard before Spark check, which is at `__init__` time, before any try/except block.

---

## Code Examples

Verified patterns from project source:

### Existing issubclass Guard (Source: `syntho_hive/relational/orchestrator.py:81–86`)

```python
if not issubclass(model_cls, ConditionalGenerativeModel):
    raise TypeError(
        f"model_cls must be a subclass of ConditionalGenerativeModel, "
        f"got {model_cls!r}. Implement fit(), sample(), save(), load() "
        f"and subclass ConditionalGenerativeModel."
    )
```

### Safe issubclass Guard for Synthesizer (handles non-class inputs)

```python
# Source: Pattern derived from orchestrator.py guard + isinstance safety wrapper
# Place this at the start of Synthesizer.__init__(), before the self.spark assignment
if not (isinstance(model, type) and issubclass(model, ConditionalGenerativeModel)):
    raise TypeError(
        f"model_cls must be a subclass of ConditionalGenerativeModel, "
        f"got {model!r}. Implement fit(), sample(), save(), load() "
        f"and subclass ConditionalGenerativeModel."
    )
```

### validate_schema with real_data (Source: `syntho_hive/interface/config.py:97`)

```python
# Full data-level check (FK types, column existence, dtype compatibility)
self.metadata.validate_schema(real_data=dfs)  # dfs: Dict[str, pd.DataFrame]

# Structural-only check (FK format, parent table existence)
self.metadata.validate_schema()  # or real_data=None
```

### Detecting DataFrames vs Path Strings in fit() data argument

```python
# Place before orchestrator.fit_all() call, within the validate block
if validate:
    if isinstance(data, dict) and data and isinstance(next(iter(data.values())), pd.DataFrame):
        # User passed actual DataFrames — data-level FK type checks possible
        self.metadata.validate_schema(real_data=data)
    else:
        # String (DB name) or dict of path strings — structural checks only
        self.metadata.validate_schema()
```

### Exception Hierarchy Verification (Source: `syntho_hive/exceptions.py`)

```python
# SchemaValidationError propagates unchanged through fit() exception handler:
# except SynthoHiveError: raise   ← SchemaValidationError hits this (it IS a SynthoHiveError)
# except Exception as exc: raise TrainingError(...)  ← SchemaValidationError NEVER hits this
assert issubclass(SchemaValidationError, SchemaError)     # True
assert issubclass(SchemaValidationError, SynthoHiveError)  # True
```

### Test Pattern for TD-04 (New Test Required)

```python
def test_synthesizer_rejects_invalid_model_cls_without_spark(metadata, privacy_config):
    """TD-04 fix verification: TypeError fires at __init__ even without SparkSession."""
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="ConditionalGenerativeModel"):
        Synthesizer(metadata, privacy_config, spark_session=None, model=NotAModel)
```

### Test Pattern for TD-01 (New Test Required)

```python
def test_synthesizer_fit_validate_catches_fk_type_mismatch(metadata, privacy_config):
    """TD-01 fix verification: fit(validate=True, data=dfs) raises SchemaValidationError."""
    # users.user_id is int, orders.user_id is str — type mismatch
    users_df = pd.DataFrame({"user_id": [1, 2, 3], "name": ["a", "b", "c"]})
    orders_df = pd.DataFrame({"order_id": [10, 11], "user_id": ["1", "2"]})  # str FK

    syn = Synthesizer(metadata, privacy_config, spark_session=None)
    with pytest.raises(SchemaValidationError):
        syn.fit(data={"users": users_df, "orders": orders_df}, validate=True)
```

Note: This test requires `fit()` to not immediately fail on `not self.orchestrator` before reaching the validate block. The validate block currently comes AFTER the orchestrator check (line 98–99). The planner must decide the call order: validate can run without Spark (it operates on in-memory DataFrames via `metadata.validate_schema`), so moving it before the Spark check is correct and enables this test.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `validate_schema()` always structural-only | `validate_schema(real_data=dfs)` for data-level checks | Phase 02 (Plan 02-01) | Data-level FK type checking now exists — just not wired at Synthesizer level |
| No issubclass guard anywhere | Guard in StagedOrchestrator.__init__ | Phase 03 (Plan 03-01) | Guard fires when Spark provided — gap when Spark=None |
| CTGAN hardcoded | `model_cls` parameter injected | Phase 03 | Any ConditionalGenerativeModel subclass works |

**Not deprecated:** Nothing in Phase 6 removes or deprecates existing behavior.

---

## Open Questions

1. **Should `fit()` accept `Dict[str, pd.DataFrame]` as the `data` argument explicitly?**
   - What we know: The current `data: Any` type annotation covers it, and the type check in the fix can detect DataFrames
   - What's unclear: Whether allowing DataFrame input to `fit()` should be formally documented or just handled silently
   - Recommendation: Update the `data` parameter docstring to document `Dict[str, pd.DataFrame]` as a valid input when `validate=True` is used for data-level checking; this makes the feature discoverable

2. **Should the validate block move before or after the Spark check?**
   - What we know: Current order is: Spark check → validate → path resolution → fit_all. With DataFrames as input, validate doesn't need Spark at all.
   - What's unclear: Should `validate=True` with a string `data` argument still validate (structurally) before failing on Spark?
   - Recommendation: Move validate before the Spark check. Structural validation (`real_data=None`) is always Spark-free. Data-level validation (DataFrames) is also Spark-free. This gives the best user experience: schema errors are surfaced even without a Spark session.

3. **Error message consistency between Synthesizer and Orchestrator guards?**
   - What we know: The audit says "copy exact message from orchestrator" — both should say the same thing
   - What's unclear: Whether to use `model` or `model_cls` in the message (Synthesizer parameter is `model`, Orchestrator parameter is `model_cls`)
   - Recommendation: Use `model_cls` in the message body for both, since that's the conceptual name in the error message text. The `{model!r}` repr will show the actual value passed.

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `syntho_hive/interface/synthesizer.py` — current state, both gap locations
- Direct code inspection: `syntho_hive/interface/config.py` — `validate_schema()` full signature and implementation
- Direct code inspection: `syntho_hive/relational/orchestrator.py:81–86` — existing guard pattern to replicate
- Direct code inspection: `syntho_hive/exceptions.py` — exception hierarchy confirming SchemaValidationError propagation
- Direct code inspection: `syntho_hive/core/models/base.py` — ConditionalGenerativeModel ABC
- `.planning/v1.1-MILESTONE-AUDIT.md` — precise gap descriptions with file/line/fix specifications

### Secondary (MEDIUM confidence)
- `.planning/ROADMAP.md` — Phase 6 plan description confirming 1-plan scope
- `.planning/STATE.md` — Phase 3 decisions confirming issubclass guard placement rationale

### Tertiary (LOW confidence)
- None required — all facts sourced from direct code inspection

---

## Metadata

**Confidence breakdown:**
- Gap diagnosis: HIGH — audit provided file, line number, and exact fix for both TD-01 and TD-04
- Fix approach (TD-04): HIGH — direct copy of existing orchestrator guard pattern with `isinstance(model, type)` safety wrapper
- Fix approach (TD-01): MEDIUM — the detection of DataFrames vs path strings at `fit()` entry is a design choice not specified by the audit; recommendation documented with rationale
- Call ordering (validate before/after Spark check): MEDIUM — functionally equivalent but affects test coverage; moving before the Spark check is the right UX

**Research date:** 2026-02-23
**Valid until:** 2026-03-25 (stable internal codebase — no external dependencies change)
