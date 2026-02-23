# Phase 3: Model Pluggability - Research

**Researched:** 2026-02-22
**Domain:** Python ABC-based dependency injection for ML model strategy pattern
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODEL-01 | `StagedOrchestrator` accepts `model_cls` parameter — no hardcoded CTGAN in orchestration logic | Architecture Patterns §1 (DI via constructor param), Code Examples §1 (orchestrator refactor) |
| MODEL-02 | `Synthesizer` exposes `model` parameter with documented supported classes; existing callers omitting `model` get CTGAN by default | Architecture Patterns §2 (façade passthrough), Code Examples §2 (Synthesizer init) |
| MODEL-03 | Any class implementing `ConditionalGenerativeModel` ABC (`fit`, `sample`, `save`, `load`) routes correctly through the full multi-table pipeline | Standard Stack §ABC, Architecture Patterns §3 (stub model test), Common Pitfalls §1 |
</phase_requirements>

---

## Summary

Phase 3 is a **pure refactor with no new library dependencies.** The `ConditionalGenerativeModel` ABC already exists at `syntho_hive/core/models/base.py` and defines the correct contract (`fit`, `sample`, `save`, `load`). The only work is wiring it through the orchestration layer by:

1. Removing the hardcoded `CTGAN` import from `StagedOrchestrator.__init__` and replacing it with a `model_cls` parameter that defaults to `CTGAN`.
2. Propagating that `model_cls` parameter upward through `Synthesizer.__init__` as `model=CTGAN` so public API callers see no change.
3. Writing an integration test that instantiates the pipeline with a hand-rolled stub implementing `ConditionalGenerativeModel` — proving the ABC contract is sufficient without CTGAN.

The critical finding from reading the codebase is that `StagedOrchestrator` stores its typed dict as `Dict[str, CTGAN]` (line 83) and calls `CTGAN(...)` at three sites in `fit_all()` (lines 117, 169). Both the type annotation and the constructor calls must change. `Synthesizer` passes `model_cls` to nothing today — it has a `backend: str = "CTGAN"` string flag but never uses it to select the model class. That dead parameter should be replaced.

**Primary recommendation:** Use Python's built-in `abc.ABC` + `@abstractmethod` pattern (already in place) as the interface contract. Pass the class (not an instance) as `model_cls: Type[ConditionalGenerativeModel]` into both `StagedOrchestrator` and `Synthesizer`. Instantiate inside `fit_all()` where model kwargs are available. No new packages required.

---

## Standard Stack

### Core (all already in pyproject.toml)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python `abc` | stdlib | `ABC` + `@abstractmethod` for `ConditionalGenerativeModel` | Already used in `base.py`; zero new dependencies |
| Python `typing` | stdlib | `Type[ConditionalGenerativeModel]` for the class-as-parameter pattern | Type hint enforces contract at call sites |
| `pydantic` v2 | `>=2.0.0` | `Metadata`, `TableConfig` passed to model constructors | Already in deps; no changes needed |
| `structlog` | `>=21.1.0` | Already used for logging; no change | Already in deps |

### No New Dependencies

This phase requires zero new packages. The ABC is already defined. The pattern is standard Python.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pass class `model_cls` | Pass instance `model_instance` | Instance approach cannot be reinstantiated per-table; class approach allows per-table construction with different kwargs — class is correct |
| `Type[ConditionalGenerativeModel]` type hint | `Any` | `Any` loses static analysis; `Type[...]` documents the contract and enables mypy checking |
| Keeping `backend: str` flag | Use `model_cls: Type[...]` | String flag requires if/else dispatch and doesn't extend without code changes; class injection extends to N models without touching orchestrator |

---

## Architecture Patterns

### Current State (what must change)

```
StagedOrchestrator.__init__:
  line 15: from syntho_hive.core.models.ctgan import CTGAN   # ← HARDCODED IMPORT
  line 83: self.models: Dict[str, CTGAN] = {}                # ← HARDCODED TYPE

fit_all():
  line 117: model = CTGAN(self.metadata, ...)                # ← HARDCODED CONSTRUCTOR (root table)
  line 169: model = CTGAN(self.metadata, ...)                # ← HARDCODED CONSTRUCTOR (child table)

Synthesizer.__init__:
  line 35: backend: str = "CTGAN"                            # ← DEAD STRING FLAG
  line 54: self.orchestrator = StagedOrchestrator(metadata, self.spark)  # ← NO MODEL PARAM FORWARDED
```

### Target State (what must be true after phase)

```
StagedOrchestrator:
  - No CTGAN import in orchestration logic
  - Constructor: model_cls: Type[ConditionalGenerativeModel] = CTGAN  (default import only at module level for default)
  - self.models: Dict[str, ConditionalGenerativeModel] = {}
  - fit_all(): model = self.model_cls(self.metadata, ...)

Synthesizer:
  - Constructor: model: Type[ConditionalGenerativeModel] = CTGAN
  - StagedOrchestrator(metadata, self.spark, model_cls=self.model_cls)
```

### Pattern 1: Class-as-Parameter Dependency Injection

**What:** Accept a class object (not an instance) as a constructor parameter. Store as `self.model_cls`. Instantiate inside the method that has all the kwargs.

**When to use:** When the same class must be instantiated multiple times (once per table), with varying constructor arguments. An instance cannot be reused across tables without reset.

**Example:**

```python
# Source: Python stdlib abc + typing — standard dependency injection pattern
from typing import Type
from syntho_hive.core.models.base import ConditionalGenerativeModel
from syntho_hive.core.models.ctgan import CTGAN  # kept for the default value only

class StagedOrchestrator:
    def __init__(
        self,
        metadata: Metadata,
        spark: Optional[SparkSession] = None,
        io: Optional[Any] = None,
        on_write_failure: Literal['raise', 'cleanup', 'retry'] = 'raise',
        model_cls: Type[ConditionalGenerativeModel] = CTGAN,  # ← NEW PARAM
    ):
        self.model_cls = model_cls
        self.models: Dict[str, ConditionalGenerativeModel] = {}  # ← updated type
        # ... rest of init unchanged

    def fit_all(self, real_data_paths, epochs=300, batch_size=500, **model_kwargs):
        # Root table
        model = self.model_cls(self.metadata, batch_size=batch_size, epochs=epochs, **model_kwargs)
        # Child table
        model = self.model_cls(self.metadata, batch_size=batch_size, epochs=epochs, **model_kwargs)
```

**Key constraint:** `CTGAN.__init__` takes `(metadata, batch_size, epochs, ...)`. Any custom model class must accept the same positional/keyword args, OR the orchestrator must document the expected constructor signature in the ABC or in `Synthesizer`'s docstring.

### Pattern 2: Facade Passthrough

**What:** `Synthesizer` accepts `model` param and passes it to `StagedOrchestrator`. It does not instantiate or inspect the class itself.

**Example:**

```python
# Source: existing synthesizer.py pattern + new model param
class Synthesizer:
    def __init__(
        self,
        metadata: Metadata,
        privacy_config: PrivacyConfig,
        spark_session: Optional[SparkSession] = None,
        model: Type[ConditionalGenerativeModel] = CTGAN,  # ← replaces 'backend: str'
        embedding_threshold: int = 50,
    ):
        self.model_cls = model
        if self.spark:
            self.orchestrator = StagedOrchestrator(
                metadata, self.spark, model_cls=self.model_cls  # ← forward it
            )
```

**Backward compatibility:** Callers that currently pass `Synthesizer(metadata, privacy)` with no `model` arg get CTGAN by default. The old `backend: str = "CTGAN"` param should be removed — it was never functional. If callers pass `backend=...`, they will get a TypeError (breakage). Assess whether `backend` is used externally; if in doubt, keep it as a deprecated kwarg (`**kwargs`) or add a deprecation warning.

### Pattern 3: Stub Model for Integration Testing

**What:** Define a minimal `ConditionalGenerativeModel` subclass in the test file. It does not train anything — it returns empty DataFrames of the correct shape. Use it to verify routing without running CTGAN.

**Example:**

```python
# Source: standard Python ABC stub pattern for integration tests
import pandas as pd
from syntho_hive.core.models.base import ConditionalGenerativeModel

class StubModel(ConditionalGenerativeModel):
    """Minimal model for testing ABC contract — no real training."""

    def fit(self, data: pd.DataFrame, context=None, table_name=None, **kwargs) -> None:
        # Record columns seen during fit for sample to return correct shape
        self._columns = [c for c in data.columns]
        self._nrows_seen = len(data)

    def sample(self, num_rows: int, context=None, **kwargs) -> pd.DataFrame:
        # Return a DataFrame with the same columns, filled with zeros
        return pd.DataFrame(
            {col: [0] * num_rows for col in self._columns}
        )

    def save(self, path: str) -> None:
        pass  # no-op for testing

    def load(self, path: str) -> None:
        pass  # no-op for testing
```

**What this test proves:** That `StagedOrchestrator` never calls `CTGAN()` directly — it uses `self.model_cls()` — and that the full multi-table pipeline routes through any conforming class.

### Anti-Patterns to Avoid

- **Keeping `backend: str` alongside `model_cls`:** Two ways to select the model creates confusion. Remove `backend` or mark it deprecated with a warning that redirects to `model`.
- **Storing `model_cls` on `Synthesizer` and re-importing it in `StagedOrchestrator`:** The class should flow downward via constructor, not be re-resolved by name.
- **Instantiating the model in `Synthesizer.__init__`:** The orchestrator needs one instance per table, so instantiation belongs in `fit_all()`, not in `Synthesizer`.
- **Changing the ABC to require `metadata` as a constructor arg:** The ABC defines method contracts (`fit`, `sample`, `save`, `load`), not constructor signatures. Constructor arg requirements should be documented in a docstring on `Synthesizer.model`, not encoded in the ABC.
- **Importing CTGAN at the top of `orchestrator.py` for anything other than the default value:** After the refactor, the only acceptable CTGAN import in `orchestrator.py` is `from syntho_hive.core.models.ctgan import CTGAN` used solely as `model_cls: Type[ConditionalGenerativeModel] = CTGAN`. If that default is moved elsewhere, the import can be removed entirely.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Interface contract enforcement | Custom `isinstance()` checks | Python `abc.ABC` + `@abstractmethod` (already in `base.py`) | ABC raises `TypeError` with a clear message at instantiation time if abstract methods are missing — no manual checking needed |
| Type safety for `model_cls` | Runtime `hasattr()` checks | `Type[ConditionalGenerativeModel]` type hint + mypy | Static checking catches wrong types before runtime |
| Test isolation | A full CTGAN training run in integration tests | `StubModel` (see Pattern 3) | Integration tests should verify routing logic, not ML convergence — CTGAN tests already exist in `test_validation.py` |

**Key insight:** The ABC contract already exists and is enforced by Python at object instantiation. Adding `model_cls` is a 3-site change in `orchestrator.py` and a 2-site change in `synthesizer.py`. The complexity here is minimal — the only real risk is missing one of the CTGAN call sites or mishandling the `backend` param migration.

---

## Common Pitfalls

### Pitfall 1: Missing CTGAN Call Site in fit_all()

**What goes wrong:** There are TWO `CTGAN(...)` constructor calls in `fit_all()` — one for root tables (line 117) and one for child tables (line 169). A developer who only replaces one leaves a partial hardcoding that passes tests using only root or only child tables.

**Why it happens:** The code paths are inside an `if config.has_dependencies` branch and look structurally similar but are distinct.

**How to avoid:** Search for all occurrences of `CTGAN(` in `orchestrator.py` before and after the change. Both must become `self.model_cls(`.

**Warning signs:** Tests using 2-table schemas (one parent, one child) fail when using a custom model even though 1-table tests pass.

### Pitfall 2: Constructor Signature Mismatch

**What goes wrong:** `CTGAN.__init__` accepts `(metadata, embedding_dim, generator_dim, ..., batch_size, epochs, ...)`. If a custom model's constructor has a different signature, `self.model_cls(self.metadata, batch_size=..., epochs=..., **model_kwargs)` raises `TypeError`.

**Why it happens:** The ABC only enforces method contracts, not constructor signatures.

**How to avoid:** Document the expected constructor signature clearly in `Synthesizer.model` parameter docstring. The recommended interface: `__init__(self, metadata, batch_size=500, epochs=300, **kwargs)`. The stub model in tests should use this signature.

**Warning signs:** `TypeError: __init__() got an unexpected keyword argument 'epochs'` when running `fit_all`.

### Pitfall 3: `backend: str` Parameter Left Active

**What goes wrong:** If `Synthesizer` keeps `backend: str = "CTGAN"` alongside the new `model` parameter, callers passing `backend="CTGAN"` expect the old behavior but nothing happens — `backend` is ignored.

**Why it happens:** Incremental refactors often leave dead parameters to avoid churn.

**How to avoid:** Either remove `backend` entirely (breaking change, acceptable since it was functionally inert) or add a deprecation warning that redirects to `model`. Since `backend` was never functional (it never influenced which class was instantiated), removing it is safe.

**Warning signs:** `test_synthesizer_init_no_spark` or similar tests passing `backend=...` start failing.

### Pitfall 4: Type Annotation `Dict[str, CTGAN]` Not Updated

**What goes wrong:** `self.models: Dict[str, CTGAN] = {}` on line 83 of `orchestrator.py` causes mypy errors when non-CTGAN models are stored. Tests still pass (Python ignores type annotations at runtime) but the codebase signals incorrect invariants to tooling.

**Why it happens:** Type annotations are easy to overlook during a refactor.

**How to avoid:** Update to `Dict[str, ConditionalGenerativeModel] = {}`.

**Warning signs:** mypy reports `Dict item "x" has incompatible type "StubModel"; expected "CTGAN"`.

### Pitfall 5: ABC Methods Not Matching CTGAN Signature

**What goes wrong:** `ConditionalGenerativeModel.fit()` signature in `base.py` is `fit(self, data, context=None, **kwargs)`. `CTGAN.fit()` has additional named params: `table_name`, `checkpoint_dir`, `log_metrics`, `seed`. If the ABC's signature is taken as the strict contract, custom models may not know about `table_name` (which `orchestrator.py` passes as a kwarg at line 123/178).

**Why it happens:** `CTGAN.fit()` was extended with orchestrator-specific params. The ABC was not updated to match.

**How to avoid:** Verify that `orchestrator.py` only passes `context=...` and `table_name=...` to `model.fit()`. The stub test model must accept `**kwargs` in `fit()` to swallow `table_name`. Document that `table_name` is forwarded.

**Warning signs:** `TypeError: fit() got an unexpected keyword argument 'table_name'` in stub model tests.

---

## Code Examples

### 1. StagedOrchestrator Refactored __init__ and fit_all Skeleton

```python
# Source: existing orchestrator.py + standard Python DI pattern
from typing import Type, Dict, Any, Optional
from syntho_hive.core.models.base import ConditionalGenerativeModel
from syntho_hive.core.models.ctgan import CTGAN  # default only — not used in logic

class StagedOrchestrator:
    def __init__(
        self,
        metadata: Metadata,
        spark: Optional[SparkSession] = None,
        io: Optional[Any] = None,
        on_write_failure: Literal['raise', 'cleanup', 'retry'] = 'raise',
        model_cls: Type[ConditionalGenerativeModel] = CTGAN,  # ← new parameter
    ):
        self.metadata = metadata
        self.spark = spark
        self.model_cls = model_cls  # ← stored, used in fit_all
        if io is not None:
            self.io = io
        else:
            self.io = SparkIO(spark)
        self.on_write_failure = on_write_failure
        self.graph = SchemaGraph(metadata)
        self.models: Dict[str, ConditionalGenerativeModel] = {}  # ← type updated
        self.linkage_models: Dict[str, LinkageModel] = {}

    def fit_all(self, real_data_paths, epochs=300, batch_size=500, **model_kwargs):
        for table_name in self.metadata.tables:
            # ...data loading...
            if not config.has_dependencies:
                model = self.model_cls(             # ← was: CTGAN(
                    self.metadata,
                    batch_size=batch_size,
                    epochs=epochs,
                    **model_kwargs
                )
                model.fit(target_pdf, table_name=table_name)
                self.models[table_name] = model
            else:
                # ...linkage setup...
                model = self.model_cls(             # ← was: CTGAN(
                    self.metadata,
                    batch_size=batch_size,
                    epochs=epochs,
                    **model_kwargs
                )
                model.fit(target_pdf, context=context_df, table_name=table_name)
                self.models[table_name] = model
```

### 2. Synthesizer model Parameter

```python
# Source: existing synthesizer.py — new model param replaces backend: str
from syntho_hive.core.models.base import ConditionalGenerativeModel
from syntho_hive.core.models.ctgan import CTGAN

class Synthesizer:
    def __init__(
        self,
        metadata: Metadata,
        privacy_config: PrivacyConfig,
        spark_session: Optional[SparkSession] = None,
        model: Type[ConditionalGenerativeModel] = CTGAN,  # ← replaces backend: str
        embedding_threshold: int = 50,
    ):
        """
        Args:
            model: Generative model class to use for synthesis. Must be a class
                   (not an instance) that implements ``ConditionalGenerativeModel``.
                   The class constructor must accept ``(metadata, batch_size, epochs,
                   **kwargs)`` and its instances must implement ``fit()``, ``sample()``,
                   ``save()``, and ``load()``.

                   Supported classes:
                   - ``syntho_hive.core.models.ctgan.CTGAN`` (default)
                   - Any custom class implementing ``ConditionalGenerativeModel``

                   Existing callers that omit this parameter receive CTGAN behavior
                   unchanged.
        """
        self.metadata = metadata
        self.privacy = privacy_config
        self.spark = spark_session
        self.model_cls = model
        self.embedding_threshold = embedding_threshold

        if self.spark:
            self.orchestrator = StagedOrchestrator(
                metadata, self.spark, model_cls=self.model_cls
            )
        else:
            self.orchestrator = None
```

### 3. Integration Test with StubModel

```python
# Source: standard Python ABC stub pattern — test goes in test_interface.py
import pandas as pd
from syntho_hive.core.models.base import ConditionalGenerativeModel
from syntho_hive.relational.orchestrator import StagedOrchestrator
from syntho_hive.interface.config import Metadata

class StubModel(ConditionalGenerativeModel):
    """Minimal stub that satisfies ConditionalGenerativeModel without training."""

    def fit(self, data: pd.DataFrame, context=None, table_name=None, **kwargs) -> None:
        self._columns = list(data.columns)

    def sample(self, num_rows: int, context=None, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({col: [0] * num_rows for col in self._columns})

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


def test_stub_model_routes_through_pipeline(tmp_path):
    """MODEL-03: StubModel implementing ConditionalGenerativeModel routes correctly."""
    meta = Metadata()
    meta.add_table("users", pk="id")

    # Build a minimal IO mock
    import pandas as pd
    from unittest.mock import MagicMock

    users_df = pd.DataFrame({"id": range(5), "age": range(5)})

    class MockSparkDF:
        def __init__(self, pdf): self.pdf = pdf
        def toPandas(self): return self.pdf

    mock_io = MagicMock()
    mock_io.read_dataset.return_value = MockSparkDF(users_df)
    mock_io.write_pandas.side_effect = lambda pdf, path, **kw: (
        (tmp_path / "users").mkdir(parents=True, exist_ok=True) or
        pdf.to_csv(tmp_path / "users" / "data.csv", index=False)
    )

    orch = StagedOrchestrator(metadata=meta, io=mock_io, model_cls=StubModel)
    orch.fit_all({"users": "path/users"}, epochs=1, batch_size=5)
    result = orch.generate({"users": 3})

    assert "users" in result
    assert len(result["users"]) == 3
    # Verify no CTGAN was imported or used
    assert isinstance(orch.models["users"], StubModel)
```

### 4. Verifying No Hardcoded CTGAN in Orchestration

```python
# Negative-space test: verify that CTGAN class is NOT stored in orch.models
# when a different model_cls is used — this is the MODEL-01 acceptance test.
assert not any(
    type(m).__name__ == "CTGAN" for m in orch.models.values()
), "StagedOrchestrator stored a CTGAN instance despite model_cls=StubModel"
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded model class in orchestrators | Dependency injection via `Type[Model]` parameter | This phase | Any conforming class works without forking the library |
| `backend: str` string selector | `model: Type[ConditionalGenerativeModel]` class selector | This phase | Eliminates if/else dispatch and scales to N models |
| `Dict[str, CTGAN]` models dict | `Dict[str, ConditionalGenerativeModel]` | This phase | Correct type annotation; enables mypy and IDEs |

**No deprecated libraries involved.** This phase is pure internal restructuring.

---

## Open Questions

1. **What happens to `embedding_threshold` parameter on `Synthesizer`?**
   - What we know: `Synthesizer.__init__` accepts `embedding_threshold: int = 50` but it is never forwarded to `StagedOrchestrator` or to `CTGAN` today. It was likely intended for future use.
   - What's unclear: Should it be dropped, kept as a passthrough `model_kwarg`, or documented as CTGAN-specific?
   - Recommendation: Keep it for now, document it as CTGAN-specific in the `model` param docstring, pass it via `**model_kwargs` in `fit_all()`.

2. **Should the ABC define a recommended constructor signature?**
   - What we know: Python ABCs cannot enforce `__init__` signatures.
   - What's unclear: Whether a docstring convention is sufficient, or whether a factory method pattern is needed.
   - Recommendation: Document the expected constructor signature in a docstring on `ConditionalGenerativeModel` and in `Synthesizer.model` param docs. A factory method would over-engineer for current needs.

3. **Should `backend: str` be removed or deprecated?**
   - What we know: `backend` is set in `__init__` (`self.backend = backend`) but only used in a log string in `sample()` (`print(f"Generating data with {self.backend} backend...")`). It has never influenced which class is instantiated.
   - What's unclear: Whether any external callers pass `backend=...`.
   - Recommendation: Remove `backend` from `__init__` signature. Update the `print()` in `sample()` to use `self.model_cls.__name__`. Low risk — it was functionally inert.

4. **Should `StagedOrchestrator` validate that `model_cls` is a subclass of `ConditionalGenerativeModel`?**
   - What we know: Python's ABC mechanism raises `TypeError` at instantiation (`self.model_cls(...)`) if the class doesn't implement all abstract methods. This catches missing methods but not non-subclass imposters.
   - What's unclear: Whether an explicit `issubclass()` check at `__init__` time gives better errors than the ABC's own TypeError.
   - Recommendation: Add an `issubclass()` guard in `StagedOrchestrator.__init__` with a clear error message. This surfaces the problem at construction time (before any data loading) rather than at first `fit_all()` call.

---

## Sources

### Primary (HIGH confidence)

- Direct codebase read — `syntho_hive/relational/orchestrator.py` lines 15, 83, 117, 169 (CTGAN import sites confirmed)
- Direct codebase read — `syntho_hive/core/models/base.py` (ABC already defined; `ConditionalGenerativeModel` covers `fit`, `sample`, `save`, `load`)
- Direct codebase read — `syntho_hive/interface/synthesizer.py` lines 35, 54–57 (`backend` dead param and orchestrator construction)
- Python stdlib `abc` documentation — ABCs enforce method presence at instantiation; cannot enforce constructor signatures
- Python `typing.Type[X]` — standard pattern for accepting a class that produces instances of X

### Secondary (MEDIUM confidence)

- `.planning/PROJECT.md` — confirmed "ML Framework: PyTorch — any new models must also use PyTorch" constraint (impacts future models, not this phase's stub test)
- `.planning/ROADMAP.md` Phase 3 plan notes — confirmed 2-plan scope (03-01: orchestrator refactor; 03-02: Synthesizer + integration test)
- `.planning/v1.1-MILESTONE-AUDIT.md` — confirmed MODEL-01/02/03 requirements, no existing coverage

### Tertiary (LOW confidence)

None — all findings are grounded in direct codebase reads.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; ABC already exists; `typing.Type` is stdlib
- Architecture: HIGH — all CTGAN call sites identified by direct code read; injection pattern is textbook Python
- Pitfalls: HIGH — all pitfalls grounded in actual line numbers in the existing code

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (stable internal codebase; no external dependency volatility)

---

## Implementation Scope Summary

This phase touches exactly 5 files:

| File | Change |
|------|--------|
| `syntho_hive/relational/orchestrator.py` | Add `model_cls` param; remove hardcoded `CTGAN(` calls (2 sites); update `Dict[str, CTGAN]` type annotation |
| `syntho_hive/interface/synthesizer.py` | Replace `backend: str` with `model: Type[ConditionalGenerativeModel]`; forward to `StagedOrchestrator` |
| `syntho_hive/core/models/base.py` | Optional: add docstring documenting expected constructor signature convention |
| `syntho_hive/tests/test_interface.py` | Add `test_stub_model_routes_through_pipeline` (MODEL-03 acceptance test) |
| `syntho_hive/tests/test_relational.py` | Optional: add assertion that `model_cls=StubModel` produces no CTGAN instances |

**No changes needed to:** `exceptions.py`, `config.py`, `ctgan.py`, `linkage.py`, `graph.py`, or any connector.
