# Phase 1: Core Reliability - Research

**Researched:** 2026-02-22
**Domain:** Python exception design, PyTorch serialization, structured logging, seed reproducibility, SQL injection prevention, pytest
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Exception Hierarchy:**
- Define a custom exception hierarchy rooted at `SynthoHiveError`
- Four typed subclasses: `SchemaError`, `TrainingError`, `SerializationError`, `ConstraintViolationError`
- Each exception carries: human-readable message + original exception chained via `raise ... from ...`
- Wrap at **public API boundaries only**: `Synthesizer.fit()`, `Synthesizer.sample()`, `Synthesizer.save()`, `Synthesizer.load()`, `Synthesizer.generate_validation_report()`
- Internal modules raise their own typed errors freely; the Synthesizer boundary wraps into top-level SynthoHive type

**Serialization Contract:**
- Save artifact is a **directory** (e.g., `./syntho_models/customers/`), not a single file
- Directory contains separate files per component: network weights, DataTransformer state, context_transformer state, data_column_info, embedding_layers, metadata (version, schema, timestamp)
- **Default save path**: `./syntho_models/{table_name}/`
- `save(path)` **raises `SerializationError` if the path already exists** unless `overwrite=True`
- Version mismatch on `load()`: **warn via structlog** and attempt load — do not fail
- `torch.load()` must use `weights_only=False` explicitly (required for PyTorch 2.6+)

**Constraint Violation Behavior:**
- Constraint checking is **opt-in**: `sample(enforce_constraints=True)` — off by default
- When enabled: validate full generated batch, collect all violations, raise `ConstraintViolationError` with summary
- Error message format: `"ConstraintViolationError: 3 violations — age: got -3 (min=0); price: got -12.50 (min=0.01); quantity: got 1001 (max=1000)"`
- **Return valid rows, warn about violations** rather than failing the entire call
- `transformer.py:222-246` bare `except Exception: pass` blocks must raise `ConstraintViolationError` with column name and observed value

**Seed & Reproducibility:**
- **Separate seeds**: `fit(seed=42)` controls training; `sample(seed=7)` controls generation
- When **no seed provided** for fit: auto-generate, log via structlog at INFO with `"Training seed: {seed}"`
- Seed **not exposed** on synthesizer object — log is the single source of truth
- Seed scope: **PyTorch only** — `torch.manual_seed()`, `numpy.random.seed()`, `random.seed()`. Spark excluded (document in docstrings)

### Claude's Discretion
- Exact structlog log level and format for training seed and constraint warning messages
- Internal error message wording for each exception type
- Whether `overwrite=True` on `save()` is a positional argument or keyword-only
- File naming convention for components within the save directory (e.g., `weights.pt`, `transformer.joblib`)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CORE-01 | Training completes without crashes, hangs, or silent failures — all exceptions surface with actionable error messages | Exception hierarchy design; bare `except:` audit findings; structlog pattern |
| CORE-02 | `save()` persists full model state (DataTransformer, context_transformer, data_column_info, embedding_layers, generator/discriminator weights) | joblib for sklearn objects; torch.save for network weights; directory-based artifact layout |
| CORE-03 | `load()` restores a saved model and runs `sample()` without requiring original training data or retraining | Complete checkpoint includes all DataTransformer state; torch.load weights_only=False for PyTorch 2.6+ |
| CORE-04 | Bare `except:` and silent `except Exception: pass` blocks eliminated project-wide; all exceptions typed and logged | Audit: 1 bare `except:` in synthesizer.py:153; 2 silent `except Exception:` in transformer.py:228,245; others re-raise |
| QUAL-04 | Numeric constraints (min, max, dtype) enforced on generated output; violations raise with column name and observed value | `ConstraintViolationError` pattern; current transformer.py clipping approach to be converted to raise |
| QUAL-05 | `Synthesizer.fit()` accepts a `seed` parameter producing deterministic DataTransformer encoding, training init, and sample output | torch.manual_seed + numpy.random.seed + random.seed; BayesianGMM random_state already fixed at 42 — needs to be parameterized |
| CONN-04 | `save_to_hive()` database name validated against allowlist pattern before SQL interpolation | Regex `^[a-zA-Z0-9_]+$` applied before any spark.sql() call; raise `SchemaError` on invalid |
| TEST-01 | End-to-end test: single-table fit → sample → validate — passes with a known small dataset | pytest fixture using iris/synthetic data; CTGAN with small epochs; assert output shape and column names |
| TEST-03 | Serialization round-trip test: fit → save → load → sample produces output matching pre-save distribution | Uses directory-based save; load in same process (fresh CTGAN instance); TVD tolerance check |
| TEST-05 | Regression test: training with fixed seed produces bit-identical synthetic output across two independent runs | Two separate CTGAN.fit(seed=42).sample(100) calls; assert DataFrame equality |
</phase_requirements>

---

## Summary

Phase 1 is a surgical reliability pass on the existing codebase. The codebase is structurally sound — the bugs are specific and locatable. Four distinct problem areas need fixing: (1) bare exception swallowing in `synthesizer.py` and `transformer.py`, (2) broken serialization in `CTGAN.save()`/`load()` that only persists network weights and discards all DataTransformer state, (3) missing seed control in both `DataTransformer` and the CTGAN training loop, and (4) SQL injection in `save_to_hive()`.

The serialization fix is the most complex task. `CTGAN.save()` currently only writes generator and discriminator state dicts to a `.pt` file. A cold load is impossible because: `DataTransformer` (with fitted sklearn transformers like `BayesianGaussianMixture`, `OneHotEncoder`, `LabelEncoder`) is entirely absent from the checkpoint; `context_transformer` is absent; `data_column_info` (the column layout list) is absent; `embedding_layers` (an `nn.ModuleDict`) are absent. Rebuilding any of these without the original data is impossible. The fix uses a directory layout: joblib for sklearn objects and embedding layers (which contain `nn.Module` instances), torch.save for generator/discriminator weights.

The seed fix has a subtle complication: `ClusterBasedNormalizer` in `transformer.py` hardcodes `random_state=42` in its `BayesianGaussianMixture`. This must be parameterized to accept the caller's seed so that `fit(seed=42)` produces deterministic DataTransformer encoding as required by QUAL-05.

**Primary recommendation:** Create `syntho_hive/exceptions.py` first, then fix serialization (most complex), then seed + constraint violations, then SQL injection, then write the three new test files. Tackle in this order to unblock test-writing which validates all other fixes.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| joblib | >=1.0 (transitively installed via sklearn) | Serialize sklearn objects (BayesianGMM, OneHotEncoder, LabelEncoder) and nn.ModuleDict | Efficient NumPy array serialization; official sklearn recommendation; already available |
| torch (existing) | >=1.10.0 (pinned in pyproject.toml) | Serialize generator/discriminator state dicts | Native format; `torch.save`/`torch.load` with explicit `weights_only=False` for PyTorch 2.6+ compat |
| structlog (existing) | >=21.1.0 (pinned in pyproject.toml); current latest is 25.5.0 | Structured logging for seed announcements, constraint warnings, version mismatch warnings | Already in project deps; structured key-value output matches data-engineer audience |
| pytest (existing) | >=7.0.0 (in dev deps) | Test runner for TEST-01, TEST-03, TEST-05 | Already configured; `testpaths = ["tests"]` in pyproject.toml |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| re (stdlib) | any | Regex allowlist validation for `save_to_hive()` database names | Pattern `^[a-zA-Z0-9_]+$` to reject injection attempts |
| random (stdlib) | any | Set Python RNG seed alongside torch and numpy for full determinism | Must be set alongside `torch.manual_seed()` and `np.random.seed()` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| joblib for sklearn objects | pickle / cloudpickle | joblib is faster for numpy-heavy objects and is sklearn's own recommendation; cloudpickle handles closures better but is overkill here |
| directory layout for save artifacts | single .tar or .zip bundle | Directory is human-inspectable (user decision); bundling adds complexity with no benefit |
| torch.load weights_only=False | torch.serialization.add_safe_globals() | allowlisting is more secure but complex to maintain across custom classes; weights_only=False is fine for trusted internal checkpoints |

**Installation:**
All required libraries are already declared in `pyproject.toml`. No new pip installs needed. `joblib` is installed transitively via `scikit-learn`.

```bash
# Verify joblib is available
python -c "import joblib; print(joblib.__version__)"
```

## Architecture Patterns

### Recommended Project Structure Changes

```
syntho_hive/
├── exceptions.py          # NEW: SynthoHiveError hierarchy (all 4 subclasses)
├── interface/
│   ├── config.py          # Unchanged
│   └── synthesizer.py     # MODIFIED: wrap public APIs, add save/load dir logic, SQL injection fix
├── core/
│   ├── models/
│   │   └── ctgan.py       # MODIFIED: fix save/load, add seed param to fit/sample
│   └── data/
│       └── transformer.py # MODIFIED: parameterize ClusterBasedNormalizer random_state; raise ConstraintViolationError
tests/
├── test_e2e_single_table.py    # NEW: TEST-01 single-table e2e
├── test_serialization.py       # NEW: TEST-03 round-trip
└── test_seed_regression.py     # NEW: TEST-05 seed bit-identity
```

### Pattern 1: Exception Hierarchy Definition

**What:** A single `exceptions.py` module defines the full hierarchy. All other modules import from it.
**When to use:** All typed raises across the codebase import from `syntho_hive.exceptions`.

```python
# syntho_hive/exceptions.py
# Source: Python official docs + project CONTEXT.md decisions

class SynthoHiveError(Exception):
    """Base exception for all SynthoHive errors."""
    pass

class SchemaError(SynthoHiveError):
    """Raised for invalid metadata, missing FK, unsupported column types, or invalid identifier names."""
    pass

class TrainingError(SynthoHiveError):
    """Raised for NaN loss, training divergence, GPU OOM, or other fit-time failures."""
    pass

class SerializationError(SynthoHiveError):
    """Raised for save/load failures, corrupt checkpoints, or version mismatches that prevent loading."""
    pass

class ConstraintViolationError(SynthoHiveError):
    """Raised when generated output violates numeric constraints (min, max, dtype)."""
    pass
```

### Pattern 2: Public API Boundary Wrapping

**What:** Synthesizer public methods catch internal exceptions and re-raise as typed SynthoHive exceptions using `raise ... from ...` to chain the cause.
**When to use:** Only at the five public API boundaries.

```python
# syntho_hive/interface/synthesizer.py
# Source: Python docs on exception chaining

from syntho_hive.exceptions import TrainingError, SerializationError, SchemaError

def fit(self, data, seed=None, ...):
    try:
        self.orchestrator.fit_all(real_paths, seed=seed, ...)
    except SynthoHiveError:
        raise  # Already typed — let it through
    except Exception as exc:
        raise TrainingError(
            f"TrainingError: fit() failed on data source '{data}'. "
            f"Original error: {exc}"
        ) from exc
```

### Pattern 3: CTGAN Directory-Based Serialization

**What:** `CTGAN.save(path)` writes a directory with separate files per component. `CTGAN.load(path)` reconstructs the full object without requiring original data.
**When to use:** Replacing the current single-file `.pt` approach.

```python
# syntho_hive/core/models/ctgan.py
# Source: joblib docs (scikit-learn.org/stable/model_persistence.html)

import joblib
import json
import torch
from pathlib import Path
from syntho_hive import __version__

def save(self, path: str, overwrite: bool = False) -> None:
    p = Path(path)
    if p.exists() and not overwrite:
        raise SerializationError(
            f"SerializationError: Save path '{path}' already exists. "
            f"Pass overwrite=True to replace it."
        )
    p.mkdir(parents=True, exist_ok=overwrite)

    # Network weights — torch native format
    torch.save(self.generator.state_dict(), p / "generator.pt")
    torch.save(self.discriminator.state_dict(), p / "discriminator.pt")

    # sklearn objects — joblib for NumPy-heavy objects
    joblib.dump(self.transformer, p / "transformer.joblib")
    joblib.dump(self.context_transformer, p / "context_transformer.joblib")

    # Embedding layers (nn.ModuleDict) — joblib handles nn.Module via pickle
    joblib.dump(self.embedding_layers, p / "embedding_layers.joblib")

    # Column layout
    joblib.dump(self.data_column_info, p / "data_column_info.joblib")

    # Metadata for version tracking
    meta = {
        "synthohive_version": __version__,
        "embedding_dim": self.embedding_dim,
        "generator_dim": list(self.generator_dim),
        "discriminator_dim": list(self.discriminator_dim),
        "embedding_threshold": self.embedding_threshold,
        "saved_at": "<timestamp>",
    }
    with open(p / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def load(self, path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SerializationError(f"SerializationError: Path '{path}' does not exist.")

    # Check version
    meta_path = p / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        saved_ver = meta.get("synthohive_version", "unknown")
        if saved_ver != __version__:
            log.warning("version_mismatch",
                        saved_version=saved_ver,
                        current_version=__version__,
                        path=path)
        # Reconstruct architecture from metadata if needed
        self.embedding_dim = meta.get("embedding_dim", self.embedding_dim)

    # Load sklearn objects
    self.transformer = joblib.load(p / "transformer.joblib")
    self.context_transformer = joblib.load(p / "context_transformer.joblib")
    self.data_column_info = joblib.load(p / "data_column_info.joblib")
    self.embedding_layers = joblib.load(p / "embedding_layers.joblib")

    # Rebuild network architecture from loaded data_column_info
    # (derive dims from transformer state)
    context_dim = self.context_transformer.output_dim if hasattr(self.context_transformer, 'output_dim') else 0
    self._build_model(self.transformer.output_dim, context_dim)

    # Load network weights — weights_only=False required for PyTorch 2.6+
    self.generator.load_state_dict(
        torch.load(p / "generator.pt", weights_only=False)
    )
    self.discriminator.load_state_dict(
        torch.load(p / "discriminator.pt", weights_only=False)
    )
```

### Pattern 4: Seed Setting for Full Determinism

**What:** A single `_set_seed(seed)` helper sets all three RNGs (torch, numpy, random). Called at start of `fit()` and `sample()`.
**When to use:** Whenever a seed is provided. When no seed provided to `fit()`, auto-generate one and log it.

```python
# Source: PyTorch reproducibility docs (docs.pytorch.org/docs/stable/notes/randomness.html)
import random
import numpy as np
import torch

def _set_seed(seed: int) -> None:
    """Set all RNG seeds for deterministic behavior (PyTorch, NumPy, Python random)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # CuDNN determinism (CPU-only training not affected, but set for GPU safety)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# In CTGAN.fit():
def fit(self, data, ..., seed=None, ...):
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
        log.info("training_seed", seed=seed,
                 message="No seed provided — auto-generated. Log this to reproduce the run.")
    else:
        log.info("training_seed", seed=seed)
    _set_seed(seed)
    # Also pass seed to DataTransformer so BayesianGMM uses it
    self.transformer.fit(data, table_name=table_name, seed=seed)
    ...
```

**Critical detail:** `ClusterBasedNormalizer` hardcodes `BayesianGaussianMixture(random_state=42)`. This must become `BayesianGaussianMixture(random_state=seed)` where seed flows from `DataTransformer.fit(seed=)`.

### Pattern 5: structlog Usage

**What:** Module-level logger obtained via `structlog.get_logger()`. Bind structured context as kwargs.
**When to use:** Seed announcements (INFO), version mismatch warnings (WARNING), constraint violation summaries (WARNING).

```python
# Source: structlog 25.5.0 docs (structlog.org/en/stable/getting-started.html)
import structlog

log = structlog.get_logger()

# Seed announcement
log.info("training_seed", seed=42)

# Version mismatch warning
log.warning("checkpoint_version_mismatch",
             saved_version="1.2.2",
             current_version="1.2.3",
             path="/syntho_models/customers/")

# Constraint warning (when returning valid rows with some violations)
log.warning("constraint_violations_detected",
             violation_count=3,
             violations="age: got -3 (min=0); price: got -12.50 (min=0.01)",
             action="returning valid rows only")
```

### Pattern 6: SQL Injection Prevention via Allowlist

**What:** Validate database/table name against `^[a-zA-Z0-9_]+$` before any SQL interpolation. Raise immediately, before Spark context is touched.
**When to use:** At the start of `save_to_hive()`, before any `self.spark.sql()` call.

```python
# Source: OWASP input validation guidance; re stdlib
import re

_SAFE_IDENTIFIER = re.compile(r'^[a-zA-Z0-9_]+$')

def save_to_hive(self, synthetic_data, target_db, overwrite=True):
    if not _SAFE_IDENTIFIER.match(target_db):
        raise SchemaError(
            f"SchemaError: Database name '{target_db}' contains invalid characters. "
            f"Only [a-zA-Z0-9_] are allowed."
        )
    # Also validate table names from synthetic_data keys
    for table in synthetic_data:
        if not _SAFE_IDENTIFIER.match(table):
            raise SchemaError(
                f"SchemaError: Table name '{table}' contains invalid characters."
            )
    # Proceed with spark.sql() safely
    ...
```

### Pattern 7: ConstraintViolationError with Batch Summary

**What:** After `inverse_transform`, when `enforce_constraints=True`, scan all columns for violations, collect them, and either raise (if configured to fail) or warn and return only valid rows.
**When to use:** In `CTGAN.sample()` when `enforce_constraints=True`.

```python
# Source: project CONTEXT.md decisions

def sample(self, num_rows, context=None, enforce_constraints=True, seed=None, **kwargs):
    if seed is not None:
        _set_seed(seed)

    # ... generation logic ...
    result_df = self.transformer.inverse_transform(fake_data_np)

    if enforce_constraints and self.table_name:
        table_config = self.metadata.get_table(self.table_name)
        if table_config:
            violations = []
            valid_mask = pd.Series([True] * len(result_df))

            for col, constraint in table_config.constraints.items():
                if col not in result_df.columns:
                    continue
                col_data = result_df[col]
                if constraint.min is not None:
                    bad = col_data < constraint.min
                    if bad.any():
                        violations.append(
                            f"{col}: got {col_data[bad].min():.4g} (min={constraint.min})"
                        )
                        valid_mask &= ~bad
                if constraint.max is not None:
                    bad = col_data > constraint.max
                    if bad.any():
                        violations.append(
                            f"{col}: got {col_data[bad].max():.4g} (max={constraint.max})"
                        )
                        valid_mask &= ~bad

            if violations:
                summary = "; ".join(violations)
                log.warning("constraint_violations_detected",
                             count=len(violations), summary=summary,
                             valid_rows=int(valid_mask.sum()),
                             total_rows=len(result_df))
                # Return only valid rows per CONTEXT.md decision
                result_df = result_df[valid_mask].reset_index(drop=True)

    return result_df
```

### Anti-Patterns to Avoid

- **Bare `except:` without re-raise:** The one instance in `synthesizer.py:153` (inside `generate_validation_report()` fallback to delta read) must be replaced with a typed exception or explicit re-raise.
- **Silent `except Exception: pass` in transformer.py:228,245:** These are in the int-casting section of `inverse_transform`. They must become explicit raises of `ConstraintViolationError` with column name and observed value rather than swallowing.
- **Single-file `.pt` save:** The current `CTGAN.save()` writes only weights. Loading into a fresh model requires re-fitting first (confirmed by `test_models.py:71`), which defeats the purpose of serialization.
- **Seeding only torch, not numpy or random:** The transformer's `BayesianGaussianMixture` uses sklearn/numpy RNG internally. Without `np.random.seed()`, encoding is non-deterministic even when PyTorch is seeded.
- **`torch.load()` without `weights_only=False`:** PyTorch 2.6+ changed the default from `False` to `True`. Loading custom objects (state dicts with embedding layers) will fail without explicit `weights_only=False`. The CONTEXT.md decision locks this as required.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| sklearn object serialization | custom pickle wrapper | `joblib.dump()` / `joblib.load()` | joblib handles numpy arrays with memory efficiency and protocol=5 for large arrays; already available |
| Structured logging | `print()` with f-strings | `structlog.get_logger()` | Key-value pairs are machine-parseable; already in project deps; structlog 25.5.0 is current |
| Regex-based identifier validation | manual character iteration | `re.compile(r'^[a-zA-Z0-9_]+$').match()` | stdlib; correct semantics; avoids subtle off-by-one errors |

**Key insight:** Every tool needed is already installed. This phase is about wiring existing capabilities correctly, not adding dependencies.

## Common Pitfalls

### Pitfall 1: Loading a CTGAN Without Rebuilding Architecture

**What goes wrong:** `load()` tries to call `load_state_dict()` on `self.generator`, but `self.generator` is `None` (never built) because no `fit()` was called after instantiating a fresh `CTGAN`.
**Why it happens:** `_build_model()` depends on knowing the transformer output dim and context dim. Current code has no path to recover this information from disk.
**How to avoid:** The save directory must include enough information to call `_build_model()` without original data. The `metadata.json` stores `embedding_dim`, `generator_dim`, `discriminator_dim`. The transformer's `output_dim` is recoverable from the loaded `DataTransformer` object. Load sequence: load transformer → load context_transformer → call `_build_model(transformer.output_dim, context_transformer.output_dim)` → load state dicts.
**Warning signs:** `AttributeError: 'NoneType' object has no attribute 'load_state_dict'` during `load()`.

### Pitfall 2: ClusterBasedNormalizer Fixed random_state Breaking Seed Determinism

**What goes wrong:** Even with `torch.manual_seed(42)` and `np.random.seed(42)`, two `fit()` calls produce different DataTransformer encodings because `ClusterBasedNormalizer` always creates `BayesianGaussianMixture(random_state=42)` (hardcoded).
**Why it happens:** The hardcoded value means encoding is reproducible in isolation but ignores the caller's seed intent. Two runs with different seeds will still produce the same encoding.
**How to avoid:** `DataTransformer.fit(seed=N)` must pass `seed` down to each `ClusterBasedNormalizer` construction. Each column may need a distinct but deterministic seed (e.g., `seed + hash(col_name) % 1000`).
**Warning signs:** TEST-05 (bit-identical output) fails even though PyTorch training is seeded correctly.

### Pitfall 3: Embedding Layers Not Saved, Causing Architecture Mismatch on Load

**What goes wrong:** Embedding layer dimensions (`emb_dim` for each categorical column) are computed in `_compile_layout()` based on `num_categories`. If the embedding layers aren't persisted, `_build_model()` on load re-creates them with fresh random weights, making sample output meaningless.
**Why it happens:** `nn.ModuleDict` is not a sklearn object — it cannot be reconstructed from just `data_column_info` without the original weights.
**How to avoid:** Use `joblib.dump(self.embedding_layers, "embedding_layers.joblib")`. joblib serializes `nn.Module` via pickle. On load, restore embedding layers before calling `_build_model()` — or skip calling `_build_model()` for embedding layers entirely (restore from checkpoint, then build only generator/discriminator from scratch using dims, then load their weights).
**Warning signs:** Synthetic data from a loaded model has completely different category distributions than the pre-save model.

### Pitfall 4: torch.load Default Change in PyTorch 2.6+

**What goes wrong:** `torch.load(path)` without explicit `weights_only` raises an error or silently uses `weights_only=True` in PyTorch >= 2.6, failing to load custom objects.
**Why it happens:** PyTorch 2.6 made `weights_only=True` the default as a security measure. Custom objects in checkpoints (like those containing generator state dicts that reference custom layer classes) may not be allowlisted.
**How to avoid:** Always pass `weights_only=False` explicitly when loading SynthoHive checkpoints. Document this in code comments. The CONTEXT.md decision locks this.
**Warning signs:** `UnpicklingError` or `WeightsUnpickler error` when calling `torch.load()` after upgrading PyTorch.

### Pitfall 5: Path Already Exists — Accidental Model Overwrite

**What goes wrong:** A second training run with the same output path silently overwrites a trained model, destroying it.
**Why it happens:** `mkdir(exist_ok=True)` does not check if the directory already has contents.
**How to avoid:** At the start of `save()`, check `if path.exists()` and raise `SerializationError` unless `overwrite=True` is explicitly passed. Use `Path(path).exists()` not `os.path.exists()` for clarity.
**Warning signs:** No error on second save, but the first model is gone.

### Pitfall 6: Bare `except:` in generate_validation_report()

**What goes wrong:** The bare `except:` in `synthesizer.py:153` catches `KeyboardInterrupt` and `SystemExit`, making it impossible to interrupt a long-running validation.
**Why it happens:** `except:` with no type catches all `BaseException` subclasses.
**How to avoid:** Replace with `except (AnalysisException, Exception) as exc:` where `AnalysisException` is the Spark table-not-found exception. Re-raise as `SynthoHiveError` at the Synthesizer boundary.
**Warning signs:** Ctrl+C does not stop the process during `generate_validation_report()`.

## Code Examples

Verified patterns from official sources:

### Structlog Logger Setup (Module Level)
```python
# Source: structlog 25.5.0 docs (structlog.org/en/stable/getting-started.html)
import structlog

log = structlog.get_logger()

# Usage:
log.info("training_seed", seed=42, table="customers")
log.warning("checkpoint_version_mismatch", saved="1.2.2", current="1.2.3")
log.error("serialization_failed", path="/syntho_models/customers/", error=str(exc))
```

### Full Seed Function
```python
# Source: PyTorch reproducibility docs (docs.pytorch.org/docs/stable/notes/randomness.html)
import random
import numpy as np
import torch

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### joblib Save/Load Pattern
```python
# Source: scikit-learn model persistence docs (scikit-learn.org/stable/model_persistence.html)
import joblib

# Save
joblib.dump(transformer_object, "transformer.joblib")

# Load
transformer_object = joblib.load("transformer.joblib")
```

### Exception Chaining
```python
# Source: Python 3 official docs (docs.python.org/3/tutorial/errors.html)
try:
    risky_operation()
except ValueError as exc:
    raise TrainingError(
        f"TrainingError: Fit failed — {exc}"
    ) from exc  # chains __cause__, preserves traceback
```

### TEST-01 Skeleton (Single-Table E2E)
```python
# tests/test_e2e_single_table.py
import pandas as pd
import numpy as np
import pytest
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata

@pytest.fixture
def small_dataset():
    np.random.seed(0)
    return pd.DataFrame({
        "id": range(200),
        "age": np.random.randint(18, 80, 200),
        "income": np.random.exponential(50_000, 200),
        "city": np.random.choice(["NY", "SF", "LA", "CHI"], 200),
    })

@pytest.fixture
def meta():
    m = Metadata()
    m.add_table("customers", pk="id")
    return m

def test_single_table_e2e(small_dataset, meta):
    model = CTGAN(meta, batch_size=32, epochs=3, embedding_dim=16,
                  generator_dim=(32, 32), discriminator_dim=(32, 32))
    model.fit(small_dataset, table_name="customers", seed=42)
    result = model.sample(50)
    assert len(result) == 50
    assert set(result.columns) == {"age", "income", "city"}
    assert not result.isnull().all().any(), "All-null column in output"
```

### TEST-03 Skeleton (Serialization Round-Trip)
```python
# tests/test_serialization.py
import shutil, tempfile, os
import pandas as pd
import numpy as np
import pytest
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata

def test_serialization_round_trip(tmp_path, small_dataset, meta):
    model = CTGAN(meta, batch_size=32, epochs=3, embedding_dim=16,
                  generator_dim=(32, 32), discriminator_dim=(32, 32))
    model.fit(small_dataset, table_name="customers", seed=42)
    pre_save_sample = model.sample(100, seed=7)

    save_dir = str(tmp_path / "customers")
    model.save(save_dir)

    # Load in fresh instance — NO retraining
    new_model = CTGAN(meta, embedding_dim=16,
                      generator_dim=(32, 32), discriminator_dim=(32, 32))
    new_model.load(save_dir)
    post_load_sample = new_model.sample(100, seed=7)

    assert len(post_load_sample) == 100
    assert set(post_load_sample.columns) == set(pre_save_sample.columns)
```

### TEST-05 Skeleton (Seed Regression)
```python
# tests/test_seed_regression.py
import pandas as pd
import numpy as np
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata

def test_seed_produces_identical_output(small_dataset, meta):
    def run_once():
        model = CTGAN(meta, batch_size=32, epochs=3, embedding_dim=16,
                      generator_dim=(32, 32), discriminator_dim=(32, 32))
        model.fit(small_dataset, table_name="customers", seed=42)
        return model.sample(100, seed=7)

    run1 = run_once()
    run2 = run_once()

    pd.testing.assert_frame_equal(run1, run2, check_exact=True,
                                   obj="Seed-identical runs should produce bit-identical output")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.load()` defaults to `weights_only=False` | `torch.load()` defaults to `weights_only=True` | PyTorch 2.6 (Jan 2025) | Must pass `weights_only=False` explicitly; failure mode is `UnpicklingError` or silent refusal |
| pickle for sklearn objects | joblib recommended by sklearn | sklearn >=0.23 | joblib is faster for NumPy-heavy objects; no new installation needed |

**Deprecated/outdated:**
- `torch.load(path)` without `weights_only`: Fails silently or loudly in PyTorch 2.6+. Always pass `weights_only=False` for full-object checkpoints.
- Hardcoded `random_state=42` in `ClusterBasedNormalizer`: Incompatible with parameterized seed control. Must become a constructor parameter.

## Open Questions

1. **`nn.ModuleDict` in joblib — cross-PyTorch-version compatibility**
   - What we know: joblib serializes via pickle; nn.Module pickles fine within the same PyTorch version
   - What's unclear: Whether embedding layer weights loaded from a different PyTorch version produce identical outputs
   - Recommendation: Document in `load()` docstring that same PyTorch version is required for bit-identical sample output; structlog warning on version mismatch is sufficient mitigation

2. **BayesianGMM seed — per-column vs global**
   - What we know: `ClusterBasedNormalizer` creates one `BayesianGaussianMixture` per column; each needs a seed
   - What's unclear: Whether all columns should share the same seed (simpler) or derive distinct seeds per column (avoids correlated RNG sequences across columns)
   - Recommendation: Use `seed + abs(hash(col_name)) % 100_000` for each column. This produces distinct but deterministic seeds per column from a single user-facing seed. Document this derivation.

3. **`context_transformer.output_dim` on fresh load**
   - What we know: `DataTransformer.output_dim` is set during `fit()` and persisted by joblib
   - What's unclear: Whether `output_dim` survives the joblib round-trip correctly (it's a plain Python int attribute)
   - Recommendation: Write a unit assertion in TEST-03 that checks `loaded_model.transformer.output_dim > 0` to catch this early.

## Sources

### Primary (HIGH confidence)
- PyTorch reproducibility docs — https://docs.pytorch.org/docs/stable/notes/randomness.html — seed setting pattern, CUDA caveats
- scikit-learn model persistence — https://scikit-learn.org/stable/model_persistence.html — joblib.dump/load, protocol=5, versioning warnings
- structlog 25.5.0 getting started — https://www.structlog.org/en/stable/getting-started.html — basic config, get_logger(), structured key-value logging
- Python official exception docs — https://docs.python.org/3/tutorial/errors.html — exception hierarchy, raise ... from ... chaining
- PyTorch 2.6 release blog — https://pytorch.org/blog/pytorch2-6/ — weights_only=True default change confirmed

### Secondary (MEDIUM confidence)
- PyTorch 2.6 torch.load breaking change discussion — https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573 — confirmed default flip to weights_only=True
- structlog best practices — https://www.structlog.org/en/stable/logging-best-practices.html — canonical log lines, JSON vs console renderer, context binding

### Tertiary (LOW confidence)
- Community articles on SQL injection prevention via regex allowlist — validated by re stdlib documentation; pattern `^[a-zA-Z0-9_]+$` is standard and verified through multiple sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries (joblib, structlog, torch, pytest, re) are already installed or stdlib; versions verified from pyproject.toml and official docs
- Architecture: HIGH — serialization patterns verified against sklearn and PyTorch official docs; exception hierarchy is pure Python standard patterns
- Pitfalls: HIGH — PyTorch 2.6 weights_only change is documented in official release notes; ClusterBasedNormalizer hardcoded seed issue verified by direct codebase inspection; embedding layer serialization gap verified by reading ctgan.py save() directly

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (stable libraries — 30-day window; PyTorch breaking change is already confirmed landed)
