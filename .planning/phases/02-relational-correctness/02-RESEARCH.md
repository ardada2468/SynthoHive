# Phase 2: Relational Correctness - Research

**Researched:** 2026-02-22
**Domain:** Multi-table synthetic data correctness — FK conditioning, cardinality modeling, schema validation, memory management, Spark/Delta version pinning
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### FK type mismatch errors
- Raise a new `SchemaValidationError` exception class (not reuse existing SynthoHiveError)
- `validate_schema()` catches both FK type mismatches (int vs string) AND missing FK columns in the child table
- Collect all schema problems before raising — do not fail on the first mismatch — so engineers can fix the whole schema at once
- Error message includes: the mismatched table/column pair, both detected types, and a concrete fix hint (e.g., "cast parent PK to string" or "add column X to child table Y")

#### Cardinality distribution
- Replace GMM with a configurable distribution: empirical or NegBinom, selectable per-table in schema config
- Default (when not specified): empirical — always matches training data distribution exactly
- Configuration is per-table (in the schema config), not a global switch
- Stale context fix (freshly drawn parent context per training step) is always-on but includes an opt-out flag for backwards compatibility with existing workflows

#### Memory-safe generation
- When a generated table is written to disk and released from memory: log at DEBUG level only (not INFO — doesn't clutter normal runs)
- On disk write failure: configurable via `on_write_failure` parameter with three modes:
  - `'raise'` — fail immediately, leave partial files on disk (default)
  - `'cleanup'` — delete all files written so far, then raise
  - `'retry'` — retry the write once before failing
- Default: `'raise'` — matches fail-fast behavior and avoids surprising cleanup

#### TEST-02 coverage
- Cover a 3-table FK chain AND a 4-table chain (deeper hierarchy to catch cascade orphan issues)
- Include FK type mismatch test case (int vs string) — verifies `SchemaValidationError` is raised correctly
- Include missing FK column test case — verifies missing-column detection in `validate_schema()`
- Verify FK referential integrity (zero orphans on join) AND cardinality distribution accuracy (child counts per parent within tolerance of empirical distribution)

### Claude's Discretion
- Whether to collect-all vs fail-fast internally — decided: collect-all for better UX (user left this to Claude)
- Exact cardinality tolerance threshold for the TEST-02 cardinality check
- Retry count and delay for `on_write_failure='retry'` mode
- Exact error message wording beyond the required components
- The opt-out flag for stale context fix: name and exact API surface

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REL-01 | Child table generation uses freshly sampled parent context per generator training step — not the discriminator's stale last-batch context | ctgan.py:399-408 is the exact site; fix is a two-line resample in the generator block |
| REL-02 | FK cardinality uses empirical distribution or Negative Binomial — not Gaussian Mixture, which produces negative sample counts | linkage.py uses GaussianMixture; replace `fit()`/`sample_counts()` with empirical histogram or `scipy.stats.nbinom` |
| REL-03 | FK type mismatches are detected and raised at `validate_schema()` time, before training begins | config.py validate_schema() only checks table existence now; extend to check dtype and column presence; add `SchemaValidationError` to exceptions.py |
| REL-04 | Multi-table generation releases DataFrames from memory after writing to disk when `output_path_base` is set — no OOM accumulation | orchestrator.py:219-226 is the site; when output_path_base set, delete from generated_tables after write |
| REL-05 | Generated parent/child tables can be joined on FK columns with zero orphaned references and zero missing parents | Achieved by REL-01 + REL-02 + REL-03 working together; verified by TEST-02 |
| CONN-02 | Spark/Delta Lake connector works correctly with PySpark 4.0+ and delta-spark 4.0+; `pyproject.toml` pins updated to match actual installed versions | venv has pyspark==4.0.1 and delta-spark==4.0.0; pyproject.toml pins pyspark>=3.2.0 and delta-spark>=2.0.0 — both must be updated |
| TEST-02 | End-to-end test: multi-table fit → sample → FK join validates zero orphans on a 3-table schema (parent → child → grandchild) | Extends to 4-table chain per CONTEXT.md; uses pytest with MockSparkDF pattern already established in test_relational.py |
</phase_requirements>

---

## Summary

Phase 2 addresses three distinct correctness failures in multi-table synthesis, plus a Spark/Delta version pin, plus a test suite. All four work areas have exact file locations and line numbers already identified in `.planning/codebase/CONCERNS.md` and `.planning/research/PITFALLS.md`. No new libraries need to be added to the stack — all fixes use existing dependencies (scipy, numpy, structlog).

The stale context bug (REL-01) is a two-line fix in `ctgan.py` generator training block. The bug is explicitly acknowledged in a code comment at lines 402-405 ("Ideally yes, but reusing batch is fine for conditional stability — We'll stick to reusing the last seen batch"). The fix: independently sample context indices for the generator step instead of reusing `real_context_batch` from the discriminator loop.

The cardinality distribution (REL-02) requires replacing `GaussianMixture` in `linkage.py` with either an empirical histogram sampler (numpy only, simplest) or `scipy.stats.nbinom` (already a dependency via scipy 1.7.0+, installed at 1.16.3). Both are supported — the decision makes empirical the default with NegBinom as an opt-in per-table. The schema validation (REL-03) requires a new `SchemaValidationError` exception class and extending `Metadata.validate_schema()` to accept actual `pd.DataFrame` arguments for type-checking (the current method only checks table existence). Memory safety (REL-04) is a one-block change in `orchestrator.py` to delete from `generated_tables` after disk write when `output_path_base` is set.

**Primary recommendation:** Fix the four files in dependency order — exceptions.py → config.py → linkage.py → ctgan.py → orchestrator.py — then update pyproject.toml, then write TEST-02.

---

## Standard Stack

### Core (all already in project dependencies)

| Library | Version (installed) | Purpose | Why Standard |
|---------|---------------------|---------|--------------|
| scipy.stats | 1.16.3 | NegBinom distribution — `scipy.stats.nbinom` for MLE fit and `rvs()` sampling | Already a dependency; nbinom is the canonical count distribution in scipy |
| numpy | 2.3.5 | Empirical distribution via `np.random.choice(observed_counts)` | Trivial implementation; zero new dependencies |
| structlog | declared, used in ctgan.py | Structured DEBUG logging for DataFrame release events | Already used — `log = structlog.get_logger()` at ctgan.py line 16 |
| pytest | >=7.0.0 | TEST-02 test framework | Established project test runner |
| unittest.mock | stdlib | MockSparkDF pattern for Spark-free testing | Pattern already used in test_relational.py |

### Supporting

| Library | Version (installed) | Purpose | When to Use |
|---------|---------------------|---------|-------------|
| pyspark | 4.0.1 | Spark DataFrame I/O in orchestrator | Used in StagedOrchestrator; CONN-02 requires pin update |
| delta-spark | 4.0.0 | Delta Lake table format support | CONN-02 requires pin update to match installed version |
| pydantic | 2.0+ | Metadata/TableConfig models | Hosts `validate_schema()` method being extended |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Empirical (default) | KDE (kernel density) | KDE produces continuous density, requires rounding and clipping — same problem as GMM. Empirical is exact. |
| scipy.stats.nbinom | statsmodels NegativeBinomial | statsmodels is not a project dependency; scipy is. Use scipy. |
| structlog DEBUG for release logs | print() statement | print() can't be suppressed; structlog matches existing ctgan.py logging pattern |

**Installation:** No new dependencies needed. All required libraries are already declared in `pyproject.toml`.

**pyproject.toml pin update required for CONN-02:**
```toml
# FROM:
"pyspark>=3.2.0",
"delta-spark>=2.0.0",

# TO:
"pyspark>=4.0.0",
"delta-spark>=4.0.0",
```

---

## Architecture Patterns

### Recommended File Modification Order

```
syntho_hive/exceptions.py          # Add SchemaValidationError (REL-03)
syntho_hive/interface/config.py    # Extend validate_schema() (REL-03)
syntho_hive/interface/config.py    # Add linkage_method field to TableConfig (REL-02)
syntho_hive/relational/linkage.py  # Replace GMM with empirical/NegBinom (REL-02)
syntho_hive/core/models/ctgan.py   # Fix stale context resample (REL-01)
syntho_hive/relational/orchestrator.py  # Memory-safe generation + on_write_failure (REL-04)
pyproject.toml                     # Update pyspark and delta-spark pins (CONN-02)
tests/test_relational.py           # TEST-02: 3-table and 4-table FK chain tests
```

### Pattern 1: Collect-All Schema Validation

**What:** Accumulate all schema errors into a list, then raise once with all problems combined.
**When to use:** REL-03 — `validate_schema()` must report all mismatches so users fix everything at once.

```python
# Source: Existing pattern in config.py extended with collect-all
def validate_schema(self, real_data: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    errors: List[str] = []
    for table_name, table_config in self.tables.items():
        for local_col, parent_ref in table_config.fk.items():
            if "." not in parent_ref:
                errors.append(f"Invalid FK reference '{parent_ref}' in table '{table_name}'.")
                continue
            parent_table, parent_col = parent_ref.split(".", 1)
            if parent_table not in self.tables:
                errors.append(f"Table '{table_name}' references non-existent parent table '{parent_table}'.")
                continue
            # Type mismatch check (when real_data provided)
            if real_data is not None:
                if table_name in real_data and parent_table in real_data:
                    child_df = real_data[table_name]
                    parent_df = real_data[parent_table]
                    if local_col not in child_df.columns:
                        errors.append(
                            f"FK column '{local_col}' missing from table '{table_name}'. "
                            f"Add column '{local_col}' to child table '{table_name}'."
                        )
                    elif parent_col not in parent_df.columns:
                        errors.append(
                            f"Parent PK column '{parent_col}' missing from table '{parent_table}'."
                        )
                    else:
                        child_dtype = str(child_df[local_col].dtype)
                        parent_dtype = str(parent_df[parent_col].dtype)
                        if not _dtypes_compatible(child_dtype, parent_dtype):
                            errors.append(
                                f"FK type mismatch: '{table_name}.{local_col}' is {child_dtype} "
                                f"but '{parent_table}.{parent_col}' is {parent_dtype}. "
                                f"Fix: cast '{table_name}.{local_col}' to {parent_dtype} or "
                                f"cast '{parent_table}.{parent_col}' to {child_dtype}."
                            )
    if errors:
        raise SchemaValidationError("\n".join(errors))
```

**Note on dtype compatibility:** Integer variants (`int64`, `int32`, `Int64`) should be considered compatible with each other. String/object and integer types are incompatible. A helper `_dtypes_compatible(a, b)` that checks the numpy kind character (`np.dtype(a).kind`) is the correct approach.

### Pattern 2: Empirical Cardinality Distribution

**What:** Replace GMM with direct histogram resampling for child counts.
**When to use:** Default (`linkage_method = "empirical"` or not specified in TableConfig).

```python
# Source: PITFALLS.md recommendation; numpy docs
def fit(self, parent_df, child_df, fk_col, pk_col="id"):
    counts = child_df[fk_col].value_counts()
    parent_ids = pd.DataFrame(parent_df[pk_col].unique(), columns=[pk_col])
    count_df = parent_ids.merge(
        counts.rename("child_count"),
        left_on=pk_col, right_index=True, how="left"
    ).fillna(0)
    self._observed_counts = count_df["child_count"].to_numpy(dtype=int)
    self.max_children = int(self._observed_counts.max())

def sample_counts(self, parent_context):
    n_samples = len(parent_context)
    return np.random.choice(self._observed_counts, size=n_samples, replace=True)
```

### Pattern 3: NegBinom Cardinality Distribution

**What:** Fit scipy.stats.nbinom via method of moments; use `rvs()` to sample.
**When to use:** When `linkage_method = "negbinom"` is specified in TableConfig.

```python
# Source: scipy.stats.nbinom official docs (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html)
from scipy import stats

def fit(self, parent_df, child_df, fk_col, pk_col="id"):
    # ... same count aggregation as empirical ...
    X = count_df["child_count"].to_numpy(dtype=float)
    mu = X.mean()
    var = X.var()
    # Method of moments for NegBinom: n = mu^2/(var-mu), p = mu/var
    # Guard: if var <= mu, fall back to empirical (Poisson regime, NegBinom ill-defined)
    if var <= mu or mu == 0:
        self._fallback_to_empirical(X)
    else:
        p = mu / var
        n = mu * p / (1 - p)
        self._nbinom_n = max(n, 0.1)  # nbinom n must be > 0
        self._nbinom_p = p
        self.method = "negbinom"  # confirm method after successful fit

def sample_counts(self, parent_context):
    n_samples = len(parent_context)
    samples = stats.nbinom.rvs(self._nbinom_n, self._nbinom_p, size=n_samples)
    return np.clip(samples, 0, None)
```

**Critical:** `scipy.stats.nbinom` does not have a direct `fit()` class method for discrete distributions that accepts raw count data. Use method-of-moments estimation instead: `p = mean/variance`, `n = mean * p / (1 - p)`.

### Pattern 4: Stale Context Fix (REL-01)

**What:** Resample context indices independently in the generator training step.
**When to use:** Always (opt-out flag for backwards compatibility).

```python
# Source: PITFALLS.md recommendation; ctgan.py lines 399-408
# BEFORE (stale — reuses discriminator batch):
if real_context_batch is not None:
    gen_input = torch.cat([noise, real_context_batch], dim=1)

# AFTER (correct — independent resample):
if context_data is not None and not self.legacy_context_conditioning:
    gen_ctx_idx = np.random.randint(0, len(context_data), self.batch_size)
    gen_context_batch = context_data[gen_ctx_idx]
    gen_input = torch.cat([noise, gen_context_batch], dim=1)
else:
    gen_input = noise
```

The opt-out flag should be a constructor parameter named `legacy_context_conditioning: bool = False` on `CTGAN`. Default `False` means the fix is active. Setting `True` restores old behavior for backwards compatibility.

### Pattern 5: Memory-Safe Generation with on_write_failure

**What:** Release DataFrames after disk write; handle write failures with configurable policy.
**When to use:** REL-04 — whenever `output_path_base` is set.

```python
# Source: orchestrator.py lines 219-226; PITFALLS.md recommendation
# on_write_failure: Literal['raise', 'cleanup', 'retry'] = 'raise'

if output_path_base:
    output_path = f"{output_path_base}/{table_name}"
    _write_with_failure_handling(
        self.io, generated_pdf, output_path,
        on_write_failure=self.on_write_failure,
        written_paths=written_paths  # track for cleanup mode
    )
    # Release from memory — child tables use disk path instead
    log.debug("table_released_from_memory", table=table_name, path=output_path)
    # Do NOT store in generated_tables if output_path_base is set
else:
    generated_tables[table_name] = generated_pdf
```

The `on_write_failure` parameter should be added to `StagedOrchestrator.__init__()` and stored as `self.on_write_failure`. Child tables that need parent context during generation already have a disk-read code path (orchestrator.py lines 167-169) — they read from `output_path_base` when set.

### Anti-Patterns to Avoid

- **Silent string fallback in LinkageModel.fit():** Remove the `except Exception` block at linkage.py:36-44 that falls back to string conversion. Replace with early detection in `validate_schema()`.
- **Storing generated_tables unconditionally:** The comment at orchestrator.py:224-226 acknowledges the risk. Remove the unconditional storage when `output_path_base` is set.
- **Using generator loss as quality signal in context of this phase:** Not in scope for Phase 2, but do not add any new quality-based checkpointing logic — that is Phase 4.
- **DataFrame.dtype vs numpy kind comparison:** Use `np.dtype(col.dtype).kind` for type comparison, not string matching. `'i'` = integer, `'f'` = float, `'U'` or `'O'` = string/object.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Count distribution sampling | Custom distribution class | `numpy.random.choice(observed_counts)` for empirical; `scipy.stats.nbinom.rvs()` for NegBinom | Both are one-liners; numpy choice is the canonical empirical resampler |
| Method-of-moments NegBinom fitting | Custom MLE optimizer | scipy moments: `p = mean/var`, `n = mean*p/(1-p)` | NegBinom MoM closed form; no optimization loop needed |
| Error collection | Custom error accumulator class | Python `list` + `"\n".join(errors)` | Trivial; no utility class needed |
| Retry logic in write failure | Custom retry decorator | Inline `for attempt in range(2)` | One retry; no need for tenacity or backoff library |

**Key insight:** All correctness problems in this phase are algorithm/logic bugs, not missing libraries. The tools are already present.

---

## Common Pitfalls

### Pitfall 1: NegBinom Variance ≤ Mean (Poisson Regime)

**What goes wrong:** If `var(child_counts) <= mean(child_counts)`, NegBinom method-of-moments gives `p >= 1`, which is outside `(0, 1]`. `scipy.stats.nbinom.rvs()` will raise or produce NaN.

**Why it happens:** NegBinom is only well-defined for overdispersed data (variance > mean). In tables where every parent has exactly K children, variance = 0 and the formula breaks.

**How to avoid:** Guard: if `var <= mu` or if `mu == 0`, fall back to empirical. Log a WARNING that NegBinom requested but empirical used due to distribution shape.

**Warning signs:** `n` comes out negative, `p >= 1`, or `scipy.stats.nbinom.rvs()` raises `ValueError`.

---

### Pitfall 2: FK dtype Compatibility Check Fragility

**What goes wrong:** Pandas dtype strings are non-trivial — `"int64"`, `"Int64"` (nullable), `"int32"`, `"float64"`, `"object"`, `"string"` (Arrow-backed). Naive string comparison `child_dtype != parent_dtype` will flag `int64` vs `Int64` as a mismatch.

**Why it happens:** Pandas 2.x introduced nullable integer dtypes. The venv has numpy 2.3.5, which may affect dtype reporting for Arrow-backed columns.

**How to avoid:** Use numpy kind comparison: `np.dtype(child_col.dtype).kind` for numeric types. For object/string, check if kind is `'U'`, `'O'`, or pandas StringDtype. Define two categories: integer-like (`kind in 'iuI'`) and string-like (`kind in 'UO'` or isinstance StringDtype). Flag as mismatch only when one is integer-like and the other is string-like.

**Warning signs:** `validate_schema()` raises false positives on `int64` vs `Int64` columns.

---

### Pitfall 3: context_data Indexing After Resample

**What goes wrong:** After the stale context fix, the generator training block samples `gen_ctx_idx` as NumPy indices and indexes `context_data` (a PyTorch tensor). PyTorch tensor indexing with a NumPy int array requires conversion or specific indexing syntax.

**Why it happens:** The discriminator block already does this correctly for `idx` (line 321: `idx = np.random.randint(...)`; line 324: `real_context_batch = context_data[idx]`). The generator block just needs the same pattern.

**How to avoid:** Copy the exact discriminator pattern — `np.random.randint(0, len(context_data), self.batch_size)` produces a NumPy array; PyTorch tensors support NumPy array indexing directly.

**Warning signs:** `IndexError` or `TypeError` in generator training step after applying the fix.

---

### Pitfall 4: Memory Release Breaking In-Memory Generation Path

**What goes wrong:** If `output_path_base` is None, child tables still need parent data from `generated_tables`. If the memory release is applied unconditionally (not gated on `output_path_base`), child table generation fails with `KeyError`.

**Why it happens:** The `generated_tables` dict serves two purposes: (1) return value to caller, (2) parent data cache for child generation. The memory release must only apply when `output_path_base` is set (child tables read from disk in that path).

**How to avoid:** Gate the `del generated_tables[table_name]` behind `if output_path_base:`. The orchestrator already has the dual code path at lines 167-171. The delete must happen AFTER the child tables that depend on this parent have been generated — which is guaranteed by topological order.

**Warning signs:** `KeyError: 'parent_table'` in the `else: parent_df = generated_tables[driver_parent_table]` branch.

---

### Pitfall 5: pyproject.toml Pin Semantics

**What goes wrong:** Setting `pyspark>=4.0.0` may break users on Spark 3.5 clusters who haven't upgraded. Delta-spark has strict Spark version coupling (delta-spark 4.0.x ONLY works with Spark 4.0.x per the compatibility matrix).

**Why it happens:** CONN-02 says "works correctly with PySpark 4.0+ and delta-spark 4.0+". The venv has 4.0.1/4.0.0 installed.

**How to avoid:** Pin to `pyspark>=4.0.0,<5.0.0` and `delta-spark>=4.0.0,<5.0.0` to stay in the correct major-version band without locking to a specific patch. This follows the delta-spark compatibility matrix exactly.

**Warning signs:** Import errors when loading SparkSession with mismatched major versions; Java-level `ClassNotFoundException` for delta classes.

---

### Pitfall 6: SchemaValidationError vs SchemaError

**What goes wrong:** The project already has `SchemaError` in `exceptions.py`. The decision says raise a NEW `SchemaValidationError` (not reuse `SynthoHiveError`). Creating `SchemaValidationError` as a subclass of `SchemaError` (not of `SynthoHiveError` directly) preserves the exception hierarchy and lets callers `except SchemaError` to catch both.

**How to avoid:** Add to exceptions.py:
```python
class SchemaValidationError(SchemaError):
    """
    Raised by validate_schema() when FK type mismatches, missing FK columns,
    or invalid FK references are detected. Collects all errors before raising.
    """
    pass
```

**Warning signs:** Tests that catch `SchemaError` break if `SchemaValidationError` inherits from `SynthoHiveError` directly instead of `SchemaError`.

---

### Pitfall 7: TEST-02 Requires Spark-Free Execution

**What goes wrong:** The existing test_relational.py uses `MockSparkDF` and mocks `orchestrator.io.read_dataset` / `orchestrator.io.write_pandas`. TEST-02 must follow the same pattern — not require a real Spark session — so it runs in CI without a Spark cluster.

**Why it happens:** Spark is optional in the test environment (TESTING.md: "PySpark optional; tests gracefully skip if unavailable").

**How to avoid:** Use the same mock pattern as `TestOrchestrator.test_orchestrator_flow()` in test_relational.py. Do NOT use `pytest.importorskip("pyspark")` — the test should run even without Spark by using the existing mock infrastructure.

---

## Code Examples

### 1: SchemaValidationError in exceptions.py

```python
# Source: exceptions.py — add after SchemaError
class SchemaValidationError(SchemaError):
    """
    Raised by validate_schema() when FK type mismatches or missing FK columns
    are detected. Error message lists all problems found (collect-all strategy).
    """
    pass
```

### 2: dtype compatibility helper

```python
# Source: Standard numpy dtype kind approach
import numpy as np

def _dtypes_compatible(dtype_a: str, dtype_b: str) -> bool:
    """Return True if both dtypes belong to the same broad category (numeric or string)."""
    try:
        kind_a = np.dtype(dtype_a).kind
        kind_b = np.dtype(dtype_b).kind
    except TypeError:
        # pandas extension types (StringDtype, Int64Dtype) fall back to 'O' or 'i'
        return True  # conservative: don't false-positive on unknown types
    integer_kinds = {'i', 'u'}  # signed and unsigned int
    string_kinds = {'U', 'O', 'S'}  # unicode, object, bytes
    # Both numeric → compatible; both string → compatible; mixed → not compatible
    if kind_a in integer_kinds and kind_b in integer_kinds:
        return True
    if kind_a in string_kinds and kind_b in string_kinds:
        return True
    if kind_a == 'f' and kind_b == 'f':
        return True
    return False
```

### 3: Empirical LinkageModel

```python
# Source: PITFALLS.md; numpy.random.choice docs
class LinkageModel:
    def __init__(self, method: str = "empirical"):
        self.method = method
        self._observed_counts = None
        self._nbinom_n = None
        self._nbinom_p = None
        self.max_children = 0

    def fit(self, parent_df, child_df, fk_col, pk_col="id"):
        counts = child_df[fk_col].value_counts()
        parent_ids = pd.DataFrame(parent_df[pk_col].unique(), columns=[pk_col])
        count_df = parent_ids.merge(
            counts.rename("child_count"), left_on=pk_col, right_index=True, how="left"
        ).fillna(0)
        X = count_df["child_count"].to_numpy(dtype=int)
        self.max_children = int(X.max())
        self._observed_counts = X

        if self.method == "negbinom":
            mu = float(X.mean())
            var = float(X.var())
            if var > mu and mu > 0:
                p = mu / var
                n = mu * p / (1.0 - p)
                self._nbinom_n = max(n, 0.1)
                self._nbinom_p = p
            else:
                import structlog
                structlog.get_logger().warning(
                    "negbinom_fallback_to_empirical",
                    reason="variance <= mean or mean is zero",
                    table_method=self.method,
                )
                self.method = "empirical"  # fallback

    def sample_counts(self, parent_context):
        n_samples = len(parent_context)
        if self._observed_counts is None:
            raise ValueError("LinkageModel not fitted")
        if self.method == "negbinom" and self._nbinom_n is not None:
            from scipy import stats
            counts = stats.nbinom.rvs(self._nbinom_n, self._nbinom_p, size=n_samples)
            return np.clip(counts, 0, None).astype(int)
        # Default: empirical
        return np.random.choice(self._observed_counts, size=n_samples, replace=True)
```

### 4: Generator context resample fix (ctgan.py)

```python
# Source: ctgan.py lines 399-408 — replace the generator context block
# --- Train Generator ---
noise = torch.randn(self.batch_size, self.embedding_dim, device=self.device)
if context_data is not None:
    if self.legacy_context_conditioning:
        # Backwards-compatible: reuse last discriminator batch context
        gen_context_batch = real_context_batch
    else:
        # Correct: independently sample context for generator
        gen_ctx_idx = np.random.randint(0, len(context_data), self.batch_size)
        gen_context_batch = context_data[gen_ctx_idx]
    gen_input = torch.cat([noise, gen_context_batch], dim=1)
else:
    gen_input = noise
```

### 5: Memory-safe generate() (orchestrator.py)

```python
# Source: orchestrator.py lines 219-226 — replace the storage block
if generated_pdf is not None:
    if output_path_base:
        output_path = f"{output_path_base}/{table_name}"
        _write_with_failure_policy(
            io=self.io,
            pdf=generated_pdf,
            path=output_path,
            policy=self.on_write_failure,
            written_paths=written_paths,
        )
        log.debug("table_released_from_memory", table=table_name, path=output_path)
        # Do NOT keep in generated_tables — child tables read from disk
    else:
        # In-memory path: keep for downstream child generation and return
        generated_tables[table_name] = generated_pdf
```

### 6: TEST-02 structure (tests/test_relational.py addition)

```python
# Source: existing test_relational.py MockSparkDF pattern
class TestFKChainIntegrity(unittest.TestCase):
    """TEST-02: Multi-table FK chain — zero orphans, cardinality accuracy."""

    def _make_mock_io(self, orchestrator, data_map, output_dir):
        """Wire MockSparkDF read/write into orchestrator.io."""
        class MockSparkDF:
            def __init__(self, pdf): self.pdf = pdf
            def toPandas(self): return self.pdf

        def read_side_effect(path):
            for key, df in data_map.items():
                if key in path:
                    # Try reading generated output first
                    csv_path = os.path.join(path, "data.csv")
                    if os.path.exists(csv_path):
                        return MockSparkDF(pd.read_csv(csv_path))
                    return MockSparkDF(df)
            return MockSparkDF(pd.DataFrame())

        def write_side_effect(pdf, path, **kwargs):
            os.makedirs(path, exist_ok=True)
            pdf.to_csv(os.path.join(path, "data.csv"), index=False)

        orchestrator.io.read_dataset = MagicMock(side_effect=read_side_effect)
        orchestrator.io.write_pandas = MagicMock(side_effect=write_side_effect)

    def test_3_table_chain_zero_orphans(self):
        """3-table chain: users → orders → items — zero orphans after generation."""
        # Build metadata, data, run fit + generate, assert zero orphans
        ...

    def test_4_table_chain_zero_orphans(self):
        """4-table chain: users → orders → items → reviews — cascade integrity."""
        ...

    def test_fk_type_mismatch_raises_schema_validation_error(self):
        """REL-03: int PK vs string FK raises SchemaValidationError."""
        ...

    def test_fk_missing_column_raises_schema_validation_error(self):
        """REL-03: missing FK column in child raises SchemaValidationError."""
        ...

    def test_cardinality_within_tolerance(self):
        """REL-02: empirical cardinality — mean child count within 20% of training."""
        ...
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| GaussianMixture for child counts | Empirical histogram or NegBinom | Phase 2 | Eliminates negative counts and distributional drift |
| Stale discriminator batch context in generator | Independent context resample per generator step | Phase 2 | Unbiased conditional distribution; REL-05 achievable |
| Unconditional generated_tables dict | Memory-released dict when output_path_base set | Phase 2 | OOM-safe for large schemas |
| ValueError for all schema errors | SchemaValidationError (collect-all) | Phase 2 | Single error surface showing all schema problems |
| pyspark>=3.2.0, delta-spark>=2.0.0 | pyspark>=4.0.0, delta-spark>=4.0.0 | Phase 2 | Matches installed venv; eliminates version mismatch |

**Deprecated/outdated:**
- `GaussianMixture` import in linkage.py: Remove entirely after Phase 2. Replaced by numpy empirical resampler and optional scipy NegBinom.
- `real_context_batch` reuse in generator block: Remove the comment "We'll stick to reusing the last seen batch for simplicity/stability" — this is the bug, not a feature.

---

## Open Questions

1. **TableConfig field for `linkage_method`**
   - What we know: CONTEXT.md says configuration is per-table in the schema config. `TableConfig` is a Pydantic model in `config.py`.
   - What's unclear: Should `linkage_method` be a `Literal["empirical", "negbinom"]` field on `TableConfig` directly, or a nested config object?
   - Recommendation: Add `linkage_method: Literal["empirical", "negbinom"] = "empirical"` directly to `TableConfig`. Simple, explicit, follows the existing `fk` and `parent_context_cols` pattern.

2. **Cardinality tolerance threshold for TEST-02**
   - What we know: CONTEXT.md delegates the exact threshold to Claude's discretion.
   - What's unclear: What tolerance is tight enough to catch regressions but loose enough to not be flaky on small test data?
   - Recommendation: Use 20% relative tolerance on mean child count (e.g., `abs(synth_mean - real_mean) / real_mean < 0.20`). With 50-parent test data, empirical resampling should hit well within 20%. NegBinom may be slightly looser — keep the same threshold.

3. **Retry delay for `on_write_failure='retry'`**
   - What we know: CONTEXT.md delegates retry count and delay to Claude's discretion.
   - Recommendation: One retry with no delay (immediate). Disk I/O failures are either transient (file lock) or persistent (no space); a delay does not help the persistent case. Keep it simple.

4. **`legacy_context_conditioning` flag placement**
   - What we know: The flag is on `CTGAN` per the locked decision (opt-out for backwards compatibility).
   - Recommendation: Constructor parameter `legacy_context_conditioning: bool = False`. Store as `self.legacy_context_conditioning`. Include in `save()`/`load()` serialized state.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis — `syntho_hive/core/models/ctgan.py` lines 399-408 (stale context bug, confirmed with exact comment text)
- Direct codebase analysis — `syntho_hive/relational/linkage.py` (full file, 88 lines; GaussianMixture confirmed)
- Direct codebase analysis — `syntho_hive/relational/orchestrator.py` (full file, 229 lines; memory issue at lines 219-226 confirmed)
- Direct codebase analysis — `syntho_hive/interface/config.py` (full file; validate_schema() confirmed to only check table existence)
- Direct codebase analysis — `syntho_hive/exceptions.py` (full file; SchemaError confirmed present, SchemaValidationError absent)
- Direct codebase analysis — `pyproject.toml` (pyspark>=3.2.0 and delta-spark>=2.0.0 confirmed)
- `pip show pyspark delta-spark` in venv — pyspark 4.0.1, delta-spark 4.0.0 confirmed installed
- `.planning/codebase/CONCERNS.md` — bug locations and line numbers
- `.planning/research/PITFALLS.md` — stale context fix pattern (WRONG/CORRECT code examples verified)
- Delta Lake compatibility matrix at https://docs.delta.io/releases/ — delta-spark 4.0.x ↔ Spark 4.0.x confirmed

### Secondary (MEDIUM confidence)
- scipy.stats.nbinom documentation at https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html — rvs() method signature verified; fit() method confirmed not available for discrete distributions (use method-of-moments)
- PyPI delta-spark release history — 4.0.1 and 4.1.0 available as of Feb 2026; Python >=3.10 required

### Tertiary (LOW confidence)
- None.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already present in project; versions confirmed from pip
- Architecture: HIGH — all file locations and line numbers verified from source
- Pitfalls: HIGH — pitfalls 1-4 verified against actual source code; pitfall 5 verified against delta-spark docs
- Test patterns: HIGH — MockSparkDF pattern confirmed in existing test_relational.py

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (stable Python/scipy/numpy APIs; delta-spark version pinning should be re-verified if Spark 4.1 releases before implementation)
