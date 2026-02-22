# Architecture Research

**Domain:** Synthetic tabular data generation Python SDK (multi-table, relational, pluggable model strategy)
**Researched:** 2026-02-22
**Confidence:** HIGH — based on direct source code analysis of the full SynthoHive codebase plus domain knowledge of production synthetic data systems (SDV, Gretel, Mostly AI architectural patterns)

---

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INTERFACE LAYER                               │
│   Synthesizer (Façade: fit / sample / generate_validation_report)   │
│   Metadata + PrivacyConfig (Pydantic schema definitions)            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ delegates to
┌────────────────────────────▼────────────────────────────────────────┐
│                   RELATIONAL ORCHESTRATION LAYER                     │
│   StagedOrchestrator                                                 │
│   ├── SchemaGraph (topological sort of FK DAG)                      │
│   ├── LinkageModel (per child table: parent→child cardinality GMM)  │
│   └── ModelRegistry  {table_name → GenerativeModel instance}        │
└────┬──────────────────────────────────────────┬───────────────────-─┘
     │ fit/sample per table                      │ read/write data
┌────▼──────────────────────┐    ┌──────────────▼─────────────────────┐
│     MODEL LAYER            │    │        CONNECTOR LAYER              │
│  GenerativeModel (ABC)     │    │   SparkIO (parquet, delta, CSV)    │
│  ConditionalGenerativeModel│    │   SQLConnector (future)            │
│  CTGAN (default impl)      │    │   sampling utilities               │
│  [TVAE, etc. — pluggable]  │    └────────────────────────────────────┘
└────┬──────────────────────┘
     │ uses
┌────▼──────────────────────────────────────────────────────────────--┐
│                     CORE DATA LAYER                                  │
│   DataTransformer (fit / transform / inverse_transform)              │
│   ├── ClusterBasedNormalizer (Bayesian GMM per continuous column)    │
│   ├── OneHotEncoder (low-cardinality categoricals)                   │
│   └── LabelEncoder + EntityEmbeddingLayer (high-cardinality cats)   │
└────────────────────────────────────────────────────────────────────-┘
     ↑ runs before model training
┌────────────────────────────────────────────────────────────────────-┐
│                      PRIVACY LAYER                                   │
│   PIISanitizer (regex + name heuristics, samples 100 rows)          │
│   ContextualFaker (replacement generation)                           │
└────────────────────────────────────────────────────────────────────-┘
     ↑ runs after generation
┌────────────────────────────────────────────────────────────────────-┐
│                    VALIDATION LAYER                                  │
│   StatisticalValidator (KS test for continuous, TVD for categorical) │
│   ValidationReport (HTML / JSON output)                              │
└────────────────────────────────────────────────────────────────────-┘
```

### Component Boundaries

| Component | Responsibility | Communicates With | Current Location |
|-----------|---------------|-------------------|-----------------|
| `Synthesizer` | Public API façade: orchestrates fit/sample/validate workflow | `StagedOrchestrator`, `ValidationReport`, `SparkIO` | `syntho_hive/interface/synthesizer.py` |
| `Metadata` / `TableConfig` | Immutable schema: PK/FK relationships, constraints, PII columns | All downstream layers (read-only) | `syntho_hive/interface/config.py` |
| `PrivacyConfig` | Privacy rule configuration (strategy, epsilon, k-anon threshold) | `PIISanitizer` | `syntho_hive/interface/config.py` |
| `StagedOrchestrator` | Multi-table training coordination and generation pipeline | `SchemaGraph`, `LinkageModel`, model instances, `SparkIO` | `syntho_hive/relational/orchestrator.py` |
| `SchemaGraph` | FK dependency DAG; topological sort for generation order | `Metadata` | `syntho_hive/relational/graph.py` |
| `LinkageModel` | Per-child GMM for sampling parent→child cardinality distribution | Called by `StagedOrchestrator` during generate | `syntho_hive/relational/linkage.py` |
| `GenerativeModel` (ABC) | Contract: `fit`, `sample`, `save`, `load` | Implemented by `CTGAN`; used by `StagedOrchestrator` | `syntho_hive/core/models/base.py` |
| `ConditionalGenerativeModel` (ABC) | Extends contract with optional `context` param on fit/sample | Implemented by `CTGAN` | `syntho_hive/core/models/base.py` |
| `CTGAN` | Conditional tabular GAN with WGAN-GP; trains per table | `DataTransformer`, PyTorch layers | `syntho_hive/core/models/ctgan.py` |
| `DataTransformer` | Reversible encode/decode of raw tabular data to tensor | Used by `CTGAN` (fit, transform, inverse_transform) | `syntho_hive/core/data/transformer.py` |
| `ClusterBasedNormalizer` | Bayesian GMM normalization for continuous columns | Owned by `DataTransformer` | `syntho_hive/core/data/transformer.py` |
| `PIISanitizer` | PII column detection and replacement | Called by `StagedOrchestrator` during data ingestion | `syntho_hive/privacy/sanitizer.py` |
| `StatisticalValidator` | KS test + TVD + Frobenius correlation comparison | Called by `Synthesizer.generate_validation_report()` | `syntho_hive/validation/statistical.py` |
| `ValidationReport` | HTML/JSON report rendering | Called by `Synthesizer.generate_validation_report()` | `syntho_hive/validation/report_generator.py` |
| `SparkIO` | Storage abstraction: read/write parquet, delta, CSV, Hive | Called by `StagedOrchestrator` and `Synthesizer` | `syntho_hive/connectors/spark_io.py` |

---

## Recommended Project Structure

The existing structure is correct and should be preserved. The structure rationale:

```
syntho_hive/
├── interface/          # User-facing API only (Synthesizer, config Pydantic models)
│   ├── synthesizer.py  # Façade — single public entry point
│   └── config.py       # Metadata, TableConfig, PrivacyConfig (Pydantic)
├── relational/         # Multi-table orchestration logic
│   ├── orchestrator.py # StagedOrchestrator — table-by-table pipeline
│   ├── graph.py        # SchemaGraph — FK DAG + topological sort
│   └── linkage.py      # LinkageModel — cardinality GMM per child table
├── core/
│   ├── models/         # Generative model implementations
│   │   ├── base.py     # GenerativeModel, ConditionalGenerativeModel (ABC)
│   │   ├── ctgan.py    # CTGAN — default implementation
│   │   ├── layers.py   # PyTorch modules: Discriminator, ResidualLayer, EntityEmbeddingLayer
│   │   └── [tvae.py]   # Future: TVAE as drop-in pluggable alternative
│   └── data/           # Data transformation (independent of any model)
│       └── transformer.py  # DataTransformer + ClusterBasedNormalizer
├── privacy/            # PII detection and sanitization
│   ├── sanitizer.py    # PIISanitizer — pattern matching + replacement
│   └── faker_contextual.py # ContextualFaker — realistic fake value generation
├── validation/         # Output quality assurance
│   ├── statistical.py  # StatisticalValidator — KS, TVD, correlation
│   └── report_generator.py # ValidationReport — HTML/JSON rendering
├── connectors/         # Storage and I/O abstraction
│   ├── spark_io.py     # SparkIO — parquet, delta, CSV, Hive
│   └── sampling.py     # Sampling utilities
└── tests/
    ├── test_transformer.py
    ├── test_interface.py
    ├── test_relational.py
    ├── test_validation.py
    └── e2e_scenarios/
        └── retail_test.py
```

### Structure Rationale

- **`interface/`:** The only layer users import. Keeps the public API surface minimal and stable. `Synthesizer` delegates everything; users never import from `relational/` or `core/` directly.
- **`relational/`:** Isolated orchestration that understands multi-table dependencies. This layer is the "manager" — it owns the training and generation pipeline but delegates model decisions to `core/models/`.
- **`core/models/`:** Pure model logic. Should not know about tables, foreign keys, or Spark. Models receive a `pd.DataFrame` and a `context: Optional[pd.DataFrame]` — nothing else.
- **`core/data/`:** Encoding/decoding is separate from model architecture. This separation is critical: `DataTransformer` must be serialized alongside the model for inference without retraining.
- **`privacy/`:** Runs before training. Stateless per invocation (detect + replace) but `PIISanitizer` stores detection decisions for logging.
- **`validation/`:** Runs after generation, on separate data. No dependencies on models; purely statistical.
- **`connectors/`:** The only layer that knows about Spark or file formats. Models and transformers deal in `pd.DataFrame` only.

---

## Architectural Patterns

### Pattern 1: Abstract Base Class for Pluggable Model Strategy

**What:** `GenerativeModel` and `ConditionalGenerativeModel` are Python ABCs defining the contract every synthesis backend must satisfy: `fit(data, **kwargs)`, `sample(num_rows, **kwargs)`, `save(path)`, `load(path)`. `ConditionalGenerativeModel` extends with optional `context` parameter on both `fit` and `sample`.

**When to use:** Whenever a second model (TVAE, GAN variant, flow-based) needs to be added. The orchestrator should only reference the abstract type, never a concrete class.

**Trade-offs:** The ABC contract is already defined correctly. The problem is that `StagedOrchestrator` currently hardcodes `CTGAN` instantiation (`from syntho_hive.core.models.ctgan import CTGAN`). The fix is to pass a model factory or class reference into the orchestrator rather than importing CTGAN directly.

**Correct pattern:**

```python
# In StagedOrchestrator.__init__:
def __init__(self, metadata: Metadata, spark: SparkSession,
             model_cls: Type[ConditionalGenerativeModel] = CTGAN):
    self.model_cls = model_cls
    ...

# In fit_all, replace direct CTGAN() calls with:
model = self.model_cls(self.metadata, batch_size=batch_size, epochs=epochs, **model_kwargs)
```

This allows users to pass `model_cls=TVAE` without touching orchestrator internals. The `Synthesizer` `backend` parameter (currently a string) should map to a model class registry, or accept a class directly:

```python
MODEL_REGISTRY = {
    "CTGAN": CTGAN,
    "TVAE": TVAE,  # future
}

# In Synthesizer.__init__:
model_cls = MODEL_REGISTRY.get(backend)
self.orchestrator = StagedOrchestrator(metadata, spark, model_cls=model_cls)
```

**Confidence:** HIGH — this is the standard Strategy pattern; verified from codebase that the ABC exists correctly but orchestrator bypasses it.

---

### Pattern 2: Correct Relational Conditioning (Parent Context Flow)

**What:** For child table generation, each child row must be conditioned on the specific parent row it belongs to. The generator must receive the same parent context used to assign that child's foreign key — not a stale batch-level context from the most recent discriminator step.

**The current bug (ctgan.py:359-362):** During generator training, `real_context_batch` is reused from the last discriminator iteration. The correct approach independently samples context for each generator training step.

**Correct training pattern:**

```python
# --- Train Generator ---
noise = torch.randn(self.batch_size, self.embedding_dim, device=self.device)
if context_data is not None:
    # CORRECT: independently sample context, not reuse discriminator's last batch
    gen_ctx_idx = np.random.randint(0, len(context_data), self.batch_size)
    gen_context_batch = context_data[gen_ctx_idx]
    gen_input = torch.cat([noise, gen_context_batch], dim=1)
else:
    gen_input = noise
```

**Correct generation pattern (orchestrator.py):** Already implemented correctly for the generation path — `context_repeated_vals` repeats each parent row's context attributes once per sampled child count, then passes the full context DataFrame aligned row-for-row with child generation. This must not be changed.

**Data flow for child conditioning:**
```
Parent synthetic data (generated in prior step)
    ↓
linkage.sample_counts(parent_df)  → counts: [n0, n1, n2, ...]
    ↓
np.repeat(parent_ids, counts)      → parent_ids_repeated [aligned per child row]
np.repeat(parent_attrs, counts)    → context_df          [aligned per child row]
    ↓
model.sample(total_child_rows, context=context_df)
    ↓
generated_pdf[driver_fk] = parent_ids_repeated   [foreign key assignment]
```

**Confidence:** HIGH — both the bug and the correct pattern are directly visible in the source.

---

### Pattern 3: Model Serialization (What Must Be Saved and Loaded)

**What:** A fitted CTGAN model for inference requires four components. The current `save()` only persists two (generator and discriminator `state_dict`). Loading without the other two makes inference impossible.

**The four required components:**

| Component | Why Needed | Current Status |
|-----------|-----------|----------------|
| `generator.state_dict()` | Generator network weights for `sample()` | Saved |
| `discriminator.state_dict()` | Discriminator weights (needed for resume training) | Saved |
| `transformer` (DataTransformer object) | Encodes input for inference; inverse_transform output | **NOT saved — bug** |
| `context_transformer` (DataTransformer) | Encodes parent context at sample time | **NOT saved — bug** |
| `data_column_info` | Layout map: column positions, types, embedding dims | **NOT saved — bug** |
| `embedding_layers.state_dict()` | Entity embedding weights for categorical columns | **NOT saved — bug** |
| Hyperparameters (dims, thresholds) | Required to reconstruct network architecture before loading weights | **NOT saved — bug** |

**Correct serialization pattern (use `pickle` or `joblib` for full object):**

```python
def save(self, path: str) -> None:
    import pickle
    checkpoint = {
        "generator_state": self.generator.state_dict(),
        "discriminator_state": self.discriminator.state_dict(),
        "embedding_layers_state": self.embedding_layers.state_dict(),
        "transformer": self.transformer,        # full fitted object
        "context_transformer": self.context_transformer,
        "data_column_info": self.data_column_info,
        "hyperparams": {
            "embedding_dim": self.embedding_dim,
            "generator_dim": self.generator_dim,
            "discriminator_dim": self.discriminator_dim,
            "embedding_threshold": self.embedding_threshold,
            "discriminator_steps": self.discriminator_steps,
        }
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

def load(self, path: str) -> None:
    import pickle
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    self.transformer = checkpoint["transformer"]
    self.context_transformer = checkpoint["context_transformer"]
    self.data_column_info = checkpoint["data_column_info"]
    hp = checkpoint["hyperparams"]
    # Reconstruct networks using saved hyperparams before loading weights
    self._build_model(
        transformer_output_dim=self.transformer.output_dim,
        context_dim=self.context_transformer.output_dim
    )
    self.generator.load_state_dict(checkpoint["generator_state"])
    self.discriminator.load_state_dict(checkpoint["discriminator_state"])
    self.embedding_layers.load_state_dict(checkpoint["embedding_layers_state"])
```

**The `save/load` methods must also be defined on the ABC** to enforce that any pluggable model implements full-state serialization, not just weights.

**Confidence:** HIGH — directly confirmed by reading ctgan.py:482-503 and the TODO comment on line 491.

---

### Pattern 4: Memory-Safe Multi-Table Generation

**What:** When generating multi-table schemas with an `output_path_base`, generated DataFrames should not be accumulated in memory after writing to disk. Only child tables that need their parent's data for FK assignment should retain the parent in memory.

**The current bug (orchestrator.py:224-226):** Line 226 always stores `generated_pdf` in `generated_tables`, regardless of whether disk output is enabled. For a 10-table schema with 100M rows total, this causes OOM.

**Correct pattern (disk-backed generation):**

```python
# After writing to disk, free the DataFrame from memory unless a downstream child
# table that hasn't been generated yet depends on this table as its driver parent.

pending_parent_tables = _compute_which_tables_still_need_parent(
    table_name, generation_order, config_map
)

if output_path_base and not pending_parent_tables:
    # Safe to discard — no downstream child needs this in memory
    del generated_pdf
else:
    generated_tables[table_name] = generated_pdf
```

**Simpler alternative (always read from disk when output_path_base is set):**

```python
# When output_path_base is set, child tables always read their parent from disk.
# Never store in generated_tables at all.
if output_path_base:
    parent_path = f"{output_path_base}/{driver_parent_table}"
    parent_df = self.io.read_dataset(parent_path).toPandas()
else:
    parent_df = generated_tables[driver_parent_table]
```

The generation path already has this logic for reading parents (lines 167-171) — the fix is simply to not store `generated_pdf` in `generated_tables` when `output_path_base` is set. The return value changes to return paths instead of DataFrames in that mode, which is already the intended API per `Synthesizer.sample()`.

**Confidence:** HIGH — confirmed bug in orchestrator.py:224-226 with comment acknowledging the risk.

---

### Pattern 5: Explicit Error Propagation

**What:** All exception handling must use specific exception types and must propagate errors with context. Bare `except:` clauses that swallow `KeyboardInterrupt` and `SystemExit` must be eliminated.

**Current violations:**
- `synthesizer.py:153`: `except:` with no type — catches everything, silently falls back
- `transformer.py:228-229`: `except Exception: pass` — constraint conversion failure silently ignored
- `transformer.py:245-246`: `except Exception: pass` — type cast failure silently ignored

**Correct pattern:**

```python
# synthesizer.py: replace bare except with typed
try:
    df = self.spark.read.table(path)
except AnalysisException:
    # Expected: table not found, fall back to path read
    df = self.spark.read.format("delta").load(path)

# transformer.py: replace silent pass with warning/raise
try:
    original_values = original_values.astype(int)
except (ValueError, TypeError) as e:
    raise ValueError(
        f"Constraint dtype='int' could not be applied to column '{col}': {e}"
    ) from e
```

**Confidence:** HIGH — both violations are directly identified in CONCERNS.md and confirmed in source code.

---

## Data Flow

### Training Flow (fit)

```
User: synth.fit(data, epochs=300, batch_size=500)
    ↓
Synthesizer.fit()
    ↓ validates SparkSession, resolves data paths
StagedOrchestrator.fit_all(real_paths, epochs, batch_size)
    ↓ iterates: metadata.tables (any order — training is independent)
    ↓ for each table:
        SparkIO.read_dataset(path) → Spark DataFrame
            ↓ .toPandas() → pandas DataFrame
        PIISanitizer.analyze(df) [detects PII columns]
            ↓ [if PII found: ContextualFaker.replace()]
        DataTransformer.fit(df, table_name)   [profiles columns, builds encoders]
        DataTransformer.transform(df)          [raw data → numpy tensor]
        [if child table]:
            LinkageModel.fit(parent_df, child_df, fk_col, pk_col)
            join child + parent on FK → context_df
        CTGAN.fit(df_tensor, context=context_df)
            ↓ [epochs × steps_per_epoch]:
                Discriminator training (WGAN-GP, discriminator_steps iterations)
                Generator training (independently sampled context — after fix)
                [if checkpoint_dir]: save best model checkpoint
        StagedOrchestrator.models[table_name] = fitted CTGAN
        StagedOrchestrator.linkage_models[table_name] = fitted LinkageModel
```

### Generation Flow (sample)

```
User: synth.sample(num_rows={root: 1000}, output_path="/out")
    ↓
Synthesizer.sample()
    ↓
StagedOrchestrator.generate(num_rows_root, output_path_base)
    ↓ iterates: SchemaGraph.get_generation_order() [topological: parents before children]
    ↓ for each table (root first):
        [ROOT TABLE]:
            CTGAN.sample(n_rows) → generated_pdf
            generated_pdf[pk] = range(1, n_rows + 1)
            SparkIO.write_pandas(generated_pdf, output_path/table)
            [after fix: do NOT keep in memory if output_path_base is set]
        [CHILD TABLE]:
            read parent_df from disk (or memory if no output_path_base)
            LinkageModel.sample_counts(parent_df) → counts[]
            np.repeat(parent_ids, counts) → parent_ids_repeated
            np.repeat(parent_attrs, counts) → context_df
            CTGAN.sample(total_child_rows, context=context_df) → generated_pdf
            generated_pdf[driver_fk] = parent_ids_repeated
            [secondary FKs]: np.random.choice(secondary_parent_pks, size=total_child_rows)
            generated_pdf[pk] = range(1, total_child_rows + 1)
            SparkIO.write_pandas(generated_pdf, output_path/table)
```

### Key Data Flow Invariants

1. **Context alignment is row-for-row**: `context_df` passed to `CTGAN.sample(n, context)` must have exactly `n` rows, where row `i` of context corresponds to the parent attributes for generated row `i`. The `np.repeat` pattern in the orchestrator enforces this correctly.
2. **Topological generation order**: `SchemaGraph.get_generation_order()` guarantees that a table's parent data is always available before the child's generation step. This must not be violated.
3. **FK column exclusion**: `DataTransformer` excludes PK and FK columns from transformation (they are identity-assigned by the orchestrator). Any future model must honor this contract — models receive and return DataFrames without FK columns.
4. **Transformer state persists beyond training**: `DataTransformer` fitted during `CTGAN.fit()` is reused during `CTGAN.sample()`. It must not be refitted. After serialization fix, it must be saved with the model.

### State Management

```
StagedOrchestrator (lives for session duration):
    .models = {table_name: CTGAN}          # one per table, fitted during fit_all()
    .linkage_models = {table_name: LinkageModel}  # one per child table

CTGAN (one per table):
    .transformer = DataTransformer         # fitted on training data
    .context_transformer = DataTransformer # fitted on parent context cols
    .generator = nn.Sequential            # PyTorch Generator network
    .discriminator = Discriminator         # PyTorch Discriminator network
    .embedding_layers = nn.ModuleDict     # one EntityEmbeddingLayer per high-card column
    .data_column_info = List[dict]         # layout map: column → position, type, dims

DataTransformer (one per CTGAN, one as context_transformer):
    ._transformers = {col: sklearn transformer}  # per-column fitted encoder
    ._column_info = {col: {type, dim, ...}}      # layout for transform/inverse
    ._excluded_columns = [pk, fk1, ...]          # columns not modeled
```

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Apache Spark / Delta Lake | `SparkIO` wraps `spark.read` / `spark.write` | SparkSession injected at `Synthesizer` init; optional dependency |
| Hive Metastore | `spark.sql("CREATE TABLE ... LOCATION ...")` via `Synthesizer.save_to_hive()` | Requires Delta format |
| Future SQL databases | Planned `SQLConnector` (Postgres, MySQL, Snowflake, BigQuery) | Not yet implemented |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `Synthesizer` → `StagedOrchestrator` | Direct method calls, passes `Metadata` at init | Orchestrator should never be imported by user code |
| `StagedOrchestrator` → `GenerativeModel` | Should be through ABC type only — currently hardcoded CTGAN import (bug) | Fix: inject model class |
| `CTGAN` → `DataTransformer` | Owns two transformer instances (data + context); calls fit/transform/inverse_transform | Transformers are NOT shared across CTGAN instances |
| `Orchestrator` → `SparkIO` | `SparkIO` injected at orchestrator init; used for all data reads/writes | Models never call SparkIO directly |
| `Metadata` → all layers | Read-only Pydantic object; passed by reference | Never mutated after construction |

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single table, <100K rows | Default Pandas path; no Spark needed; `Synthesizer(spark_session=None)` future state |
| Multi-table, <10M rows total | Current architecture works; OOM bug must be fixed first (disk-backed generation) |
| Multi-table, 10M-1B rows | Spark I/O already in place; must eliminate `toPandas()` round-trips (CTGAN in Spark UDFs — future) |
| Training data >GPU VRAM | Gradient accumulation to simulate larger batch sizes; currently missing |

### Scaling Priorities

1. **First bottleneck (current):** Memory accumulation — all generated tables held in RAM. Fix: disk-backed generation (described in Pattern 4 above). Blocking for production use on multi-table schemas.
2. **Second bottleneck:** `toPandas()` serialization — orchestrator calls `toPandas()` at lines 60, 87, 169, 207. For 10M+ row tables, each conversion is expensive. Fix: cache the Pandas representation within a single `fit_all()` call to avoid repeated conversion.
3. **Third bottleneck:** WGAN-GP gradient penalty — computed every discriminator step. For large batches this dominates training time. Fix: reduce frequency (every N steps) or switch to spectral normalization.

---

## Anti-Patterns

### Anti-Pattern 1: Hardcoding the Model Class in the Orchestrator

**What people do:** `from syntho_hive.core.models.ctgan import CTGAN` at the top of `orchestrator.py`; then `model = CTGAN(...)` inline in `fit_all()`.

**Why it's wrong:** Makes swapping models require editing orchestrator internals. Violates the open/closed principle. The ABC (`ConditionalGenerativeModel`) exists precisely to avoid this — bypassing it defeats its purpose.

**Do this instead:** Inject the model class as a constructor parameter. The orchestrator instantiates `self.model_cls(...)` without importing any concrete class. Concrete class binding happens at the `Synthesizer` level via a model registry.

---

### Anti-Pattern 2: Saving Only Neural Network Weights Without the Transformer

**What people do:** `torch.save({"generator": self.generator.state_dict(), ...}, path)`.

**Why it's wrong:** A loaded CTGAN without its `DataTransformer` cannot call `inverse_transform()` on generated tensors — it has no column layout, no cluster means, no encoder state. The model is unusable for inference.

**Do this instead:** Serialize the full CTGAN object state including fitted transformers, `data_column_info`, and hyperparameters. Use `pickle` for the full bundle; `torch.save` for `state_dict` entries within it.

---

### Anti-Pattern 3: Reusing Discriminator's Context Batch for Generator Training

**What people do:** After the discriminator training loop, reuse `real_context_batch` (the last batch sampled for the discriminator) when training the generator.

**Why it's wrong:** The generator sees context drawn from a single small batch of the parent distribution. Over many steps, it overfits to frequently appearing context values while rarely seeing rare parents. This causes the conditional generation quality to degrade for minority parent segments, which is exactly the class imbalance problem CTGAN is designed to address.

**Do this instead:** For each generator training step, independently sample a fresh random minibatch of context rows from `context_data`. This ensures the generator sees the full parent distribution during training.

---

### Anti-Pattern 4: Accumulating All Generated DataFrames in Memory

**What people do:** `generated_tables[table_name] = generated_pdf` unconditionally, even when `output_path_base` is set (data already written to disk).

**Why it's wrong:** For a multi-table schema with 100M total child rows, all tables are held in memory simultaneously. The OOM failure happens silently with no useful error message.

**Do this instead:** When `output_path_base` is set, do not store in `generated_tables`. Child tables read their parent data from disk via `SparkIO`. Memory usage stays bounded to one table's data at a time.

---

### Anti-Pattern 5: Bare Exception Clauses That Swallow SystemExit

**What people do:** `except:` with no type specification, or `except Exception: pass` with no logging.

**Why it's wrong:** `except:` catches `KeyboardInterrupt` and `SystemExit`, making the process unresponsive to Ctrl+C. Silent `pass` means constraint violations produce wrong data with no warning — the user sees plausibly-formatted output that silently violates their stated schema constraints.

**Do this instead:** Use typed `except` clauses (`except AnalysisException`, `except ValueError`). For constraint violations, either raise with an actionable message or log a WARNING so the user can investigate. Never `pass` silently on a path that affects output correctness.

---

## Suggested Build Order

This ordering minimizes risk and maximizes testability at each step. Later phases depend on earlier phases being stable.

### Phase 1: Core Correctness (Blocking — nothing works reliably without these)

Fix these four bugs before adding any features. In order:

1. **Fix bare exception handling** (`synthesizer.py:153`, `transformer.py:228-229`, `transformer.py:245-246`): Unblocks debugging of every subsequent fix. Lowest risk change. Prerequisite for everything.

2. **Fix context data mismatch in generator training** (`ctgan.py:359-362`): Independently sample context for generator training step. This directly affects relational conditioning quality. Moderate risk — requires understanding the training loop. Test: train a simple parent-child schema and verify child attributes correlate with parent context attributes in the output.

3. **Fix model serialization** (`ctgan.py:482-503`): Replace `torch.save` of state dicts with full-object pickle including transformers, `data_column_info`, and embedding layers. Requires adding load validation. Test: save after fit, load, call sample without refitting — verify output matches schema.

4. **Fix memory accumulation in multi-table generation** (`orchestrator.py:224-226`): When `output_path_base` is set, do not store `generated_pdf` in `generated_tables`; read parents from disk. Test: generate a 3-table schema with large row counts and verify peak memory stays bounded.

### Phase 2: Model Pluggability (Enables Extension)

5. **Refactor orchestrator to inject model class**: Remove direct CTGAN import from orchestrator. Add `model_cls` parameter with `CTGAN` as default. Add model registry to `Synthesizer`. No behavior change — only wiring change. Test: existing tests pass unchanged. Then add a trivial `PassthroughModel` as a second implementation to verify the plug works.

6. **Resolve/remove `_apply_embeddings()` stub** (`ctgan.py:132-185`): Either complete it to replace the inline embedding logic in the training loop, or remove it and document why embedding processing is inline-only. This is a clarity fix, not a bug — but leaving a dead stub with confused comments creates confusion for any contributor adding a second model.

### Phase 3: Validation and Quality Gates (Enables Trustworthy Output)

7. **Add training quality metrics emission**: The training loop already logs `loss_G` and `loss_D` to CSV. Extend to emit a final summary quality score post-training. Define a threshold-based quality gate that `fit()` can optionally enforce.

8. **Add post-generation validation auto-run**: Allow `sample()` to optionally run `StatisticalValidator` on the generated output against a reference sample and return quality metrics alongside the data.

### Phase 4: SQL Connector (Widens Usability)

9. **Add SQL database connector**: `Postgres/MySQL/Snowflake/BigQuery` connector that reads tables into `pd.DataFrame` and passes them to the existing fit pipeline. This requires no changes to the model or orchestrator — only a new `connectors/sql_io.py` class.

### Phase 5: Test Coverage (Locks in Correctness)

10. **CTGAN embedding roundtrip tests**: Unit tests for single categorical column, multiple categoricals, mixed continuous+categorical, null handling. Verify output shapes at each stage.

11. **Relational integrity end-to-end tests**: Cover referential integrity (all FKs reference valid PKs), cardinality distribution match, and secondary FK assignment.

12. **Model serialization roundtrip test**: Save after fit, load fresh instance, call sample — verify output schema matches expected and matches sample output from original fitted model.

---

## Sources

- Direct source code analysis of `syntho_hive/` package (confidence: HIGH)
  - `syntho_hive/core/models/ctgan.py` (CTGAN training loop, save/load, embedding handling)
  - `syntho_hive/core/models/base.py` (GenerativeModel, ConditionalGenerativeModel ABCs)
  - `syntho_hive/relational/orchestrator.py` (StagedOrchestrator fit_all and generate pipelines)
  - `syntho_hive/relational/graph.py` (SchemaGraph topological sort)
  - `syntho_hive/relational/linkage.py` (LinkageModel GMM cardinality)
  - `syntho_hive/core/data/transformer.py` (DataTransformer, ClusterBasedNormalizer)
  - `syntho_hive/interface/synthesizer.py` (Synthesizer façade)
  - `syntho_hive/interface/config.py` (Metadata, TableConfig, Pydantic models)
- `.planning/codebase/ARCHITECTURE.md` — existing architecture documentation (confidence: HIGH)
- `.planning/codebase/CONCERNS.md` — known bugs and tech debt (confidence: HIGH)
- `.planning/PROJECT.md` — project goals and constraints (confidence: HIGH)

---

*Architecture research for: SynthoHive — Python SDK for synthetic tabular data generation*
*Researched: 2026-02-22*
