# Pitfalls Research

**Domain:** Synthetic tabular data generation — Python SDK with GAN training, relational synthesis, model persistence, Spark connectors, and statistical validation
**Researched:** 2026-02-22
**Confidence:** HIGH (primary evidence from direct codebase analysis of SynthoHive; supplemented by domain knowledge of CTGAN, WGAN-GP, SDV ecosystem, and Spark/Pandas interop patterns)

---

## Critical Pitfalls

### Pitfall 1: Partial Model Serialization — Saving Weights Without State

**What goes wrong:**
The model checkpoint saves only PyTorch `state_dict()` for generator and discriminator. It does not save the `DataTransformer`, `context_transformer`, `embedding_layers`, or `data_column_info`. When the model is loaded, `inverse_transform()` cannot run because the transformer was never restored. The loaded model is structurally unusable for inference. This is the exact bug at `ctgan.py:482-503`.

**Why it happens:**
Developers familiar with image/NLP GANs think of "the model" as the neural network weights. Tabular synthesis is different: the transformer is co-equal in importance to the network — it is the inverse mapping that converts generated tensors back to a human-readable schema. Without it, generated tensors cannot be decoded. The transformer is also stateful (fitted `BayesianGaussianMixture`, `OneHotEncoder`, `LabelEncoder`, `ClusterBasedNormalizer`), so it cannot be reconstructed from metadata alone without the original data.

**How to avoid:**
Serialize the entire CTGAN object using `pickle` or `joblib.dump()` rather than using `torch.save()` with only state dicts. Alternatively, implement a structured checkpoint that explicitly includes: (1) `DataTransformer` pickle, (2) `context_transformer` pickle, (3) `embedding_layers` state dict, (4) `data_column_info`, (5) hyperparameters, (6) generator state dict, (7) discriminator state dict. Add a `load()` validation check that verifies all required keys are present before attempting inference.

**Warning signs:**
- `load()` runs successfully but `sample()` immediately raises `AttributeError` or `ValueError`
- Checkpoint file size is suspiciously small (weights only = kilobytes; full object should be megabytes for real data)
- Test for model serialization passes using a freshly-fit model object (transformer already in memory) rather than truly loading from cold disk

**Phase to address:**
Model Serialization fix (Milestone: fixing/productizing SynthoHive — immediate priority, blocks all deployment use cases)

---

### Pitfall 2: Stale Context in Conditional Generator Training

**What goes wrong:**
During the generator training step, `real_context_batch` is reused from the most recent discriminator batch rather than being independently sampled. This means the generator trains on context drawn from the discriminator's random index selection, not from a fresh uniform sample of the full context distribution. The generator learns to generate outputs conditioned on a biased subsample of parent contexts, which skews the conditional distribution. This is the bug at `ctgan.py:359-362`.

**Why it happens:**
In vanilla unconditional WGAN-GP, context does not exist, so generator training loops do not sample it. When context conditioning is bolted on, reusing the last discriminator batch's context feels natural and harmless ("it's the same data"). The difference is subtle: discriminator batches are sampled with replacement, so in any given epoch some parent contexts appear repeatedly while others are underrepresented. The generator's gradient updates become correlated with the discriminator's sampling order, not the true parent marginal distribution.

**How to avoid:**
Independently sample context indices for the generator step:
```python
# WRONG — reuses discriminator batch context
gen_input = torch.cat([noise, real_context_batch], dim=1)

# CORRECT — independent resample
gen_ctx_idx = np.random.randint(0, len(context_data), self.batch_size)
gen_context_batch = context_data[gen_ctx_idx]
gen_input = torch.cat([noise, gen_context_batch], dim=1)
```
The fix is a single additional line in the generator training block. Add a regression test that verifies generated child data has a parent-FK distribution matching the sampling distribution, not a clumped distribution.

**Warning signs:**
- Child table generation produces rows heavily skewed toward a subset of parent IDs
- Validation report shows good marginal distributions per column but poor cross-table correlation
- Training loss converges, but generated child data visually looks "clustered" around a few parent contexts

**Phase to address:**
Context conditioning fix (Milestone: fixing/productizing SynthoHive — high priority, directly causes referential integrity failures in non-trivial schemas)

---

### Pitfall 3: Memory Accumulation During Multi-Table Generation

**What goes wrong:**
`StagedOrchestrator.generate()` stores every generated `pd.DataFrame` in `generated_tables` dict regardless of whether `output_path_base` is set. When writing to disk for large schemas, the orchestrator simultaneously holds all tables in memory — root table + all child tables — while also writing each to Spark. For a schema with a 10M-row root and 100M-row child tables, this causes OOM before the pipeline completes. The acknowledged bug is at `orchestrator.py:224-226`.

**Why it happens:**
The dict is needed for sequential table generation (child tables need parent data to build context). The simplest implementation retains everything. The "write to disk then read back" pattern for large datasets requires a structural change — you must read the previously written parent from disk rather than memory — which is a less obvious code path than just keeping the dict.

**How to avoid:**
When `output_path_base` is set, immediately release tables from the in-memory dict after writing them to disk. For child tables that need parent context during generation, read from the written disk path rather than from `generated_tables`:
```python
# After writing to disk:
self.io.write_pandas(generated_pdf, output_path)
if output_path_base:
    del generated_tables[table_name]  # Release memory
# Child tables already read parent from disk via output_path_base branch
```
The orchestrator already has two code paths (lines 167-171) that handle reading parent data from disk vs. memory — the memory release is the missing half.

**Warning signs:**
- OOM crashes on multi-table generation when `output_path_base` is set
- Memory usage grows linearly with total generated rows rather than staying bounded to the largest single table
- `generated_tables` dict length equals number of tables at end of generation loop

**Phase to address:**
Memory-safe multi-table generation fix (Milestone: fixing/productizing SynthoHive — immediate priority, blocks large schema use cases)

---

### Pitfall 4: Silent Error Swallowing Masking Real Failures

**What goes wrong:**
Two categories of silent failures exist in the codebase:

1. Bare `except:` in `synthesizer.py:153` catches all exceptions (including `KeyboardInterrupt`, `SystemExit`) during the Spark table read fallback. A permission error, network failure, or schema mismatch silently falls through to a path that may silently corrupt or produces wrong data.

2. `except Exception: pass` blocks in `transformer.py:228-230` and `transformer.py:245-246` allow numeric type conversion failures to proceed silently. Generated data with unconvertible constraint types passes through without warning, violating user-specified constraints (e.g., a column that should be `int` remains `float`).

**Why it happens:**
Two distinct root causes. For (1): defensive try/except blocks written to handle a known expected case (table-vs-path duality) are written too broadly. For (2): constraint enforcement was added after the core transformer worked, with `pass` blocks as a placeholder to avoid crashes during development that were never replaced with proper handling.

**How to avoid:**
Replace bare `except:` with `except (AnalysisException, Exception) as e:` with explicit logging. Replace `except Exception: pass` with at minimum a `logger.warning()` call, and ideally a strict raise when constraint violations occur. Establish a project-wide rule: no bare `except:` and no `except Exception: pass` — enforce with a linting rule (`flake8 --select=E722`).

**Warning signs:**
- Debugging a pipeline failure produces no error message, only wrong output
- Training completes but generated data silently violates constraint bounds
- Stack traces from downstream consumers (e.g., SQL inserts) fail with schema errors when data was expected to be valid

**Phase to address:**
Error handling hardening (Milestone: fixing/productizing SynthoHive — high priority, makes all other bugs unfindable when they occur)

---

### Pitfall 5: GAN Training Instability — Using Generator Loss as Best-Model Criterion

**What goes wrong:**
The checkpointing logic at `ctgan.py:414-416` saves the "best model" checkpoint when `current_loss_g < best_loss`. For WGAN-GP, generator loss is not a reliable proxy for output quality. Generator loss decreasing can mean the generator found an adversarial shortcut (mode collapse or gradient exploitation) rather than genuinely improving distribution fidelity. Saving based on minimum generator loss can checkpoint a collapsed model as the "best."

**Why it happens:**
In supervised learning, minimum validation loss = best model. This intuition breaks down in adversarial training where generator and discriminator losses are in dynamic tension. WGAN-GP theory says the Wasserstein distance (approximated by `d_real - d_fake`) is the meaningful loss, but even that is not a direct quality metric for downstream use.

**How to avoid:**
Replace checkpoint criterion with a validation-based approach: after every N epochs, generate a small sample and compute a statistical quality metric (KS divergence, TVD) against the real training data. Save the checkpoint that achieves the best validation metric, not the best generator loss. Alternatively, use a fixed-epoch checkpoint cadence (e.g., save every 50 epochs) and let the user choose the best checkpoint via the validation report post-training.

**Warning signs:**
- Best checkpoint produces visually worse data than the final checkpoint
- Generator loss continues decreasing while discriminator loss plateaus (mode collapse signature)
- Generated data has one or two dominant categories per categorical column (mode collapse in categorical features)

**Phase to address:**
Statistical quality gates (Milestone: fixing/productizing SynthoHive — medium priority, affects model quality visibility)

---

### Pitfall 6: Linkage Model Using GMM on Child Counts — Wrong Distribution Family

**What goes wrong:**
`LinkageModel` uses `GaussianMixture` to model the distribution of child counts per parent. Child count distributions are non-negative integers that typically follow a Negative Binomial or Zero-Inflated Poisson distribution (many parents have 0-5 children; a long tail of parents with many). GMM can produce negative samples and non-integer samples, both of which are clipped/rounded in `sample_counts()` (lines 84-85). For heavily skewed distributions (e.g., 80% of parents have exactly 1 child), GMM severely underestimates the zero/one peak and generates spurious counts in the 5-20 range. The result: synthetic schemas have wrong parent-to-child cardinality ratios.

**Why it happens:**
GMM is the go-to density estimator for continuous distributions. Child count is technically discrete and bounded (non-negative integers). GMM is a valid first approximation but not the right tool. The clipping and rounding in `sample_counts()` are band-aids that acknowledge this mismatch but do not fix it.

**How to avoid:**
Replace GMM with a count-distribution model. Options in order of increasing complexity:
1. **Empirical distribution**: Sample directly from the observed count histogram using `np.random.choice(observed_counts)`. No model needed, captures exact distribution.
2. **Negative Binomial**: Fit `scipy.stats.nbinom` via method of moments. Works well for overdispersed count data (variance > mean).
3. **Zero-Inflated Poisson**: Use when many parents have exactly 0 children.

The empirical approach is sufficient for most real-world cases and is trivial to implement correctly.

**Warning signs:**
- `sample_counts()` returns many negative values before clipping (logged as zeros)
- Child table has significantly different total rows than expected given parent size
- Cardinality distribution between real and synthetic data diverges in validation report
- GMM `n_components` is capped at `min(5, len(np.unique(X)))` — small datasets hit this cap often

**Phase to address:**
Relational synthesis fidelity (Milestone: fixing/productizing SynthoHive — medium priority)

---

### Pitfall 7: FK Type Mismatch With Silent String Fallback

**What goes wrong:**
In `linkage.py:36-44`, when parent PK dtype does not match child FK dtype, the code falls back to converting both to strings. This can cause silent data corruption: integer PK `1234` becomes string `"1234"`, and the FK assignment in the orchestrator (`orchestrator.py:197`) sets the FK column to the integer `parent_ids_repeated`. If downstream joins or writes expect integer FKs, the type mismatch causes join failures or schema errors. The CONCERNS.md documents this at `linkage.py:35-44`.

**Why it happens:**
Type coercion as a fallback is a pattern that feels robust ("handle whatever comes in"). In a relational context, silent type coercion means referential integrity can be violated in ways that are not detectable until a consumer tries to join the tables.

**How to avoid:**
Remove the silent fallback. Detect FK/PK type mismatches at `Metadata.validate_schema()` time and raise a clear error requiring the user to explicitly align types in their metadata definition. Add a post-generation assertion: after assigning FKs, verify that `all(fk_value in parent_pk_set for fk_value in generated_pdf[fk_col].values)`.

**Warning signs:**
- Warning message "Could not cast FK to match PK type" appears in logs
- Downstream joins between generated tables fail with key errors
- Parent-child join after generation produces empty result sets

**Phase to address:**
Relational synthesis fidelity / error handling hardening (Milestone: fixing/productizing SynthoHive)

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Save only PyTorch state_dict for checkpointing | Simpler code, no pickle dependency | Model cannot be loaded for inference; defeats purpose of checkpointing | Never — breaks the core use case |
| Reuse last discriminator context batch for generator training | Avoids one sampling call per step | Biased conditional distribution; referential integrity failures | Never — trivial to fix correctly |
| `except: pass` for constraint conversion failures | Avoids crashes on edge case data | Silent constraint violations; generated data violates user-specified rules | Never in production; acceptable only during prototyping with comment |
| All generated DataFrames in memory dict | Simpler code; no disk read-back needed | OOM on any real-scale multi-table schema | Never for multi-table; fine for single-table unit tests |
| `print()` statements for logging | Zero setup; visible in notebooks | Cannot be suppressed, configured, or captured by callers; unprofessional SDK behavior | Never in a published SDK; switch to `logging.getLogger()` |
| GMM for child count modeling | One library; works for simple cases | Wrong distribution family for integer counts; cardinality drift | Only for prototype; replace before production |
| Bare `except:` in Spark table read fallback | Handles both table-name and path inputs | Swallows all exceptions including `KeyboardInterrupt`; debug impossible | Never — specify exception types explicitly |
| `np.random.choice(valid_pks)` for secondary FK assignment | One line; always produces valid FK | Completely random — no correlation with real FK assignment patterns | Acceptable for MVP; needs cardinality-aware assignment for fidelity |

---

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PySpark + Pandas interop | Calling `.toPandas()` multiple times on the same Spark DF in one pipeline stage | Cache the Pandas result; `orchestrator.py` calls `toPandas()` at lines 60, 87, 169, 207 for same tables |
| Spark table write then read-back | Writing Parquet then reading in same job without committing a new Spark session context | Use `spark.catalog.refreshTable()` or explicit cache invalidation after write before re-read |
| Spark optional import | `from pyspark.sql import SparkSession` at top-level; module fails to import without Spark | Use `try/except ImportError` pattern (already done in `synthesizer.py`) — extend to ALL connector files consistently |
| Delta Lake format vs Parquet | `write_pandas()` writes Parquet; `sample()` return says "delta" in output format string | Align format strings to actual format used; ensure write format and read format match across methods |
| Hive SQL injection in `save_to_hive()` | `f"CREATE DATABASE IF NOT EXISTS {target_db}"` — `target_db` is user-supplied and unsanitized | Validate `target_db` against `[a-zA-Z0-9_]+` pattern before executing SQL |
| SparkSession held by Synthesizer | Synthesizer stores `self.spark` reference; if caller's session closes, all subsequent operations fail with opaque Java stack traces | Document that SparkSession must remain active for the lifetime of the Synthesizer object |
| `inferSchema=True` for CSV reads | Schema inference reads entire CSV to determine types; slow on large files | Provide explicit schema via metadata where CSV column types are known |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| All generated tables in memory simultaneously | OOM crash during multi-table `generate()` with large schemas | Release tables from dict after writing to disk when `output_path_base` is set | ~1GB total generated data (10M root + 100M children) |
| WGAN-GP gradient penalty on every discriminator step | Training time dominated by `torch.autograd.grad()` calls; 5x overhead from `discriminator_steps=5` | Compute gradient penalty every N steps (lazy gradient penalty) or use spectral normalization | Batch size > 500 or `embedding_dim` > 256 |
| `toPandas()` without caching in multi-stage pipeline | Each stage re-reads and re-converts the same Spark DF; 2-3x pipeline slowdown | Cache Pandas conversion result in-memory for tables used across multiple pipeline stages | Tables > 1M rows |
| PII detection with regex on every row in sample | Wide tables (100+ columns) with 100-row sample trigger O(columns x patterns x rows x regex) | Pre-compile regex; use column-name heuristics first (already partial in `sanitizer.py:58-62`) | 50+ columns with string content |
| `BayesianGaussianMixture` with `n_components=10` for every continuous column | Slow fit on large cardinality continuous columns; convergence issues with sparse data | Profile and tune `n_components` per column; reduce for low-cardinality continuous columns | Datasets with > 20 continuous columns and < 1000 rows |
| High-cardinality categorical columns with embeddings | Training instability; embedding lookup table underfit with sparse data | Document recommended cardinality/dataset size ratio; add validation warning when `n_unique / n_samples < threshold` | Categorical columns with > 5000 unique values and < 50K training rows |
| Argmax for continuous cluster reconstruction in `ClusterBasedNormalizer` | Mode collapse: all values assigned to most probable cluster; destroys within-cluster variance | During training, use soft sampling (sample cluster from probabilities); use argmax only at inference | Any dataset with multimodal continuous distributions |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| SHA256 hash without salt for PII values (`sanitizer.py:154-156`) | Rainbow table attacks recover small PII spaces (SSNs, zip codes). SHA256 of "123-45-6789" is always the same hash — precomputed table cracks it instantly | Use HMAC with an environment-provided secret key; document that "hash" action is for deduplication consistency, not cryptographic security; recommend "fake" or "drop" for compliance use cases |
| Parent context columns copied verbatim into child synthetic data | If parent contains sensitive attributes (salary, health status), they appear unchanged in child rows and can re-identify individuals via join attacks | Warn when `parent_context_cols` overlap with `pii_cols`; apply privacy rules to context columns before using them in generation |
| Unsanitized SQL in `save_to_hive()` | `target_db` and potentially `table` names passed directly into Spark SQL — SQL injection if values come from external configuration | Validate table and database names against `[a-zA-Z0-9_]+` pattern before constructing SQL |
| PII detection using only first 100 rows | PII in low-frequency rows (e.g., sporadically filled SSN field) missed by 50%-threshold heuristic | Allow users to declare PII columns explicitly in `TableConfig.pii_cols` (field already exists); treat explicit declarations as authoritative; use sampling only for undeclared columns |
| Regex patterns matching non-PII values (false positives) | Numeric columns (product codes, order IDs matching `\d{4}[-\s]?\d{4}...`) flagged as credit cards and replaced with fake data, silently corrupting the schema | Add column name exclusion list; require content pattern match AND column name heuristic before flagging |

---

## UX Pitfalls

Common user experience mistakes in a data engineering SDK context.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| `print()` statements as the only observability | Callers cannot suppress output, route to file, or integrate with their logging framework | Replace with `logging.getLogger(__name__)` throughout; expose a `verbose=True` flag on `Synthesizer` that sets log level |
| SparkSession required for all operations | Users without Spark clusters cannot use the SDK at all — no Pandas-native path | Separate the Spark-dependent orchestration from the single-table CTGAN path; allow `Synthesizer` to work with `Dict[str, pd.DataFrame]` input for non-Spark use |
| No progress indication during training | 300-epoch training with print every 10 epochs looks like a hung process | Use `tqdm` progress bar with live loss display; expose `on_epoch_end` callback for custom monitoring |
| `fit()` fails silently mid-training | If training crashes at table 3 of 5, no partial checkpoint is saved; user must restart from scratch | Save per-table checkpoints as each table completes training; allow `fit()` to resume from partial state |
| No dry-run validation before training | Users discover metadata errors (wrong FK format, missing table configs) only after hours of training | Add `Metadata.validate_schema()` call at the top of `fit()` before loading any data; add `Synthesizer.validate()` dry-run method |
| Validation report has no pass/fail summary | Users must read every row of the column metrics table to assess overall quality | Add overall quality score (e.g., "X/Y columns pass KS test at p=0.05") to report header |
| `sample()` returns different types depending on `output_path` | Returns `Dict[str, pd.DataFrame]` if `output_path=None`, returns `Dict[str, str]` otherwise — callers must branch on return type | Pick one return type or use a typed result object; `Dict[str, pd.DataFrame]` as always-available is more composable |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Model save/load**: The `save()` and `load()` methods exist and run without error — but verify that a model loaded from disk can actually generate data (call `sample()` on a cold-loaded model with no in-memory transformer state)
- [ ] **Checkpointing**: `fit()` creates `best_model.pt` in `checkpoint_dir` — but verify the "best" checkpoint survives loading and produces non-degenerate samples (currently saves based on generator loss, which does not guarantee quality)
- [ ] **Referential integrity**: Multi-table generation completes and produces parent + child CSVs — but verify that every FK value in child tables actually appears in parent PK column (run `assert all(child[fk_col].isin(parent[pk_col]))` after generation)
- [ ] **Constraint enforcement**: `DataTransformer.inverse_transform()` applies min/max clipping — but verify constraints are actually applied (the `except Exception: pass` blocks at lines 228-230 and 245-246 silently skip enforcement on conversion failure)
- [ ] **Context conditioning**: CTGAN trains with parent context — but verify that generated child data actually reflects parent context (e.g., children of "premium" parents should have higher-value order amounts if that correlation exists in training data)
- [ ] **PII sanitization**: `PIISanitizer.analyze()` detects PII columns — but verify detection is applied before training, not only in the validation report (the orchestrator's `fit_all()` does not call `PIISanitizer` — it is not wired into the training path at all)
- [ ] **Statistical validation**: `ValidationReport.generate()` writes a report file — but verify the report reflects correct metrics (TVD threshold of 0.1 is arbitrary and hardcoded; KS p-value threshold of 0.05 is also hardcoded without documentation)
- [ ] **Spark optional**: `SparkIO` and `StagedOrchestrator` import `pyspark` in `try/except ImportError` blocks — but verify that non-Spark users get a helpful error message rather than `AttributeError: 'type' object has no attribute 'read'` at runtime

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Broken model serialization (saved model unusable) | HIGH | Must retain original training data; refit model with corrected serialization; no way to recover transformer state from weights-only checkpoint |
| Stale context in child generation (wrong FK distribution) | MEDIUM | Refit child table model with corrected context sampling; parent table models are unaffected; retrain only child table CTGANs |
| OOM during multi-table generation | LOW | Reduce `num_rows_root` to fit in available memory; or implement disk-spillover fix and rerun; generated tables written before OOM can be reused if generation order is deterministic |
| Silent constraint violations in output | MEDIUM | Re-run `inverse_transform()` with strict mode after fixing exception handling; no model retraining needed if the transformer is the only broken component |
| Mode collapse during GAN training | HIGH | Restart training with reduced learning rate (`2e-5` instead of `2e-4`); increase `discriminator_steps` to 10; add gradient clipping; reduce batch size; no fast recovery path — must retrain |
| FK type mismatch causing orphaned child rows | LOW | Re-run generation only (no retraining); fix type alignment in metadata and regenerate |
| GMM generating wrong cardinality | MEDIUM | Refit linkage model only (fast); rerun child table generation; parent table generation is unaffected |
| PII not sanitized before training (privacy leak) | HIGH | Discard the trained model (it has seen PII); sanitize data; retrain from scratch; audit generated outputs for PII leakage |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Partial model serialization | Phase: Model Serialization Fix | Load a saved model in a fresh Python process with no training data available; call `sample(10)` — must succeed |
| Stale context in child generation | Phase: Relational Synthesis Fix | Generate 1000 child rows; verify FK distribution matches parent PK distribution within 5% |
| Memory accumulation in multi-table generation | Phase: Memory-Safe Generation | Generate 10x scale test; memory profiler shows peak memory equals 2 tables, not all tables simultaneously |
| Silent error swallowing | Phase: Error Handling Hardening | Run `flake8 --select=E722` (bare except); run with intentionally broken Spark config — must see actionable error, not silent failure |
| Generator loss as checkpoint criterion | Phase: Statistical Quality Gates | Best checkpoint produces lower TVD/KS than final checkpoint (or equal); not worse |
| Wrong distribution family for child counts | Phase: Relational Synthesis Fix | Cardinality distribution of generated child counts matches real distribution within 10% Wasserstein distance |
| FK type mismatch with silent fallback | Phase: Error Handling Hardening / Relational Synthesis Fix | Pass mismatched FK/PK types in metadata — must raise `ValueError` at `validate_schema()` time, not silently corrupt data |
| PII not wired into training path | Phase: Privacy Integration | Verify `PIISanitizer.analyze()` is called during `fit_all()` before `CTGAN.fit()`; training data in any checkpoint must not contain raw PII |
| Unsanitized SQL in `save_to_hive()` | Phase: Error Handling Hardening | Pass `target_db="malicious; DROP DATABASE"` — must raise validation error, not execute |
| No progress indication during training | Phase: UX / SDK Productization | Training a 50-epoch model produces visible epoch-by-epoch progress output that can be suppressed via `verbose=False` |

---

## Sources

- Direct codebase analysis: `syntho_hive/core/models/ctgan.py` (lines 359-362, 414-416, 482-503)
- Direct codebase analysis: `syntho_hive/relational/orchestrator.py` (lines 60, 87, 169, 207, 224-226)
- Direct codebase analysis: `syntho_hive/relational/linkage.py` (lines 35-44, 58-87)
- Direct codebase analysis: `syntho_hive/core/data/transformer.py` (lines 209-255)
- Direct codebase analysis: `syntho_hive/interface/synthesizer.py` (lines 86-123, 152-155)
- Direct codebase analysis: `syntho_hive/privacy/sanitizer.py` (lines 25-31, 154-156)
- `.planning/codebase/CONCERNS.md` — full bug audit (2026-02-22)
- `.planning/PROJECT.md` — product requirements and known bugs
- Domain knowledge: CTGAN (Xu et al., 2019) — WGAN-GP training dynamics, conditional vector construction, VGM normalization
- Domain knowledge: SDV (Synthetic Data Vault) ecosystem — multi-table synthesis patterns, HMA1 relational synthesis approach
- Domain knowledge: WGAN-GP (Gulrajani et al., 2017) — gradient penalty properties and training stability requirements
- Domain knowledge: Spark-Pandas interop patterns and `toPandas()` performance characteristics

---
*Pitfalls research for: SynthoHive — synthetic tabular data generation Python SDK*
*Researched: 2026-02-22*
