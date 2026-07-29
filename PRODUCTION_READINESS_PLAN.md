# SynthoHive Production Readiness Plan

This document is the result of a full audit of the SynthoHive codebase (v1.4.0,
~4,700 LOC). Every finding below was verified by reading the code (and, where
noted, by executing the failing path). Findings are grouped by theme and marked
with a status:

- ✅ **Fixed in this branch** — implemented, with regression tests where applicable.
- 🗺️ **Roadmap** — designed but deferred; larger features that deserve their own PR.

Severity: **C** = critical, **H** = high, **M** = medium, **L** = low.

---

## 1. Model correctness (CTGAN core)

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 1.1 | C | **Generator has no output activations.** The generator ends in a bare `nn.Linear`; the CTGAN paper applies `tanh` to continuous scalars and gumbel-softmax to every one-hot block (categorical, GMM mode indicator, null indicator). The discriminator could trivially separate real (exact 0/1 one-hots) from fake (unbounded floats), crippling training, and `inverse_transform` argmaxed raw noise. | ✅ Activations (`tanh` on scalars, `gumbel_softmax` on one-hot blocks, `sigmoid` on null indicators) now applied via a single `_apply_activations()` helper used in the D step, G step, and `sample()`. |
| 1.2 | C | **`IndexError` on generated data when a column has fewer distinct values than GMM components.** `ClusterBasedNormalizer.fit()` clamps the mixture to `effective_components`, but `transform`/`inverse_transform` used the full `n_components` width, so an argmax landing on a dead component indexed `means`/`stds` out of bounds. | ✅ `means`/`stds` padded to `n_components`; dead components can no longer crash inverse transform. |
| 1.3 | H | **Fake batch not detached in the discriminator step** — every one of the 5 D steps backpropagated through the full generator for no reason (~5× wasted compute/memory; correct only by accident of `zero_grad` ordering). | ✅ Fake batch built under `torch.no_grad()` in the D loop. |
| 1.4 | H | **NaN inputs at transform time crash or silently poison training** when fit data had no nulls; ±inf never handled. A NaN loss trained silently to garbage (no divergence detection despite `TrainingError` documenting it). | ✅ Transform always imputes; non-finite values rejected with a clear error at fit; NaN loss now raises `TrainingError` with the epoch number. |
| 1.5 | H | **`sample()` before `fit()`** died with `AttributeError: 'NoneType' object has no attribute 'training'`. | ✅ Raises `GenerationError("Model is not fitted…")`. |
| 1.6 | M | Normalized scalar unclipped (CTGAN clips to ±0.99); outliers destabilized training. | ✅ Clipped. |
| 1.7 | M | Re-`fit()` with a different schema silently reused the old network and crashed deep in the forward pass. | ✅ Model rebuilt on every `fit()`. |
| 1.8 | M | `sample()` left the model in eval mode if anything raised mid-sample. | ✅ `try/finally` restores training mode. |
| 1.9 | M | `_apply_embeddings` was ~50 lines of dead code with stream-of-consciousness comments. | ✅ Deleted. |
| 1.10 | M | `enforce_constraints` docstring promised row-dropping, code raised; `valid_mask` computed and unused. | ✅ Docstring and behavior aligned (raises, documented; dead mask removed). |
| 1.11 | M | Checkpoint validation consumed the training RNG stream — same seed gave different models depending on checkpointing settings. | ✅ RNG state saved/restored around validation sampling. |
| 1.12 | H | **The model is not actually CTGAN**: the paper's conditional vector + training-by-sampling (the fix for imbalanced categoricals) is absent. This is a WGAN-GP tabular GAN with entity embeddings and parent-context conditioning. | 🗺️ Implement cond-vector + log-frequency sampling; until then the docs state the architecture honestly. |
| 1.13 | M | Whole dataset moved to device up front (GPU OOM on large tables); `sample()` generates all rows in one forward pass. | 🗺️ Batch-wise device transfer and chunked sampling. |
| 1.14 | L | Unseen categories at transform silently mapped to alphabetically-first class. | ✅ Mapped to null sentinel when present, with a warning. |
| 1.15 | L | Datetime columns fall into the categorical path (stringified timestamps). | 🗺️ Epoch-numeric datetime transform. |

## 2. Serialization & security

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 2.1 | C | **`torch.load(..., weights_only=False)` on plain state dicts** — reopened arbitrary-code-execution via pickle for zero benefit (the files are pure tensors; the comment claiming otherwise was wrong). | ✅ `weights_only=True` + `map_location=self.device`. |
| 2.2 | C | **`Synthesizer.save()` failed in any real Spark deployment**: `__getstate__` dropped its own `spark`, but the pickled orchestrator still carried the live `SparkSession` and `SparkIO`. And after `load()` there was no way to re-attach a session at all. | ✅ `StagedOrchestrator.__getstate__/__setstate__` drop `spark`/`io`; `Synthesizer.load(path, spark_session=...)` rebinds a session (or falls back to the local pandas backend). |
| 2.3 | H | Loaded embedding layers never moved to `self.device`; CUDA checkpoints could not load on CPU hosts. | ✅ Embedding layers saved as `state_dict` tensors and loaded with `map_location`, then moved to device. |
| 2.4 | M | `metadata.json` omitted `embedding_threshold`/`batch_size`/`discriminator_steps`, so `load()` restored an incomplete config. | ✅ Full constructor config persisted and restored. |
| 2.5 | M | `transformer.joblib` etc. remain pickle-based (RCE if a checkpoint comes from an untrusted source). | ✅ Documented loudly in `load()`; 🗺️ migrate to a manifest + arrays format. |

## 3. Privacy (the actual point of the library)

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 3.1 | C | **`PrivacyConfig` was a no-op.** The object passed to `Synthesizer` was stored and never read — the API promised PII guardrails it never applied. | ✅ `Synthesizer.fit()` now sanitizes each table's declared `pii_cols` (per `pii_strategy`) before training, so raw PII never reaches the model. Setting `enable_differential_privacy=True` raises `NotImplementedError` instead of silently doing nothing. |
| 3.2 | H | **Unknown sanitizer rule names and unrecognized `action` values silently left raw PII in the output.** | ✅ Both now raise `PrivacyError`. |
| 3.3 | H | **The DOB mask leaked the full birth year** (`"01/15/1990"` → `"******1990"`), and last-4 SSN is itself the sensitive verification token. | ✅ Masking is now full-mask by default; preserved-suffix length is an explicit opt-in per rule. |
| 3.4 | M | Hash salt was random per instance with no way to supply it — broke referential integrity across tables/runs. | ✅ Optional `salt` parameter, documented. |
| 3.5 | M | Faker replacement filled NaN cells with fake values (changing null distribution) and had no seeding support (irreproducible). | ✅ NaNs preserved; `seed` parameter threaded to Faker. |
| 3.6 | M | Row-wise `apply`/`iterrows` faker paths were O(n) Python loops; `.at` with non-unique index corrupted rows. | ✅ Batch fast-path when no context columns; positional assignment. |
| 3.7 | L | Unknown locales silently degraded to `en_US`. | ✅ Raw locale string tried first; fallback logged. |

## 4. Validation & reporting

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 4.1 | C | **Stored XSS in the HTML report** — table/column names, error strings, and raw data values (`top_value` = the mode of a column) were interpolated unescaped; a `<script>` payload in source data executed on report open (verified empirically). | ✅ Every interpolated value passes through `html.escape()`. |
| 4.2 | H | Report generation **hard-crashed on any empty input table** (`AttributeError` via the `{"error": ...}` shape), and all-NaN categorical columns crashed `value_counts().iloc[0]` (verified). | ✅ Both handled; regression tests added. |
| 4.3 | H | **KS-test misuse**: `p > 0.05` as a pass criterion means production-scale data always "fails" and tiny samples always "pass"; discrete/bool columns violate KS assumptions. | ✅ Pass criterion switched to effect size (KS statistic `D < 0.1`); low-cardinality numeric/bool columns routed to the TVD branch. |
| 4.4 | H | Correlation distance silently became `NaN` when column sets differed; `.fillna(0)` fabricated zero correlations for constant columns. | ✅ Columns intersected; undefined pairs masked out and counted. |
| 4.5 | M | Empty-DataFrame return shape `{"error": ...}` broke the per-column contract. | ✅ Typed status shape; report generator handles it. |

## 5. Relational layer

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 5.1 | C | **Parquet written, Delta read**: the default flow wrote parquet via `SparkIO.write_pandas` but `generate_validation_report`/`save_to_hive` read/registered Delta — the documented end-to-end pipeline was broken. `sample(output_format=...)` was a dead parameter. | ✅ `output_format` plumbed through orchestrator → IO → Hive registration; read paths use the same format. |
| 5.2 | C | **Zero-child-row schema fallback was wrong** (built from `constraints.keys()` — missing PK/FK/feature columns) and grandchildren crashed (`LinkageModel` on empty parents raised on `X.max()`). | ✅ Real training schema captured at fit time; `LinkageModel` handles empty parents. |
| 5.3 | H | Self-referencing FKs silently dropped from the graph (then crashed at generation); malformed FK refs (`"users"`, `"db.users.id"`) crashed `SchemaGraph.__init__` with raw `ValueError` *before* validation could run. | ✅ FK-ref parsing centralized; `validate_schema()` rejects self-references, unknown parents, malformed refs, **and cycles** (previously undetected until after hours of training). |
| 5.4 | H | Tables missing from `real_data_paths` were skipped at fit but crashed `generate()` with a bare `KeyError`; missing driver-parent path crashed with `TypeError` on `None`. | ✅ Typed `SchemaError` at fit time naming the missing table. |
| 5.5 | H | **Generation was irreproducible**: linkage counts, Poisson/negbinom draws, and secondary-FK assignment all used the unseeded global NumPy RNG. | ✅ `seed` accepted by `Synthesizer.sample()`/`generate()` and threaded to a local `np.random.Generator` in linkage + FK sampling. |
| 5.6 | M | Generated PKs always `range(1, n+1)` int64 regardless of source dtype. | 🗺️ Record PK dtype at fit; cast/format generated keys. |
| 5.7 | M | Driver parent chosen by alphabetical FK order (rename a column, silently model a different relationship). | ✅ Documented + explicit `driver_fk` field on `TableConfig` (falls back to sorted order). |
| 5.8 | M | Secondary parents re-read from disk per child per FK. | ✅ Parent PK arrays cached per generation run. |
| 5.9 | M | `SparkIO(None)` constructed when no session; failure surfaced as deep `AttributeError`. | ✅ Falls back to `LocalIO` (see §6); explicit error if neither is possible. |

## 6. Architecture: Spark should be optional

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 6.1 | C | **`pyspark`+`delta-spark` (~400 MB + Java 17) were hard requirements** even though every pyspark import is try/except-guarded and the README says components work without Spark. `Synthesizer.fit()`'s docstring implied DataFrame input worked; it crashed (`"/" in DataFrame` → column check → `AttributeError`). There was no pandas-native path at all. | ✅ New `LocalIO` backend (pandas + pyarrow): `Synthesizer` works end-to-end without Spark — fit on dicts of DataFrames or parquet/csv paths, sample to memory or parquet. `pyspark`/`delta-spark` moved to the `[spark]` extra. |
| 6.2 | M | `Synthesizer` not exported from the package root; `interface/__init__.py` empty; deep-path imports required everywhere. | ✅ `from syntho_hive import Synthesizer` now works (lazy import keeps `import syntho_hive` light); subpackages export their public classes. |
| 6.3 | M | Default `sampling_strategy="relational_stratified"` warned "not implemented" on every single `fit()` call. | ✅ Default is `"full"`; unimplemented strategies raise `NotImplementedError`. |
| 6.4 | M | `print()` scattered through library code. | ✅ Replaced with structlog. |
| 6.5 | M | Pydantic models silently dropped typo'd kwargs (`extra="ignore"`); `Constraint` never validated `min <= max`; identical datetime/bool dtypes reported as FK "type mismatch". | ✅ `extra="forbid"`, min/max validation, dtype-kind equality check. |
| 6.6 | M | `num_rows` for forgotten root tables silently defaulted to 1000; child-table entries silently ignored. | ✅ Missing root entries raise; child entries warn. |
| 6.7 | H | Stratified relational sampler ignored `sample_size` entirely (hard-coded 10% fractions) and double-parent children lost one parent's filter. | 🗺️ Sampler rework (documented as experimental; not on the default path). |

## 7. Packaging, CI, repo hygiene

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 7.1 | C | Heavy Spark deps required (see 6.1). | ✅ `[project.optional-dependencies] spark`. |
| 7.2 | H | **PyPI publish workflow had no test gate**, no `twine check`, no tag↔version check — a release from a broken commit published immediately. | ✅ Publish gated on tests + `twine check` + tag/version match; build and publish split. |
| 7.3 | H | **`syntho_hive/tests/` (the only orchestrator/FK-integrity tests) never ran in CI** (`testpaths=["tests"]`); five more "test" files never collected (one calls `sys.exit()` at import). | ✅ Tests consolidated under `tests/`; manual scripts moved to `scripts/`. |
| 7.4 | H | Tautological tests: null-handling asserted `x >= 0` on counts; seed-regression only warned when outputs were identical. | ✅ Real assertions. |
| 7.5 | H | No lint or type-check anywhere (tools listed in dev extra, zero config, never run in CI). | ✅ Ruff config + CI job; mypy config added (advisory). |
| 7.6 | M | No `readme`, `urls`, license/version classifiers, `py.typed`; version duplicated in two places; package exclude for `syntho_hive.tests` only worked by accident. | ✅ All fixed; version single-sourced from `syntho_hive.__version__`. |
| 7.7 | M | Committed artifacts: nested `examples/demos/*/examples/demos/*/outputs/` (path-doubling bug in `run.py`), `test_output/report.html`, `.planning/` (106 internal AI-planning files), root scratch scripts (`debug_import.py`, `verify_quickstart.py`). | ✅ Removed; `run.py` output path now `__file__`-relative; `.gitignore` covers all generated dirs. |
| 7.8 | M | CI matrix stopped at 3.11 while `requires-python` claims ≥3.9 open-ended; outdated action versions; no pip cache; no coverage. | ✅ Matrix 3.9–3.12, actions bumped, pip cache, coverage via pytest-cov. |

---

## Suggested roadmap (post-branch)

1. **True CTGAN conditioning** (1.12): conditional vector + training-by-sampling; largest fidelity win available.
2. **Chunked sampling + CPU-resident training data** (1.13) for large-scale generation.
3. **PK dtype preservation** (5.6) and composite-key support.
4. **Differential privacy** (DP-SGD via opacus) behind `PrivacyConfig.enable_differential_privacy` — currently fails loudly instead of silently.
5. **Safe checkpoint format** (2.5): manifest + `np.save`/`state_dict` only, no pickle.
6. **Datetime column support** (1.15).
7. **Stratified sampler rework** (6.7).
