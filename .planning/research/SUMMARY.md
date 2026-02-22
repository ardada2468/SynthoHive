# Project Research Summary

**Project:** SynthoHive — Production-Grade Synthetic Tabular Data Python SDK
**Domain:** Synthetic tabular data generation with GAN training, relational synthesis, Spark I/O, and statistical validation
**Researched:** 2026-02-22
**Confidence:** HIGH (stack verified against installed venv artifacts; architecture from direct codebase analysis; features/pitfalls cross-validated against codebase + domain knowledge)

---

## Executive Summary

SynthoHive is a Python SDK for generating realistic synthetic tabular data from relational schemas using a custom CTGAN (Conditional Tabular GAN) implementation with PyTorch. The codebase architecture is fundamentally sound — the layered design (Synthesizer facade, StagedOrchestrator, pluggable model ABCs, DataTransformer, SparkIO connectors, StatisticalValidator) matches how production-grade synthetic data tools are built. This is a fix-and-extend milestone, not a rebuild. The research consistently confirms that the structure is right but four specific bugs render the SDK unreliable enough that engineers would abandon it in first serious use: broken model serialization, stale context conditioning for child tables, silent exception swallowing, and memory accumulation during multi-table generation.

The recommended approach is to fix these four blocking bugs in a strict dependency order before extending any capabilities. Error handling must be hardened first because all other bugs produce invisible failures without it. Model serialization must be fixed before SQL connectors or checkpoint resume are meaningful. FK context conditioning must be correct before cardinality preservation (LinkageModel) is trustworthy. Memory-safe generation must be fixed before large-scale production use is viable. After the reliability foundation is stable, the natural extensions are automated quality gates wired into `sample()`, SQL database connectors (the only major missing capability versus competitors), and validation of the pluggable model architecture by adding TVAE as a second model.

The primary risks are: (1) the CTGAN context conditioning bug is subtle and any test that uses a freshly-fit in-memory model will pass while a loaded model fails — test coverage must verify cold-load inference, not just warm-state inference; (2) PII sanitization is not wired into the training path at all, meaning trained models have seen raw PII and any checkpoint is a potential compliance liability; (3) the Hive SQL path has SQL injection exposure via unsanitized `target_db` parameter that must be patched regardless of phase ordering. These three issues should be addressed in Phase 1 even though they are not the primary blocking bugs.

---

## Key Findings

### Recommended Stack

The installed venv (Python 3.14) already has the correct major libraries at production-ready versions. The primary work is tightening pyproject.toml version pins to reflect the actual dependencies (torch 2.9.1, pyspark 4.0.1, delta-spark 4.0.0, pandas 2.3.3, numpy 2.3.5, scipy 1.16.3, scikit-learn 1.8.0, pydantic 2.12.5). The single most significant new dependency addition is SQLAlchemy 2.0 + dialect drivers (psycopg2-binary, pymysql) to build the SQL connector. Serialization should be fixed using joblib (already transitively installed via scikit-learn) rather than adding new packages. Three new dev dependencies are needed: pytest-cov, pytest-mock, and hypothesis — the last of which is particularly valuable for property-based testing of the constraint roundtrip bugs.

**Core technologies:**
- PyTorch 2.9.1: CTGAN generator/discriminator networks — keep custom implementation, do not replace with SDV/CTGAN package
- PySpark 4.0.1 + delta-spark 4.0.0: Spark-native I/O and Delta Lake output — must be bumped together; incompatible with Spark 3.x
- pandas 2.3.3 + numpy 2.3.5: Primary data structures throughout all modules
- pydantic 2.12.5: Schema validation for Metadata, TableConfig, PrivacyConfig — already at v2, no v3 exists
- joblib 1.5.2 (transitive, already installed): Full CTGAN object serialization — replaces the broken partial torch.save approach
- SQLAlchemy 2.0 (new): SQL database connector abstraction layer for PostgreSQL/MySQL/Snowflake/BigQuery
- scikit-learn 1.8.0: DataTransformer internals (GaussianMixture, OneHotEncoder) + mutual_info_score for validation
- structlog 25.5.0: Replace all bare print() calls with structured log events
- hypothesis 6.100+ (new dev dep): Property-based testing for DataTransformer constraint roundtrips

**What not to add:**
- SDV as a library dependency (~300MB, duplicates CTGAN/TVAE, creates dependency conflicts)
- sdmetrics or table_evaluator (redundant with existing SciPy validators; heavy installs)
- cloudpickle or dill (joblib is already available and better tested with sklearn objects)

See `.planning/research/STACK.md` for full version compatibility matrix and pyproject.toml change set.

### Expected Features

The research draws a sharp line between features whose absence causes immediate engineer rejection ("table stakes") and those that differentiate the product. Currently, SynthoHive fails on four of the seven table-stakes features, which explains why it cannot be trusted in production regardless of its architectural quality.

**Must have — v1 (table stakes, currently failing):**
- Reliable fit() without silent crashes — bare except clauses in synthesizer.py:153 and transformer.py:228-246 must be replaced
- Working save() / load() round-trip — DataTransformer state not saved; loaded models cannot run inference
- Correct FK / referential integrity — stale context bug at ctgan.py:359-362 causes wrong child conditioning
- Memory-safe generation on large schemas — orchestrator.py:224-226 accumulates all DataFrames in RAM
- Reproducible output via random seed — not found anywhere in codebase; must be added
- Explicit, actionable error messages — tied to error handling hardening above

**Must have — v1 (table stakes, currently working):**
- Single-table synthesis — architecturally in place; reliability depends on above fixes
- Pandas DataFrame I/O — stable
- Column type auto-detection — BayesianGMM for continuous, OneHotEncoder for categorical; exists
- CSV and Parquet I/O — exists via SparkIO
- Per-column statistical quality metrics — exists but not auto-enforced during sample()

**Should have — v1.x (competitive differentiators):**
- Automated quality gates on sample() — wire StatisticalValidator into sample() as optional threshold gate
- SQL database connectors (Postgres/MySQL/Snowflake/BigQuery) — the largest capability gap vs. competitors
- Progress callbacks / tqdm during training — currently print()-only; engineers need liveness signal
- PII column auto-detection accuracy improvements — salted HMAC hashing, better regex + library-backed detection
- Pluggable model: TVAE — validates the abstract base class strategy works end-to-end

**Defer — v2+:**
- Checkpoint resume for interrupted training
- Differential privacy (DP-SGD as explicit opt-in)
- Distributed CTGAN training (multi-GPU or Spark UDFs)
- Fairness metrics in validation report
- Delta Lake time-travel output versioning

**Anti-features to avoid building:**
- REST API / server mode (orthogonal to SDK; callers can wrap it)
- CLI interface with YAML config (complex schemas become unmanageable; out of scope)
- Auto-remediation of constraint violations (masks model quality problems)
- Real-time / streaming synthesis (CTGAN requires batch inference)
- Auto-tuning of hyperparameters (premature before basic reliability is fixed)

See `.planning/research/FEATURES.md` for full feature prioritization matrix and competitor comparison table.

### Architecture Approach

The existing 6-layer architecture (Interface, Relational Orchestration, Model, Core Data, Privacy, Validation) is correct and should not be restructured. The Synthesizer is a clean facade, the abstract base classes exist, the DataTransformer is properly separated from model logic, and SparkIO is appropriately isolated in the connectors layer. The bugs are in specific implementation details within existing components, not in the layering itself. The two architectural improvements needed are: (1) remove the hardcoded CTGAN import from StagedOrchestrator and inject it as a model class parameter so the pluggable strategy pattern actually works, and (2) add a SQLConnector in `syntho_hive/connectors/sql_io.py` using the same read-interface as SparkIO but returning Pandas DataFrames directly for non-Spark users.

**Major components:**
1. `Synthesizer` (interface/synthesizer.py) — public API facade; only entry point for user code
2. `StagedOrchestrator` (relational/orchestrator.py) — multi-table training and generation pipeline; currently hardcodes CTGAN (bug)
3. `SchemaGraph` (relational/graph.py) — FK dependency DAG with topological sort for correct generation order
4. `LinkageModel` (relational/linkage.py) — GMM-based parent-to-child cardinality modeling (needs distribution family fix)
5. `CTGAN` (core/models/ctgan.py) — conditional tabular GAN; owns DataTransformer and PyTorch network weights
6. `DataTransformer` (core/data/transformer.py) — reversible encode/decode; must be serialized with the model
7. `SparkIO` (connectors/spark_io.py) — all storage reads/writes; models never call SparkIO directly
8. `StatisticalValidator` (validation/statistical.py) — KS test + TVD; needs wiring into sample() pipeline
9. `PIISanitizer` (privacy/sanitizer.py) — currently not wired into fit_all() training path (critical gap)

**Key patterns to follow:**
- Context alignment is row-for-row: context_df passed to CTGAN.sample(n, context) must have exactly n rows
- Topological generation order must not be violated: SchemaGraph guarantees parent data is available before child generation
- FK columns are excluded from transformation and identity-assigned by the orchestrator — models must never receive or return FK columns

See `.planning/research/ARCHITECTURE.md` for complete data flow diagrams and anti-pattern documentation.

### Critical Pitfalls

1. **Partial model serialization (saving weights without transformer state)** — Use joblib.dump() on a structured checkpoint dict that includes DataTransformer, context_transformer, data_column_info, embedding_layers.state_dict(), and hyperparameters alongside the neural network state_dicts. Verify correctness by loading in a fresh Python process with no training data available and calling sample() cold.

2. **Stale context reuse in generator training** — For each generator training step, independently sample a fresh random minibatch of context rows from context_data rather than reusing real_context_batch from the last discriminator iteration. One additional np.random.randint line fixes this; add a regression test verifying child FK distribution matches parent PK distribution.

3. **Silent exception swallowing masking all other bugs** — Replace all bare `except:` and `except Exception: pass` blocks with typed catches and either explicit raises or logger.warning() calls. This is prerequisite to debugging every other fix; enforce with flake8 --select=E722.

4. **Memory accumulation during multi-table generation** — When output_path_base is set, do not store generated_pdf in generated_tables after writing to disk; child tables must read parent data from disk. Peak memory should be bounded to one or two tables, not all tables simultaneously.

5. **PII not wired into training path** — PIISanitizer.analyze() is defined but not called during fit_all(). Trained models have seen raw PII. Fix: add PIISanitizer call at the top of the per-table training block before CTGAN.fit(), treating explicit pii_cols declarations in TableConfig as authoritative.

6. **GMM wrong distribution family for child counts** — Replace GaussianMixture in LinkageModel with empirical distribution sampling (np.random.choice on observed counts) or scipy.stats.nbinom. GMM produces negative and non-integer samples; clipping/rounding are band-aids, not a fix.

7. **SQL injection in save_to_hive()** — target_db and table name parameters are interpolated directly into Spark SQL without sanitization. Validate against [a-zA-Z0-9_]+ pattern before executing.

See `.planning/research/PITFALLS.md` for the complete pitfall catalog including recovery strategies and "looks done but isn't" checklist.

---

## Implications for Roadmap

The research from all four files converges on a clear 5-phase structure. The first two phases address blocking reliability issues; phases 3-5 add capability and quality in a dependency-respecting order.

### Phase 1: Core Reliability — Error Handling and Serialization

**Rationale:** Without typed exception handling, every other bug is invisible. Without working serialization, the SDK cannot be used in any real deployment where training and inference are separate processes. These are the two changes with the highest fix-to-impact ratio: both are low-to-medium complexity changes that unblock trust in everything downstream.

**Delivers:** A Synthesizer where fit() surfaces errors with actionable messages, save()/load() produces models that can run cold inference, and trained model state is fully preserved across sessions.

**Addresses features:** Reliable fit() without crashes, Working save()/load() round-trip, Explicit actionable error messages, Reproducible random seed

**Avoids pitfalls:** Partial model serialization (Pitfall 1), Silent error swallowing (Pitfall 4), SQL injection in save_to_hive (Pitfall 7 — fix here regardless)

**Stack elements:** joblib (already installed), structlog (already installed — replace print() calls), pytest-cov + pytest-mock (new dev deps)

**Needs research-phase:** No — patterns are well-documented; serialization approach is specified explicitly in ARCHITECTURE.md Pattern 3.

---

### Phase 2: Relational Correctness — Context Conditioning and Memory Safety

**Rationale:** The FK context conditioning bug and the memory accumulation bug are both in the relational path. Fixing them together is natural; both have clear specifications and the fixes are contained. Multi-table synthesis is the core value proposition of SynthoHive — it must work correctly before anything else in the relational stack is extended.

**Delivers:** Multi-table generation that produces correct referential integrity, with FK distributions that reflect the true parent distribution, and memory usage that stays bounded regardless of schema size.

**Addresses features:** Correct FK / referential integrity, Memory-safe multi-table generation, Cardinality preservation (partially — GMM distribution fix)

**Avoids pitfalls:** Stale context in generator training (Pitfall 2), Memory accumulation (Pitfall 3), GMM wrong distribution family (Pitfall 6), FK type mismatch with silent fallback (Pitfall 7)

**Stack elements:** numpy (already installed), scipy.stats.nbinom as GMM replacement for LinkageModel

**Needs research-phase:** No — bugs and correct patterns are explicitly documented in ARCHITECTURE.md Patterns 2 and 4.

---

### Phase 3: Model Pluggability and Privacy Integration

**Rationale:** Once the single model (CTGAN) is reliable, validate the pluggable architecture by wiring the model class injection properly in StagedOrchestrator and adding a minimal TVAE implementation. Simultaneously fix PII sanitization not being wired into the training path — this is a compliance risk that grows with each additional training run that happens before the fix. These belong together because both involve refactoring how the orchestrator calls into the model layer.

**Delivers:** An orchestrator that accepts any ConditionalGenerativeModel via dependency injection, a working TVAE as proof-of-concept for the pattern, and a training pipeline that sanitizes PII before fitting.

**Addresses features:** Pluggable model architecture, TVAE as second model, PII detection accuracy improvements, PIISanitizer wired into training path

**Avoids pitfalls:** Hardcoded model class in orchestrator (Architecture Anti-Pattern 1), PII not wired into training path (critical security/compliance gap from Pitfall checklist)

**Stack elements:** No new dependencies; uses existing PyTorch + DataTransformer interface

**Needs research-phase:** Yes — TVAE architecture (encoder/decoder structure, loss function, reparameterization trick) warrants a focused research step to get the implementation right on first attempt.

---

### Phase 4: Validation and Quality Gates

**Rationale:** After the synthesis pipeline is reliable and correct, add the observability and quality assurance layer. This phase wires existing statistical validation components into the sample() pipeline, adds tqdm progress for training, and enriches the validation report. These are additive changes with no risk of breaking the fixed core behavior.

**Delivers:** An optional quality threshold on sample() that prevents silent acceptance of bad synthetic data, visible epoch-by-epoch training progress, a richer validation report with overall pass/fail summary, and training quality metrics emission (loss + statistical score per epoch).

**Addresses features:** Automated quality gates on sample(), Progress callbacks/tqdm, Validation report enhancements, Training quality metrics

**Avoids pitfalls:** GAN generator loss as checkpoint criterion (Pitfall 5 — replace with validation-metric-based checkpointing), No progress indication (UX Pitfall), Validation report has no pass/fail summary (UX Pitfall)

**Stack elements:** tqdm (add to dependencies), existing scipy + sklearn for metric computation

**Needs research-phase:** No — the StatisticalValidator is already built; wiring patterns are straightforward.

---

### Phase 5: SQL Connectors and Extended I/O

**Rationale:** SQL connector is the largest capability gap versus competitors (Gretel, Mostly AI both have it; SDV does not). Building it last is correct because it requires no changes to the model or orchestrator — it is a pure addition to the connectors layer. It also requires a working, reliable synthesis pipeline to be meaningful; adding connectors to a broken synthesizer would create a larger surface area for bugs.

**Delivers:** Synthesizer.fit() accepting a PostgreSQL, MySQL, Snowflake, or BigQuery connection string without requiring Spark; SQLConnector class in `syntho_hive/connectors/sql_io.py` that reads tables into Pandas DataFrames and plugs into the existing fit pipeline.

**Addresses features:** SQL database connectors (PostgreSQL, MySQL, Snowflake, BigQuery)

**Uses stack elements:** SQLAlchemy 2.0 (new), psycopg2-binary (new), pymysql (new), snowflake-sqlalchemy (optional extra), google-cloud-bigquery (optional extra)

**Avoids pitfalls:** PySpark for small-to-medium SQL reads (startup overhead; SQLAlchemy is the correct path for <1M row tables)

**Needs research-phase:** Yes — SQLAlchemy dialect compatibility with Snowflake and BigQuery, and the specific API for each connector, warrant a research step before implementation to avoid dialect-specific gotchas.

---

### Phase Ordering Rationale

- **Error handling before everything:** Silent failures make all subsequent bugs invisible; this is the prerequisite for trusting any test result.
- **Serialization in Phase 1, not Phase 2+:** Every deployment pattern (train once, serve many times) requires working serialization. Deferring it means nothing built afterward can be validated in a realistic deployment scenario.
- **Relational fixes grouped:** Context conditioning and memory safety both live in the relational/orchestrator.py and ctgan.py training loop; fixing them together avoids partial states where one is fixed but the other is not.
- **Privacy before pluggability:** PII wiring into fit_all() must come before a second model is added, otherwise both CTGAN and TVAE training paths would need the fix applied twice.
- **Validation gates after core is stable:** Wiring statistical thresholds into sample() on top of a broken pipeline would block valid samples and confuse debugging.
- **SQL connectors last:** Pure addition to the connectors layer; zero risk to the fixed model/orchestrator behavior; depends on a stable pipeline to be useful.

### Research Flags

Phases needing deeper research during planning:
- **Phase 3 (TVAE implementation):** TVAE encoder/decoder architecture, KL-divergence loss formulation, reparameterization trick — warrants `/gsd:research-phase` to get the implementation right and avoid the same trap as the CTGAN embedding stub.
- **Phase 5 (SQL Connectors):** SQLAlchemy dialect-specific behavior for Snowflake and BigQuery (auth patterns, chunked reads, type mapping) — warrants `/gsd:research-phase` to catch connector-specific gotchas before implementation.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Error handling + serialization):** joblib serialization pattern is explicit in ARCHITECTURE.md; typed exception handling is standard Python practice; no novel architecture.
- **Phase 2 (Relational correctness):** Both bug fixes have exact code specifications in ARCHITECTURE.md Patterns 2 and 4; empirical distribution for LinkageModel is a one-function replacement.
- **Phase 4 (Validation gates):** StatisticalValidator is already built; tqdm integration is standard; wiring patterns are additive and well-understood.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All major library versions verified from dist-info METADATA in the installed venv. pyproject.toml update set is specific and verified. SQLAlchemy dialect recommendations are MEDIUM (training knowledge, not verified from live docs). |
| Features | MEDIUM | Table-stakes and differentiator features derive from direct codebase analysis (HIGH). Competitor feature claims (SDV, Gretel, Mostly AI) are from training knowledge through Aug 2025 — WebSearch was unavailable during research; these should be spot-checked before finalizing competitive positioning. |
| Architecture | HIGH | All bugs, patterns, and architectural recommendations derive from direct source code analysis of the SynthoHive codebase. No inference required — the bugs are visible, the correct patterns are specified. |
| Pitfalls | HIGH | All critical pitfalls identified from direct codebase reading confirmed against CONCERNS.md. Domain knowledge pitfalls (GMM distribution family, WGAN-GP checkpoint criterion) are from verified ML literature (Xu et al. 2019 CTGAN, Gulrajani et al. 2017 WGAN-GP). |

**Overall confidence:** HIGH for Phase 1-2-4 (bug fixes and quality gates). MEDIUM for Phase 3 (TVAE architecture specifics need research). MEDIUM for Phase 5 (SQL dialect specifics need research).

### Gaps to Address

- **Competitor feature validation:** Gretel's Relational SDK feature set is actively developed; SDV may have migrated away from pickle serialization in recent versions. Before finalizing competitive positioning, spot-check SDV current serialization approach and whether any competitor has shipped Spark-native training (would undercut SynthoHive's main differentiator claim).

- **PII sanitization scope:** PIISanitizer is confirmed not wired into fit_all() training path. Before Phase 3, audit whether the privacy sanitizer was ever intended to run at training time vs. as a post-generation step. The ARCHITECTURE.md diagram shows it running "before training" but the code does not reflect this — clarify the intended design.

- **SparkSession-optional single-table path:** Multiple pitfall entries note that Spark is required for all operations, but PROJECT.md implies Spark should be optional. Before Phase 1, confirm whether `Synthesizer(spark_session=None)` is a v1 requirement or a v2 goal — if v1, the single-table non-Spark path needs to be added in Phase 1.

- **Pandas copy-on-write (CoW) impact:** pandas 2.x introduced copy-on-write semantics. transformer.py uses `.values` mutations that may behave differently under CoW. This needs a targeted audit before pinning to `pandas>=2.0.0` in Phase 1.

- **torch.load weights_only default change:** PyTorch 2.6+ changed the default of torch.load() to weights_only=True. The current ctgan.py:501 load path will raise a warning or error. This must be addressed in Phase 1 alongside the full serialization fix.

---

## Sources

### Primary (HIGH confidence — verified from codebase or dist-info METADATA)

- `syntho_hive/core/models/ctgan.py` — training loop bugs (lines 359-362, 414-416, 482-503), serialization gap
- `syntho_hive/relational/orchestrator.py` — memory accumulation bug (lines 224-226), toPandas() bottlenecks
- `syntho_hive/relational/linkage.py` — GMM distribution family, FK type mismatch fallback (lines 35-44)
- `syntho_hive/core/data/transformer.py` — silent constraint violation (lines 228-246)
- `syntho_hive/interface/synthesizer.py` — bare except block (line 153), SparkSession handling
- `syntho_hive/privacy/sanitizer.py` — SHA256 without salt (line 154-156), not wired into training path
- `.planning/codebase/CONCERNS.md` — full bug audit, prioritized issue list
- `.planning/PROJECT.md` — product requirements, constraints, SQL connector requirement
- `.venv/lib/python3.14/site-packages/` dist-info METADATA — all library version constraints

### Secondary (MEDIUM confidence — training knowledge, community-validated patterns)

- SDV (sdv-dev/SDV, sdv-dev/RDT) documentation and GitHub — competitor feature set, serialization approach
- Gretel.ai documentation and GitHub (gretelai/gretel-synthetics) — connector features, relational SDK
- Mostly AI Python SDK — multi-table referential integrity patterns
- SQLAlchemy 2.0 documentation — connector API, dialect compatibility

### Tertiary (domain knowledge — well-established ML literature)

- Xu et al., 2019 — CTGAN: Modeling Tabular Data using Conditional GAN (WGAN-GP training dynamics, VGM normalization)
- Gulrajani et al., 2017 — Improved Training of Wasserstein GANs (gradient penalty properties, training stability)

---

*Research completed: 2026-02-22*
*Ready for roadmap: yes*
