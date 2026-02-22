# Feature Research

**Domain:** Synthetic tabular data generation — Python SDK
**Researched:** 2026-02-22
**Confidence:** MEDIUM (training data through Aug 2025; WebSearch/WebFetch unavailable — verified against codebase and published competitor behavior as of that date; flag for spot-checks against current SDV/Gretel docs)

---

## Context: What SynthoHive Is Fixing

This is a subsequent milestone research file. SynthoHive has an architecture that is already right but is unreliable. The four critical failure modes from CONCERNS.md are:

1. Broken model serialization — loaded models cannot run inference
2. Bad child table context conditioning — referential integrity fails on non-trivial schemas
3. Memory OOM on large multi-table schemas
4. Silent error handling that makes failures invisible

The feature question is therefore not "what to build from scratch" but "what must a production-grade synthetic data SDK guarantee?" Features are categorized by whether their absence causes engineers to reject or abandon the tool.

---

## Competitor Landscape (Training Knowledge — MEDIUM Confidence)

### SDV (Synthetic Data Vault) — RDT + SDV open source
- **Model types:** CTGAN, TVAE, CopulaGAN, GaussianCopula, PAR (time-series), HMA (hierarchical multi-table)
- **Multi-table:** Full relational support via HMASynthesizer; generates parent before children; FK integrity enforced
- **Serialization:** `save()` / `load()` via pickle; full object state round-trips correctly (transformer + model)
- **Quality:** Built-in `evaluate_quality()` with Shape score, Trend score, Coverage score, Boundary score; separate `run_diagnostic()` for FK integrity checks
- **Privacy:** Column-level anonymization transformers (rdt library); no differential privacy by default
- **Error handling:** Explicit exceptions with typed errors; training surfaced as progress bars via tqdm
- **Connectors:** Pandas-only; no Spark connector; no database connectors out of the box
- **Schema definition:** Metadata object with `add_table()`, auto-detection of column types
- **Constraints:** Column-level constraints (Positive, Between, Unique, FixedCombinations); applied during generation via rejection sampling or transformation

### Gretel.ai
- **Model types:** ACTGAN (their CTGAN fork), Amplify (for augmenting small datasets), DGAN (time-series), GPT-based text synthesizer, tabular-specific Relational model
- **Multi-table:** Relational synthesis via Gretel Relational SDK; FK integrity enforced; topological generation order
- **Serialization:** Model artifacts persisted in cloud or local storage; full state including preprocessors
- **Quality:** Synthetic Data Quality Score (SQS) — composite score; individual metrics for column shapes, pair trends, privacy distances
- **Privacy:** Privacy filters; membership inference attack testing built-in; differential privacy optional on some models
- **Error handling:** API-first design; status objects; explicit failure modes with structured error messages
- **Connectors:** CSV, JSON, SQL via connectors (Postgres, MySQL, Snowflake, BigQuery); cloud-native
- **Constraints:** Constrained generation; post-generation validation against rules

### Mostly AI
- **Model types:** Proprietary MOSTLY tabular model (transformer-based); handles mixed types natively
- **Multi-table:** First-class multi-table with referential integrity; known for being best-in-class here
- **Serialization:** Cloud-stored model artifacts; local SDK wraps API calls
- **Quality:** Accuracy report (statistical similarity), Privacy report (distance to closest record, membership inference), Fairness report
- **Privacy:** DCR (Distance to Closest Record), NNDR (Nearest Neighbor Distance Ratio); k-anonymity checks
- **Error handling:** API-driven; structured responses; progress polling
- **Connectors:** Cloud storage, SQL databases, flat files

### YData / ydata-synthetic
- **Model types:** CTGAN, WGAN, CWGAN, TimeGAN; lightweight Python SDK
- **Serialization:** save/load via pickle; transformer state included
- **Quality:** ydata-profiling integration for pre/post comparison
- **Privacy:** Basic; no differential privacy
- **Connectors:** Pandas DataFrame only

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features whose absence causes engineers to reject the tool on first serious use.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Reliable fit() without crashes** | If training fails silently or hangs, user has no idea where to debug; tool is unusable | MEDIUM | Currently broken: bare `except:` clauses swallow errors; this is the single highest rejection cause |
| **Working save() / load() round-trip** | Engineers train once, deploy inference separately; broken serialization means retraining every session | MEDIUM | Currently broken: DataTransformer state not saved; ctgan.py:482-503 |
| **Correct FK / referential integrity** | Multi-table is the core value prop; if FK links are wrong, the data is useless for pipeline testing | HIGH | Currently broken: stale context bug in ctgan.py:359-362 |
| **Memory-safe generation on large schemas** | OOM on large datasets is a hard stop; engineers do not tolerate silent OOM failures | MEDIUM | Currently broken: orchestrator.py:224-226 accumulates all DataFrames in memory |
| **Explicit, actionable error messages** | Engineers need to know what went wrong (wrong dtype, missing FK, OOM) to fix it | LOW | Currently broken: bare except in synthesizer.py:153, silent failures in transformer.py:222-246 |
| **Single-table synthesis** | Most users start here; must work perfectly before multi-table trust is established | LOW | Architecturally in place; bugs are in training reliability and serialization |
| **Pandas DataFrame input/output** | Standard currency for data engineers; required for interoperability | LOW | Exists; stable |
| **Column type auto-detection** | Engineers should not have to manually declare every column type; schema inference is expected | MEDIUM | Exists via DataTransformer; BayesianGMM for continuous, OneHotEncoder for categorical |
| **Per-column statistical quality metrics** | Engineers need to verify output is usable; KS test / TVD per column is the minimum bar | MEDIUM | Exists but quality gates are not auto-enforced during sample(); only manual via generate_validation_report() |
| **Numeric constraints (min/max/dtype)** | Generated data must honor business rules (age > 0, price > 0) or it fails downstream validation | LOW | Exists in config.py Constraint + transformer.py; but silent failure on constraint application is a bug |
| **Reproducible output via random seed** | Engineers need deterministic outputs for regression tests | LOW | Not explicitly found in codebase; missing |
| **Progress feedback during training** | Long training runs with no output are alarming; engineers need to know training is alive | LOW | Currently only print() statements; no tqdm or structured progress |
| **CSV and Parquet I/O** | Standard file formats for data engineering pipelines | LOW | Exists via SparkIO |

### Differentiators (Competitive Advantage)

Features that distinguish SynthoHive from open-source alternatives. These are not expected by default but provide strong competitive positioning.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Spark-native I/O for large datasets** | SDV is Pandas-only and can't handle 100M+ row datasets; SynthoHive's Spark connector is a genuine gap-filler for enterprise data engineers | MEDIUM | Exists in spark_io.py; needs reliability work on the toPandas() bottleneck |
| **Multi-table synthesis with cardinality preservation** | Competitors handle FK integrity but not always cardinality (how many children per parent); LinkageModel (GMM-based) is a real differentiator if it works | HIGH | LinkageModel exists; referential integrity bugs must be fixed first |
| **Pluggable model architecture** | Lock-in to CTGAN is a risk; a clean model strategy interface (base.py) allows TVAE, CopulaGAN, or future models without orchestrator rewrites | HIGH | Abstract base classes exist; "Pluggable model strategy" is in active requirements |
| **Automated quality gates on sample()** | SDV requires manual `evaluate_quality()` call; auto-enforcing quality thresholds during generation is production-grade behavior | MEDIUM | StatisticalValidator exists; needs to be wired into sample() as optional gate |
| **SQL database connectors** | Gretel and Mostly AI have DB connectors; SynthoHive targets enterprise engineers who pull from Postgres/Snowflake/BigQuery; this would close the feature gap | HIGH | Listed as active requirement; not yet built |
| **Delta Lake / Hive table output** | Data Lakehouse engineers need outputs in their native formats; S3-backed Parquet is table stakes, Delta Lake is a differentiator | MEDIUM | Exists via delta-spark integration |
| **PII detection + context-aware replacement** | Faker-based contextual replacement (ContextualFaker) that preserves field semantics (realistic emails, phone numbers) is better than simple masking | MEDIUM | Exists in privacy/; needs accuracy improvements and salted hashing |
| **Validation report (HTML + JSON)** | Shareable quality reports for data engineering teams reviewing synthetic data quality | MEDIUM | Exists in validation/report_generator.py; needs to surface more metrics |
| **Checkpoint resume for long training runs** | Large-dataset CTGAN training can take hours; ability to resume from checkpoint reduces risk | HIGH | Currently missing; identified in CONCERNS.md as blocking for large-scale deployments |

### Anti-Features (Commonly Requested, Often Problematic)

Features that sound appealing but create more problems than they solve for this domain and user.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **REST API / server mode** | "Can I call the synthesizer from my Node service?" | Adds auth, networking, deployment, versioning, and availability concerns that are orthogonal to ML quality; data engineering teams use Python natively | Expose a clean Python SDK; callers who need a service can wrap it themselves |
| **CLI interface** | "I want to run `syntho-hive fit config.yaml` from bash" | YAML config schemas diverge from Python API; error messages are worse; complex nested schemas become unmanageable in YAML; out-of-scope per PROJECT.md | Python API is the interface; shell users write a thin Python script |
| **Auto-remediation of constraint violations post-generation** | "Just fix the data if it violates constraints" | Silently correcting generated data masks model quality issues; engineers need to know if the model is producing garbage, not have it quietly patched | Raise on constraint violation; let the engineer fix the model config or retrain |
| **Real-time / streaming synthesis** | "Can I generate one row at a time as data arrives?" | CTGAN and transformer-based models require batch inference; single-row generation has poor statistical properties; conditioning requires parent context that may not exist yet | Batch generation is the correct paradigm for synthetic data; stream the results after batch generation |
| **Differential privacy as a default** | "Add DP to protect training data" | DP adds significant noise that degrades statistical fidelity; for most enterprise use cases (pipeline testing, dev environments) DP trades too much quality for marginal privacy gain | Offer DP as an explicit opt-in flag in PrivacyConfig (already modeled); document the quality tradeoffs clearly |
| **GUI / web dashboard** | "I want a visual interface" | Wrong audience; data engineers live in notebooks and scripts; a GUI requires an entirely separate product surface | Invest in a clean Jupyter notebook integration; validation reports as HTML (already exists) |
| **Auto-tuning of hyperparameters** | "Let the tool find the best epochs/batch_size/embedding_dim" | AutoML-style tuning requires a search budget; this dramatically increases complexity; premature optimization before basic training reliability is solved | Provide sensible defaults + documentation on when to tune; expose hyperparameters clearly |
| **Federated / distributed model training** | "Train across data silos without sharing data" | Federated learning adds infrastructure complexity (coordination, communication protocols, aggregation) that is entirely out of scope for v1 | Fix single-machine training reliability first; Spark I/O already handles distributed data ingestion |

---

## Feature Dependencies

```
[Reliable fit() / error handling]
    └──required by──> [Working save() / load()]
    └──required by──> [Multi-table FK integrity]
    └──required by──> [Automated quality gates on sample()]
    └──required by──> [SQL database connectors]

[Working save() / load()]
    └──required by──> [Checkpoint resume]
    └──required by──> [Inference without retraining]

[Multi-table FK integrity]
    └──required by──> [Cardinality preservation (LinkageModel)]
    └──required by──> [Spark-native large-dataset generation]

[Column type auto-detection]
    └──required by──> [Single-table synthesis]
    └──required by──> [Multi-table FK integrity]

[Numeric constraints]
    └──required by──> [Automated quality gates on sample()]

[Per-column statistical quality metrics]
    └──enhances──> [Automated quality gates on sample()]
    └──enhances──> [Validation report]

[Pluggable model architecture]
    └──enhances──> [SQL database connectors] (connectors are model-agnostic)
    └──enhances──> [Checkpoint resume] (each model serializes differently)

[Spark-native I/O]
    └──enhances──> [SQL database connectors] (can share connector abstractions)
    └──depends on──> [Multi-table FK integrity] (large schema generation requires correct ordering)

[PII detection]
    └──enhances──> [SQL database connectors] (PII must be handled on ingest)
    └──conflicts with──> [Context information leakage] (parent context columns must not contain PII; see CONCERNS.md)
```

### Dependency Notes

- **Reliable fit() is a hard prerequisite for everything:** No other feature is testable or trustworthy if training crashes silently. This is phase 0.
- **Save/load is a prerequisite for Checkpoint resume:** You cannot resume from a checkpoint that doesn't serialize the full model state.
- **Multi-table FK integrity must precede Cardinality preservation:** The LinkageModel's child counts are meaningless if the FK links themselves are wrong (stale context bug).
- **Quality metrics enhance but don't block synthesizer:** The statistical validator can be wired as a post-generation check; it's independent of the core training path.
- **PII detection conflicts with Context information leakage:** When parent context columns are copied into child rows, any PII in those parent columns leaks into child synthetic data. These two features interact in a dangerous way if context_cols selection is not guarded.

---

## MVP Definition

For a production-ready v1 of SynthoHive (this milestone, "fixing reliability"):

### Launch With (v1 — must fix to not be rejected)

- [ ] **Reliable training with explicit error surfacing** — Replace bare except clauses; structured exceptions with actionable messages; tqdm progress for training epochs
- [ ] **Full model serialization** — Save/load persists DataTransformer state + CTGAN weights; loaded models produce identical output to in-memory models
- [ ] **Correct child table context conditioning** — Fix stale context bug; child rows conditioned on freshly sampled parent context per batch
- [ ] **Memory-safe multi-table generation** — Stream DataFrames to disk when output_path_base is set; do not accumulate all tables in RAM
- [ ] **Reproducible output via random seed** — `fit(seed=42)` and `sample(seed=42)` produce deterministic output for regression testing
- [ ] **Numeric constraint enforcement with warnings** — Constraints applied strictly; violations logged as warnings, not silently swallowed
- [ ] **End-to-end test suite** — Automated tests for single-table, multi-table, and referential integrity; regression tests for serialization round-trip

### Add After Validation (v1.x — once core is stable)

- [ ] **Automated quality gates on sample()** — Optional `quality_threshold` parameter; aborts if per-column KS/TVD scores fall below threshold; prevents engineers from accidentally using bad synthetic data
- [ ] **SQL database connectors** — Postgres/MySQL connector; allow `Synthesizer.fit(source="postgresql://...")` without requiring Spark
- [ ] **Progress callback API** — `fit(on_epoch_end=callback)` for notebook-friendly training monitoring
- [ ] **PII column auto-detection accuracy improvements** — Replace regex patterns with `email-validator` / `phonenumbers` libraries; salted hashing for compliance
- [ ] **Pluggable model: TVAE** — Add TVAE as second model option; validates the pluggable architecture works

### Future Consideration (v2+ — after product-market fit)

- [ ] **Checkpoint resume** — Resume interrupted training from last saved epoch; depends on working serialization
- [ ] **Differential privacy** — DP-SGD integration as explicit opt-in; document quality tradeoffs
- [ ] **Distributed CTGAN training** — Multi-GPU or Spark-based training for >10GB datasets
- [ ] **Fairness metrics in validation report** — Demographic parity, equalized odds checks on generated data
- [ ] **Delta Lake time-travel output** — Version synthetic datasets; allow engineers to roll back to previous generation runs

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Reliable fit() / explicit errors | HIGH | LOW | P1 |
| Working save() / load() | HIGH | LOW | P1 |
| Correct FK / referential integrity | HIGH | MEDIUM | P1 |
| Memory-safe multi-table generation | HIGH | LOW | P1 |
| Reproducible random seed | HIGH | LOW | P1 |
| Constraint enforcement with warnings | HIGH | LOW | P1 |
| End-to-end test suite | HIGH | MEDIUM | P1 |
| Automated quality gates on sample() | HIGH | MEDIUM | P2 |
| SQL database connectors | HIGH | HIGH | P2 |
| Spark-native large dataset I/O | MEDIUM | MEDIUM | P2 |
| Progress callbacks | MEDIUM | LOW | P2 |
| PII detection accuracy | MEDIUM | MEDIUM | P2 |
| Pluggable model (TVAE) | MEDIUM | HIGH | P2 |
| Validation report enhancements | MEDIUM | LOW | P2 |
| Checkpoint resume | MEDIUM | HIGH | P3 |
| Differential privacy | LOW | HIGH | P3 |
| Distributed training | LOW | HIGH | P3 |
| Fairness metrics | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for this milestone (production reliability)
- P2: Add when core is stable and validated
- P3: Future consideration; do not start until P1 and P2 are solid

---

## Competitor Feature Analysis

| Feature | SDV | Gretel | Mostly AI | SynthoHive (current) | SynthoHive (target) |
|---------|-----|--------|-----------|---------------------|---------------------|
| Single-table synthesis | Yes | Yes | Yes | Yes (unreliable) | Yes (reliable) |
| Multi-table FK integrity | Yes (HMA) | Yes | Yes (best-in-class) | Partial (context bug) | Yes |
| Serialization | Yes (pickle full state) | Yes (cloud artifacts) | Yes (cloud API) | Broken (no transformer state) | Yes (full object) |
| Pandas I/O | Yes | Yes | Via SDK | Yes | Yes |
| Spark / large dataset I/O | No | No | No | Yes | Yes (differentiator) |
| SQL connectors | No | Yes | Yes | No | Planned |
| Auto quality metrics | Yes (built-in score) | Yes (SQS) | Yes (composite) | Manual only | Semi-auto gate |
| PII detection | Via RDT anonymizers | Yes | Yes | Yes (needs accuracy work) | Improved |
| Differential privacy | No (RDT transforms only) | Optional | No | Config exists; untested | Optional opt-in |
| Cardinality preservation | Partial | Partial | Good | LinkageModel (GMM) | Yes (after bug fix) |
| Reproducible seed | Yes | Yes | N/A | Not found | Yes |
| Pluggable models | CTGAN/TVAE/GaussianCopula | ACTGAN/DGAN/GPT | Proprietary | Abstract base (not validated) | CTGAN + TVAE |
| Checkpoint resume | Partial | Yes | Yes | Missing | Planned v2 |
| Progress feedback | tqdm | API status | API polling | print() only | tqdm or callback |
| Explicit error messages | Yes | Yes | Yes | No (silent failures) | Yes |

---

## Sources

- SDV documentation and GitHub (sdv-dev/SDV, sdv-dev/RDT) — training knowledge through Aug 2025 — MEDIUM confidence
- Gretel.ai documentation and GitHub (gretelai/gretel-synthetics, gretelai/gretel-client) — training knowledge through Aug 2025 — MEDIUM confidence
- Mostly AI Python SDK documentation (mostly-ai/mostlyai) — training knowledge through Aug 2025 — MEDIUM confidence
- YData ydata-synthetic GitHub — training knowledge through Aug 2025 — MEDIUM confidence
- SynthoHive codebase direct analysis (CONCERNS.md, ARCHITECTURE.md, config.py, statistical.py) — HIGH confidence (direct code read)
- SynthoHive PROJECT.md requirements and constraints — HIGH confidence (direct read)

**Confidence note:** WebSearch and WebFetch were unavailable during this research session. Competitor feature claims are based on training data (through Aug 2025). Before finalizing the roadmap, spot-check the following against current docs:
1. SDV's current serialization approach (they migrated from pickle to a custom format in some versions)
2. Gretel's current Relational SDK feature set (actively developed)
3. Whether any competitor has shipped Spark-native training (would undercut the differentiator claim)

---

*Feature research for: Synthetic tabular data generation — Python SDK*
*Researched: 2026-02-22*
