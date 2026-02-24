# Milestones

## v1.0 Core Reliability (Shipped: 2026-02-22)

**Phases completed:** 1 phase, 5 plans, 11 tasks
**Files changed:** 21 (+2,626 / -167 lines)
**LOC (syntho_hive/):** 3,347 Python
**Test suite:** 12/12 passing

**Key accomplishments:**
- Established `SynthoHiveError` exception hierarchy (5 typed classes: SchemaError, TrainingError, SerializationError, ConstraintViolationError); eliminated all bare/silent except blocks from `syntho_hive/`
- Replaced CTGAN single-file save with 7-file directory checkpoint enabling cold `load()` + `sample()` without any training data or retraining
- Added deterministic seed control (`fit(seed=N)`, `sample(seed=N)`) propagated through CTGAN → DataTransformer → ClusterBasedNormalizer via per-column hash-derived seeds; `pd.testing.assert_frame_equal(check_exact=True)` across independent runs passes
- Patched `save_to_hive()` SQL injection vulnerability with `_SAFE_IDENTIFIER` allowlist regex validating both DB name and all table name keys before any `spark.sql()` call
- Shipped 12-test permanent regression harness (TEST-01 E2E, TEST-03 serialization round-trip, TEST-05 seed regression, QUAL-04 constraint violation raising)

**Tech debt carried forward:**
- Stale `.venv` — `pip install -e .` required before Phase 2
- No `tests/` regression for `SchemaError` on invalid DB name (CONN-04 gap, low risk)
- Two misleading `# noqa` comments (`ctgan.py:14`, `transformer.py:7`)

**Git range:** `6a4dd10` → `e864017`
**Archive:** `.planning/milestones/v1.0-ROADMAP.md`, `.planning/milestones/v1.0-REQUIREMENTS.md`

---

## v1.1 Relational Correctness (Shipped: 2026-02-24)

**Phases completed:** 4 phases, 10 plans
**Files changed:** 11 (+803 / -174 lines)
**LOC (syntho_hive/):** 3,976 Python
**Test suite:** 14/14 passing

**Key accomplishments:**
- Multi-table FK integrity proven end-to-end: empirical-histogram cardinality model eliminates FK cardinality drift; `TestFKChainIntegrity` suite confirms zero-orphan join guarantee across generated parent/child tables (REL-01, REL-02, REL-05)
- Schema validation at definition time: `validate_schema(real_data=)` with collect-all FK type/column checks raises `SchemaValidationError` before training begins (REL-03, REL-04)
- Pluggable model architecture: `ConditionalGenerativeModel` ABC; `Synthesizer(model=CustomClass)` API; `issubclass` guard at `__init__` time fires regardless of Spark session presence (MODEL-01, MODEL-02, MODEL-03)
- Validation hardening: `fit(validate=True, data=DataFrames)` passes real DataFrames to `validate_schema()` enabling data-level FK type checks; issubclass guard moved from deferred orchestrator construction to Synthesizer init (TD-01, TD-04 closed)
- Clean test suite: all 4 pre-existing failures fixed — `TrainingError` assertions aligned with exception boundaries, `call_args` checks aligned with current call signatures (14/14 passing, TD-02 closed)
- PySpark 4.0+ / delta-spark 4.0+ version pins resolved; pytest `--import-mode=importlib` configured for stable test discovery (CONN-02, TEST-02)

**Tech debt carried forward:**
- REL-03 partial wiring: `validate_schema()` data-level checks require explicit `data=` argument — not enforced at API boundary; documented behavior
- Production-scale FK chain test (10k+ rows) outstanding — zero-orphan guarantee unverified at realistic dataset sizes
- Stale CTGAN-specific docstrings in `orchestrator.py` (lines 100, 104, 110, 124, 161, 188) — documentation-only, no functional impact

**Git range:** `4048f9e` → `97e5814`
**Archive:** `.planning/milestones/v1.1-ROADMAP.md`, `.planning/milestones/v1.1-MILESTONE-AUDIT.md`

---

