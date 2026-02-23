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
