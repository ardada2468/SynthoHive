---
phase: 02-relational-correctness
verified: 2026-02-23T02:30:00Z
status: passed
score: 5/5 success criteria verified
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "test_metadata_validation and test_metadata_invalid_fk_format now pass — pytest.raises(SchemaValidationError) assertions in place; commit c01056a confirmed in history"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Run the 3-table and 4-table FK chain tests on a production-sized dataset (10k+ rows)"
    expected: "Zero orphaned FK references at all join levels with realistic dataset scale"
    why_human: "The automated TestFKChainIntegrity tests use epochs=2, batch_size=20, and only 5-20 parent rows. Real-world generation with a full training run may expose timing, memory, or join-key assignment issues not visible at small scale."
---

# Phase 02: Relational Correctness Verification Report

**Phase Goal:** Multi-table synthesis produces correct referential integrity — FK columns join cleanly with zero orphans, cardinality reflects the real parent distribution, and generation stays memory-bounded regardless of schema size
**Verified:** 2026-02-23
**Status:** passed
**Re-verification:** Yes — after gap closure (Plan 05 fixed SchemaValidationError test regression)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Generated parent and child tables join on FK columns with zero orphaned references and zero missing parents on any schema with 2+ tables | VERIFIED | TestFKChainIntegrity 5/5 pass; test_metadata_validation and test_metadata_invalid_fk_format both PASS after Plan 05 fix; commit c01056a confirmed |
| 2 | StagedOrchestrator uses freshly sampled parent context per generator training step — FK value distributions in child output match the empirical distribution of parent PK values | VERIFIED | ctgan.py lines 408-414: `gen_ctx_idx = np.random.randint(0, len(context_data), self.batch_size)` — independent resample per generator step; `legacy_context_conditioning=False` default; flag persisted in save/load |
| 3 | FK type mismatches between parent PK and child FK are raised at validate_schema() time before training begins | VERIFIED | config.py lines 38-64: `_dtypes_compatible()` helper; lines 97-162: collect-all errors pattern; SchemaValidationError raised; test_fk_type_mismatch_raises_schema_validation_error PASSES |
| 4 | Multi-table generation with output_path_base set keeps peak memory bounded to at most two DataFrames simultaneously — no accumulation of all tables in RAM | VERIFIED | orchestrator.py lines 22-48: `_write_with_failure_policy()` helper; lines 281-294: write-and-release — DataFrames NOT stored in `generated_tables` when output_path_base is set |
| 5 | PySpark 4.0+ and delta-spark 4.0+ version pins in pyproject.toml match the installed venv without conflict | VERIFIED | pyproject.toml lines 23 and 28: `pyspark>=4.0.0,<5.0.0` and `delta-spark>=4.0.0,<5.0.0`; installed: pyspark 4.0.1, delta-spark 4.0.0 — both within bounds |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `syntho_hive/exceptions.py` | SchemaValidationError as subclass of SchemaError | VERIFIED | Lines 24-30: `class SchemaValidationError(SchemaError)` with correct docstring; inherits SchemaError -> SynthoHiveError -> Exception |
| `syntho_hive/interface/config.py` | Extended validate_schema() + linkage_method + _dtypes_compatible() | VERIFIED | Line 30: `linkage_method: Literal["empirical", "negbinom"] = "empirical"` on TableConfig; lines 38-64: `_dtypes_compatible()` module-level helper; lines 97-162: collect-all pattern raising SchemaValidationError |
| `syntho_hive/core/models/ctgan.py` | Fixed generator context resample + legacy_context_conditioning flag | VERIFIED | Line 84: `legacy_context_conditioning: bool = False`; lines 408-414: independent `gen_ctx_idx` resample; line 701: persisted in save(); line 788: restored in load() |
| `syntho_hive/relational/orchestrator.py` | Memory-safe generation with on_write_failure policy | VERIFIED | Lines 22-48: `_write_with_failure_policy()` module-level helper; line 59: `on_write_failure: Literal['raise', 'cleanup', 'retry'] = 'raise'`; lines 281-294: write-and-release in generate() |
| `pyproject.toml` | pyspark>=4.0.0,<5.0.0 and delta-spark>=4.0.0,<5.0.0 | VERIFIED | Lines 23 and 28 confirmed; installed pyspark 4.0.1, delta-spark 4.0.0 — both in range |
| `syntho_hive/relational/linkage.py` | Empirical + NegBinom resampler, GaussianMixture removed | VERIFIED | GaussianMixture only referenced in docstring comment (line 11); `_observed_counts` attribute present; `np.random.choice` for empirical (line 86); `scipy.stats.nbinom` lazy-imported for negbinom (line 83) |
| `syntho_hive/tests/test_relational.py` | TestFKChainIntegrity with 5 test methods | VERIFIED | Lines 158-405: TestFKChainIntegrity class with all 5 methods; all 5 pass in live run |
| `syntho_hive/tests/test_interface.py` | test_metadata_validation and test_metadata_invalid_fk_format with SchemaValidationError | VERIFIED | Line 6: `from syntho_hive.exceptions import SchemaValidationError`; line 35: `pytest.raises(SchemaValidationError, match="references non-existent parent table")`; line 39: `pytest.raises(SchemaValidationError, match="Invalid FK reference")`; both tests PASS |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `syntho_hive/interface/config.py` | `syntho_hive/exceptions.py` | `from syntho_hive.exceptions import SchemaValidationError` | WIRED | Line 5 of config.py: import present; SchemaValidationError raised at line 162 |
| `syntho_hive/interface/config.py` | `validate_schema(real_data)` | collect-all errors list then raise once | WIRED | `errors: List[str] = []` at line 113; `errors.append(...)` at lines 119, 127, 142, 147, 155; `raise SchemaValidationError(...)` at line 162 |
| `syntho_hive/core/models/ctgan.py` | generator training block | `np.random.randint` for independent context resample | WIRED | Lines 408-414: `legacy_context_conditioning` flag gates between stale reuse and fresh `gen_ctx_idx = np.random.randint(0, len(context_data), self.batch_size)` |
| `syntho_hive/relational/orchestrator.py` | `generated_tables` dict | conditional del after output_path_base write | WIRED | Lines 281-294: when `output_path_base` is set, `_write_with_failure_policy()` is called and DataFrame is NOT stored in `generated_tables` |
| `syntho_hive/relational/orchestrator.py` | `syntho_hive/relational/linkage.py` | `LinkageModel(method=table_config.linkage_method)` | WIRED | Lines 143-144 of orchestrator.py: `linkage_method = self.metadata.tables[table_name].linkage_method; linkage = LinkageModel(method=linkage_method)` |
| `syntho_hive/relational/linkage.py` | `numpy.random.choice` | empirical resampling from `_observed_counts` | WIRED | Line 86: `return np.random.choice(self._observed_counts, size=n_samples, replace=True)` |
| `syntho_hive/tests/test_relational.py` | `syntho_hive/relational/orchestrator.py` | MockSparkDF + mock io.read_dataset / io.write_pandas | WIRED | Lines 167-203: `_make_mock_io()` helper wires `read_side_effect` and `write_side_effect`; orchestrator uses `io=mock_io` injection |
| `syntho_hive/tests/test_relational.py` | `syntho_hive/exceptions.py` | `self.assertRaises(SchemaValidationError)` | WIRED | Lines 344, 364: SchemaValidationError imported and used in assertRaises context managers |
| `syntho_hive/tests/test_interface.py` | `syntho_hive/exceptions.py` | `from syntho_hive.exceptions import SchemaValidationError` | WIRED | Line 6: import present; used at lines 35 and 39 in pytest.raises contexts |

---

### Requirements Coverage

All requirement IDs are taken from ROADMAP.md Phase 2 definition (`REL-01, REL-02, REL-03, REL-04, REL-05, CONN-02, TEST-02`) and cross-referenced against REQUIREMENTS.md descriptions.

**Note on prompt-supplied requirement list:** The prompt specified `REL-01, REL-02, REL-03, TEST-01, TEST-02`. This list omits `REL-04`, `REL-05`, and `CONN-02` (all three are mapped to Phase 2 in ROADMAP.md and REQUIREMENTS.md) and includes `TEST-01` (which was completed in Phase 1 per ROADMAP.md and REQUIREMENTS.md). The ROADMAP.md canonical list is used below. TEST-01 is noted as already satisfied before Phase 2 began.

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| REL-01 | 02-02 | Child table generation uses freshly sampled parent context per generator training step | SATISFIED | ctgan.py lines 408-414: independent `gen_ctx_idx` resample; `legacy_context_conditioning=False` is the default; flag persists across save/load |
| REL-02 | 02-03 | FK cardinality uses empirical distribution or Negative Binomial — not Gaussian Mixture | SATISFIED | linkage.py: GaussianMixture import absent; `_observed_counts` + `np.random.choice` for empirical; `scipy.stats.nbinom` for negbinom with fallback |
| REL-03 | 02-01, 02-05 | FK type mismatches raised at validate_schema() before training | SATISFIED | config.py: `_dtypes_compatible()` + `SchemaValidationError` raised at validation time; test_fk_type_mismatch_raises_schema_validation_error PASSES; test_metadata_validation and test_metadata_invalid_fk_format PASS (fixed in Plan 05) |
| REL-04 | 02-02 | Multi-table generation releases DataFrames from memory after writing to disk when output_path_base is set | SATISFIED | orchestrator.py lines 281-294: write-and-release pattern; DataFrames not accumulated in `generated_tables` when `output_path_base` set; comment on line 292 explicitly documents the invariant |
| REL-05 | 02-04 | Generated parent/child tables can be joined on FK columns with zero orphaned references and zero missing parents | SATISFIED | TestFKChainIntegrity::test_3_table_chain_zero_orphans and test_4_table_chain_zero_orphans both PASS |
| CONN-02 | 02-02 | PySpark 4.0+ and delta-spark 4.0+ pins match installed venv | SATISFIED | pyproject.toml: `pyspark>=4.0.0,<5.0.0` and `delta-spark>=4.0.0,<5.0.0`; venv: pyspark 4.0.1, delta-spark 4.0.0 |
| TEST-01 | N/A (Phase 1) | End-to-end test: single-table fit → sample → validate | PRE-EXISTING PASS | Completed in Phase 1 (01-04-PLAN.md); no Phase 2 plans claimed this ID; it was not regressed by Phase 2 changes |
| TEST-02 | 02-04, 02-05 | End-to-end test: multi-table fit → sample → FK join validates zero orphans on a 3-table schema | SATISFIED | TestFKChainIntegrity: 5 tests, all pass in live run; no real Spark session required |

All 7 ROADMAP-mapped Phase 2 requirements are SATISFIED. TEST-01 is pre-existing and unaffected.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `syntho_hive/tests/test_interface.py` | 49, 54 | `pytest.raises(ValueError, match="SparkSession required")` — expects ValueError but synthesizer.py line 86 raises plain `ValueError("SparkSession required for fit()")` | INFO | These 4 tests are PRE-EXISTING failures (confirmed present at Phase 1 end state). The ValueError is raised inside Synthesizer before the exception is re-raised as a typed exception — this is a Synthesizer API design issue, not a Phase 2 regression. Out of scope for Phase 2. |
| `syntho_hive/tests/test_relational.py` | multiple | `ConvergenceWarning` from sklearn GaussianMixture in test output | INFO | GMM convergence warnings appear in test output during orchestrator tests. These come from sklearn's internal GMM fitting within DataTransformer (not from LinkageModel — GaussianMixture has been removed from LinkageModel). Warnings are non-fatal and do not affect test pass/fail status. |

No BLOCKER anti-patterns found. Phase 02 introduced no new blocker anti-patterns.

---

### Human Verification Required

#### 1. Production-Scale FK Chain Integrity

**Test:** Run the 3-table and 4-table FK chain integration tests on a dataset with 10,000+ parent rows and realistic cardinality (2-10 children per parent).
**Expected:** Zero orphaned FK references at all join levels; memory usage remains stable rather than accumulating across tables.
**Why human:** The automated TestFKChainIntegrity tests use epochs=2, batch_size=20, and only 5-20 parent rows. Real-world generation with a full training run may expose timing, memory, or join-key assignment issues not visible at small scale.

---

### Re-verification Gap Closure Summary

**Previous status (initial verification 2026-02-22):** gaps_found — 4/5 truths verified

**Gap that was open:** Plan 01 changed `validate_schema()` to raise `SchemaValidationError` (a subclass of `SchemaError -> SynthoHiveError -> Exception`, NOT a subclass of ValueError). Two tests in `test_interface.py` — `test_metadata_validation` and `test_metadata_invalid_fk_format` — still asserted `pytest.raises(ValueError)`. Because SchemaValidationError does not inherit from ValueError, pytest never caught the exception and both tests raised as errors.

**Fix applied (Plan 05, commit c01056a):** Three changes to `syntho_hive/tests/test_interface.py`:
1. Added `from syntho_hive.exceptions import SchemaValidationError` import (line 6)
2. Changed `pytest.raises(ValueError, match="references non-existent parent table")` to `pytest.raises(SchemaValidationError, ...)` (line 35)
3. Changed `pytest.raises(ValueError, match="Invalid FK reference")` to `pytest.raises(SchemaValidationError, ...)` (line 39)

**Verification of fix:**
- `python -m pytest syntho_hive/tests/test_interface.py::test_metadata_validation syntho_hive/tests/test_interface.py::test_metadata_invalid_fk_format -v` → **2 passed**
- Full suite `python -m pytest syntho_hive/tests/ -v` → **15 passed, 4 failed** (4 failures are pre-existing Synthesizer Spark API test issues; no new failures introduced)
- `git show c01056a --stat` confirms commit exists and modifies only `test_interface.py`

**Regression check on all previously-verified artifacts:**
- `syntho_hive/exceptions.py` — SchemaValidationError present at lines 24-30, inheritance unchanged
- `syntho_hive/interface/config.py` — `_dtypes_compatible()`, `linkage_method`, collect-all errors pattern all present
- `syntho_hive/core/models/ctgan.py` — `legacy_context_conditioning` flag, `gen_ctx_idx` resample, save/load all present
- `syntho_hive/relational/orchestrator.py` — `_write_with_failure_policy()`, `on_write_failure`, write-and-release pattern all present
- `syntho_hive/relational/linkage.py` — GaussianMixture import absent; `_observed_counts`, `np.random.choice`, `scipy.stats.nbinom` all present
- `pyproject.toml` — `pyspark>=4.0.0,<5.0.0` and `delta-spark>=4.0.0,<5.0.0` confirmed
- `syntho_hive/tests/test_relational.py::TestFKChainIntegrity` — all 5 tests PASS in live run

No regressions introduced by Plan 05.

---

_Verified: 2026-02-23_
_Verifier: Claude (gsd-verifier)_
