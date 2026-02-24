---
phase: "06"
phase_name: synthesizer-validation-hardening
status: passed
verified_at: 2026-02-23
verifier: orchestrator (agent rate-limited; manual spot-check)
---

# Phase 06: Synthesizer Validation Hardening — Verification

## Goal

The Synthesizer public façade enforces both structural and data-level FK validation at fit time, and invalid model class injection fails immediately at `__init__` regardless of Spark session presence.

## Must-Haves Checklist

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `Synthesizer.fit(validate=True, data=dfs)` calls `validate_schema(real_data=dfs)` and raises `SchemaValidationError` when FK type mismatches exist | ✓ | `test_synthesizer_fit_validate_catches_fk_type_mismatch` PASSED |
| 2 | `Synthesizer(model=InvalidClass, spark_session=None)` raises `TypeError` at `__init__` time — not deferred to `fit()` | ✓ | `test_synthesizer_rejects_invalid_model_cls_without_spark` PASSED |

## Requirement Coverage

| Requirement | Description | Status |
|-------------|-------------|--------|
| REL-03 | FK type mismatches detected at validate_schema() time before training | ✓ Closed — TD-01 fix wires real_data into validate_schema() through Synthesizer facade |
| MODEL-02 | Invalid model_cls raises TypeError at __init__ | ✓ Closed — TD-04 fix adds issubclass guard in Synthesizer.__init__() unconditionally |

## Test Results

```
pytest syntho_hive/tests/test_interface.py -v --tb=short
14 collected

PASSED  test_metadata_validation
PASSED  test_metadata_invalid_fk_format
PASSED  test_synthesizer_init_no_spark
FAILED  test_synthesizer_fit_requires_spark          ← pre-existing TD-02 (Phase 7)
FAILED  test_synthesizer_sample_requires_spark       ← pre-existing TD-02 (Phase 7)
FAILED  test_synthesizer_fit_call                    ← pre-existing TD-02 (Phase 7)
FAILED  test_synthesizer_sample_call                 ← pre-existing TD-02 (Phase 7)
PASSED  test_save_to_hive
PASSED  test_stub_model_routes_through_pipeline
PASSED  test_synthesizer_accepts_model_parameter
PASSED  test_synthesizer_default_model_is_ctgan
PASSED  test_issubclass_guard_rejects_invalid_model_cls
PASSED  test_synthesizer_rejects_invalid_model_cls_without_spark  ← NEW Phase 6
PASSED  test_synthesizer_fit_validate_catches_fk_type_mismatch    ← NEW Phase 6

10 passed, 4 failed
```

The 4 failures are pre-existing TD-02 issues confirmed present before this phase (Phase 7 scope). No regressions introduced.

## Codebase Spot-Checks

### TD-04 fix — `syntho_hive/interface/synthesizer.py:__init__()`
- Guard: `isinstance(model, type) and issubclass(model, ConditionalGenerativeModel)` present before `self.metadata = metadata`
- Fires unconditionally — no Spark session required
- `CTGAN` (valid subclass) passes without error ✓

### TD-01 fix — `syntho_hive/interface/synthesizer.py:fit()`
- Validate block moved before `if not self.orchestrator` check
- `validate_schema(real_data=data)` called when `data` is a `dict` of `pd.DataFrame`
- `SchemaValidationError` propagates unchanged via `except SynthoHiveError: raise` ✓

## Verdict

**PASSED** — Phase 06 goal achieved. TD-01 and TD-04 from the v1.1 audit are closed. REL-03 and MODEL-02 E2E flows are no longer broken through the Synthesizer facade. No regressions to previously-passing tests.
