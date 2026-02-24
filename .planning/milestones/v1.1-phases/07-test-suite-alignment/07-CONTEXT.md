# Phase 7: Test Suite Alignment - Context

**Gathered:** 2026-02-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix 4 pre-existing test failures in `syntho_hive/tests/test_interface.py`. All changes are confined to the test file — no production code changes. Phase closes TD-02 from the v1.1 audit.

The 4 failing tests are:
1. `test_synthesizer_fit_requires_spark` — asserts `ValueError`, code now raises `TrainingError`
2. `test_synthesizer_sample_requires_spark` — same exception mismatch
3. `test_synthesizer_fit_call` — `assert_called_with(paths)` fails because actual call includes `epochs=300, batch_size=500`
4. `test_synthesizer_sample_call` — `rows, output_base = args` fails; `output_path_base=None` is passed as a keyword arg, leaving `args` with only 1 element

</domain>

<decisions>
## Implementation Decisions

### Exception assertions
- Replace `pytest.raises(ValueError)` with `pytest.raises(TrainingError)` — strict replacement, no ValueError fallback
- Keep the `match="SparkSession required"` string — it is a valid substring of the TrainingError message (`"fit() failed. Original error: SparkSession required for fit()"`)
- Do NOT change the match pattern to be more specific; `"SparkSession required"` is sufficient and forward-stable

### Signature assertion style
- Fix `test_synthesizer_fit_call` by asserting only the table paths positional arg, not the kwargs
- Use `call_args.args[0] == expected_paths` (or equivalent) instead of `assert_called_with(paths)` — ignore `epochs` and `batch_size` kwargs
- Do NOT assert `sample_size` behavior; test only verifies fit_all routing

### generate() args unpacking
- Fix `test_synthesizer_sample_call` by updating the args unpack to handle keyword args
- Claude's Discretion: choose the assertion that best validates the routing contract — either assert `rows` only, or assert `rows` + verify `output_path_base=None` in kwargs
- Remove or fix the stale `"/tmp/syntho_hive_output/delta"` assertion (this value is never passed in the no-output-path branch)
- No new test cases added — fix the existing test only

### Test run cleanliness
- Fix all 4 failures; do not touch the 10 passing tests
- No warnings currently exist — no warning cleanup needed
- Plan must include a verification step: run `pytest syntho_hive/tests/test_interface.py` and confirm 0 failures before the phase is complete

### Claude's Discretion
- Exact assertion style for `generate()` call (rows-only vs rows + output_path_base=None kwarg check)
- Whether to add a second `sample()` call in `test_synthesizer_sample_call` to cover the `output_path` branch (only if the gap warrants it within this phase scope)

</decisions>

<specifics>
## Specific Ideas

- The test for `fit_all` should check `call_args.args[0]` rather than using `assert_called_with` so that future additions to kwargs don't break the test again
- The `generate()` assertion should be robust to keyword vs positional arg placement

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 07-test-suite-alignment*
*Context gathered: 2026-02-23*
