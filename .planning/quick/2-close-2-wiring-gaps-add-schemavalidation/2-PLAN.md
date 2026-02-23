---
phase: quick
plan: 2
type: execute
wave: 1
depends_on: []
files_modified:
  - syntho_hive/__init__.py
  - syntho_hive/interface/synthesizer.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "from syntho_hive import SchemaValidationError succeeds without ImportError"
    - "Synthesizer.fit(data, validate=True) calls metadata.validate_schema() before training"
    - "Synthesizer.fit(data, validate=False) (default) does not call validate_schema() — no behavior change for existing callers"
  artifacts:
    - path: "syntho_hive/__init__.py"
      provides: "Public export of SchemaValidationError"
      contains: "SchemaValidationError"
    - path: "syntho_hive/interface/synthesizer.py"
      provides: "validate_schema wiring inside fit()"
      contains: "validate_schema"
  key_links:
    - from: "syntho_hive/__init__.py"
      to: "syntho_hive/exceptions.py"
      via: "from syntho_hive.exceptions import SchemaValidationError"
      pattern: "SchemaValidationError"
    - from: "syntho_hive/interface/synthesizer.py"
      to: "syntho_hive/interface/config.py"
      via: "self.metadata.validate_schema() inside fit() when validate=True"
      pattern: "validate_schema"
---

<objective>
Close two wiring gaps: export SchemaValidationError from the package public API, and wire the already-defined `validate` parameter in `Synthesizer.fit()` to actually call `self.metadata.validate_schema()`.

Purpose: SchemaValidationError is defined and tested but unreachable via `from syntho_hive import SchemaValidationError`. The `validate=True` code path in `fit()` is dead — no schema check runs even when the caller explicitly requests it.
Output: Two targeted file edits. No new logic, no new files.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Export SchemaValidationError from syntho_hive/__init__.py</name>
  <files>syntho_hive/__init__.py</files>
  <action>
    Add SchemaValidationError to the imports block in syntho_hive/__init__.py.

    Current imports from syntho_hive.exceptions:
      SynthoHiveError, SchemaError, TrainingError, SerializationError, ConstraintViolationError

    Add SchemaValidationError after SchemaError (it is a subclass of SchemaError — keep the hierarchy visible in the ordering):
      SynthoHiveError, SchemaError, SchemaValidationError, TrainingError, SerializationError, ConstraintViolationError

    No other changes to the file.
  </action>
  <verify>
    Run from the repo root:
      python -c "from syntho_hive import SchemaValidationError; print('OK')"

    Must print "OK" with no ImportError.
  </verify>
  <done>
    `from syntho_hive import SchemaValidationError` succeeds.
    SchemaValidationError is listed in the exceptions import block directly after SchemaError.
  </done>
</task>

<task type="auto">
  <name>Task 2: Wire validate_schema into Synthesizer.fit()</name>
  <files>syntho_hive/interface/synthesizer.py</files>
  <action>
    In the fit() method of syntho_hive/interface/synthesizer.py, add the validate_schema call.

    Current fit() structure (inside the try block, after the sample_size check and before the real_paths resolution):

      if sample_size <= 0:
          raise ValueError("sample_size must be positive")

      print(f"Fitting on data source...")
      ...
      if isinstance(data, str):
          ...

    Add the validate_schema call immediately after the sample_size guard and before the print statement:

      if validate:
          self.metadata.validate_schema()

    This means validate_schema runs before any data is loaded or paths are resolved, which is correct — schema problems should surface before training begins. SchemaValidationError (a SynthoHiveError subclass) will propagate unchanged through the existing `except SynthoHiveError: raise` guard at the bottom of fit().

    No import changes needed — validate_schema is a method on self.metadata which is already present. SchemaValidationError already propagates through the existing SynthoHiveError re-raise block without wrapping.

    Do NOT call validate_schema with real_data here — that requires DataFrames which fit() does not have at this point. The no-arg call performs structural FK checks only (table existence, FK format), which is the correct pre-flight check.
  </action>
  <verify>
    Run from the repo root:
      python -m pytest syntho_hive/tests/test_interface.py -v

    All existing tests must pass. Specifically:
      - test_metadata_validation PASS
      - test_metadata_invalid_fk_format PASS
      - test_synthesizer_fit_call PASS (validate=False default, no schema call)

    Then manually confirm the wiring exists:
      grep -n "validate_schema" syntho_hive/interface/synthesizer.py

    Must show the new if-validate block inside fit().
  </verify>
  <done>
    fit() calls self.metadata.validate_schema() when validate=True.
    fit() does NOT call validate_schema() when validate=False (default — no regression).
    All existing tests in test_interface.py pass.
  </done>
</task>

</tasks>

<verification>
After both tasks:

1. python -c "from syntho_hive import SchemaValidationError; print(SchemaValidationError.__bases__)"
   Output must include SchemaError.

2. python -m pytest syntho_hive/tests/test_interface.py syntho_hive/tests/test_relational.py -v
   All tests pass.

3. grep "SchemaValidationError" syntho_hive/__init__.py
   Must match.

4. grep -n "validate_schema" syntho_hive/interface/synthesizer.py
   Must match inside the fit() method body.
</verification>

<success_criteria>
- `from syntho_hive import SchemaValidationError` works without error
- `Synthesizer.fit(..., validate=True)` triggers schema validation before training
- Default `validate=False` behavior unchanged — zero regression for existing callers
- All pytest tests in test_interface.py and test_relational.py pass
</success_criteria>

<output>
After completion, create `.planning/quick/2-close-2-wiring-gaps-add-schemavalidation/2-SUMMARY.md`
</output>
