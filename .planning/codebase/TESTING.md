# Testing Patterns

**Analysis Date:** 2026-02-22

## Test Framework

**Runner:**
- `pytest` (version ≥7.0.0, listed in `pyproject.toml` dev dependencies)
- Config: `pyproject.toml` contains test configuration:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = "test_*.py"
  ```

**Assertion Library:**
- Native `assert` statements (no custom assertion library)
- `pytest` built-in assertions used for exception testing:
  ```python
  with pytest.raises(ValueError, match="references non-existent parent table"):
  ```

**Run Commands:**
```bash
pytest tests/                  # Run all tests
pytest tests/ -v               # Run with verbose output
pytest tests/ --tb=short       # Run with shorter traceback
pytest tests/ -k test_ctgan    # Run specific test by keyword
```

Note: No coverage command, watch mode, or CI config present.

## Test File Organization

**Location:**
- Primary test directory: `tests/` (root level)
- Nested structure: `tests/core/`, `tests/e2e_scenarios/`
- Parallel structure to source: Test mirrors `syntho_hive/` structure where applicable
- Co-located tests: Some test files in `syntho_hive/tests/` (legacy, alongside source)

**Naming:**
- Pattern: `test_*.py` (enforced via `python_files = "test_*.py"`)
- Examples: `test_models.py`, `test_transformer.py`, `test_privacy.py`, `test_validation.py`
- Helper/utility tests: `manual_privacy_test.py`, `repro_quickstart.py` (outside standard pattern)

**Structure:**
```
tests/
├── test_models.py              # Unit tests for core models
├── test_transformer.py         # Data transformer tests
├── test_privacy.py             # Privacy module tests
├── test_validation.py          # Validation logic tests
├── test_checkpointing.py       # Checkpointing feature tests
├── core/
│   ├── test_constraints.py     # Constraint handling
│   └── test_embeddings.py      # Embedding logic
├── e2e_scenarios/
│   └── retail_test.py          # End-to-end retail scenario
└── [manual/integration tests]
```

## Test Structure

**Suite Organization:**
```python
import pytest
import pandas as pd
import numpy as np
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata

def test_ctgan_full_cycle():
    """Test CTGAN fit, sample, save, and load lifecycle."""
    # 1. Setup Data
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.normal(30, 5, 100),
        'income': np.random.exponential(50000, 100),
        'city': np.random.choice(['NY', 'SF', 'LA'], 100)
    })

    # 2. Setup Metadata
    meta = Metadata()
    meta.add_table("users", "id")

    # 3. Init Model
    model = CTGAN(metadata=meta, batch_size=20, epochs=2)

    # 4. Fit
    model.fit(data, table_name="users")
    assert model.generator is not None

    # 5. Sample
    synthetic_data = model.sample(50)
    assert len(synthetic_data) == 50

    # 6. Save and Load
    model.save("test_ctgan_model.pth")
    assert os.path.exists("test_ctgan_model.pth")
```

**Patterns:**
- **Setup:** Seed RNG (`np.random.seed(42)`) for reproducibility
- **Teardown:** Manual cleanup with `shutil.rmtree()`, `os.remove()` (seen in `test_checkpointing.py`, `test_models.py`)
- **Assertion:** Direct `assert` statements with descriptive messages
- **Phases:** Tests organized in logical steps with comments (1. Setup, 2. Config, 3. Init, 4. Fit, 5. Sample, 6. Save)
- **Test isolation:** Each test creates independent fixtures; no shared state across tests

## Mocking

**Framework:** `unittest.mock` (built-in Python library)
- `MagicMock` for creating mock objects
- `patch` for replacing functions/classes

**Patterns:**
```python
from unittest.mock import MagicMock, patch

# Mock SparkSession for tests without Spark
@patch('syntho_hive.interface.synthesizer.SparkSession')
def test_synthesizer_without_spark(mock_spark):
    metadata = Metadata()
    synthesizer = Synthesizer(metadata, PrivacyConfig())
    # Mock will be used as the SparkSession
```

**What to Mock:**
- PySpark dependencies (optional, may not be in test environment)
- External I/O (file reads, database connections)
- Large/expensive operations (when testing logic in isolation)

**What NOT to Mock:**
- Core model logic (test real behavior)
- Data transformations (verify actual output)
- Configuration objects (use real Pydantic models)

**Example of NOT mocking:**
```python
def test_statistical_validation():
    # Real dataframes, real validator
    real_df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})
    synth_df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})

    validator = StatisticalValidator()  # Real object
    results = validator.compare_columns(real_df, synth_df)

    assert "val" in results
    assert bool(results["val"]["passed"]) is True
```

## Fixtures and Factories

**Test Data:**
```python
# Seed-based generation for reproducibility
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(30, 5, 100),
    'income': np.random.exponential(50000, 100),
    'city': np.random.choice(['NY', 'SF', 'LA'], 100)
})
```

**Factory Pattern (Metadata):**
```python
def setup_metadata():
    meta = Metadata()
    meta.add_table("users", "id", pii_cols=["email", "phone"])
    meta.add_table("orders", "order_id", fk={"user_id": "users.id"})
    return meta
```

**Location:**
- Fixtures defined inline within test functions (no `conftest.py` detected)
- No pytest `@pytest.fixture` decorators used; manual setup in test bodies
- Metadata setup repeated across tests (high duplication)

## Coverage

**Requirements:** No explicit coverage threshold enforced
- No `.coveragerc` or coverage config file present
- `# pragma: no cover` directives used in abstract base classes (see `syntho_hive/core/models/base.py`)

**View Coverage:**
```bash
# Not configured; would require manual setup:
pytest --cov=syntho_hive tests/
pytest --cov=syntho_hive --cov-report=html tests/
```

## Test Types

**Unit Tests:**
- **Scope:** Individual class methods (CTGAN, DataTransformer, StatisticalValidator)
- **Approach:** Direct instantiation, method calls, assertion of outputs
- **Examples:** `test_ctgan_full_cycle()` in `tests/test_models.py`, `test_cluster_normalizer_flow()` in `tests/test_transformer.py`
- **Data:** Small synthetic data (100-1000 rows) for speed
- **Isolation:** No external dependencies (Spark mocked or skipped)

**Integration Tests:**
- **Scope:** Multiple components working together (models + transformers, privacy + validation)
- **Approach:** Full pipeline testing (fit → sample → validate)
- **Examples:** `test_ctgan_full_cycle()` (saves/loads model), `test_data_transformer_relational()` (transformer + metadata)
- **Data:** Synthetic data matching relational structure
- **Spark:** Some marked with `try/except ImportError` for optional Spark components (see `tests/test_validation.py`)

**E2E Tests:**
- **Scope:** End-to-end scenarios (retail dataset synthesis)
- **Location:** `tests/e2e_scenarios/retail_test.py`
- **Approach:** Full pipeline from metadata setup → fit → sample → validation
- **Framework:** Not automated CI; manual/exploratory style

**Specialized Tests:**
- **Checkpointing:** `tests/test_checkpointing.py` - verifies artifact generation during training
- **Constraints:** `tests/core/test_constraints.py` - verifies min/max clamping
- **Embeddings:** `tests/core/test_embeddings.py` - embedding layer behavior
- **Null handling:** `tests/test_null_handling.py` - missing value handling
- **Project setup:** `tests/test_project_setup.py` - import/installation validation

## Common Patterns

**Async Testing:**
Not applicable (no async code in codebase).

**Error Testing:**
```python
# Using pytest.raises context manager
with pytest.raises(ValueError, match="references non-existent parent table"):
    metadata = Metadata()
    metadata.add_table("orders", "id", fk={"user_id": "missing_parent.id"})
    metadata.validate_schema()
```

**Fixture Teardown:**
```python
def test_checkpointing():
    checkpoint_dir = "./test_checkpoints"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    # ... test logic ...

    # Cleanup
    shutil.rmtree(checkpoint_dir)
```

**Comparison Testing (Real vs Synthetic):**
```python
def test_statistical_validation_bad_match():
    real_df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})
    synth_df = pd.DataFrame({"val": np.random.normal(10, 1, 1000)})  # Different distribution

    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)

    assert bool(results["val"]["passed"]) is False  # Should fail (different dists)
```

**Print Debugging:**
```python
# Tests use print() for debugging output
print(f"Metrics head:\n{metrics_df.head()}")
print(f"Files in {checkpoint_dir}: {files}")
print("\nRecovered Data Summary:")
print(recovered_df.describe())
```

## Test Execution Notes

- Tests are designed to run without GPU (device="cpu" explicit in many tests)
- Random seeds set for reproducibility (`np.random.seed(42)`)
- Temporary files/directories created and cleaned up within tests
- No test database or external service setup required
- PySpark optional; tests gracefully skip if unavailable

---

*Testing analysis: 2026-02-22*
