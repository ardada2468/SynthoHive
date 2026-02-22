# Coding Conventions

**Analysis Date:** 2026-02-22

## Naming Patterns

**Files:**
- Lowercase with underscores: `synthesizer.py`, `transformer.py`, `faker_contextual.py`
- Module files match class/functionality names
- Test files follow pattern: `test_*.py` (e.g., `test_models.py`, `test_transformer.py`)

**Functions:**
- Public methods: `snake_case` (e.g., `fit()`, `sample()`, `generate_validation_report()`)
- Private methods: Leading underscore `_snake_case` (e.g., `_compile_layout()`, `_apply_embeddings()`, `_build_model()`)
- Factory methods: `default()` class method pattern (see `PrivacyConfig.default()` in `syntho_hive/interface/config.py`)

**Variables:**
- Instance attributes: `snake_case` (e.g., `self.metadata`, `self.embedding_dim`, `self.models`)
- Class attributes: `UPPER_CASE` for constants (not prevalent in codebase)
- Loop/temp variables: Short `snake_case` (e.g., `col`, `table`, `result`)
- Config/metadata objects: descriptive names like `real_dfs`, `synth_dfs`, `table_config`

**Types/Classes:**
- PascalCase: `GenerativeModel`, `ConditionalGenerativeModel`, `CTGAN`, `DataTransformer`, `StatisticalValidator`
- Abstract base classes: `ABC` subclass with `pass # pragma: no cover` in abstract methods (see `syntho_hive/core/models/base.py`)
- Data classes: Use `@dataclass` decorator (e.g., `PiiRule`, `PrivacyConfig` in `syntho_hive/privacy/sanitizer.py`)
- Pydantic models: Use `BaseModel` (e.g., `Metadata`, `TableConfig` in `syntho_hive/interface/config.py`)

## Code Style

**Formatting:**
- No explicit formatter configured in project
- Dependencies include `black` for formatting (in `pyproject.toml` dev dependencies)
- Line length: Implicit ~100-120 chars (no config file enforced)
- Imports: 4-space indentation (Python standard)

**Linting:**
- `flake8` configured as dev dependency (in `pyproject.toml`)
- No `.flake8` configuration file present
- Code contains `# pragma: no cover` directives for coverage exclusion

**Type Hints:**
- Extensive use of type hints throughout (imported from `typing` module)
- Common patterns:
  - `Optional[Type]` for nullable values
  - `Dict[str, Type]` for mappings
  - `List[Type]` for sequences
  - `Union[Type1, Type2]` for multiple types
  - `Tuple[int, int]` for fixed-size tuples (generator/discriminator dims)
  - `Any` for flexible types (framework-dependent parameters like PySpark types)
  - `Callable` for function type hints (see `syntho_hive/privacy/sanitizer.py`)

## Import Organization

**Order:**
1. Standard library: `import os`, `import json`, `import csv`, `from abc import ABC, abstractmethod`
2. Third-party: `import numpy`, `import pandas`, `import torch`, `from sklearn.preprocessing import OneHotEncoder`
3. Local/internal: `from syntho_hive.core.models.ctgan import CTGAN`, `from .base import GenerativeModel`

**Path Aliases:**
- Full module paths used: `from syntho_hive.interface.config import Metadata`
- Relative imports with single dot: `from .base import ConditionalGenerativeModel` (within same package)
- No centralized path aliases file detected; full paths preferred

**Pattern:**
- Try/except for optional imports (PySpark, which may not be available):
```python
try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = Any
```

## Error Handling

**Patterns:**
- Explicit `ValueError` for parameter validation (most common):
  ```python
  if sample_size <= 0:
      raise ValueError("sample_size must be positive")
  ```
- Schema validation errors: Custom `ValueError` with descriptive messages (see `syntho_hive/interface/config.py`:
  ```python
  raise ValueError(f"Invalid FK reference '{parent_ref}' in table '{table_name}'. Format should be 'parent_table.parent_col'.")
  ```
- Try/except blocks for recovery or fallback (e.g., locale loading in `syntho_hive/privacy/faker_contextual.py`):
  ```python
  try:
      # Attempt operation
  except Exception as e:
      self.logger.warning(f"Could not load locale {mapped_locale}, falling back to default. Error: {e}")
  ```
- Bare `except:` used in rare cases for recovery (see `syntho_hive/interface/synthesizer.py` line 153-154):
  ```python
  try:
      df = self.spark.read.table(path)
  except:
      df = self.spark.read.format("delta").load(path)
  ```
- Re-raise pattern: `except Exception as e: ... raise e` to preserve original traceback

## Logging

**Framework:** Standard Python `logging` module (see `syntho_hive/privacy/faker_contextual.py`)
- Logger initialized: `self.logger = logging.getLogger(__name__)`
- Levels used: `.warning()`, `.error()`

**Fallback:** `print()` statements used extensively for user-facing output (not ideal, but pervasive):
- Training progress: `print(f"Epoch {epoch}: Loss D={loss_D.item():.4f}, Loss G={loss_G.item():.4f}")`
- Data loading: `print(f"Loading real data for {table} from {path}...")`
- Status messages: `print(f"Fitting model for table: {table_name}")`

**Observation:** `structlog` is listed in dependencies (`pyproject.toml`) but rarely used in codebase. Codebase inconsistently mixes `logging`, `print()`, and unused `structlog`.

## Comments

**When to Comment:**
- Docstrings on all public classes and methods (see pattern below)
- Inline comments for non-obvious logic (e.g., "Random weight term for interpolation" in `syntho_hive/core/models/ctgan.py`)
- TODO comments sparse but present (e.g., in `syntho_hive/core/data/transformer.py`)

**Docstring/JSDoc Pattern:**
- Google-style docstrings (not NumPy or Sphinx style):
  ```python
  def fit(self, data: pd.DataFrame, **kwargs: Any) -> None:
      """Train the model on the provided data.

      Args:
          data: Training dataframe for the model to learn from.
          **kwargs: Model-specific hyperparameters or training options.
      """
  ```
- Sections: `Args:`, `Returns:`, `Raises:` (if applicable)
- Exception documentation: Methods document expected `ValueError`, `ImportError` in docstrings
- No type hints in docstrings (redundant with code annotations)

## Function Design

**Size:**
- Mix of sizes; public methods tend to be 10-50 lines
- Complex methods like `fit()` or `generate()` can span 100+ lines
- Internal `_` prefixed methods tend toward 30-60 lines
- No strict line limit enforced

**Parameters:**
- Required first: `self` (methods), then required parameters
- Optional with defaults: `Optional[Type] = None` or specific defaults
- Flexible kwargs: `**kwargs: Union[int, str, Tuple[int, int]]` for forwarding model hyperparameters
- Named parameters preferred over positional (seen in `fit()` calls with explicit `table_name=`, `epochs=`)

**Return Values:**
- Explicit return types in annotations: `-> pd.DataFrame`, `-> None`, `-> Dict[str, Any]`
- Mixed returns allowed: `-> Union[Dict[str, str], Dict[str, pd.DataFrame]]` (see `syntho_hive/interface/synthesizer.py` line 86)
- Methods returning `None` use explicit `pass` in abstract definitions

## Module Design

**Exports:**
- `__init__.py` files typically import top-level classes only:
  ```python
  # syntho_hive/__init__.py
  from .interface.config import Metadata, PrivacyConfig
  ```
- No `__all__` definitions detected; implicit exports

**Barrel Files:**
- Present in some packages (e.g., `syntho_hive/interface/__init__.py`, `syntho_hive/core/__init__.py`)
- Used for consolidating module imports, not universal pattern

**Package Structure:**
- Layer-based: `core/`, `interface/`, `validation/`, `privacy/`, `relational/`, `connectors/`
- Clear separation of concerns (no god modules)
- Related functionality grouped (e.g., all privacy logic in `syntho_hive/privacy/`)

---

*Convention analysis: 2026-02-22*
