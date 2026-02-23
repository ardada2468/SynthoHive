"""
QUAL-04 regression test: enforce_constraints=True raises ConstraintViolationError.

Gap closure for Phase 1 verification failure — the original Plan 03 implementation
logged a warning and returned valid rows instead of raising. This test verifies the
corrected behavior: violations raise ConstraintViolationError with column and value details.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata
from syntho_hive.exceptions import ConstraintViolationError


@pytest.fixture
def small_dataset():
    np.random.seed(0)
    return pd.DataFrame({
        "id": range(200),
        "age": np.random.randint(18, 80, 200),
        "income": np.random.exponential(50_000, 200),
        "city": np.random.choice(["NY", "SF", "LA"], 200),
    })


@pytest.fixture
def meta():
    m = Metadata()
    m.add_table("customers", pk="id")
    return m


@pytest.fixture
def trained_model(small_dataset, meta):
    """A trained CTGAN model for use across constraint tests."""
    model = CTGAN(
        meta,
        batch_size=32,
        epochs=3,
        embedding_dim=16,
        generator_dim=(32, 32),
        discriminator_dim=(32, 32),
    )
    model.fit(small_dataset, table_name="customers", seed=42)
    return model


def _make_constraint(min_val=None, max_val=None):
    """Build a simple constraint mock with .min and .max attributes."""
    c = MagicMock()
    c.min = min_val
    c.max = max_val
    return c


def _inject_constraints(model, table_name, constraints_dict):
    """
    Inject a fake table config with the given constraints dict into model.metadata
    so that sample(enforce_constraints=True) will process them.

    constraints_dict: {col_name: mock_constraint_with_min_max}
    """
    table_config = MagicMock()
    table_config.constraints = constraints_dict
    model.metadata = MagicMock()
    model.metadata.get_table.return_value = table_config
    # Make transformer expose table_name so the lookup is triggered
    model.transformer.table_name = table_name


def _make_df_with_violation():
    """Return a DataFrame with a column that will violate a min=0 constraint."""
    return pd.DataFrame({
        "age": [-5, 25, 30],       # -5 violates min=0
        "income": [50000, 60000, 70000],
        "city": ["NY", "SF", "LA"],
    })


def test_enforce_constraints_raises_on_violation(trained_model):
    """
    sample(enforce_constraints=True) must raise ConstraintViolationError when
    generated data contains a constraint violation — not warn-and-return.

    This test patches the sample output to guarantee a violation so the test
    does not depend on probabilistic model output.
    """
    # Inject a constraint: age must be >= 0
    _inject_constraints(
        trained_model,
        "customers",
        {"age": _make_constraint(min_val=0)},
    )

    # Patch inverse_transform to return a DataFrame that definitely violates age >= 0
    violating_df = _make_df_with_violation()
    with patch.object(trained_model.transformer, "inverse_transform",
                      return_value=violating_df):
        with pytest.raises(ConstraintViolationError) as exc_info:
            trained_model.sample(3, enforce_constraints=True, seed=7)

    # Error message must include column name and observed value
    error_msg = str(exc_info.value)
    assert "age" in error_msg, (
        f"ConstraintViolationError message must name the violating column 'age'. Got: {error_msg}"
    )
    assert "min=" in error_msg or "got" in error_msg, (
        f"ConstraintViolationError message must include observed value context. Got: {error_msg}"
    )


def test_enforce_constraints_false_does_not_raise(trained_model):
    """
    sample(enforce_constraints=False) (the default) must not raise even when
    generated data contains constraint violations — it returns whatever the model generates.
    """
    # Inject the same constraint
    _inject_constraints(
        trained_model,
        "customers",
        {"age": _make_constraint(min_val=0)},
    )

    violating_df = _make_df_with_violation()
    with patch.object(trained_model.transformer, "inverse_transform",
                      return_value=violating_df):
        # Must not raise — constraint checking is opt-in
        result = trained_model.sample(3, enforce_constraints=False, seed=7)

    assert result is not None
    assert len(result) == 3


def test_enforce_constraints_no_violation_does_not_raise(trained_model):
    """
    sample(enforce_constraints=True) must not raise when no violations exist.
    """
    _inject_constraints(
        trained_model,
        "customers",
        {"age": _make_constraint(min_val=0)},
    )

    clean_df = pd.DataFrame({
        "age": [25, 30, 45],        # all >= 0 — no violation
        "income": [50000, 60000, 70000],
        "city": ["NY", "SF", "LA"],
    })
    with patch.object(trained_model.transformer, "inverse_transform",
                      return_value=clean_df):
        result = trained_model.sample(3, enforce_constraints=True, seed=7)

    assert result is not None
    assert len(result) == 3


def test_enforce_constraints_error_message_format(trained_model):
    """
    ConstraintViolationError message follows the expected format:
    'ConstraintViolationError: N violation(s) found — col: got X (min/max=Y)'
    """
    _inject_constraints(
        trained_model,
        "customers",
        {
            "age": _make_constraint(min_val=0),
            "income": _make_constraint(max_val=100_000),
        },
    )

    # Both age and income violate their constraints
    double_violation_df = pd.DataFrame({
        "age": [-10, 25, 30],         # -10 violates min=0
        "income": [200_000, 60_000, 70_000],  # 200_000 violates max=100_000
        "city": ["NY", "SF", "LA"],
    })
    with patch.object(trained_model.transformer, "inverse_transform",
                      return_value=double_violation_df):
        with pytest.raises(ConstraintViolationError) as exc_info:
            trained_model.sample(3, enforce_constraints=True, seed=7)

    error_msg = str(exc_info.value)
    # Both columns must appear in the error message
    assert "age" in error_msg
    assert "income" in error_msg
