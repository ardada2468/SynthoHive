"""TEST-01: Single-table end-to-end test — fit -> sample -> basic validation."""
import pandas as pd
import numpy as np
import pytest
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata


@pytest.fixture
def small_dataset():
    np.random.seed(0)
    return pd.DataFrame({
        "id": range(200),
        "age": np.random.randint(18, 80, 200),
        "income": np.random.exponential(50_000, 200),
        "city": np.random.choice(["NY", "SF", "LA", "CHI"], 200),
    })


@pytest.fixture
def meta():
    m = Metadata()
    m.add_table("customers", pk="id")
    return m


def test_single_table_e2e(small_dataset, meta):
    """Fit a CTGAN model on a small dataset, sample, and verify output shape."""
    model = CTGAN(
        meta,
        batch_size=32,
        epochs=3,
        embedding_dim=16,
        generator_dim=(32, 32),
        discriminator_dim=(32, 32),
    )
    model.fit(small_dataset, table_name="customers", seed=42)
    result = model.sample(50, seed=7)

    assert len(result) == 50, f"Expected 50 rows, got {len(result)}"
    # id is the PK — may or may not appear in output depending on config
    # At minimum, non-PK columns must appear
    expected_cols = {"age", "income", "city"}
    missing_cols = expected_cols - set(result.columns)
    assert not missing_cols, f"Missing columns in sample output: {missing_cols}"
    assert not result[list(expected_cols)].isnull().all().any(), \
        "At least one column is entirely null — model output is degenerate"


def test_single_table_fit_does_not_raise(small_dataset, meta):
    """Fit should complete without raising any exception."""
    model = CTGAN(
        meta,
        batch_size=32,
        epochs=2,
        embedding_dim=8,
        generator_dim=(16, 16),
        discriminator_dim=(16, 16),
    )
    # Should not raise
    model.fit(small_dataset, table_name="customers", seed=0)
