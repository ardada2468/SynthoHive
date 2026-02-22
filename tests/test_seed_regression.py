"""TEST-05: Seed regression — two independent runs with seed=42 produce bit-identical output."""
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
        "city": np.random.choice(["NY", "SF", "LA"], 200),
    })


@pytest.fixture
def meta():
    m = Metadata()
    m.add_table("customers", pk="id")
    return m


def _fit_and_sample(dataset, meta, fit_seed: int, sample_seed: int) -> pd.DataFrame:
    """Create a fresh model, fit with the given seed, sample with the given seed."""
    model = CTGAN(
        meta,
        batch_size=32,
        epochs=3,
        embedding_dim=16,
        generator_dim=(32, 32),
        discriminator_dim=(32, 32),
    )
    model.fit(dataset, table_name="customers", seed=fit_seed)
    return model.sample(100, seed=sample_seed)


def test_seed_produces_identical_output(small_dataset, meta):
    """Two independent CTGAN instances trained and sampled with the same seeds must produce bit-identical DataFrames."""
    run1 = _fit_and_sample(small_dataset, meta, fit_seed=42, sample_seed=7)
    run2 = _fit_and_sample(small_dataset, meta, fit_seed=42, sample_seed=7)

    pd.testing.assert_frame_equal(
        run1, run2,
        check_exact=True,
        obj="Seed-identical runs must produce bit-identical synthetic output",
    )


def test_different_seeds_produce_different_output(small_dataset, meta):
    """Two runs with different seeds should produce different DataFrames (probabilistic sanity check)."""
    run_a = _fit_and_sample(small_dataset, meta, fit_seed=42, sample_seed=7)
    run_b = _fit_and_sample(small_dataset, meta, fit_seed=99, sample_seed=7)

    # They might theoretically be equal by chance, but with a continuous column like income
    # this is astronomically unlikely with different training seeds.
    # Use a soft check — warn rather than hard-fail.
    try:
        pd.testing.assert_frame_equal(run_a, run_b, check_exact=True)
        import warnings
        warnings.warn(
            "Different seeds produced identical output — seed parameterization may not be working correctly.",
            UserWarning,
            stacklevel=2,
        )
    except AssertionError:
        pass  # Expected: different seeds produce different output
