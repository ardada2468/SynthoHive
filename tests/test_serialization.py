"""TEST-03: Serialization round-trip — fit -> save -> load -> sample without retraining."""
import os
import pandas as pd
import numpy as np
import pytest
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata
from syntho_hive.exceptions import SerializationError


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


def _make_model(meta):
    return CTGAN(
        meta,
        batch_size=32,
        epochs=3,
        embedding_dim=16,
        generator_dim=(32, 32),
        discriminator_dim=(32, 32),
    )


def test_serialization_round_trip(tmp_path, small_dataset, meta):
    """fit -> save to directory -> load into fresh CTGAN -> sample produces output."""
    model = _make_model(meta)
    model.fit(small_dataset, table_name="customers", seed=42)
    pre_save_sample = model.sample(100, seed=7)

    save_dir = str(tmp_path / "customers")
    model.save(save_dir)

    # Verify directory structure
    required_files = [
        "generator.pt", "discriminator.pt",
        "transformer.joblib", "context_transformer.joblib",
        "embedding_layers.joblib", "data_column_info.joblib",
        "metadata.json",
    ]
    saved_files = os.listdir(save_dir)
    for f in required_files:
        assert f in saved_files, f"Missing checkpoint file: {f}"

    # Cold load — fresh CTGAN instance, NO fit() called
    new_model = _make_model(meta)
    new_model.load(save_dir)
    post_load_sample = new_model.sample(100, seed=7)

    assert len(post_load_sample) == 100, \
        f"Expected 100 rows after load, got {len(post_load_sample)}"
    assert set(post_load_sample.columns) == set(pre_save_sample.columns), \
        f"Column mismatch after load: {set(post_load_sample.columns)} vs {set(pre_save_sample.columns)}"
    # transformer.output_dim must have survived the joblib round-trip
    assert hasattr(new_model.transformer, 'output_dim'), \
        "Loaded transformer missing output_dim — joblib round-trip failed"
    assert new_model.transformer.output_dim > 0, \
        f"Loaded transformer.output_dim={new_model.transformer.output_dim} — invalid"


def test_save_raises_on_existing_path(tmp_path, small_dataset, meta):
    """save() raises SerializationError when path exists and overwrite=False."""
    model = _make_model(meta)
    model.fit(small_dataset, table_name="customers", seed=42)

    save_dir = str(tmp_path / "customers")
    model.save(save_dir)

    with pytest.raises(SerializationError, match="already exists"):
        model.save(save_dir)  # overwrite=False is the default


def test_save_overwrite_true_succeeds(tmp_path, small_dataset, meta):
    """save(overwrite=True) succeeds when path already exists."""
    model = _make_model(meta)
    model.fit(small_dataset, table_name="customers", seed=42)

    save_dir = str(tmp_path / "customers")
    model.save(save_dir)
    model.save(save_dir, overwrite=True)  # Must not raise


def test_load_raises_on_missing_path(meta):
    """load() raises SerializationError when path does not exist."""
    model = _make_model(meta)
    with pytest.raises(SerializationError, match="does not exist"):
        model.load("/tmp/definitely_does_not_exist_syntho_test_path_xyz/")
