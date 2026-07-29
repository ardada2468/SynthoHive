"""Regression tests for core correctness fixes (transformer + CTGAN)."""

import numpy as np
import pandas as pd
import pytest

from syntho_hive.core.data.transformer import ClusterBasedNormalizer, DataTransformer
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.exceptions import GenerationError, TrainingError
from syntho_hive.interface.config import Metadata


def _metadata(name="t", pk="id"):
    meta = Metadata()
    meta.add_table(name, pk=pk)
    return meta


class TestClusterBasedNormalizer:
    def test_few_distinct_values_inverse_no_indexerror(self):
        """Generated one-hots may activate dead components; inverse must not crash."""
        norm = ClusterBasedNormalizer(n_components=10, seed=0)
        norm.fit(pd.Series([1.0, 2.0, 1.0, 2.0]))  # n_active < 10

        n = 8
        fake = np.zeros((n, norm.output_dim))
        # Put all mass on the LAST (dead) component column.
        fake[:, norm.n_components - 1] = 1.0
        fake[:, norm.n_components] = 0.5  # scalar
        result = norm.inverse_transform(fake)
        assert len(result) == n
        assert np.isfinite(result).all()

    def test_nulls_at_transform_when_fit_had_none(self):
        norm = ClusterBasedNormalizer(seed=0)
        norm.fit(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        out = norm.transform(pd.Series([1.0, np.nan, 3.0]))
        assert np.isfinite(out).all()

    def test_transform_before_fit_raises(self):
        norm = ClusterBasedNormalizer()
        with pytest.raises(ValueError, match="not been fitted"):
            norm.transform(pd.Series([1.0]))

    def test_inf_rejected_at_fit(self):
        norm = ClusterBasedNormalizer()
        with pytest.raises(ValueError, match="non-finite"):
            norm.fit(pd.Series([1.0, np.inf, 3.0]))

    def test_scalar_clipped(self):
        norm = ClusterBasedNormalizer(seed=0)
        data = pd.Series([0.0] * 50 + [1000.0])  # extreme outlier
        norm.fit(data)
        out = norm.transform(data)
        scalars = out[:, norm.n_components]
        assert np.abs(scalars).max() <= 0.99 + 1e-9


class TestDataTransformer:
    def test_non_string_column_names(self):
        meta = _metadata()
        t = DataTransformer(meta)
        df = pd.DataFrame({123: [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
        t.fit(df)
        arr = t.transform(df)
        out = t.inverse_transform(arr)
        assert 123 in out.columns

    def test_unseen_category_maps_to_sentinel_when_present(self):
        meta = _metadata()
        t = DataTransformer(meta, embedding_threshold=2)
        train = pd.DataFrame({"c": ["a", "b", "c", None]})  # embedding path
        t.fit(train)
        arr = t.transform(pd.DataFrame({"c": ["zzz-unseen"]}))
        classes = list(t._transformers["c"].classes_)
        assert classes[int(arr[0, 0])] == "<NAN>"

    def test_excluded_columns_reset_between_fits(self):
        meta = Metadata()
        meta.add_table("t", pk="id")
        t = DataTransformer(meta)
        t.fit(pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0]}), table_name="t")
        assert t._excluded_columns == ["id"]
        t.fit(pd.DataFrame({"x": [1.0, 2.0]}))
        assert t._excluded_columns == []


class TestCTGAN:
    def test_sample_before_fit_raises_typed_error(self):
        model = CTGAN(_metadata(), epochs=1, batch_size=10)
        with pytest.raises(GenerationError, match="not fitted"):
            model.sample(5)

    def test_constructor_validation(self):
        meta = _metadata()
        with pytest.raises(ValueError):
            CTGAN(meta, batch_size=0)
        with pytest.raises(ValueError):
            CTGAN(meta, epochs=0)
        with pytest.raises(ValueError):
            CTGAN(meta, discriminator_steps=0)

    def test_continuous_output_within_training_range(self):
        """With tanh-bounded scalars, generated values stay near the data range."""
        meta = _metadata()
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.uniform(0, 10, 200)})
        model = CTGAN(meta, epochs=1, batch_size=32)
        model.fit(df, table_name="t", progress_bar=False, seed=0)
        out = model.sample(100, seed=1)
        # tanh bounds the scalar to ±1 → reconstructed x within mean ± 4σ of a
        # component; loose sanity bound on the whole column:
        assert out["x"].between(-50, 60).all()

    def test_refit_with_different_schema(self):
        meta = _metadata()
        model = CTGAN(meta, epochs=1, batch_size=16)
        model.fit(pd.DataFrame({"x": np.arange(40.0)}), progress_bar=False, seed=0)
        # Refit with a completely different schema must not reuse the old net.
        df2 = pd.DataFrame({"a": np.arange(40.0), "b": ["u", "v"] * 20})
        model.fit(df2, progress_bar=False, seed=0)
        out = model.sample(10, seed=0)
        assert set(out.columns) == {"a", "b"}

    def test_sample_restores_train_mode_on_error(self):
        meta = _metadata()
        model = CTGAN(meta, epochs=1, batch_size=16)
        model.fit(pd.DataFrame({"x": np.arange(40.0)}), progress_bar=False, seed=0)
        model.generator.train()
        with pytest.raises(ValueError):
            model.sample(5, context=pd.DataFrame({"c": [1]}))  # wrong context len
        assert model.generator.training

    def test_save_load_roundtrip_safe_format(self, tmp_path):
        meta = _metadata()
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.uniform(0, 10, 120),
                "hc": [f"cat_{i % 60}" for i in range(120)],  # embedding path
            }
        )
        model = CTGAN(meta, epochs=1, batch_size=32, embedding_threshold=10)
        model.fit(df, table_name="t", progress_bar=False, seed=0)

        path = tmp_path / "ckpt"
        model.save(str(path))
        assert (path / "embedding_layers.pt").exists()  # safe state_dict format
        assert not (path / "embedding_layers.joblib").exists()

        fresh = CTGAN(meta, epochs=1, batch_size=32, embedding_threshold=10)
        fresh.load(str(path))
        out = fresh.sample(20, seed=1)
        assert set(out.columns) == {"x", "hc"}
        assert len(out) == 20

    def test_nan_loss_raises_training_error(self):
        meta = _metadata()
        model = CTGAN(meta, epochs=1, batch_size=8)
        df = pd.DataFrame({"x": np.arange(16.0)})

        # Force divergence by poisoning generator weights after build via a
        # monkeypatched _build_model wrapper.
        orig_build = model._build_model

        def poisoned_build(*args, **kwargs):
            orig_build(*args, **kwargs)
            for p in model.generator.parameters():
                p.data.fill_(float("nan"))

        model._build_model = poisoned_build
        with pytest.raises(TrainingError, match="diverged"):
            model.fit(df, progress_bar=False, seed=0)
