import pytest
import pandas as pd
import numpy as np
from syntho_hive.validation.statistical import StatisticalValidator, TABLE_STATUS_KEY

try:
    from syntho_hive.connectors.sampling import RelationalSampler
except ImportError:
    pass  # Spark might not be available in test env


def test_statistical_validation():
    np.random.seed(42)
    # Mock data: Real vs Synthetic (Good match)
    real_df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})
    synth_df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})

    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)

    assert "val" in results
    # KS statistic (effect size) should be small for the same distribution
    assert bool(results["val"]["passed"]) is True


def test_statistical_validation_bad_match():
    np.random.seed(42)
    # Mock data: Real vs Synthetic (Bad match)
    real_df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})
    synth_df = pd.DataFrame({"val": np.random.normal(10, 1, 1000)})

    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)

    assert bool(results["val"]["passed"]) is False


def test_ks_effect_size_criterion():
    """Pass/fail is driven by the KS statistic D, not the p-value.

    With very large samples, a tiny distribution shift gives a significant
    p-value (< 0.05) but a small effect size (D < 0.1) — this must PASS.
    """
    rng = np.random.default_rng(0)
    n = 50_000
    real_df = pd.DataFrame({"val": rng.normal(0, 1, n)})
    synth_df = pd.DataFrame({"val": rng.normal(0.05, 1, n)})

    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)

    res = results["val"]
    assert res["test"] == "ks_test"
    assert "statistic" in res and "p_value" in res
    assert res["statistic"] < 0.1
    assert res["p_value"] < 0.05  # statistically "significant"...
    assert bool(res["passed"]) is True  # ...but effect size is negligible


def test_ks_effect_size_fail():
    rng = np.random.default_rng(1)
    real_df = pd.DataFrame({"val": rng.normal(0, 1, 2000)})
    synth_df = pd.DataFrame({"val": rng.normal(0.8, 1, 2000)})

    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)

    assert results["val"]["statistic"] >= 0.1
    assert bool(results["val"]["passed"]) is False


def test_low_cardinality_numeric_uses_tvd():
    rng = np.random.default_rng(2)
    real_df = pd.DataFrame({"rating": rng.integers(1, 6, 500)})
    synth_df = pd.DataFrame({"rating": rng.integers(1, 6, 500)})

    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)

    assert results["rating"]["test"] == "tvd"
    assert "statistic" in results["rating"]


def test_bool_column_uses_tvd():
    real_df = pd.DataFrame({"flag": [True, False] * 50})
    synth_df = pd.DataFrame({"flag": [True, False] * 50})

    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)

    assert results["flag"]["test"] == "tvd"
    assert bool(results["flag"]["passed"]) is True


def test_category_vs_object_strings_comparable():
    """category-of-strings vs plain object strings must not be a type mismatch."""
    real_df = pd.DataFrame({"c": pd.Series(["a", "b", "a", "b"], dtype="category")})
    synth_df = pd.DataFrame({"c": pd.Series(["a", "b", "a", "b"], dtype=object)})

    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)

    assert "error" not in results["c"]
    assert results["c"]["test"] == "tvd"
    assert bool(results["c"]["passed"]) is True


def test_empty_dataframe_typed_shape():
    validator = StatisticalValidator()
    results = validator.compare_columns(pd.DataFrame(), pd.DataFrame())

    assert TABLE_STATUS_KEY in results
    assert "error" in results[TABLE_STATUS_KEY]
    # Every value in the mapping is a dict (per-column contract).
    assert all(isinstance(v, dict) for v in results.values())


def test_statistic_key_present_for_all_successful_columns():
    """Checkpoint validation averages results[col]["statistic"] — keep the key."""
    rng = np.random.default_rng(3)
    real_df = pd.DataFrame(
        {
            "num": rng.normal(0, 1, 200),
            "cat": rng.choice(["x", "y", "z"], 200),
            "small": rng.integers(0, 3, 200),
        }
    )
    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, real_df.copy())

    for col in ["num", "cat", "small"]:
        assert "statistic" in results[col]
        assert isinstance(results[col]["statistic"], float)


def test_correlation_column_mismatch_not_nan():
    rng = np.random.default_rng(4)
    real_df = pd.DataFrame(
        {
            "a": rng.normal(size=100),
            "b": rng.normal(size=100),
            "c": rng.normal(size=100),
        }
    )
    # Synthetic is missing "c" and has an extra column "d".
    synth_df = pd.DataFrame(
        {
            "a": rng.normal(size=100),
            "b": rng.normal(size=100),
            "d": rng.normal(size=100),
        }
    )

    validator = StatisticalValidator()
    result = validator.check_correlations(real_df, synth_df)

    assert not np.isnan(result["frobenius_distance"])
    assert result["columns_compared"] == 2  # only a, b are shared


def test_correlation_constant_column_masked_not_zeroed():
    rng = np.random.default_rng(5)
    real_df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    synth_df = pd.DataFrame({"a": rng.normal(size=100), "b": np.ones(100)})

    validator = StatisticalValidator()
    result = validator.check_correlations(real_df, synth_df)

    # Constant column -> NaN correlations in the synth matrix; those cells are
    # excluded (counted), and the distance stays finite.
    assert not np.isnan(result["frobenius_distance"])
    assert result["undefined_pairs"] > 0


def test_correlation_single_common_column():
    real_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    synth_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    validator = StatisticalValidator()
    result = validator.check_correlations(real_df, synth_df)

    assert result["frobenius_distance"] == 0.0
    assert result["columns_compared"] == 1
