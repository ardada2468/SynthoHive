import pytest
import pandas as pd
import numpy as np
from syntho_hive.validation.statistical import StatisticalValidator
try:
    from syntho_hive.connectors.sampling import RelationalSampler
except ImportError:
    pass # Spark might not be available in test env

def test_statistical_validation():
    # Mock data: Real vs Synthetic (Good match)
    real_df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})
    synth_df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})
    
    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)
    
    assert "val" in results
    # KS test should pass (p > 0.05) for same dist
    assert bool(results["val"]["passed"]) is True

def test_statistical_validation_bad_match():
    # Mock data: Real vs Synthetic (Bad match)
    real_df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})
    synth_df = pd.DataFrame({"val": np.random.normal(10, 1, 1000)})
    
    validator = StatisticalValidator()
    results = validator.compare_columns(real_df, synth_df)
    
    assert bool(results["val"]["passed"]) is False
