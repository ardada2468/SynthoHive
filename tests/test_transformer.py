import pytest
import pandas as pd
import numpy as np
from syntho_hive.core.data.transformer import DataTransformer, ClusterBasedNormalizer
from syntho_hive.interface.config import Metadata

def test_cluster_normalizer_flow():
    """Test full flow of ClusterBasedNormalizer with bimodal data."""
    # Generate bimodal data: mixture of N(0, 1) and N(10, 1)
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 500)
    data2 = np.random.normal(10, 1, 500)
    data = pd.Series(np.concatenate([data1, data2]))
    
    # Init and Fit
    norm = ClusterBasedNormalizer(n_components=5)
    norm.fit(data)
    
    assert norm.means is not None
    assert norm.stds is not None
    
    # Transform
    transformed = norm.transform(data)
    # Shape should be (N, n_components + 1)
    assert transformed.shape == (1000, 6)
    
    # Check that vectors are largely one-hot (sum of first 5 cols is 1)
    one_hots = transformed[:, :-1]
    assert np.allclose(np.sum(one_hots, axis=1), 1)
    
    # Inverse Transform
    reconstructed = norm.inverse_transform(transformed)
    
    # Check correlation (should be high)
    corr = np.corrcoef(data, reconstructed)[0, 1]
    assert corr > 0.99
    
    # Check MSE (should be low)
    mse = np.mean((data - reconstructed) ** 2)
    # It won't be zero because of the mode specific normalization (it's reversible but assumes correct cluster assignment)
    # Actually, if we use argmax for assignment and same for inverse, it should be EXACTLY reversible
    # distinct from the generative process where we might sample.
    assert mse < 1e-10

def test_data_transformer_relational():
    """Test DataTransformer with table context (PK exclusion)."""
    # Setup Metadata
    meta = Metadata()
    meta.add_table("users", pk="user_id", pii_cols=[])
    
    # Setup Data
    df = pd.DataFrame({
        'user_id': range(100), # Should be excluded
        'age': np.random.normal(30, 10, 100), # Continuous
        'city': np.random.choice(['NY', 'SF', 'LA'], 100) # Categorical
    })
    
    transformer = DataTransformer(metadata=meta)
    
    # Fit with table name
    transformer.fit(df, table_name="users")
    
    # Check that 'user_id' was excluded
    assert 'user_id' not in transformer._column_info
    assert 'age' in transformer._column_info
    assert 'city' in transformer._column_info
    
    # Transform
    # Note: Transformer expects DataFrame. If we pass df, it will ignore user_id based on fit.
    # But usually transform input should match trained cols? 
    # Current impl iterates over _column_info keys, so if df has user_id, it just ignores it.
    output = transformer.transform(df)
    
    # Expected dims:
    # Age: 10 + 1 = 11
    # City (3 unique vals): 3 (OneHot)
    # Total: 14
    expected_dim = 11 + 3
    assert output.shape == (100, expected_dim)
    
    # Inverse Transform
    recon_df = transformer.inverse_transform(output)
    
    # Verify content
    assert 'user_id' not in recon_df.columns
    assert 'age' in recon_df.columns
    assert 'city' in recon_df.columns
    
    # Check continuous values
    assert np.allclose(df['age'], recon_df['age'], atol=1e-5)
    
    # Check categorical values
    assert (df['city'] == recon_df['city']).all()

def test_data_transformer_missing_col_error():
    meta = Metadata()
    meta.add_table("simple", "id")
    df = pd.DataFrame({'val': range(20)})
    
    transformer = DataTransformer(metadata=meta)
    transformer.fit(df, table_name="simple")
    
    bad_df = pd.DataFrame({'wrong': range(20)})
    with pytest.raises(ValueError, match="Column val missing"):
        transformer.transform(bad_df)
