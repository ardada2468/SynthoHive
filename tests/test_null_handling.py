
import pandas as pd
import numpy as np
import os
import shutil
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.core.data.transformer import DataTransformer

class MockMetadata:
    def __init__(self):
        self.tables = {"test_table": "config"} # minimal config
        self.constraints = {}
    def get_table(self, name):
        return None

def test_null_handling():
    print("Creating data with nulls...")
    data = pd.DataFrame({
        'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0] * 100,
        'categorical_col': ['A', 'B', None, 'A', 'C'] * 100
    })
    
    metadata = MockMetadata()
    
    print("Initializing CTGAN...")
    model = CTGAN(metadata=metadata, epochs=1, batch_size=50) # Fast run
    
    print("Fitting model...")
    model.fit(data)
    
    print("Sampling data...")
    sampled = model.sample(100)
    
    print("Sampled Data Head:")
    print(sampled.head())
    
    # Check for NaNs
    num_nulls_numeric = sampled['numeric_col'].isnull().sum()
    num_nulls_cat = sampled['categorical_col'].isnull().sum()
    
    print(f"Nulls in numeric_col: {num_nulls_numeric}")
    print(f"Nulls in categorical_col: {num_nulls_cat}")
    
    assert num_nulls_numeric > 0, "Numeric column should preserve nulls (via learning)"
    # Note: Categorical nulls might be rare if the model learns 'None' is unlikely, but with 20% nulls it should appear.
    assert num_nulls_cat > 0, "Categorical column should preserve nulls (via learning)"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_null_handling()
