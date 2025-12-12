import unittest
import pandas as pd
import numpy as np
from syntho_hive.interface.config import Metadata, Constraint
from syntho_hive.core.data.transformer import DataTransformer

class TestConstraints(unittest.TestCase):
    def test_constraints_enforcement(self):
        # 1. Setup Metadata with Constraints
        meta = Metadata()
        meta.add_table(
            "test_table", 
            pk="id", 
            constraints={
                "age": Constraint(dtype="int", min=18, max=100),
                "score": Constraint(dtype="float", min=0.0, max=1.0)
            }
        )
        
        # 2. Setup Transformer
        transformer = DataTransformer(meta, embedding_threshold=10)
        
        # 3. Fit on Mock Data
        data = pd.DataFrame({
            "id": range(10),
            "age": np.random.randint(18, 100, 10),
            "score": np.random.rand(10)
        })
        transformer.fit(data, table_name="test_table")
        
        # 4. Mock Transformed Data (Inverse Transform)
        # Create some transformed data that would be out of bounds or float for int
        # We need to manually construct what 'transform' would output to feed 'inverse_transform'
        # But simpler: we can just patch/hack the inverse_transform logic? 
        # No, let's actually run a full cycle but intercept the values just before clamping?
        # Or better: We trust the transformer structure, we just check if it clamps.
        
        # Let's trust that inverse_transform takes an array and returns a dataframe.
        # We need to know the output dimension.
        dim = transformer.output_dim
        
        # Create dummy model output "z"
        # We need valid one-hot + normalized scalar for the continuous columns.
        # This is complex to manually construct.
        
        # A better unit test strategy for this specific feature might be to specificially call the logic?
        # But we want to test `inverse_transform`. 
        
        # Let's generate "random" model output, invoke inverse_transform, and inspect results.
        # The VMG components might produce wild values, which SHOULD be clamped.
        
        # Need enough samples to likely hit out of bounds naturally?
        # Or we can just use the transformer's own transformed data, but modify it to be out of bounds.
        
        transformed = transformer.transform(data)
        
        # Modify the scalar parts to be out of bounds.
        # Layout: 
        # Age: 10 components (10 cols) + 1 scalar (1 col) = 11?
        # Score: 10 components + 1 scalar = 11?
        # Total approx 22 cols.
        
        # Let's just mess up the whole array with huge values.
        # The one-hot part being wrong doesn't matter much for scalar reconstruction value magnitude usually,
        # it just selects which mean/std to use.
        
        # Force high values
        transformed[:, -1] = 1000.0 # Last column (likely 'score' scalar)
        transformed[:, 10] = -1000.0 # Middle column (likely 'age' scalar) -> Should map to very low age
        
        # Inverse Transform
        recovered_df = transformer.inverse_transform(transformed)
        
        # 5. Assertions
        print("\nRecovered Data Summary:")
        print(recovered_df.describe())
        
        # Check Age Constraints
        self.assertTrue(pd.api.types.is_integer_dtype(recovered_df["age"]), "Age should be integer")
        self.assertTrue((recovered_df["age"] >= 18).all(), "Age min constraint failed")
        self.assertTrue((recovered_df["age"] <= 100).all(), "Age max constraint failed")
        
        # Check Score Constraints
        # Note: Score is float, so just min/max
        self.assertTrue((recovered_df["score"] >= 0.0).all(), "Score min constraint failed")
        self.assertTrue((recovered_df["score"] <= 1.0).all(), "Score max constraint failed")

if __name__ == "__main__":
    unittest.main()
