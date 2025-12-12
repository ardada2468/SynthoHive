
import unittest
import pandas as pd
import numpy as np
import shutil
import os
from unittest.mock import MagicMock, patch

from syntho_hive.relational.linkage import LinkageModel
from syntho_hive.relational.orchestrator import StagedOrchestrator
from syntho_hive.interface.config import Metadata

class TestLinkageModel(unittest.TestCase):
    def test_fit_sample(self):
        # Parent: 10 users
        # Child: Users have [0, 1, 2, ..., 9] children respectively (just for fun)
        parent_ids = np.arange(10)
        parent_df = pd.DataFrame({'id': parent_ids})
        
        child_rows = []
        for pid in parent_ids:
            # Create 'pid' children for parent 'pid'
            for _ in range(pid):
                child_rows.append({'id': len(child_rows), 'user_id': pid})
                
        child_df = pd.DataFrame(child_rows)
        # Handle case where no children exists (pid=0 causes no rows in child_df with user_id=0)
        
        model = LinkageModel()
        model.fit(parent_df, child_df, fk_col='user_id', pk_col='id')
        
        # Test sampling
        # Create a new parent context
        new_parents = pd.DataFrame({'id': [100, 101, 102]})
        counts = model.sample_counts(new_parents)
        
        self.assertEqual(len(counts), 3)
        self.assertTrue(np.all(counts >= 0))
        # Basic check that max_children was learned roughly correctly
        print(f"Learned max children: {model.max_children}")
        self.assertGreaterEqual(model.max_children, 9)

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_output_relational"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_orchestrator_flow(self):
        # Mock metadata
        metadata = Metadata()
        metadata.add_table("users", pk="user_id", pii_cols=["name"])
        metadata.add_table("orders", pk="order_id", 
                           fk={"user_id": "users.user_id"},
                           parent_context_cols=["region"])
        
        # Mock Spark Session and SparkIO
        mock_spark = MagicMock()
        
        # Mock DataFrames
        users_data = pd.DataFrame({
            'user_id': range(100),
            'region': np.random.choice(['US', 'EU'], 100),
            'age': np.random.randint(18, 80, 100)
        })
        
        orders_data = []
        for _, user in users_data.iterrows():
            # 0 to 3 orders per user
            n_orders = np.random.randint(0, 4)
            for _ in range(n_orders):
                orders_data.append({
                    'order_id': len(orders_data),
                    'user_id': user['user_id'],
                    'amount': np.random.uniform(10, 100),
                    'region': user['region'] # Correlated with parent
                })
        orders_data = pd.DataFrame(orders_data)
        
        # Mock SparkIO methods to return Pandas DFs when read is called (or mock object with toPandas)
        # We need to patch the internal SparkIO of the orchestrator, or pass a mocked spark that produces mocks
        
        orchestrator = StagedOrchestrator(metadata, mock_spark)
        
        # Mock the IO read_dataset to return objects that behave like Spark DFs (have toPandas)
        class MockSparkDF:
            def __init__(self, pdf):
                self.pdf = pdf
            def toPandas(self):
                return self.pdf
            def createOrReplaceTempView(self, name):
                pass
            def write(self):
                return MagicMock() # Mock writer
                
        # Setup side effects for read_dataset
        def read_side_effect(path):
            # If path points to the test output dir, read it from disk (Generated Data)
            if self.output_dir in path and os.path.exists(os.path.join(path, "data.csv")):
                return MockSparkDF(pd.read_csv(os.path.join(path, "data.csv")))
                
            # Else return mock training data
            if "users" in path:
                return MockSparkDF(users_data)
            if "orders" in path:
                return MockSparkDF(orders_data)
            return MockSparkDF(pd.DataFrame())
            
        orchestrator.io.read_dataset = MagicMock(side_effect=read_side_effect)
        
        # Mock write_dataset to just save to parquet/csv or do nothing
        def write_side_effect(sdf, path, mode="overwrite", partition_by=None):
            if hasattr(sdf, "toPandas"):
                pdf = sdf.toPandas()
            else:
                pdf = sdf # already pandas?
            
            # Save properly to verify later
            os.makedirs(path, exist_ok=True)
            pdf.to_csv(os.path.join(path, "data.csv"), index=False)
            
        orchestrator.io.write_dataset = MagicMock(side_effect=write_side_effect)
        # Also mock write_pandas
        orchestrator.io.write_pandas = MagicMock(side_effect=lambda pdf, path, **kwargs: write_side_effect(pdf, path))

        # Run fit_all
        real_data_paths = {"users": "path/to/users", "orders": "path/to/orders"}
        orchestrator.fit_all(real_data_paths)
        
        # Verify models are trained
        self.assertIn("users", orchestrator.models)
        self.assertIn("orders", orchestrator.models)
        self.assertIn("orders", orchestrator.linkage_models)
        
        # Run generate
        orchestrator.generate({"users": 50}, self.output_dir)
        
        # Verify output
        users_out = pd.read_csv(os.path.join(self.output_dir, "users", "data.csv"))
        orders_out = pd.read_csv(os.path.join(self.output_dir, "orders", "data.csv"))
        
        self.assertEqual(len(users_out), 50)
        # Orders check
        # Check FK integrity
        user_ids = set(users_out['user_id'])
        order_user_ids = set(orders_out['user_id'])
        self.assertTrue(order_user_ids.issubset(user_ids), "Generated orders have user_ids not in generated users")
        
        print("Generated Users:", len(users_out))
        print("Generated Orders:", len(orders_out))

if __name__ == '__main__':
    unittest.main()
