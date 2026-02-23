
import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
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

class TestFKChainIntegrity(unittest.TestCase):
    """TEST-02: Multi-table FK chain — zero orphans, cardinality accuracy, schema validation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_mock_io(self, training_data: dict, output_dir: str):
        """Build a mock IO object wiring read_dataset and write_pandas for the orchestrator."""

        class MockSparkDF:
            def __init__(self, pdf):
                self.pdf = pdf

            def toPandas(self):
                return self.pdf

            def createOrReplaceTempView(self, name):
                pass

            def write(self):
                return MagicMock()

        def read_side_effect(path):
            # If path points to the temp output dir, read the previously written CSV.
            for table_name in list(training_data.keys()):
                table_out_path = os.path.join(output_dir, table_name)
                if path == table_out_path and os.path.exists(
                    os.path.join(table_out_path, "data.csv")
                ):
                    return MockSparkDF(pd.read_csv(os.path.join(table_out_path, "data.csv")))
            # Fall back to training data keyed by table name found in path.
            for table_name, df in training_data.items():
                if table_name in path:
                    return MockSparkDF(df)
            return MockSparkDF(pd.DataFrame())

        def write_side_effect(pdf, path, **kwargs):
            os.makedirs(path, exist_ok=True)
            pdf.to_csv(os.path.join(path, "data.csv"), index=False)

        mock_io = MagicMock()
        mock_io.read_dataset = MagicMock(side_effect=read_side_effect)
        mock_io.write_pandas = MagicMock(side_effect=write_side_effect)
        return mock_io

    def test_3_table_chain_zero_orphans(self):
        """3-table chain users → orders → items produces zero orphaned FK references."""
        np.random.seed(42)

        # Build training data: 20 users, ~3 orders each, ~3 items each order.
        n_users = 20
        users_df = pd.DataFrame({"id": range(n_users), "age": np.random.randint(18, 60, n_users)})

        order_rows = []
        for uid in range(n_users):
            n_orders = np.random.randint(2, 5)
            for _ in range(n_orders):
                order_rows.append({"id": len(order_rows), "user_id": uid, "amount": np.random.uniform(10, 100)})
        orders_df = pd.DataFrame(order_rows)

        item_rows = []
        for _, order in orders_df.iterrows():
            n_items = np.random.randint(2, 5)
            for _ in range(n_items):
                item_rows.append({"id": len(item_rows), "order_id": order["id"], "price": np.random.uniform(1, 50)})
        items_df = pd.DataFrame(item_rows)

        meta = Metadata()
        meta.add_table("users", pk="id")
        meta.add_table("orders", pk="id", fk={"user_id": "users.id"})
        meta.add_table("items", pk="id", fk={"order_id": "orders.id"})

        training_data = {"users": users_df, "orders": orders_df, "items": items_df}
        mock_io = self._make_mock_io(training_data, self.temp_dir)

        orch = StagedOrchestrator(metadata=meta, io=mock_io)
        real_data_paths = {
            "users": "path/users",
            "orders": "path/orders",
            "items": "path/items",
        }
        orch.fit_all(real_data_paths, epochs=2, batch_size=20)
        orch.generate({"users": 10}, output_path_base=self.temp_dir)

        # Load generated tables from disk.
        generated_users = pd.read_csv(os.path.join(self.temp_dir, "users", "data.csv"))
        generated_orders = pd.read_csv(os.path.join(self.temp_dir, "orders", "data.csv"))
        generated_items = pd.read_csv(os.path.join(self.temp_dir, "items", "data.csv"))

        # Zero-orphan check: orders → users.
        merged_orders = generated_orders.merge(generated_users, left_on="user_id", right_on="id", how="inner")
        self.assertEqual(
            len(merged_orders), len(generated_orders),
            f"Orphaned orders: {len(generated_orders) - len(merged_orders)}"
        )

        # Zero-orphan check: items → orders.
        merged_items = generated_items.merge(generated_orders, left_on="order_id", right_on="id", how="inner")
        self.assertEqual(
            len(merged_items), len(generated_items),
            f"Orphaned items: {len(generated_items) - len(merged_items)}"
        )

    def test_4_table_chain_zero_orphans(self):
        """4-table chain users → orders → items → reviews produces zero orphans at all join levels."""
        np.random.seed(0)

        n_users = 10
        users_df = pd.DataFrame({"id": range(n_users), "age": np.random.randint(18, 60, n_users)})

        order_rows = []
        for uid in range(n_users):
            n_orders = np.random.randint(2, 5)
            for _ in range(n_orders):
                order_rows.append({"id": len(order_rows), "user_id": uid, "amount": np.random.uniform(10, 100)})
        orders_df = pd.DataFrame(order_rows)

        item_rows = []
        for _, order in orders_df.iterrows():
            n_items = np.random.randint(2, 5)
            for _ in range(n_items):
                item_rows.append({"id": len(item_rows), "order_id": order["id"], "price": np.random.uniform(1, 50)})
        items_df = pd.DataFrame(item_rows)

        review_rows = []
        for _, item in items_df.iterrows():
            n_reviews = np.random.randint(2, 5)
            for _ in range(n_reviews):
                review_rows.append({"id": len(review_rows), "item_id": item["id"], "rating": np.random.randint(1, 6)})
        reviews_df = pd.DataFrame(review_rows)

        meta = Metadata()
        meta.add_table("users", pk="id")
        meta.add_table("orders", pk="id", fk={"user_id": "users.id"})
        meta.add_table("items", pk="id", fk={"order_id": "orders.id"})
        meta.add_table("reviews", pk="id", fk={"item_id": "items.id"})

        training_data = {
            "users": users_df,
            "orders": orders_df,
            "items": items_df,
            "reviews": reviews_df,
        }
        mock_io = self._make_mock_io(training_data, self.temp_dir)

        orch = StagedOrchestrator(metadata=meta, io=mock_io)
        real_data_paths = {
            "users": "path/users",
            "orders": "path/orders",
            "items": "path/items",
            "reviews": "path/reviews",
        }
        orch.fit_all(real_data_paths, epochs=2, batch_size=20)
        orch.generate({"users": 5}, output_path_base=self.temp_dir)

        generated_users = pd.read_csv(os.path.join(self.temp_dir, "users", "data.csv"))
        generated_orders = pd.read_csv(os.path.join(self.temp_dir, "orders", "data.csv"))
        generated_items = pd.read_csv(os.path.join(self.temp_dir, "items", "data.csv"))
        generated_reviews = pd.read_csv(os.path.join(self.temp_dir, "reviews", "data.csv"))

        # Level 1: orders → users.
        merged_orders = generated_orders.merge(generated_users, left_on="user_id", right_on="id", how="inner")
        self.assertEqual(
            len(merged_orders), len(generated_orders),
            f"Orphaned orders: {len(generated_orders) - len(merged_orders)}"
        )

        # Level 2: items → orders.
        merged_items = generated_items.merge(generated_orders, left_on="order_id", right_on="id", how="inner")
        self.assertEqual(
            len(merged_items), len(generated_items),
            f"Orphaned items: {len(generated_items) - len(merged_items)}"
        )

        # Level 3: reviews → items.
        merged_reviews = generated_reviews.merge(generated_items, left_on="item_id", right_on="id", how="inner")
        self.assertEqual(
            len(merged_reviews), len(generated_reviews),
            f"Orphaned reviews: {len(generated_reviews) - len(merged_reviews)}"
        )

    def test_fk_type_mismatch_raises_schema_validation_error(self):
        """FK type mismatch (int PK vs str FK) raises SchemaValidationError."""
        from syntho_hive.exceptions import SchemaValidationError

        meta = Metadata()
        meta.add_table("users", pk="id")
        meta.add_table("orders", pk="order_id", fk={"user_id": "users.id"})

        parent_df = pd.DataFrame({"id": [1, 2, 3]})          # int PK
        child_df = pd.DataFrame({"order_id": [1, 2], "user_id": ["1", "2"]})  # str FK

        with self.assertRaises(SchemaValidationError) as ctx:
            meta.validate_schema(real_data={"users": parent_df, "orders": child_df})

        msg = str(ctx.exception).lower()
        self.assertTrue(
            "mismatch" in msg or "type" in msg,
            f"Expected 'mismatch' or 'type' in error message, got: {msg!r}"
        )

    def test_fk_missing_column_raises_schema_validation_error(self):
        """Missing FK column in child table raises SchemaValidationError."""
        from syntho_hive.exceptions import SchemaValidationError

        meta = Metadata()
        meta.add_table("users", pk="id")
        meta.add_table("orders", pk="order_id", fk={"user_id": "users.id"})

        parent_df = pd.DataFrame({"id": [1, 2]})
        child_df = pd.DataFrame({"order_id": [1, 2]})  # no user_id column

        with self.assertRaises(SchemaValidationError) as ctx:
            meta.validate_schema(real_data={"users": parent_df, "orders": child_df})

        msg = str(ctx.exception).lower()
        self.assertTrue(
            "missing" in msg or "user_id" in msg,
            f"Expected 'missing' or 'user_id' in error message, got: {msg!r}"
        )

    def test_cardinality_within_tolerance(self):
        """Empirical LinkageModel produces child counts within 20% of training mean."""
        np.random.seed(7)

        # 50 parents, each with exactly 3 children → observed mean = 3.0
        parent_df = pd.DataFrame({"id": range(50)})
        child_df = pd.DataFrame({
            "parent_id": np.repeat(range(50), 3),
            "id": range(150),
        })

        model = LinkageModel(method="empirical")
        model.fit(parent_df, child_df, fk_col="parent_id", pk_col="id")

        # Generate counts for 50 parents — sampled mean should be near 3.0.
        sampled = model.sample_counts(parent_df)
        sampled_mean = float(np.mean(sampled))
        rel_error = abs(sampled_mean - 3.0) / 3.0

        self.assertLess(
            rel_error,
            0.20,
            f"Cardinality drift: {sampled_mean:.2f} vs expected 3.0 (rel error {rel_error:.2%})"
        )


if __name__ == '__main__':
    unittest.main()
