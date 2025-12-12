
import unittest
import pandas as pd
import numpy as np
import torch
import shutil
import os
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata
from syntho_hive.core.data.transformer import DataTransformer

class TestEntityEmbeddings(unittest.TestCase):
    def setUp(self):
        self.metadata = Metadata()
        # High cardinality column
        self.metadata.add_table("test_table", pk="id", pii_cols=[])
        
        # Create data with high cardinality column 'category'
        self.data = pd.DataFrame({
            "id": range(100),
            "category": [f"cat_{i}" for i in range(100)], # 100 unique values
            "value": np.random.randn(100)
        })
        
    def test_transformer_embedding_logic(self):
        # Threshold 10 -> 'category' (100) shld be embedding
        transformer = DataTransformer(self.metadata, embedding_threshold=10)
        transformer.fit(self.data)
        
        info = transformer._column_info['category']
        self.assertEqual(info['type'], 'categorical_embedding')
        self.assertEqual(info['num_categories'], 100)
        self.assertEqual(info['dim'], 1)
        
        # Check Transform
        transformed = transformer.transform(self.data)
        # Output: [id(cont), category(1), value(cont)]
        # Actually ID might be excluded if logic in transformer excludes PK?
        # Transform logic: "columns_to_transform = data.columns.tolist()"
        # Unless table_name passed to fit.
        # Let's pass table_name='test_table' to fit to trigger PK exclusion if any.
        # But PK exclusion relies on metadata get_table.
        
        # Let's just check the shape.
        # category -> 1 dim
        # value -> 11 dims (10 comps + 1 scalar)
        # id -> 11 dims (treated as numerical continuous)
        
        # total = 1 + 11 + 11 = 23?
        self.assertTrue(transformed.shape[1] > 20)
        
        # Check Inverse
        reconstructed = transformer.inverse_transform(transformed)
        self.assertTrue("category" in reconstructed.columns)
        self.assertEqual(len(reconstructed), 100)
        # Values should be close (strings should be exact)
        self.assertEqual(reconstructed['category'].iloc[0], "cat_0")

    def test_ctgan_embedding_flow(self):
        # Threshold 10 -> Force embedding
        model = CTGAN(
            self.metadata, 
            embedding_threshold=10, 
            epochs=1, 
            batch_size=10
        )
        
        # Fit
        # Note: fit calls transformer.fit which needs table_name to exclude PK?
        # If we don't pass table_name, it transforms PK as continuous. That's fine for this test.
        model.fit(self.data)
        
        # Check if Embedding Layers created
        self.assertTrue('category' in model.embedding_layers)
        layer = model.embedding_layers['category']
        self.assertIsInstance(layer, torch.nn.Module)
        # Dim heuristic: min(50, (100+1)//2) = 50
        self.assertEqual(layer.embedding.embedding_dim, 50)
        
        # Sample
        samples = model.sample(10)
        self.assertEqual(len(samples), 10)
        self.assertTrue('category' in samples.columns)
        self.assertTrue(isinstance(samples['category'].iloc[0], str))
        
    def test_ctgan_no_embedding_flow(self):
        # Threshold 200 -> No embedding (OHE)
        model = CTGAN(
            self.metadata, 
            embedding_threshold=200, 
            epochs=1, 
            batch_size=10
        )
        model.fit(self.data)
        
        # Check NO Embedding Layers
        self.assertFalse('category' in model.embedding_layers)
        
        samples = model.sample(10)
        self.assertEqual(len(samples), 10)

if __name__ == '__main__':
    unittest.main()
