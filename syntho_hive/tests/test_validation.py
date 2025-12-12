import unittest
import pandas as pd
import numpy as np
import os
import shutil
from syntho_hive.validation.statistical import StatisticalValidator
from syntho_hive.validation.report_generator import ValidationReport

class TestValidation(unittest.TestCase):
    def setUp(self):
        self.validator = StatisticalValidator()
        self.report_gen = ValidationReport()
        self.test_dir = "test_output"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            # shutil.rmtree(self.test_dir)
            print(f"\nTest output persisted at: {os.path.abspath(self.test_dir)}")

    def test_perfect_match(self):
        df = pd.DataFrame({"A": np.random.randn(100), "B": np.random.choice(["x", "y"], 100)})
        results = self.validator.compare_columns(df, df)
        
        self.assertTrue(results["A"]["passed"])
        self.assertTrue(results["B"]["passed"])

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        results = self.validator.compare_columns(df, df)
        self.assertIn("error", results)

    def test_type_mismatch(self):
        real_df = pd.DataFrame({"A": [1, 2, 3]})
        synth_df = pd.DataFrame({"A": ["a", "b", "c"]})
        results = self.validator.compare_columns(real_df, synth_df)
        self.assertIn("error", results["A"])

    def test_html_report_generation(self):
        real_data = {"table1": pd.DataFrame({"A": np.random.randn(100)})}
        synth_data = {"table1": pd.DataFrame({"A": np.random.randn(100)})}
        
        output_path = os.path.join(self.test_dir, "report.html")
        self.report_gen.generate(real_data, synth_data, output_path)
        
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r") as f:
            content = f.read()
            self.assertIn("<html>", content)
            self.assertIn("Validation Report", content)
            self.assertIn("table1", content)

if __name__ == "__main__":
    unittest.main()
