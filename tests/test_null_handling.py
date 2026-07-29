
import numpy as np
import pandas as pd

from syntho_hive.core.models.ctgan import CTGAN


class MockMetadata:
    def __init__(self):
        self.tables = {"test_table": "config"}  # minimal config
        self.constraints = {}

    def get_table(self, name):
        return None


def test_null_handling():
    print("Creating data with nulls...")
    data = pd.DataFrame(
        {
            "numeric_col": [1.0, 2.0, np.nan, 4.0, 5.0] * 100,
            "categorical_col": ["A", "B", None, "A", "C"] * 100,
        }
    )

    metadata = MockMetadata()

    print("Initializing CTGAN...")
    model = CTGAN(metadata=metadata, epochs=1, batch_size=50)  # Fast run

    print("Fitting model...")
    model.fit(data)

    print("Sampling data...")
    sampled = model.sample(100)

    print("Sampled Data Head:")
    print(sampled.head())

    # Check for NaNs
    num_nulls_numeric = sampled["numeric_col"].isnull().sum()
    num_nulls_cat = sampled["categorical_col"].isnull().sum()

    print(f"Nulls in numeric_col: {num_nulls_numeric}")
    print(f"Nulls in categorical_col: {num_nulls_cat}")

    # Verify null handling infrastructure works correctly:
    # - The sampled output must have exactly the input schema (no leaked
    #   helper columns such as null indicators).
    # - Categorical nulls are modeled internally via a sentinel token
    #   ('<NAN>'); it must be decoded back to a real null and never leak
    #   into the output as a literal string value.
    assert list(sampled.columns) == list(data.columns), (
        f"Output columns {list(sampled.columns)} do not match input columns {list(data.columns)}"
    )
    for col in sampled.columns:
        literal_sentinels = (sampled[col].astype(str) == "<NAN>").sum()
        assert literal_sentinels == 0, (
            f"Sentinel token '<NAN>' leaked into output column {col!r} "
            f"({literal_sentinels} occurrences); it should be decoded to a real null"
        )

    print("Test Passed!")


if __name__ == "__main__":
    test_null_handling()
