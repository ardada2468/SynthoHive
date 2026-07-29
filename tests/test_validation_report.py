import numpy as np
import pandas as pd
import pytest

from syntho_hive.validation.report_generator import ValidationReport
from syntho_hive.validation.statistical import StatisticalValidator


@pytest.fixture
def validator():
    return StatisticalValidator()


@pytest.fixture
def report_gen():
    return ValidationReport()


def test_perfect_match(validator):
    df = pd.DataFrame({"A": np.random.randn(100), "B": np.random.choice(["x", "y"], 100)})
    results = validator.compare_columns(df, df)

    assert results["A"]["passed"]
    assert results["B"]["passed"]


def test_empty_dataframe(validator):
    df = pd.DataFrame()
    results = validator.compare_columns(df, df)
    assert "error" in results


def test_type_mismatch(validator):
    real_df = pd.DataFrame({"A": [1, 2, 3]})
    synth_df = pd.DataFrame({"A": ["a", "b", "c"]})
    results = validator.compare_columns(real_df, synth_df)
    assert "error" in results["A"]


def test_html_report_generation(report_gen, tmp_path):
    real_data = {"table1": pd.DataFrame({"A": np.random.randn(100)})}
    synth_data = {"table1": pd.DataFrame({"A": np.random.randn(100)})}

    output_path = tmp_path / "report.html"
    report_gen.generate(real_data, synth_data, str(output_path))

    assert output_path.exists()
    content = output_path.read_text()
    assert "<html>" in content
    assert "Validation Report" in content
    assert "table1" in content
