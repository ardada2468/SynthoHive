import json

import numpy as np
import pandas as pd
import pytest

from syntho_hive.validation.report_generator import ValidationReport


@pytest.fixture
def report_gen():
    return ValidationReport()


XSS = "<script>alert(1)</script>"


def test_html_report_escapes_data_derived_strings(report_gen, tmp_path):
    """Data-derived strings (column names, values) must be HTML-escaped."""
    real_df = pd.DataFrame(
        {
            XSS: ["<img src=x onerror=alert(1)>", "<b>bold</b>", "plain"] * 10,
            "num": np.arange(30, dtype=float),
        }
    )
    synth_df = real_df.copy()

    output_path = tmp_path / "report.html"
    report_gen.generate({XSS: real_df}, {XSS: synth_df}, str(output_path))

    content = output_path.read_text()
    assert XSS not in content
    assert "<img src=x onerror=alert(1)>" not in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content


def test_html_report_escapes_error_messages(report_gen, tmp_path):
    """Type-mismatch error rows carry dtype text and must be escaped safely."""
    real_df = pd.DataFrame({"<svg onload=alert(2)>": [1, 2, 3]})
    synth_df = pd.DataFrame({"<svg onload=alert(2)>": ["a", "b", "c"]})

    output_path = tmp_path / "report.html"
    report_gen.generate({"t": real_df}, {"t": synth_df}, str(output_path))

    content = output_path.read_text()
    assert "<svg onload=alert(2)>" not in content
    assert "&lt;svg onload=alert(2)&gt;" in content


def test_empty_synth_table_does_not_crash(report_gen, tmp_path):
    """An empty table must produce a table-level error, not an AttributeError."""
    real_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    synth_df = pd.DataFrame({"a": pd.Series([], dtype=float)})

    output_path = tmp_path / "report.html"
    report_gen.generate({"t": real_df}, {"t": synth_df}, str(output_path))

    content = output_path.read_text()
    assert "Validation error" in content
    assert "empty" in content.lower()


def test_all_nan_non_numeric_column_does_not_crash(report_gen, tmp_path):
    """A non-empty, all-NaN object column previously raised IndexError."""
    real_df = pd.DataFrame({"c": pd.Series([None, None, None], dtype=object)})
    synth_df = pd.DataFrame({"c": pd.Series(["x", "y", "z"], dtype=object)})

    output_path = tmp_path / "report.html"
    report_gen.generate({"t": real_df}, {"t": synth_df}, str(output_path))

    content = output_path.read_text()
    assert "N/A" in content  # guarded top_value


def test_json_report_native_types_and_no_nan_tokens(report_gen, tmp_path):
    real_df = pd.DataFrame(
        {
            "num": np.random.default_rng(0).normal(size=50),
            "all_nan": np.full(50, np.nan),  # numeric all-NaN -> NaN mean
        }
    )
    synth_df = real_df.copy()

    output_path = tmp_path / "report.json"
    report_gen.generate({"t": real_df}, {"t": synth_df}, str(output_path))

    text = output_path.read_text()

    # Strict JSON: bare NaN/Infinity tokens are rejected by this parser.
    def _reject_constant(value):
        raise AssertionError(f"bare {value} token found in JSON report")

    parsed = json.loads(text, parse_constant=_reject_constant)
    assert '"True"' not in text  # numpy bools must not be stringified

    passed = parsed["tables"]["t"]["column_metrics"]["num"]["passed"]
    assert passed is True  # native JSON bool


def test_missing_and_extra_tables_recorded(report_gen, tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    real_data = {"t1": df, "t2": df}
    synth_data = {"t1": df.copy(), "t3": df.copy()}

    output_path = tmp_path / "report.json"
    report_gen.generate(real_data, synth_data, str(output_path))

    parsed = json.loads(output_path.read_text())
    summary = parsed["summary"]
    assert summary["tables_missing_in_synth"] == ["t2"]
    assert summary["tables_extra_in_synth"] == ["t3"]
    assert summary["tables_validated"] == ["t1"]
    assert "t2" not in parsed["tables"]

    # The HTML rendering surfaces the coverage warnings too.
    html_path = tmp_path / "report.html"
    report_gen.generate(real_data, synth_data, str(html_path))
    content = html_path.read_text()
    assert "Coverage Warnings" in content
    assert "t2" in content
