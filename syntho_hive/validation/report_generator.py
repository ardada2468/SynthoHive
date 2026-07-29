from typing import Dict, Any, Optional

import html
import json
import math
import os

import pandas as pd
import numpy as np
import structlog

from .statistical import StatisticalValidator, TABLE_STATUS_KEY

log = structlog.get_logger(__name__)


class ValidationReport:
    """Generate summary reports of validation metrics."""

    def __init__(self):
        """Initialize statistical validator and metric store."""
        self.validator = StatisticalValidator()
        self.metrics = {}

    def _calculate_detailed_stats(
        self, real_df: pd.DataFrame, synth_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate descriptive statistics for side-by-side comparison.

        Args:
            real_df: Real dataframe.
            synth_df: Synthetic dataframe aligned to the real columns.

        Returns:
            Nested dict of summary stats for each column.
        """
        stats = {}
        for col in real_df.columns:
            if col not in synth_df.columns:
                continue

            col_stats = {"real": {}, "synth": {}}

            for name, df, res in [
                ("real", real_df, col_stats["real"]),
                ("synth", synth_df, col_stats["synth"]),
            ]:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    res["mean"] = series.mean()
                    res["std"] = series.std()
                    res["min"] = series.min()
                    res["max"] = series.max()
                else:
                    res["unique_count"] = series.nunique()
                    # value_counts drops NaN, so an all-NaN column yields an
                    # empty result — guard both top_value and top_freq.
                    vc = series.value_counts()
                    res["top_value"] = str(vc.index[0]) if len(vc) else "N/A"
                    res["top_freq"] = int(vc.iloc[0]) if len(vc) else 0

            stats[col] = col_stats
        return stats

    def generate(
        self,
        real_data: Dict[str, pd.DataFrame],
        synth_data: Dict[str, pd.DataFrame],
        output_path: str,
    ):
        """Run validation and save a report.

        Args:
            real_data: Mapping of table name to real dataframe.
            synth_data: Mapping of table name to synthetic dataframe.
            output_path: Destination path for HTML or JSON report.
        """
        missing_in_synth = sorted(set(real_data) - set(synth_data))
        extra_in_synth = sorted(set(synth_data) - set(real_data))
        validated = sorted(set(real_data) & set(synth_data))

        if missing_in_synth:
            log.warning("tables_missing_in_synth", tables=missing_in_synth)
        if extra_in_synth:
            log.warning("tables_extra_in_synth", tables=extra_in_synth)

        report = {
            "tables": {},
            "summary": {
                "title": "Validation Report",
                "tables_validated": validated,
                "tables_missing_in_synth": missing_in_synth,
                "tables_extra_in_synth": extra_in_synth,
            },
        }

        for table_name, real_df in real_data.items():
            if table_name not in synth_data:
                continue

            synth_df = synth_data[table_name]

            # 1. Column comparisons
            col_metrics = self.validator.compare_columns(real_df, synth_df)

            # 2. Correlation
            corr_diff = self.validator.check_correlations(real_df, synth_df)

            # 3. Detailed Stats
            stats = self._calculate_detailed_stats(real_df, synth_df)

            # 4. Data Preview
            # Use Pandas to_html for easy formatting, strict constraints
            # (to_html escapes cell content by default).
            preview = {
                "real_html": real_df.head(10).to_html(
                    index=False, classes="scroll-table", border=0
                ),
                "synth_html": synth_df.head(10).to_html(
                    index=False, classes="scroll-table", border=0
                ),
            }

            report["tables"][table_name] = {
                "column_metrics": col_metrics,
                "correlation_distance": corr_diff,
                "detailed_stats": stats,
                "preview": preview,
            }

        if output_path.endswith(".html"):
            self._save_html(report, output_path)
        else:
            # Save to JSON for now (PDF requires more deps)
            with open(output_path, "w") as f:
                json.dump(self._to_jsonable(report), f, indent=2, allow_nan=False)

        log.info("report_saved", path=os.path.abspath(output_path))

    def _to_jsonable(self, obj: Any) -> Any:
        """Recursively convert report values to JSON-native types.

        numpy scalars become native ``bool``/``int``/``float``, NaN/Inf become
        ``None`` (instead of invalid ``NaN`` tokens), and anything unknown is
        stringified.

        Args:
            obj: Arbitrary report value.

        Returns:
            JSON-serializable equivalent of ``obj``.
        """
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            value = float(obj)
            return None if (math.isnan(value) or math.isinf(value)) else value
        if obj is None or isinstance(obj, str):
            return obj
        if isinstance(obj, np.ndarray):
            return [self._to_jsonable(v) for v in obj.tolist()]
        if obj is pd.NaT:
            return None
        return str(obj)

    @staticmethod
    def _table_level_error(col_metrics: Any) -> Optional[str]:
        """Extract a table-level error message from a column-metrics mapping.

        Args:
            col_metrics: Value produced by ``compare_columns``.

        Returns:
            Error message when the mapping carries a table-level status (or is
            not a mapping at all), otherwise ``None``.
        """
        if not isinstance(col_metrics, dict):
            return str(col_metrics)

        status = col_metrics.get(TABLE_STATUS_KEY)
        if isinstance(status, dict) and "error" in status:
            return str(status["error"])
        # Legacy flat shape: {"error": "message"}.
        if isinstance(col_metrics.get("error"), str):
            return col_metrics["error"]
        return None

    @staticmethod
    def _format_metric(value: Any, default: str = "N/A") -> str:
        """Format a metric value for HTML output, escaping non-numeric values."""
        if value is None:
            return default
        if isinstance(value, bool):
            return html.escape(str(value))
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):.4f}"
        return html.escape(str(value))

    def _save_html(self, report: Dict[str, Any], output_path: str):
        """Render a rich HTML report with metric explanations, stats, and previews.

        All data-derived strings (table/column names, error messages, metric
        values) are HTML-escaped; pandas ``to_html`` previews are already
        escaped by pandas.

        Args:
            report: Structured report dictionary produced by ``generate``.
            output_path: Filesystem path to write the HTML file.
        """
        html_content = [
            """<html>
            <head>
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }
                    h1, h2, h3 { color: #2c3e50; }
                    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }

                    /* Tables */
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; }
                    th, td { border: 1px solid #e1e4e8; padding: 10px; text-align: left; }
                    th { background-color: #f1f8ff; color: #0366d6; font-weight: 600; }
                    tr:nth-child(even) { background-color: #f8f9fa; }

                    /* Status Colors */
                    .pass { color: #28a745; font-weight: bold; }
                    .fail { color: #dc3545; font-weight: bold; }
                    .warn-box { background: #fff8e1; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #f0ad4e; }

                    /* Layout */
                    .section { margin-top: 40px; border-top: 1px solid #eee; padding-top: 20px; }
                    .metric-box { background: #f0f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #0366d6; }
                    .row { display: flex; gap: 20px; }
                    .col { flex: 1; overflow-x: auto; }

                    /* Tabs/Previews */
                    .preview-header { font-weight: bold; margin-bottom: 10px; color: #555; }
                    .scroll-table { max-height: 400px; overflow-y: auto; display: block; }
                </style>
            </head>
            <body>
            <div class="container">
                <h1>Validation Report</h1>

                <div class="metric-box">
                    <h3>Metric Explanations</h3>
                    <ul>
                        <li><strong>KS Test (Kolmogorov-Smirnov):</strong> Used for continuous numerical columns. Compares the cumulative distribution functions of the real and synthetic data. <br>
                            <em>Result:</em> The KS statistic D measures the maximum CDF gap (effect size). We consider D &lt; 0.1 as passing; the p-value is reported for reference.</li>
                        <li><strong>TVD (Total Variation Distance):</strong> Used for categorical, boolean or discrete columns. Measures the maximum difference between probabilities assigned to the same event by two distributions. <br>
                            <em>Result:</em> Value between 0 and 1. Lower is better (0 means identical). We consider &lt; 0.1 as passing.</li>
                        <li><strong>Correlation Distance:</strong> Measures how well the pairwise correlations between numerical columns are preserved. Calculated as the Frobenius norm of the difference between correlation matrices (over cells defined in both). <br>
                            <em>Result:</em> Lower is better (0 means identical correlation structure).</li>
                    </ul>
                </div>
            """
        ]

        summary = report.get("summary", {})
        if isinstance(summary, dict):
            missing = summary.get("tables_missing_in_synth") or []
            extra = summary.get("tables_extra_in_synth") or []
            if missing or extra:
                html_content.append("<div class='warn-box'><h3>Coverage Warnings</h3><ul>")
                if missing:
                    names = ", ".join(html.escape(str(t)) for t in missing)
                    html_content.append(
                        f"<li>Tables missing from synthetic data (not validated): {names}</li>"
                    )
                if extra:
                    names = ", ".join(html.escape(str(t)) for t in extra)
                    html_content.append(
                        f"<li>Tables present only in synthetic data (not validated): {names}</li>"
                    )
                html_content.append("</ul></div>")

        for table_name, data in report["tables"].items():
            safe_table = html.escape(str(table_name))
            html_content.append(f"<div class='section'><h2>Table: {safe_table}</h2>")

            # --- 1. Correlation & Overall ---
            corr = data.get("correlation_distance", 0.0)
            if isinstance(corr, dict):
                corr_dist = corr.get("frobenius_distance", 0.0)
                undefined_pairs = corr.get("undefined_pairs", 0)
            else:  # Backward compatibility with the plain-float shape.
                corr_dist, undefined_pairs = corr, 0
            corr_text = (
                f"<p><strong>Correlation Distance:</strong> "
                f"{self._format_metric(corr_dist)}"
            )
            if undefined_pairs:
                corr_text += (
                    f" <em>({int(undefined_pairs)} undefined correlation "
                    f"pair(s) excluded)</em>"
                )
            html_content.append(corr_text + "</p>")

            # --- 2. Column Metrics ---
            html_content.append("<h3>Column Validation Metrics</h3>")

            col_metrics = data.get("column_metrics", {})
            table_error = self._table_level_error(col_metrics)
            if table_error is not None:
                html_content.append(
                    f"<p class='fail'>Validation error: {html.escape(table_error)}</p>"
                )
            else:
                html_content.append(
                    "<table><tr><th>Column</th><th>Test Type</th><th>Statistic</th>"
                    "<th>P-Value / Score</th><th>Status</th></tr>"
                )

                for col, metrics in col_metrics.items():
                    if col == TABLE_STATUS_KEY:
                        continue
                    safe_col = html.escape(str(col))

                    if not isinstance(metrics, dict):
                        html_content.append(
                            f"<tr><td>{safe_col}</td><td colspan='4' class='fail'>"
                            f"Error: {html.escape(str(metrics))}</td></tr>"
                        )
                        continue

                    if "error" in metrics:
                        safe_err = html.escape(str(metrics["error"]))
                        html_content.append(
                            f"<tr><td>{safe_col}</td><td colspan='4' class='fail'>"
                            f"Error: {safe_err}</td></tr>"
                        )
                        continue

                    status = "PASS" if metrics.get("passed", False) else "FAIL"
                    cls = "pass" if status == "PASS" else "fail"

                    stat = self._format_metric(metrics.get("statistic", 0))
                    # TVD doesn't have a p-value, KS does.
                    pval = self._format_metric(metrics.get("p_value"))
                    test_name = html.escape(str(metrics.get("test", "N/A")))

                    html_content.append(
                        f"<tr><td>{safe_col}</td><td>{test_name}</td><td>{stat}</td>"
                        f"<td>{pval}</td><td class='{cls}'>{status}</td></tr>"
                    )

                html_content.append("</table>")

            # --- 3. Detailed Statistics ---
            if "detailed_stats" in data:
                html_content.append("<h3>Detailed Statistics (Real vs Synthetic)</h3>")
                html_content.append(
                    "<table><tr><th>Column</th><th>Metric</th><th>Real</th>"
                    "<th>Synthetic</th></tr>"
                )

                for col, stats in data["detailed_stats"].items():
                    safe_col = html.escape(str(col))
                    # stats has "real": {...}, "synth": {...}
                    real_s = stats.get("real", {})
                    synth_s = stats.get("synth", {})

                    # Merge keys to show
                    all_keys = sorted(list(set(real_s.keys()) | set(synth_s.keys())))
                    # Usually we want mean, std, min, max or unique, top

                    first = True
                    for k in all_keys:
                        r_val = real_s.get(k, "-")
                        s_val = synth_s.get(k, "-")

                        # Format floats; escape everything else.
                        if isinstance(r_val, (float, np.floating)):
                            r_val = f"{r_val:.4f}"
                        else:
                            r_val = html.escape(str(r_val))
                        if isinstance(s_val, (float, np.floating)):
                            s_val = f"{s_val:.4f}"
                        else:
                            s_val = html.escape(str(s_val))

                        safe_key = html.escape(str(k))
                        row_start = (
                            f"<tr><td rowspan='{len(all_keys)}'>{safe_col}</td>"
                            if first
                            else "<tr>"
                        )
                        row_end = f"<td>{safe_key}</td><td>{r_val}</td><td>{s_val}</td></tr>"
                        html_content.append(row_start + row_end)
                        first = False
                html_content.append("</table>")

            # --- 4. Data Preview ---
            if "preview" in data:
                html_content.append("<h3>Data Preview (First 10 Rows)</h3>")
                html_content.append("<div class='row'>")

                # Real (pandas to_html output is already escaped)
                html_content.append("<div class='col'>")
                html_content.append("<div class='preview-header'>Original Data (Real)</div>")
                html_content.append(data["preview"]["real_html"])
                html_content.append("</div>")

                # Synth
                html_content.append("<div class='col'>")
                html_content.append("<div class='preview-header'>Synthetic Data (Generated)</div>")
                html_content.append(data["preview"]["synth_html"])
                html_content.append("</div>")

                html_content.append("</div>")  # End row

            html_content.append("</div>")  # End section

        html_content.append("</div></body></html>")

        with open(output_path, "w") as f:
            f.write("\n".join(html_content))
