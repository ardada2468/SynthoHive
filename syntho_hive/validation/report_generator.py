from typing import Dict, Any, List
import pandas as pd
import json
import numpy as np
from .statistical import StatisticalValidator

class ValidationReport:
    """Generate summary reports of validation metrics."""

    def __init__(self):
        """Initialize statistical validator and metric store."""
        self.validator = StatisticalValidator()
        self.metrics = {}
        
    def _calculate_detailed_stats(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, Any]:
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
            
            for name, df, res in [("real", real_df, col_stats["real"]), ("synth", synth_df, col_stats["synth"])]:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    res["mean"] = series.mean()
                    res["std"] = series.std()
                    res["min"] = series.min()
                    res["max"] = series.max()
                else:
                    res["unique_count"] = series.nunique()
                    res["top_value"] = series.mode().iloc[0] if not series.mode().empty else "N/A"
                    res["top_freq"] = series.value_counts().iloc[0] if not series.empty else 0
            
            stats[col] = col_stats
        return stats

    def generate(self, real_data: Dict[str, pd.DataFrame], synth_data: Dict[str, pd.DataFrame], output_path: str):
        """Run validation and save a report.

        Args:
            real_data: Mapping of table name to real dataframe.
            synth_data: Mapping of table name to synthetic dataframe.
            output_path: Destination path for HTML or JSON report.
        """
        report = {
            "tables": {},
            "summary": "Validation Report"
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
            preview = {
                "real_html": real_df.head(10).to_html(index=False, classes='scroll-table', border=0),
                "synth_html": synth_df.head(10).to_html(index=False, classes='scroll-table', border=0)
            }
            
            report["tables"][table_name] = {
                "column_metrics": col_metrics,
                "correlation_distance": corr_diff,
                "detailed_stats": stats,
                "preview": preview
            }
            
        if output_path.endswith(".html"):
            self._save_html(report, output_path)
        else:
            # Save to JSON for now (PDF requires more deps)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        
        import os
        print(f"Report saved to {os.path.abspath(output_path)}")

    def _save_html(self, report: Dict[str, Any], output_path: str):
        """Render a rich HTML report with metric explanations, stats, and previews.

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
                            <em>Result:</em> Returns a p-value. If p > 0.05, we fail to reject the null hypothesis (i.e., distributions are likely the same).</li>
                        <li><strong>TVD (Total Variation Distance):</strong> Used for categorical or discrete columns. Measures the maximum difference between probabilities assigned to the same event by two distributions. <br>
                            <em>Result:</em> Value between 0 and 1. Lower is better (0 means identical). We consider < 0.1 as passing.</li>
                        <li><strong>Correlation Distance:</strong> Measures how well the pairwise correlations between numerical columns are preserved. Calculated as the Frobenius norm of the difference between correlation matrices. <br>
                            <em>Result:</em> Lower is better (0 means identical correlation structure).</li>
                    </ul>
                </div>
            """]

        for table_name, data in report["tables"].items():
            html_content.append(f"<div class='section'><h2>Table: {table_name}</h2>")
            
            # --- 1. Correlation & Overall ---
            corr_dist = data.get('correlation_distance', 0.0)
            html_content.append(f"<p><strong>Correlation Distance:</strong> {corr_dist:.4f}</p>")

            # --- 2. Column Metrics ---
            html_content.append("<h3>Column Validation Metrics</h3>")
            html_content.append("<table><tr><th>Column</th><th>Test Type</th><th>Statistic</th><th>P-Value / Score</th><th>Status</th></tr>")
            
            for col, metrics in data["column_metrics"].items():
                if "error" in metrics:
                    html_content.append(f"<tr><td>{col}</td><td colspan='4' class='fail'>Error: {metrics['error']}</td></tr>")
                    continue

                status = "PASS" if metrics.get("passed", False) else "FAIL"
                cls = "pass" if status == "PASS" else "fail"
                
                stat = f"{metrics.get('statistic', 0):.4f}"
                # TVD doesn't have a p-value, KS does.
                pval = f"{metrics.get('p_value', 0):.4f}" if metrics.get('p_value') is not None else "N/A"
                test_name = metrics.get('test', 'N/A')
                
                html_content.append(f"<tr><td>{col}</td><td>{test_name}</td><td>{stat}</td><td>{pval}</td><td class='{cls}'>{status}</td></tr>")
            
            html_content.append("</table>")
            
            # --- 3. Detailed Statistics ---
            if "detailed_stats" in data:
                html_content.append("<h3>Detailed Statistics (Real vs Synthetic)</h3>")
                html_content.append("<table><tr><th>Column</th><th>Metric</th><th>Real</th><th>Synthetic</th></tr>")
                
                for col, stats in data["detailed_stats"].items():
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
                        
                        # Format floats
                        if isinstance(r_val, (float, np.floating)): r_val = f"{r_val:.4f}"
                        if isinstance(s_val, (float, np.floating)): s_val = f"{s_val:.4f}"
                        
                        row_start = f"<tr><td rowspan='{len(all_keys)}'>{col}</td>" if first else "<tr>"
                        row_end = f"<td>{k}</td><td>{r_val}</td><td>{s_val}</td></tr>"
                        html_content.append(row_start + row_end)
                        first = False
                html_content.append("</table>")

            # --- 4. Data Preview ---
            if "preview" in data:
                html_content.append("<h3>Data Preview (First 10 Rows)</h3>")
                html_content.append("<div class='row'>")
                
                # Real
                html_content.append("<div class='col'>")
                html_content.append("<div class='preview-header'>Original Data (Real)</div>")
                html_content.append(data["preview"]["real_html"])
                html_content.append("</div>")
                
                # Synth
                html_content.append("<div class='col'>")
                html_content.append("<div class='preview-header'>Synthetic Data (Generated)</div>")
                html_content.append(data["preview"]["synth_html"])
                html_content.append("</div>")
                
                html_content.append("</div>") # End row

            html_content.append("</div>") # End section
            
        html_content.append("</div></body></html>")
        
        with open(output_path, "w") as f:
            f.write("\n".join(html_content))
