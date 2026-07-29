from typing import Dict, Any

import pandas as pd
import numpy as np
import structlog
from scipy.stats import ks_2samp

log = structlog.get_logger(__name__)

#: Key under which table-level (non per-column) status is reported by
#: ``StatisticalValidator.compare_columns``.
TABLE_STATUS_KEY = "__status__"


class StatisticalValidator:
    """Perform statistical checks between real and synthetic data."""

    #: KS statistic (effect size) below which a numeric column passes.
    KS_STATISTIC_THRESHOLD = 0.1
    #: Total variation distance below which a categorical column passes.
    TVD_THRESHOLD = 0.1
    #: Numeric columns with nunique at or below this are treated as discrete
    #: and validated with TVD instead of the KS test.
    LOW_CARDINALITY_THRESHOLD = 20

    def compare_columns(
        self, real_df: pd.DataFrame, synth_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compare column-wise distributions between real and synthetic data.

        Numeric columns with more than ``LOW_CARDINALITY_THRESHOLD`` distinct
        values use the two-sample KS test with an effect-size pass criterion
        (statistic ``D < KS_STATISTIC_THRESHOLD``); boolean, categorical and
        low-cardinality numeric columns use total variation distance.

        Args:
            real_df: Real dataframe.
            synth_df: Synthetic dataframe aligned to the same schema.

        Returns:
            Mapping of column name to a result dict. Every value is a dict.
            Successful results always contain a ``"statistic"`` key (relied on
            by checkpoint validation); failures contain an ``"error"`` key.
            Table-level failures (e.g. empty input) are reported under
            ``TABLE_STATUS_KEY`` (with an ``"error"`` alias entry kept for
            backward compatibility).
        """
        results: Dict[str, Any] = {}

        if real_df.empty or synth_df.empty:
            status = {"error": "One or both DataFrames are empty."}
            log.warning(
                "compare_columns_empty_input",
                real_empty=bool(real_df.empty),
                synth_empty=bool(synth_df.empty),
            )
            # "error" is a backward-compatible alias of the typed status entry;
            # both values are dicts, matching the per-column contract.
            return {TABLE_STATUS_KEY: status, "error": status}

        for col in real_df.columns:
            if col not in synth_df.columns:
                results[col] = {"error": "Column missing in synthetic data"}
                continue

            real_data = real_df[col].dropna()
            synth_data = synth_df[col].dropna()

            if real_data.empty or synth_data.empty:
                results[col] = {"error": "Column data is empty after dropping NaNs"}
                continue

            # Compare dtype families rather than strict dtypes so e.g.
            # category-of-strings vs object-strings remain comparable.
            real_family = self._dtype_family(real_data)
            synth_family = self._dtype_family(synth_data)
            families = {real_family, synth_family}

            if real_family != synth_family and families != {"bool", "numeric"}:
                results[col] = {
                    "error": (
                        f"Type mismatch: Real {real_family} ({real_df[col].dtype}) "
                        f"vs Synth {synth_family} ({synth_df[col].dtype})"
                    )
                }
                continue

            use_ks = (
                families == {"numeric"}
                and real_data.nunique() > self.LOW_CARDINALITY_THRESHOLD
            )

            if use_ks:
                # KS test, judged on effect size (statistic D), not p-value:
                # with large samples the p-value rejects on trivially small
                # differences, and with small samples it passes poor matches.
                try:
                    stat, p_value = ks_2samp(
                        np.asarray(real_data, dtype=float),
                        np.asarray(synth_data, dtype=float),
                    )
                    results[col] = {
                        "test": "ks_test",
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "passed": bool(stat < self.KS_STATISTIC_THRESHOLD),
                    }
                except Exception as e:
                    results[col] = {"error": f"KS Test failed: {str(e)}"}
            else:
                # TVD (Total Variation Distance) for categorical, boolean and
                # low-cardinality numeric columns.
                try:
                    real_counts = real_data.value_counts(normalize=True)
                    synth_counts = synth_data.value_counts(normalize=True)

                    # Align categories
                    all_cats = set(real_counts.index).union(set(synth_counts.index))

                    tvd = 0.5 * sum(
                        abs(real_counts.get(c, 0) - synth_counts.get(c, 0))
                        for c in all_cats
                    )

                    results[col] = {
                        "test": "tvd",
                        "statistic": float(tvd),
                        "passed": bool(tvd < self.TVD_THRESHOLD),
                    }
                except Exception as e:
                    results[col] = {"error": f"TVD Checks failed: {str(e)}"}

        return results

    @staticmethod
    def _dtype_family(series: pd.Series) -> str:
        """Classify a series into a coarse dtype family.

        Args:
            series: Series to classify (typically already NaN-dropped).

        Returns:
            One of ``"bool"``, ``"numeric"``, ``"datetime"``, ``"timedelta"``
            or ``"categorical"``.
        """
        if isinstance(series.dtype, pd.CategoricalDtype):
            inferred = pd.api.types.infer_dtype(series.cat.categories, skipna=True)
        else:
            inferred = pd.api.types.infer_dtype(series, skipna=True)

        if inferred == "boolean":
            return "bool"
        if inferred in (
            "integer",
            "floating",
            "mixed-integer-float",
            "decimal",
            "complex",
        ):
            return "numeric"
        if inferred in ("datetime", "datetime64", "date"):
            return "datetime"
        if inferred in ("timedelta", "timedelta64"):
            return "timedelta"
        return "categorical"

    def check_correlations(
        self, real_df: pd.DataFrame, synth_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compare correlation matrices using the Frobenius norm.

        Only numeric columns present in both frames are compared. Cells that
        are undefined (NaN) in either correlation matrix — e.g. constant
        columns — are excluded from the norm instead of being coerced to 0.

        Args:
            real_df: Real dataframe.
            synth_df: Synthetic dataframe.

        Returns:
            Dict with:
                - ``frobenius_distance``: norm over cells defined in both
                  matrices (0.0 when identical or fewer than two shared
                  numeric columns).
                - ``columns_compared``: number of shared numeric columns.
                - ``undefined_pairs``: number of matrix cells that were NaN in
                  either correlation matrix and excluded from the norm.
        """
        real_num = real_df.select_dtypes(include=[np.number])
        synth_num = synth_df.select_dtypes(include=[np.number])

        common_cols = sorted(set(real_num.columns) & set(synth_num.columns))
        result: Dict[str, Any] = {
            "frobenius_distance": 0.0,
            "columns_compared": len(common_cols),
            "undefined_pairs": 0,
        }

        if len(common_cols) < 2:
            return result

        real_corr = real_num[common_cols].corr()
        synth_corr = synth_num[common_cols].corr()

        defined = real_corr.notna().values & synth_corr.notna().values
        diff = np.where(defined, (real_corr - synth_corr).values, 0.0)

        result["frobenius_distance"] = float(np.linalg.norm(diff))
        result["undefined_pairs"] = int(defined.size - defined.sum())
        return result
