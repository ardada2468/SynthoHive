from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare

class StatisticalValidator:
    """
    Performs statistical checks between Real and Synthetic data.
    """
    
    def compare_columns(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare distributions column by column.
        """
        results = {}
        
        if real_df.empty or synth_df.empty:
            return {"error": "One or both DataFrames are empty."}

        for col in real_df.columns:
            if col not in synth_df.columns:
                results[col] = {"error": "Column missing in synthetic data"}
                continue
                
            real_data = real_df[col].dropna()
            synth_data = synth_df[col].dropna()
            
            if real_data.empty or synth_data.empty:
                results[col] = {"error": "Column data is empty after dropping NaNs"}
                continue

            # Check for type mismatch
            if real_data.dtype != synth_data.dtype:
                # Try to cast if compatible (e.g. float vs int)
                if pd.api.types.is_numeric_dtype(real_data) and pd.api.types.is_numeric_dtype(synth_data):
                    pass # Compatible enough for stats
                else:
                    results[col] = {"error": f"Type mismatch: Real {real_data.dtype} vs Synth {synth_data.dtype}"}
                    continue
            
            if pd.api.types.is_numeric_dtype(real_data):
                # KS Test
                try:
                    stat, p_value = ks_2samp(real_data, synth_data)
                    results[col] = {
                        "test": "ks_test",
                        "statistic": stat,
                        "p_value": p_value,
                        "passed": p_value > 0.05 # Null hypothesis: Same distribution
                    }
                except Exception as e:
                    results[col] = {"error": f"KS Test failed: {str(e)}"}
            else:
                # TVD (Total Variation Distance)
                try:
                    real_counts = real_data.value_counts(normalize=True)
                    synth_counts = synth_data.value_counts(normalize=True)
                    
                    # Align categories
                    all_cats = set(real_counts.index).union(set(synth_counts.index))
                    
                    tvd = 0.5 * sum(abs(real_counts.get(c, 0) - synth_counts.get(c, 0)) for c in all_cats)
                    
                    results[col] = {
                        "test": "tvd",
                        "statistic": tvd,
                        "passed": tvd < 0.1 # Threshold arbitrary
                    }
                except Exception as e:
                    results[col] = {"error": f"TVD Checks failed: {str(e)}"}
                
        return results

    def check_correlations(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """
        Compare correlation matrices (Frobenius norm).
        """
        # Numeric only
        real_corr = real_df.select_dtypes(include=[np.number]).corr().fillna(0)
        synth_corr = synth_df.select_dtypes(include=[np.number]).corr().fillna(0)
        
        if real_corr.empty or synth_corr.empty:
            return 0.0
            
        diff = real_corr - synth_corr
        frobenius_norm = np.linalg.norm(diff.values)
        
        return float(frobenius_norm)
