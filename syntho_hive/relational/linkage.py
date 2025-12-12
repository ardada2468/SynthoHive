import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from sklearn.mixture import GaussianMixture

class LinkageModel:
    """
    Models the cardinality relationship between Parent and Child tables.
    Learns 'How many children does this parent have?'
    """
    def __init__(self, method: str = "gmm"):
        self.method = method
        self.model = None
        self.max_children = 0
        
    def fit(self, parent_df: pd.DataFrame, child_df: pd.DataFrame, fk_col: str, pk_col: str = "id"):
        """
        Fit distribution of child counts per parent.
        """
        # 1. Aggregate child counts
        # Assumes parent_df has unique PKs
        counts = child_df[fk_col].value_counts()
        
        # Merge with all parents to include 0-count parents
        parent_ids = pd.DataFrame(parent_df[pk_col].unique(), columns=[pk_col])
        
        # Ensure Types Match (Cast Child FK to Parent PK type)
        try:
            target_type = parent_ids[pk_col].dtype
            if not np.issubdtype(counts.index.dtype, target_type):
                counts.index = counts.index.astype(target_type)
        except Exception as e:
            # Fallback to string if direct cast fails
            print(f"Warning: Could not cast FK to match PK type ({e}). Falling back to string.")
            parent_ids[pk_col] = parent_ids[pk_col].astype(str)
            counts.index = counts.index.astype(str)

        # Use merge to get counts, fillna(0) for parents with no children
        # Note: In real Spark env this aggregation happens differently
        count_df = parent_ids.merge(
            counts.rename("child_count"), 
            left_on=pk_col, 
            right_index=True, 
            how="left"
        ).fillna(0)
        
        X = count_df["child_count"].values.reshape(-1, 1)
        self.max_children = int(X.max())
        
        if self.method == "gmm":
            # Using GMM to learn continuous approximation of counts
            # Could also use NegativeBinomial or KDE
            self.model = GaussianMixture(n_components=min(5, len(np.unique(X))), random_state=42)
            self.model.fit(X)
            
    def sample_counts(self, parent_context: pd.DataFrame) -> np.ndarray:
        """
        Sample child counts for the given parents.
        Returns array of integers.
        """
        n_samples = len(parent_context)
        if self.model is None:
            raise ValueError("LinkageModel not fitted")
            
        # Sample from GMM
        counts, _ = self.model.sample(n_samples)
        
        # Post-process: Round to nearest int, clip to [0, max]
        counts = np.round(counts).flatten().astype(int)
        counts = np.clip(counts, 0, None) # Ensure non-negative
        
        return counts
