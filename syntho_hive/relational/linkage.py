import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from sklearn.mixture import GaussianMixture

class LinkageModel:
    """Model cardinality relationships between parent and child tables."""

    def __init__(self, method: str = "gmm"):
        """Create a linkage model.

        Args:
            method: Distribution family used to model child counts.
        """
        self.method = method
        self.model = None
        self.max_children = 0
        
    def fit(self, parent_df: pd.DataFrame, child_df: pd.DataFrame, fk_col: str, pk_col: str = "id"):
        """Fit the distribution of child counts per parent.

        Args:
            parent_df: Parent table with unique primary keys.
            child_df: Child table containing foreign keys to parents.
            fk_col: Name of the foreign key column in the child table.
            pk_col: Name of the primary key column in the parent table.
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
        """Sample child counts for a set of parents.

        Args:
            parent_context: Parent dataframe (only length is used here).

        Returns:
            Numpy array of integer child counts aligned with parents.

        Raises:
            ValueError: If called before fitting the model.
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
