import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()


class LinkageModel:
    """Model cardinality relationships between parent and child tables.

    Replaces the former GaussianMixture approach (which produced negative counts)
    with an empirical histogram resampler (default) or optional NegBinom fit.
    """

    def __init__(self, method: str = "empirical"):
        """Create a linkage model.

        Args:
            method: Cardinality distribution. 'empirical' (default) draws from observed
                    child counts. 'negbinom' fits scipy.stats.nbinom via method-of-moments.
                    Falls back to empirical if the data is not overdispersed (variance <= mean).
        """
        self.method = method
        self._observed_counts = None
        self._nbinom_n = None
        self._nbinom_p = None
        self.max_children = 0

    def fit(self, parent_df: pd.DataFrame, child_df: pd.DataFrame, fk_col: str, pk_col: str = "id"):
        """Fit the distribution of child counts per parent.

        Counts children per parent, including parents with zero children.

        Args:
            parent_df: Parent table with unique primary keys.
            child_df: Child table containing foreign keys to parents.
            fk_col: Name of the foreign key column in the child table.
            pk_col: Name of the primary key column in the parent table.
        """
        counts = child_df[fk_col].value_counts()
        parent_ids = pd.DataFrame(parent_df[pk_col].unique(), columns=[pk_col])
        count_df = parent_ids.merge(
            counts.rename("child_count"),
            left_on=pk_col, right_index=True, how="left"
        ).fillna(0)
        X = count_df["child_count"].to_numpy(dtype=int)
        self.max_children = int(X.max())
        self._observed_counts = X

        if self.method == "negbinom":
            mu = float(X.mean())
            var = float(X.var())
            if var > mu and mu > 0:
                p = mu / var
                n = mu * p / (1.0 - p)
                self._nbinom_n = max(n, 0.1)
                self._nbinom_p = p
            else:
                log.warning(
                    "negbinom_fallback_to_empirical",
                    reason="variance <= mean or mean is zero — NegBinom ill-defined for this data",
                    fk_col=fk_col,
                )
                self.method = "empirical"  # runtime fallback

    def sample_counts(self, parent_context: pd.DataFrame) -> np.ndarray:
        """Sample child counts for a set of parents.

        Args:
            parent_context: Parent dataframe (only length is used here).

        Returns:
            Numpy array of non-negative integer child counts aligned with parents.

        Raises:
            ValueError: If called before fitting the model.
        """
        if self._observed_counts is None:
            raise ValueError("LinkageModel.sample_counts() called before fit()")
        n_samples = len(parent_context)
        if self.method == "negbinom" and self._nbinom_n is not None:
            from scipy import stats
            counts = stats.nbinom.rvs(self._nbinom_n, self._nbinom_p, size=n_samples)
            return np.clip(counts, 0, None).astype(int)
        # Default: empirical — draw from observed distribution
        return np.random.choice(self._observed_counts, size=n_samples, replace=True)
