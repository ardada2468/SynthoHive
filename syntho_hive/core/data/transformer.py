import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import BayesianGaussianMixture

class DataTransformer:
    """Reversible transformer for tabular data.

    Continuous columns use a Bayesian GMM-based normalizer, while categorical
    columns are either one-hot encoded or mapped to indices for embeddings.
    """
    
    def __init__(self, metadata: Any, embedding_threshold: int = 50):
        """Create a transformer configured by table metadata.

        Args:
            metadata: Metadata object describing tables, keys, and constraints.
            embedding_threshold: Switch to embedding mode when cardinality exceeds this value.
        """
        self.metadata = metadata
        self.embedding_threshold = embedding_threshold
        self._transformers = {}
        self._column_info = {} # Maps col_name -> {'type': str, 'dim': int, 'transformer': obj}
        self.output_dim = 0
        self._excluded_columns = []
        
    def fit(self, data: pd.DataFrame, table_name: Optional[str] = None):
        """Fit per-column transformers and collect column layout metadata.

        Args:
            data: DataFrame to profile and transform.
            table_name: Optional table name for applying PK/FK exclusions and constraints.

        Raises:
            ValueError: If metadata is missing table configurations.
        """
        self.table_name = table_name # Store for constraint application later
        if not self.metadata.tables:
            raise ValueError("Metadata must be populated with table configs")
            
        columns_to_transform = data.columns.tolist()
        
        # Handle relational constraints if table_name is provided
        if table_name:
            table_config = self.metadata.get_table(table_name)
            if table_config:
                # Exclude PK and FKs from transformation
                pk = table_config.pk
                fks = list(table_config.fk.keys())
                self._excluded_columns = [pk] + fks
                columns_to_transform = [c for c in columns_to_transform if c not in self._excluded_columns]

        self.output_dim = 0
        
        for col in columns_to_transform:
            col_data = data[col]
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Continuous column
                transformer = ClusterBasedNormalizer(n_components=10)
                transformer.fit(col_data)
                
                # Dim = n_components (one-hot) + 1 (scalar)
                dim = transformer.n_components + 1
                self._transformers[col] = transformer
                self._column_info[col] = {
                    'type': 'continuous',
                    'dim': dim,
                    'transformer': transformer
                }
                self.output_dim += dim
                
            else:
                # Categorical column
                # Use OneHotEncoder for now. 
                # Categorical column
                # Check cardinality for embedding suggestion
                n_unique = col_data.nunique()
                if n_unique > self.embedding_threshold:
                    # Use LabelEncoder for Entity Embeddings
                    from sklearn.preprocessing import LabelEncoder
                    transformer = LabelEncoder()
                    # LabelEncoder expects 1D array
                    transformer.fit(col_data)
                    
                    dim = 1 # Just the index
                    self._transformers[col] = transformer
                    self._column_info[col] = {
                        'type': 'categorical_embedding',
                        'dim': dim,
                        'num_categories': n_unique,
                        'transformer': transformer
                    }
                    self.output_dim += dim
                else:
                    # Use OneHotEncoder
                    transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    values = col_data.values.reshape(-1, 1)
                    transformer.fit(values)
                    
                    dim = len(transformer.categories_[0])
                    self._transformers[col] = transformer
                    self._column_info[col] = {
                        'type': 'categorical',
                        'dim': dim,
                        'transformer': transformer
                    }
                    self.output_dim += dim

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform a dataframe into model-ready numpy arrays.

        Args:
            data: DataFrame with the same columns used during ``fit``.

        Raises:
            ValueError: If the transformer has not been fitted or a column is missing.

        Returns:
            Concatenated numpy array representing all transformed columns.
        """
        if not self._transformers:
            raise ValueError("Transformer has not been fitted.")
            
        output_arrays = []
        
        # Iterate in the same order as fit/stored in _column_info
        for col, info in self._column_info.items():
            if col not in data.columns:
                raise ValueError(f"Column {col} missing from input data")
                
            transformer = self._transformers[col]
            col_data = data[col]
            
            if info['type'] == 'continuous':
                # Returns (N, n_components + 1)
                transformed = transformer.transform(col_data)
            elif info['type'] == 'categorical_embedding':
                # Returns (N, 1)
                values = transformer.transform(col_data)
                transformed = values.reshape(-1, 1)
            else:
                # Returns (N, n_categories)
                values = col_data.values.reshape(-1, 1)
                transformed = transformer.transform(values)
            
            output_arrays.append(transformed)
            
        return np.concatenate(output_arrays, axis=1)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Convert model outputs back to the original dataframe schema.

        Args:
            data: Numpy array produced by a model, aligned to transform layout.

        Raises:
            ValueError: If called before ``fit``.

        Returns:
            DataFrame with original column names and value types (constraints applied).
        """
        if not self._transformers:
            raise ValueError("Transformer has not been fitted.")
            
        output_df = pd.DataFrame()
        start_idx = 0
        
        for col, info in self._column_info.items():
            dim = info['dim']
            end_idx = start_idx + dim
            col_data = data[:, start_idx:end_idx]
            
            transformer = self._transformers[col]
            
            if info['type'] == 'continuous':
                original_values = transformer.inverse_transform(col_data)
            elif info['type'] == 'categorical_embedding':
                 # col_data is (N, 1) floats/ints. 
                 # We need ints for LabelEncoder.
                 indices = np.clip(col_data.flatten().astype(int), 0, info['num_categories'] - 1)
                 original_values = transformer.inverse_transform(indices)
            else:
                original_values = transformer.inverse_transform(col_data).flatten()
            
            # Apply Constraints
            if self.metadata and hasattr(self, 'table_name') and self.table_name:
                table_config = self.metadata.get_table(self.table_name)
                if table_config and col in table_config.constraints:
                    constraint = table_config.constraints[col]
                    
                    # 1. Rounding/Type
                    if constraint.dtype == "int":
                        original_values = np.round(original_values).astype(int)
                    
                    # 2. Clipping
                    if constraint.min is not None or constraint.max is not None:
                        # Handle potential pandas Series or numpy array
                        if isinstance(original_values, pd.Series):
                            original_values = original_values.clip(lower=constraint.min, upper=constraint.max)
                        else:
                            original_values = np.clip(original_values, constraint.min, constraint.max)

            output_df[col] = original_values
            start_idx = end_idx
            
        return output_df

class ClusterBasedNormalizer:
    """VGM-based normalizer for continuous columns.

    Projects a value to a cluster assignment and a normalized scalar relative
    to the chosen component.
    """

    def __init__(self, n_components: int = 10):
        """Configure the number of mixture components."""
        self.n_components = n_components
        self.model = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type='dirichlet_process',
            n_init=1,
            random_state=42
        )
        self.means = None
        self.stds = None
        
    def fit(self, data: pd.Series):
        """Fit the Bayesian GMM on a continuous series.

        Args:
            data: Continuous pandas Series to normalize.
        """
        values = data.values.reshape(-1, 1)
        self.model.fit(values)
        self.means = self.model.means_.flatten() # (n_components,)
        self.stds = np.sqrt(self.model.covariances_).flatten() # (n_components,)
        
    def transform(self, data: pd.Series) -> np.ndarray:
        """Project values to one-hot cluster assignment and normalized scalar.

        Args:
            data: Continuous pandas Series to transform.

        Returns:
            Numpy array of shape ``(N, n_components + 1)`` with one-hot cluster and scaled value.
        """
        values = data.values.reshape(-1, 1)
        n_samples = len(values)
        
        # 1. Get cluster probabilities: P(c|x)
        probs = self.model.predict_proba(values) # (N, n_components)
        
        # 2. Sample component c ~ P(c|x) (Argmax for simplicity/determinism in this impl)
        # CTGAN uses argmax during interaction but sampling during training prep sometimes. 
        # Using argmax is stable.
        cluster_assignments = np.argmax(probs, axis=1)
        
        # 3. Calculate normalized scalar: v = (x - mu_c) / (4 * sigma_c)
        # Clip to [-1, 1] usually, or roughly there.
        means = self.means[cluster_assignments]
        stds = self.stds[cluster_assignments]
        
        normalized_values = (values.flatten() - means) / (4 * stds)
        normalized_values = normalized_values.reshape(-1, 1)
        
        # 4. Create One-Hot encoding of cluster assignment
        cluster_one_hot = np.zeros((n_samples, self.n_components))
        cluster_one_hot[np.arange(n_samples), cluster_assignments] = 1
        
        # Output: [one_hot_cluster, scalar]
        return np.concatenate([cluster_one_hot, normalized_values], axis=1)

    def inverse_transform(self, data: np.ndarray) -> pd.Series:
        """Reconstruct approximate original values from normalized representation.

        Args:
            data: Array shaped ``(N, n_components + 1)`` produced by ``transform``.

        Returns:
            Pandas Series of reconstructed continuous values.
        """
        # data shape: (N, n_components + 1)
        
        # Split into one-hot and scalar
        cluster_one_hot = data[:, :-1]
        scalars = data[:, -1]
        
        # Identify cluster
        cluster_assignments = np.argmax(cluster_one_hot, axis=1)
        
        means = self.means[cluster_assignments]
        stds = self.stds[cluster_assignments]
        
        # Reconstruct: x = v * 4 * sigma_c + mu_c
        reconstructed_values = scalars * 4 * stds + means
        
        return pd.Series(reconstructed_values)
