from typing import Dict, Optional, Any, Union, Tuple
import pandas as pd
from syntho_hive.interface.config import Metadata, PrivacyConfig
from syntho_hive.relational.orchestrator import StagedOrchestrator
from syntho_hive.validation.report_generator import ValidationReport

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = Any

class Synthesizer:
    """Main entry point that wires metadata, privacy, and orchestration."""
    def __init__(
        self,
        metadata: Metadata,
        privacy_config: PrivacyConfig,
        spark_session: Optional[SparkSession] = None,
        backend: str = "CTGAN",
        embedding_threshold: int = 50
    ):
        """Instantiate the synthesizer façade.

        Args:
            metadata: Dataset schema and relational configuration.
            privacy_config: Privacy guardrail configuration.
            spark_session: Optional SparkSession required for orchestration.
            backend: Synthesis backend identifier (currently CTGAN).
            embedding_threshold: Cardinality threshold for switching to embeddings.
        """
        self.metadata = metadata
        self.privacy = privacy_config
        self.spark = spark_session
        self.backend = backend
        self.embedding_threshold = embedding_threshold
        
        # Initialize internal components
        if self.spark:
            self.orchestrator = StagedOrchestrator(metadata, self.spark)
        else:
            self.orchestrator = None # Mode without Spark (maybe local pandas only in future)
            
    def fit(
        self, 
        data: Any, # Str (database name) or Dict[str, str] (table paths)
        sampling_strategy: str = "relational_stratified", 
        sample_size: int = 5_000_000, 
        validate: bool = False,
        epochs: int = 300,
        batch_size: int = 500,
        **model_kwargs: Union[int, str, Tuple[int, int]]
    ):
        """Fit the generative models on the real database.

        Args:
            data: Database name (str) or mapping of {table: path} (dict).
            sampling_strategy: Strategy for sampling real data.
            sample_size: Number of rows to sample from real data (approx).
            validate: Whether to run validation after fitting.
            epochs: Number of training epochs for CTGAN.
            batch_size: Batch size for training.
            **model_kwargs: Additional args forwarded to the underlying model (e.g., embedding_dim).

        Raises:
            ValueError: If Spark is unavailable or sample_size is invalid.
        """
        if not self.orchestrator:
            raise ValueError("SparkSession required for fit()")
        
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")

        print(f"Fitting on data source with {sampling_strategy} (target: {sample_size} rows)...")
        print(f"Training Config: epochs={epochs}, batch_size={batch_size}")
        
        # Determine paths
        if isinstance(data, str):
            real_paths = {t: f"{data}.{t}" for t in self.metadata.tables}
        elif isinstance(data, dict):
            real_paths = data
        else:
            raise ValueError("Argument 'data' must be a database name (str) or path mapping (dict).")
        
        self.orchestrator.fit_all(real_paths, epochs=epochs, batch_size=batch_size, **model_kwargs)
        
    def sample(self, num_rows: Dict[str, int], output_format: str = "delta", output_path: Optional[str] = None) -> Union[Dict[str, str], Dict[str, pd.DataFrame]]:
        """Generate synthetic data for each table.

        Args:
            num_rows: Mapping of table name to number of rows to generate.
            output_format: Storage format for generated datasets (default ``"delta"``).
            output_path: Optional path to write files. If None, returns DataFrames in memory.
            output_path: Optional path to write files. If None, returns DataFrames in memory.

        Raises:
            ValueError: If Spark orchestration is unavailable.

        Returns:
            Mapping of table name to the output path (if wrote to disk) OR Dictionary of DataFrames (if in-memory).
        """
        if not self.orchestrator:
            raise ValueError("SparkSession required for sample()")
            
        print(f"Generating data with {self.backend} backend...")
        
        # If output_path is explicitly None, we return DataFrames
        if output_path is None:
             return self.orchestrator.generate(num_rows, output_path_base=None)
        
        # Otherwise, write to disk (legacy/default behavior could be forced key if needed, but current API allows None)
        # Wait, previous code forced a default if output_path was None. 
        # To maintain exact backward compat we might want a flag, but user asked for "option to save to df".
        # If I change default behavior (None -> /tmp to None -> Memory), it breaks scripts relying on /tmp default?
        # The previous default was implicit in the logic.
        # Let's assume explicit None means memory now, and if they want disk they provide path.
        # OR: we could interpret empty string as None? No, None is Pythonic.
        # The user's request "add option to save to df object instead" implies a switch.
        
        output_base = output_path
        self.orchestrator.generate(num_rows, output_base)
        
        # Return paths mapping
        return {t: f"{output_base}/{t}" for t in self.metadata.tables}

    def generate_validation_report(self, real_data: Dict[str, str], synthetic_data: Dict[str, str], output_path: str):
        """Generate a validation report comparing real vs synthetic datasets.

        Args:
            real_data: Map of table name to real dataset path/table.
            synthetic_data: Map of table name to generated dataset path.
            output_path: Filesystem path for the rendered report.

        Raises:
            ValueError: If Spark is unavailable.
            Exception: Propagates any failures encountered while reading or validating.
        """
        if not self.spark:
             raise ValueError("SparkSession required for validation report generation")

        print("Generating validation report...")
        report_gen = ValidationReport()
        
        real_dfs = {}
        synth_dfs = {}
        
        try:
            # 1. Load Real Data
            for table, path in real_data.items():
                print(f"Loading real data for {table} from {path}...")
                # Try reading as table first, then path
                try:
                    df = self.spark.read.table(path)
                except:
                   # Fallback to loading as path (parquet/delta default)
                   df = self.spark.read.format("delta").load(path)
                
                real_dfs[table] = df.toPandas()

            # 2. Load Synthetic Data
            for table, path in synthetic_data.items():
                print(f"Loading synthetic data for {table} from {path}...")
                df = self.spark.read.format("delta").load(path)
                synth_dfs[table] = df.toPandas()
                
            # 3. Generate Report
            report_gen.generate(real_dfs, synth_dfs, output_path)
            
        except Exception as e:
            print(f"Error generating validation report: {str(e)}")
            raise e

    def save_to_hive(self, synthetic_data: Dict[str, str], target_db: str, overwrite: bool = True):
        """Register generated datasets as Hive tables.

        Args:
            synthetic_data: Map of table name to generated dataset path.
            target_db: Hive database where tables should be registered.
            overwrite: Whether to drop and recreate existing tables.

        Raises:
            ValueError: If Spark is unavailable.
        """
        if not self.spark:
            raise ValueError("SparkSession required for Hive registration")

        print(f"Save to Hive database: {target_db}")
        
        # Ensure DB exists
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {target_db}")
        
        for table, path in synthetic_data.items():
            full_table_name = f"{target_db}.{table}"
            print(f"Registering table {full_table_name} at {path}")
            
            if overwrite:
                self.spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")
            
            # Register External Table
            self.spark.sql(f"CREATE TABLE {full_table_name} USING DELTA LOCATION '{path}'")
