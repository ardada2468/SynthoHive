from typing import Dict, Optional, Any
import pandas as pd
from syntho_hive.interface.config import Metadata, PrivacyConfig
from syntho_hive.relational.orchestrator import StagedOrchestrator
from syntho_hive.validation.report_generator import ValidationReport

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = Any

class Synthesizer:
    """
    Main entry point for SynthoHive 2.0.
    """
    def __init__(
        self,
        metadata: Metadata,
        privacy_config: PrivacyConfig,
        spark_session: Optional[SparkSession] = None,
        backend: str = "CTGAN",
        embedding_threshold: int = 50
    ):
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
        database: str, 
        sampling_strategy: str = "relational_stratified", 
        sample_size: int = 5_000_000, 
        validate: bool = False,
        epochs: int = 300,
        batch_size: int = 500,
        **model_kwargs
    ):
        """
        Fit the generative models on the real database.
        
        Args:
            database: Name of the database or path prefix.
            sampling_strategy: Strategy for sampling real data.
            sample_size: Number of rows to sample from real data (approx).
            validate: Whether to run validation after fitting.
            epochs: Number of training epochs for CTGAN.
            batch_size: Batch size for training.
            **model_kwargs: Additional arguments passed to the underlying model (e.g. embedding_dim).
        """
        if not self.orchestrator:
            raise ValueError("SparkSession required for fit()")
        
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")

        print(f"Fitting on {database} with {sampling_strategy} (target: {sample_size} rows)...")
        print(f"Training Config: epochs={epochs}, batch_size={batch_size}")
        
        # Construct paths to real tables
        # Assuming database means a Spark/Hive database or a path prefix
        # For simplicity in this prototype, we'll assume it's a Hive DB name
        # If it was a file path, we'd need more logic.
        real_paths = {t: f"{database}.{t}" for t in self.metadata.tables}
        
        # In a real scenario, we might want to validate these tables exist first
        
        self.orchestrator.fit_all(real_paths, epochs=epochs, batch_size=batch_size, **model_kwargs)
        
    def sample(self, num_rows: Dict[str, int], output_format: str = "delta") -> Dict[str, str]:
        """
        Generate synthetic data.
        Returns map of table_name -> output_path
        """
        if not self.orchestrator:
            raise ValueError("SparkSession required for sample()")
            
        print(f"Generating data with {self.backend} backend...")
        
        # Define output base path - distinct per run ideally, or user specified
        # Here we use a temp path or one derived from a base config if we had it
        output_base = f"/tmp/syntho_hive_output/{output_format}"
        
        # Delegate to orchestrator
        self.orchestrator.generate(num_rows, output_base)
        
        # Return paths mapping
        return {t: f"{output_base}/{t}" for t in self.metadata.tables}

    def generate_validation_report(self, real_data: Dict[str, str], synthetic_data: Dict[str, str], output_path: str):
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
