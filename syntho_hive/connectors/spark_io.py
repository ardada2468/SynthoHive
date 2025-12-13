from typing import Any, List, Optional, Union
try:
    from pyspark.sql import SparkSession, DataFrame
    from delta.tables import DeltaTable
except ImportError:
    # Allow imports without spark for local non-spark testing
    SparkSession = Any
    DataFrame = Any
    DeltaTable = Any

import pandas as pd

class SparkIO:
    """Utility for reading and writing datasets via Spark and Delta Lake."""
    def __init__(self, spark: SparkSession):
        """Initialize the IO helper.

        Args:
            spark: Active SparkSession used for all IO.
        """
        self.spark = spark

    def read_dataset(self, path_or_table: str, format: str = None, **kwargs: Union[str, int, bool, float]) -> DataFrame:
        """Read a dataset from a table name or filesystem path.

        Args:
            path_or_table: Hive table name or filesystem/URI path.
            format: Optional explicit format override (e.g., ``"csv"``).
            **kwargs: Additional Spark read options.

        Returns:
            Spark DataFrame loaded from the specified source.
        """
        # Simple heuristic
        if "/" in path_or_table or "\\" in path_or_table or path_or_table.startswith("file://"):
            if format:
                return self.spark.read.format(format).load(path_or_table, **kwargs)
            
            if path_or_table.endswith(".csv"):
                return self.spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("multiLine", "true").load(path_or_table, **kwargs)
            elif path_or_table.endswith(".parquet"):
                return self.spark.read.format("parquet").load(path_or_table, **kwargs)
            else:
                # Default to parquet for directories/tables (matching write default)
                return self.spark.read.format("parquet").load(path_or_table, **kwargs)
        return self.spark.table(path_or_table)

    def write_dataset(self, df: DataFrame, target_path: str, mode: str = "overwrite", partition_by: Optional[str] = None, format: str = "parquet"):
        """Write a Spark DataFrame to storage.

        Args:
            df: Spark DataFrame to persist.
            target_path: Output path (directory or table location).
            mode: Save mode, e.g., ``"overwrite"`` or ``"append"``.
            partition_by: Optional column name to partition by.
            format: Output format, defaults to ``"parquet"``.
        """
        writer = df.write.format(format).mode(mode)
        if partition_by:
            writer = writer.partitionBy(partition_by)
        writer.save(target_path)
        
    def write_pandas(self, pdf: pd.DataFrame, target_path: str, mode: str = "append", format: str = "parquet"):
        """Write a Pandas DataFrame using Spark-backed persistence.

        Args:
            pdf: Pandas DataFrame to persist.
            target_path: Output path for the written dataset.
            mode: Save mode for Spark writer (default ``"append"``).
            format: Storage format, defaults to ``"parquet"``.
        """
        sdf = self.spark.createDataFrame(pdf)
        self.write_dataset(sdf, target_path, mode=mode, format=format)
