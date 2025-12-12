from typing import Any, List, Optional
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
    """
    Handles read/write operations with Spark and Delta Lake.
    """
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def read_dataset(self, path_or_table: str, format: str = None, **kwargs) -> DataFrame:
        """Read from Hive or Delta path."""
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
        """Write to Data Lake."""
        writer = df.write.format(format).mode(mode)
        if partition_by:
            writer = writer.partitionBy(partition_by)
        writer.save(target_path)
        
    def write_pandas(self, pdf: pd.DataFrame, target_path: str, mode: str = "append", format: str = "parquet"):
        """Write local Pandas DF to Distributed Storage."""
        sdf = self.spark.createDataFrame(pdf)
        self.write_dataset(sdf, target_path, mode=mode, format=format)
