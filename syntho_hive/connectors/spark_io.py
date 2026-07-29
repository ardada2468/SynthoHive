from typing import Any, Optional, Union

try:
    from pyspark.sql import SparkSession, DataFrame
except ImportError:
    # Allow imports without spark for local non-spark testing
    SparkSession = Any
    DataFrame = Any

import pandas as pd

_KNOWN_FILE_EXTENSIONS = (".csv", ".parquet", ".json")


class SparkIO:
    """Utility for reading and writing datasets via Spark and Delta Lake."""

    def __init__(self, spark: SparkSession):
        """Initialize the IO helper.

        Args:
            spark: Active SparkSession used for all IO.

        Raises:
            ValueError: If no SparkSession is provided.
        """
        if spark is None:
            raise ValueError(
                "SparkIO requires an active SparkSession. For Spark-free usage, "
                "use syntho_hive.connectors.local_io.LocalIO instead."
            )
        self.spark = spark

    def read_dataset(
        self,
        path_or_table: str,
        format: str = None,
        **kwargs: Union[str, int, bool, float],
    ) -> DataFrame:
        """Read a dataset from a table name or filesystem path.

        Args:
            path_or_table: Hive table name or filesystem/URI path.
            format: Optional explicit format override (e.g., ``"csv"``).
            **kwargs: Additional Spark read options.

        Returns:
            Spark DataFrame loaded from the specified source.
        """
        # Simple heuristic: separators, URI prefixes, or known file extensions
        # mean "path"; anything else is treated as a catalog table name.
        if (
            "/" in path_or_table
            or "\\" in path_or_table
            or path_or_table.startswith("file://")
            or path_or_table.endswith(_KNOWN_FILE_EXTENSIONS)
        ):
            if format:
                return self.spark.read.format(format).load(path_or_table, **kwargs)

            if path_or_table.endswith(".csv"):
                return (
                    self.spark.read.format("csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .option("multiLine", "true")
                    .load(path_or_table, **kwargs)
                )
            elif path_or_table.endswith(".parquet"):
                return self.spark.read.format("parquet").load(path_or_table, **kwargs)
            else:
                # Default to parquet for directories/tables (matching write default)
                return self.spark.read.format("parquet").load(path_or_table, **kwargs)
        return self.spark.table(path_or_table)

    def read_pandas(
        self,
        path_or_df: Union[str, pd.DataFrame],
        format: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read a dataset and return it as a pandas DataFrame.

        Args:
            path_or_df: A DataFrame (returned as-is) or a table name/path.
            format: Optional explicit format override.

        Returns:
            The loaded pandas DataFrame.
        """
        if isinstance(path_or_df, pd.DataFrame):
            return path_or_df
        return self.read_dataset(path_or_df, format=format).toPandas()

    def write_dataset(
        self,
        df: DataFrame,
        target_path: str,
        mode: str = "overwrite",
        partition_by: Optional[str] = None,
        format: str = "parquet",
    ):
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

    def write_pandas(
        self,
        pdf: pd.DataFrame,
        target_path: str,
        mode: str = "overwrite",
        format: str = "parquet",
    ):
        """Write a Pandas DataFrame using Spark-backed persistence.

        Args:
            pdf: Pandas DataFrame to persist.
            target_path: Output path for the written dataset.
            mode: Save mode for Spark writer (default ``"overwrite"``).
            format: Storage format, defaults to ``"parquet"``.
        """
        sdf = self.spark.createDataFrame(pdf)
        self.write_dataset(sdf, target_path, mode=mode, format=format)
