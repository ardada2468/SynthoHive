"""Pandas-native IO backend — lets the full pipeline run without Spark."""

import os
import shutil
from typing import Optional, Union

import pandas as pd
import structlog

log = structlog.get_logger()


class LocalIO:
    """Filesystem IO backend backed by pandas + pyarrow.

    Drop-in alternative to ``SparkIO`` for single-machine workloads: reads and
    writes parquet/csv files with no Spark or Java dependency. Paths written by
    ``write_pandas`` can be read back by ``read_pandas`` symmetrically.
    """

    def read_pandas(
        self, path_or_df: Union[str, pd.DataFrame], format: Optional[str] = None
    ) -> pd.DataFrame:
        """Read a dataset into a pandas DataFrame.

        Args:
            path_or_df: A DataFrame (returned as-is), or a filesystem path to a
                csv/parquet file, a bare path written by ``write_pandas``, or a
                directory of parquet files.
            format: Optional explicit format override (``"csv"`` or ``"parquet"``).

        Raises:
            FileNotFoundError: If the path does not exist in any known layout.

        Returns:
            The loaded DataFrame.
        """
        if isinstance(path_or_df, pd.DataFrame):
            return path_or_df

        path = str(path_or_df)
        fmt = format or ("csv" if path.endswith(".csv") else "parquet")

        if fmt == "csv":
            return pd.read_csv(path)

        for candidate in (path, f"{path}.parquet"):
            if os.path.exists(candidate):
                return pd.read_parquet(candidate)
        raise FileNotFoundError(f"No dataset found at '{path}' (or '{path}.parquet')")

    def write_pandas(
        self,
        pdf: pd.DataFrame,
        target_path: str,
        mode: str = "overwrite",
        format: str = "parquet",
    ) -> None:
        """Write a pandas DataFrame to the filesystem.

        Args:
            pdf: DataFrame to persist.
            target_path: Output path (a single file is written at this path).
            mode: ``"overwrite"`` (default) or ``"error"`` (fail if exists).
            format: ``"parquet"`` (default) or ``"csv"``.

        Raises:
            ValueError: If the format is unsupported.
            FileExistsError: If ``mode="error"`` and the target exists.
        """
        if format not in ("parquet", "csv"):
            raise ValueError(
                f"LocalIO supports formats 'parquet' and 'csv', got '{format}'"
            )
        if os.path.exists(target_path):
            if mode == "error":
                raise FileExistsError(f"Target path '{target_path}' already exists")
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)

        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if format == "csv":
            pdf.to_csv(target_path, index=False)
        else:
            pdf.to_parquet(target_path, index=False)
        log.debug("local_write", path=target_path, rows=len(pdf), format=format)

    def delete(self, path: str) -> None:
        """Remove a previously written dataset path (best effort)."""
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
