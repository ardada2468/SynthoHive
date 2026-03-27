from typing import Any, Dict, List, Optional

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
except ImportError:
    SparkSession = Any
    DataFrame = Any
    F = Any

from syntho_hive.interface.config import Metadata


class RelationalSampler:
    """Relational stratified sampler for parent-child table hierarchies."""

    def __init__(self, metadata: Metadata, spark: SparkSession):
        """Initialize the sampler.

        Args:
            metadata: Metadata describing tables and their keys.
            spark: Active SparkSession for table access.
        """
        self.metadata = metadata
        self.spark = spark

    def sample_relational(
        self, root_table: str, sample_size: int, stratify_by: Optional[str] = None
    ) -> Dict[str, DataFrame]:
        """Sample a root table and cascade the sample to child tables.

        Args:
            root_table: Name of the parent/root table to sample.
            sample_size: Approximate number of rows to retain from the root.
            stratify_by: Optional column for stratified sampling.

        Returns:
            Dictionary mapping table name to sampled Spark DataFrame.
        """
        sampled_data = {}

        # 1. Sample Root
        print(f"Sampling root table: {root_table}")
        # Placeholder for real table loading
        root_df = self.spark.table(root_table)

        if stratify_by:
            # Approximate stratified sampling
            fractions = (
                root_df.select(stratify_by)
                .distinct()
                .withColumn("fraction", F.lit(0.1))
                .rdd.collectAsMap()
            )
            # Note: fractions logic needs to be calculated based on sample_size / total_count
            sampled_root = root_df.sampleBy(stratify_by, fractions, seed=42)
        else:
            fraction = min(1.0, sample_size / root_df.count())
            sampled_root = root_df.sample(
                withReplacement=False, fraction=fraction, seed=42
            )

        sampled_data[root_table] = sampled_root

        # 2. Cascade to Children using BFS for multi-level hierarchies
        tables_to_process = [root_table]
        processed = set()

        while tables_to_process:
            current = tables_to_process.pop(0)
            if current in processed:
                continue
            processed.add(current)

            current_pk = self.metadata.get_table(current).pk
            current_sampled = sampled_data[current]

            # Find children of the current table
            for child_name, config in self.metadata.tables.items():
                if child_name in processed:
                    continue
                for child_col, parent_ref in config.fk.items():
                    parent_table = parent_ref.split(".")[0]
                    if parent_table == current:
                        print(f"Cascading sample to child: {child_name}")
                        child_df = self.spark.table(child_name)

                        # Semi-join: keep only child rows matching sampled parent PKs
                        # without introducing ambiguous duplicate columns
                        child_sampled = child_df.join(
                            current_sampled.select(current_pk).distinct(),
                            child_df[child_col] == current_sampled[current_pk],
                            "left_semi",
                        )

                        sampled_data[child_name] = child_sampled
                        tables_to_process.append(child_name)
                        break  # Only process the first FK match per child table

        return sampled_data
