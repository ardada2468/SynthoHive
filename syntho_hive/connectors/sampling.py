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
        self, 
        root_table: str, 
        sample_size: int, 
        stratify_by: Optional[str] = None
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
            fractions = root_df.select(stratify_by).distinct().withColumn("fraction", F.lit(0.1)).rdd.collectAsMap() 
            # Note: fractions logic needs to be calculated based on sample_size / total_count
            sampled_root = root_df.sampleBy(stratify_by, fractions, seed=42)
        else:
            fraction = min(1.0, sample_size / root_df.count())
            sampled_root = root_df.sample(withReplacement=False, fraction=fraction, seed=42)
            
        sampled_data[root_table] = sampled_root
        
        # 2. Cascade to Children
        # Simple BFS or using Graph
        parent_pk = self.metadata.get_table(root_table).pk
        
        # Find children
        for child_name, config in self.metadata.tables.items():
            for child_col, parent_ref in config.fk.items():
                if parent_ref.startswith(f"{root_table}."):
                    print(f"Cascading sample to child: {child_name}")
                    child_df = self.spark.table(child_name)
                    
                    # Semijoin
                    # Join on PK-FK to keep only rows matching sampled parents
                    sampled_child = child_df.join(
                        sampled_root.select(parent_pk),
                        child_df[child_col] == sampled_root[parent_pk],
                        "inner"
                    ).select(child_df.columns) # Keep only child cols
                    
                    sampled_data[child_name] = sampled_child
                    
        return sampled_data
