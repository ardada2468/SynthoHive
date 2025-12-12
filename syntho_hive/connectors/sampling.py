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
    """
    Implements Relational Stratified Sampling.
    Goal: Sample parent table efficiently while keeping distribution, 
    then fetch ALL related children for sampled parents.
    """
    
    def __init__(self, metadata: Metadata, spark: SparkSession):
        self.metadata = metadata
        self.spark = spark
        
    def sample_relational(
        self, 
        root_table: str, 
        sample_size: int, 
        stratify_by: Optional[str] = None
    ) -> Dict[str, DataFrame]:
        """
        Sample root table and cascade to children.
        Returns map of {table_name: sampled_dataframe}
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
