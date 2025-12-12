from typing import Dict, Any, List
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import pandas_udf, PandasUDFType
except ImportError:
    SparkSession = Any

import numpy as np

import pandas as pd
from syntho_hive.interface.config import Metadata
from syntho_hive.relational.graph import SchemaGraph
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.relational.linkage import LinkageModel
from syntho_hive.connectors.spark_io import SparkIO

class StagedOrchestrator:
    """
    Manages the multi-stage generation process.
    1. Identify Dependency Order
    2. Generate Roots
    3. Generate Children (using Parent context)
    """
    
    def __init__(self, metadata: Metadata, spark: SparkSession):
        self.metadata = metadata
        self.spark = spark
        self.graph = SchemaGraph(metadata)
        self.io = SparkIO(spark)
        self.models: Dict[str, CTGAN] = {}
        self.linkage_models: Dict[str, LinkageModel] = {}
        
    def fit_all(self, real_data_paths: Dict[str, str], epochs: int = 300, batch_size: int = 500, **model_kwargs):
        """
        Fit all models.
        real_data_paths: {table_name: 'db.table_name' or '/path/to/delta'}
        """
        # Topo sort to train parents first? Or independent?
        # Linkage model needs both parent and child data.
        # CTGAN needs Child data + Parent attributes (joined).
        
        # Training order doesn't strictly matter as long as we have data, 
        # but generation order matters.
        
        for table_name in self.metadata.tables:
            print(f"Fitting model for table: {table_name}")
            data_path = real_data_paths.get(table_name)
            if not data_path:
                print(f"Warning: No data path provided for {table_name}, skipping.")
                continue

            # Read data
            target_df = self.io.read_dataset(data_path)
            # Convert to Pandas for CTGAN (prototype limitation)
            target_pdf = target_df.toPandas()

            config = self.metadata.get_table(table_name)
            if not config.has_dependencies:
                # Root Table
                model = CTGAN(
                    self.metadata, 
                    batch_size=batch_size, 
                    epochs=epochs,
                    **model_kwargs
                )
                model.fit(target_pdf, table_name=table_name)
                self.models[table_name] = model
            else:
                # Child Table
                # 1. Identify "Driver" Parent (First FK)
                pk_map = config.fk
                # pk_map is {local_col: "parent_table.parent_col"}
                
                # Sort keys to ensure deterministic driver selection
                sorted_fks = sorted(pk_map.keys())
                driver_fk = sorted_fks[0]
                driver_ref = pk_map[driver_fk]
                
                driver_parent_table, driver_parent_pk = driver_ref.split(".")

                parent_path = real_data_paths.get(driver_parent_table)
                parent_df = self.io.read_dataset(parent_path).toPandas()
                
                # 2. Train Linkage Model on Driver Parent
                print(f"Training Linkage for {table_name} driven by {driver_parent_table}")
                linkage = LinkageModel()
                linkage.fit(parent_df, target_pdf, fk_col=driver_fk, pk_col=driver_parent_pk)
                self.linkage_models[table_name] = linkage

                # 3. Train Conditional CTGAN (Conditioning on Driver Parent Context)
                context_cols = config.parent_context_cols
                if context_cols:
                     # Prepare parent data for merge
                     right_side = parent_df[[driver_parent_pk] + context_cols].copy()
                     
                     rename_map = {c: f"__ctx__{c}" for c in context_cols}
                     right_side = right_side.rename(columns=rename_map)
                     
                     joined = target_pdf.merge(
                         right_side,
                         left_on=driver_fk,
                         right_on=driver_parent_pk,
                         how="left"
                     )
                     
                     context_df = joined[list(rename_map.values())].copy()
                     context_df.columns = context_cols
                else:
                    context_df = None

                model = CTGAN(
                    self.metadata, 
                    batch_size=batch_size, 
                    epochs=epochs,
                    **model_kwargs
                )
                # Note: We exclude ALL FK columns from CTGAN modeling to avoid them being treated as continuous/categorical features
                # The DataTransformer handles excluding PK/FK if they are marked in metadata.
                # But we must ensure metadata knows about ALL FKs. (It does via config.fk)
                model.fit(target_pdf, context=context_df, table_name=table_name)
                self.models[table_name] = model
            
    def generate(self, num_rows_root: Dict[str, int], output_path_base: str):
        """
        Execute generation pipeline.
        """
        generation_order = self.graph.get_generation_order()
        
        generated_tables = set()
        
        for table_name in generation_order:
            config = self.metadata.get_table(table_name)
            is_root = not config.fk
            
            output_path = f"{output_path_base}/{table_name}"
            model = self.models[table_name]
            
            if is_root:
                print(f"Generating root table: {table_name}")
                n_rows = num_rows_root.get(table_name, 1000)
                generated_pdf = model.sample(n_rows)
                # Assign PKs
                generated_pdf[config.pk] = range(1, n_rows + 1)
                self.io.write_pandas(generated_pdf, output_path)
            else:
                print(f"Generating child table: {table_name}")
                
                # 1. Handle Driver Parent (Cardinality & Context)
                pk_map = config.fk
                sorted_fks = sorted(pk_map.keys())
                driver_fk = sorted_fks[0]
                driver_ref = pk_map[driver_fk]
                driver_parent_table, driver_parent_pk = driver_ref.split(".")
                
                # Read Driver Parent Data
                parent_path = f"{output_path_base}/{driver_parent_table}"
                parent_df = self.io.read_dataset(parent_path).toPandas()
                
                linkage = self.linkage_models[table_name]

                # Sample Counts
                counts = linkage.sample_counts(parent_df)

                # Construct Context from Driver
                parent_ids_repeated = np.repeat(parent_df[driver_parent_pk].values, counts)
                
                context_cols = config.parent_context_cols
                if context_cols:
                    context_repeated_vals = {}
                    for col in context_cols:
                        context_repeated_vals[col] = np.repeat(parent_df[col].values, counts)
                    context_df = pd.DataFrame(context_repeated_vals)
                else:
                    context_df = None

                total_child_rows = len(parent_ids_repeated)

                # 2. Generate Data
                if total_child_rows > 0:
                     generated_pdf = model.sample(total_child_rows, context=context_df)
                     
                     # Assign Driver FK
                     generated_pdf[driver_fk] = parent_ids_repeated
                     
                     # Assign Secondary FKs (Random Sampling from respective Parents)
                     for fk_col in sorted_fks[1:]:
                         ref = pk_map[fk_col]
                         p_table, p_pk = ref.split(".")
                         
                         # Read Secondary Parent
                         p_path = f"{output_path_base}/{p_table}"
                         # Optimization: cache read dfs? For now just read.
                         p_df = self.io.read_dataset(p_path).toPandas()
                         valid_pks = p_df[p_pk].values
                         
                         # Randomly sample valid PKs for this column
                         # Note: This ignores correlations between parents, but ensures Referental Integrity.
                         generated_pdf[fk_col] = np.random.choice(valid_pks, size=total_child_rows)
                     
                     # Assign PKs
                     generated_pdf[config.pk] = range(1, total_child_rows + 1)
                     
                     self.io.write_pandas(generated_pdf, output_path)
            
            generated_tables.add(table_name)
