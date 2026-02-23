import shutil
from typing import Dict, Any, List, Union, Tuple, Optional, Literal
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import pandas_udf, PandasUDFType
except ImportError:
    SparkSession = Any

import numpy as np
import structlog

import pandas as pd
from syntho_hive.interface.config import Metadata
from syntho_hive.relational.graph import SchemaGraph
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.relational.linkage import LinkageModel
from syntho_hive.connectors.spark_io import SparkIO

log = structlog.get_logger()


def _write_with_failure_policy(io, pdf, path, policy, written_paths):
    """Write pdf to path; handle failures per policy ('raise', 'cleanup', 'retry')."""
    def _attempt_write():
        io.write_pandas(pdf, path)

    if policy == 'retry':
        try:
            _attempt_write()
            written_paths.append(path)
        except Exception:
            # One retry, no delay — transient lock release
            _attempt_write()
            written_paths.append(path)
    elif policy == 'cleanup':
        try:
            _attempt_write()
            written_paths.append(path)
        except Exception as exc:
            for p in written_paths:
                try:
                    shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
            raise exc
    else:  # 'raise' (default)
        _attempt_write()
        written_paths.append(path)


class StagedOrchestrator:
    """Manage staged relational synthesis across parent/child tables."""

    def __init__(
        self,
        metadata: Metadata,
        spark: Optional[SparkSession] = None,
        io: Optional[Any] = None,
        on_write_failure: Literal['raise', 'cleanup', 'retry'] = 'raise',
    ):
        """Initialize orchestrator dependencies.

        Args:
            metadata: Dataset metadata with relational details.
            spark: SparkSession used for IO and potential UDFs. Required when
                ``io`` is not provided.
            io: Pre-constructed IO backend. When provided, ``spark`` is ignored.
                Useful for testing and environments where SparkIO is not desired.
            on_write_failure: Policy when a write fails during generation with
                ``output_path_base`` set. Options:
                - ``'raise'`` (default): re-raise the exception immediately.
                - ``'cleanup'``: remove all previously written paths before raising.
                - ``'retry'``: attempt one additional write before raising.
        """
        self.metadata = metadata
        self.spark = spark
        if io is not None:
            self.io = io
        else:
            self.io = SparkIO(spark)
        self.on_write_failure = on_write_failure
        self.graph = SchemaGraph(metadata)
        self.models: Dict[str, CTGAN] = {}
        self.linkage_models: Dict[str, LinkageModel] = {}

    def fit_all(self, real_data_paths: Dict[str, str], epochs: int = 300, batch_size: int = 500, **model_kwargs: Union[int, str, Tuple[int, int]]):
        """Fit CTGAN and linkage models for every table.

        Args:
            real_data_paths: Mapping ``{table_name: 'db.table' or '/path'}``.
            epochs: Number of training epochs for CTGAN.
            batch_size: Training batch size.
            **model_kwargs: Extra parameters forwarded to CTGAN constructor.
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
                linkage_method = self.metadata.tables[table_name].linkage_method
                linkage = LinkageModel(method=linkage_method)
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

    def generate(self, num_rows_root: Dict[str, int], output_path_base: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Execute the multi-stage generation pipeline.

        Args:
            num_rows_root: Mapping of root table name to number of rows to generate.
            output_path_base: Base path where generated tables will be stored.
                When set, DataFrames are written to disk and released from memory
                after each table, preventing OOM on large schemas. Child tables
                read parent data from disk via this path. When None, all DataFrames
                are accumulated in memory (original behavior).

        Returns:
            Dictionary of generated DataFrames. When ``output_path_base`` is set,
            the dict contains only tables that could not be released (i.e., an empty
            dict is normal). When ``output_path_base`` is None, all tables are
            returned in memory.
        """
        generation_order = self.graph.get_generation_order()

        generated_tables = {}
        written_paths: List[str] = []
        self._written_paths = written_paths

        for table_name in generation_order:
            config = self.metadata.get_table(table_name)
            is_root = not config.fk

            model = self.models[table_name]

            generated_pdf = None

            if is_root:
                print(f"Generating root table: {table_name}")
                n_rows = num_rows_root.get(table_name, 1000)
                generated_pdf = model.sample(n_rows)
                # Assign PKs
                generated_pdf[config.pk] = range(1, n_rows + 1)
            else:
                print(f"Generating child table: {table_name}")

                # 1. Handle Driver Parent (Cardinality & Context)
                pk_map = config.fk
                sorted_fks = sorted(pk_map.keys())
                driver_fk = sorted_fks[0]
                driver_ref = pk_map[driver_fk]
                driver_parent_table, driver_parent_pk = driver_ref.split(".")

                # Read Driver Parent Data (From Output or Memory)
                if output_path_base:
                    parent_path = f"{output_path_base}/{driver_parent_table}"
                    parent_df = self.io.read_dataset(parent_path).toPandas()
                else:
                    parent_df = generated_tables[driver_parent_table]

                linkage = self.linkage_models[table_name]

                # Sample Counts
                counts = linkage.sample_counts(parent_df)

                # Construct Context from Driver
                parent_ids_repeated = np.repeat(parent_df[driver_parent_pk].to_numpy(), counts)

                context_cols = config.parent_context_cols
                if context_cols:
                    context_repeated_vals = {}
                    for col in context_cols:
                        context_repeated_vals[col] = np.repeat(parent_df[col].to_numpy(), counts)
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
                        if output_path_base:
                            p_path = f"{output_path_base}/{p_table}"
                            p_df = self.io.read_dataset(p_path).toPandas()
                        else:
                            p_df = generated_tables[p_table]

                        valid_pks = p_df[p_pk].to_numpy()

                        # Randomly sample valid PKs for this column
                        generated_pdf[fk_col] = np.random.choice(valid_pks, size=total_child_rows)

                    # Assign PKs
                    generated_pdf[config.pk] = range(1, total_child_rows + 1)

            if generated_pdf is not None:
                if output_path_base:
                    output_path = f"{output_path_base}/{table_name}"
                    _write_with_failure_policy(
                        io=self.io,
                        pdf=generated_pdf,
                        path=output_path,
                        policy=self.on_write_failure,
                        written_paths=written_paths,
                    )
                    log.debug("table_released_from_memory", table=table_name, path=output_path)
                    # Do NOT store in generated_tables — child tables read from disk via output_path_base
                else:
                    generated_tables[table_name] = generated_pdf

        return generated_tables
