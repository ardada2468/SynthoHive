import shutil
from typing import Dict, Any, List, Union, Tuple, Optional, Literal, Type

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = Any

import numpy as np
import structlog

import pandas as pd
from syntho_hive.interface.config import Metadata, PrivacyConfig, parse_fk_ref
from syntho_hive.relational.graph import SchemaGraph
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.core.models.base import ConditionalGenerativeModel
from syntho_hive.connectors.local_io import LocalIO
from syntho_hive.relational.linkage import LinkageModel
from syntho_hive.exceptions import SchemaError, GenerationError

log = structlog.get_logger()


def _write_with_failure_policy(io, pdf, path, policy, written_paths, format="parquet"):
    """Write pdf to path; handle failures per policy ('raise', 'cleanup', 'retry')."""

    def _attempt_write():
        io.write_pandas(pdf, path, format=format)

    if policy == "retry":
        try:
            _attempt_write()
            written_paths.append(path)
        except Exception:
            # One retry, no delay — transient lock release
            _attempt_write()
            written_paths.append(path)
    elif policy == "cleanup":
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
        on_write_failure: Literal["raise", "cleanup", "retry"] = "raise",
        model_cls: Type[ConditionalGenerativeModel] = CTGAN,
        privacy_config: Optional[PrivacyConfig] = None,
    ):
        """Initialize orchestrator dependencies.

        Args:
            metadata: Dataset metadata with relational details.
            spark: Optional SparkSession used for IO. When neither ``spark``
                nor ``io`` is provided, a pandas-native ``LocalIO`` backend is
                used so the pipeline runs without Spark.
            io: Pre-constructed IO backend. When provided, ``spark`` is ignored.
                Useful for testing and environments where SparkIO is not desired.
            on_write_failure: Policy when a write fails during generation with
                ``output_path_base`` set. Options:
                - ``'raise'`` (default): re-raise the exception immediately.
                - ``'cleanup'``: remove all previously written paths before raising.
                - ``'retry'``: attempt one additional write before raising.
            model_cls: Generative model class to instantiate per table. Must be a class
                (not an instance) implementing ``ConditionalGenerativeModel``. The class
                constructor must accept ``(metadata, batch_size, epochs, **kwargs)``.
                Defaults to CTGAN.
            privacy_config: Optional privacy configuration. When set, each
                table's declared ``pii_cols`` are sanitized before training.
        """
        if not (
            isinstance(model_cls, type)
            and issubclass(model_cls, ConditionalGenerativeModel)
        ):
            raise TypeError(
                f"model_cls must be a subclass of ConditionalGenerativeModel, "
                f"got {model_cls!r}. Implement fit(), sample(), save(), load() "
                f"and subclass ConditionalGenerativeModel."
            )
        self.metadata = metadata
        self.spark = spark
        if io is not None:
            self.io = io
        elif spark is not None:
            from syntho_hive.connectors.spark_io import SparkIO

            self.io = SparkIO(spark)
        else:
            self.io = LocalIO()
        self.on_write_failure = on_write_failure
        self.model_cls = model_cls
        self.privacy_config = privacy_config
        self.graph = SchemaGraph(metadata)
        self.models: Dict[str, ConditionalGenerativeModel] = {}
        self.linkage_models: Dict[str, LinkageModel] = {}
        self._table_schemas: Dict[str, List[str]] = {}

    def __getstate__(self):
        """Exclude live Spark handles from pickling (checkpoint portability)."""
        state = self.__dict__.copy()
        state["spark"] = None
        # SparkIO holds the live session and cannot be pickled; LocalIO can.
        if state.get("io") is not None and type(state["io"]).__name__ == "SparkIO":
            state["io"] = None
        return state

    def __setstate__(self, state):
        """Restore from pickle; default to the pandas-native IO backend."""
        self.__dict__.update(state)
        if self.io is None:
            self.io = LocalIO()

    def bind_spark(self, spark: Optional[SparkSession]) -> None:
        """Re-attach a SparkSession (e.g. after loading a pickled synthesizer)."""
        self.spark = spark
        if spark is not None:
            from syntho_hive.connectors.spark_io import SparkIO

            self.io = SparkIO(spark)
        else:
            self.io = LocalIO()

    def fit_all(
        self,
        real_data_paths: Dict[str, str],
        epochs: int = 300,
        batch_size: int = 500,
        progress_bar: bool = True,
        checkpoint_interval: int = 10,
        checkpoint_dir: Optional[str] = None,
        seed: Optional[int] = None,
        **model_kwargs: Union[int, str, Tuple[int, int]],
    ):
        """Fit CTGAN and linkage models for every table.

        Args:
            real_data_paths: Mapping of table name to a path/table identifier
                (``'db.table'`` or ``'/path'``) or an in-memory pandas DataFrame.
            epochs: Number of training epochs for CTGAN.
            batch_size: Training batch size.
            progress_bar: If True (default), display tqdm progress bar to stderr during training.
                Structured log events always emit regardless of this flag.
            checkpoint_interval: Save a validation checkpoint every N epochs. Default 10.
            checkpoint_dir: Optional directory to save best_checkpoint/ and final_checkpoint/
                during training.
            seed: Optional integer seed. Per-table seeds are derived from it so
                every model trains deterministically.
            **model_kwargs: Extra parameters forwarded to the model constructor.

        Raises:
            SchemaError: If a metadata table has no data source, or schema
                validation fails.
        """
        # Training order doesn't strictly matter as long as we have data,
        # but generation order matters.

        self.metadata.validate_schema()

        # Every metadata table needs a data source — a missing one used to be
        # skipped with a warning here and then crash generate() with KeyError.
        missing_tables = [t for t in self.metadata.tables if t not in real_data_paths]
        if missing_tables:
            raise SchemaError(
                f"No data source provided for metadata table(s) {missing_tables}. "
                f"Provide an entry for every table in the metadata."
            )

        for table_index, table_name in enumerate(self.metadata.tables):
            log.info("fitting_model", table=table_name)
            table_seed = (seed + table_index) if seed is not None else None

            target_pdf = self.io.read_pandas(real_data_paths[table_name])

            config = self.metadata.get_table(table_name)
            if config is None:
                raise SchemaError(f"Table '{table_name}' not found in metadata")

            # Privacy guardrail: sanitize declared PII before it reaches a model.
            if self.privacy_config is not None and config.pii_cols:
                from syntho_hive.privacy.pipeline import apply_privacy

                target_pdf = apply_privacy(
                    target_pdf,
                    config.pii_cols,
                    self.privacy_config.pii_strategy,
                    seed=table_seed,
                )

            # Record the real schema so zero-row children can be built correctly.
            self._table_schemas[table_name] = list(target_pdf.columns)

            if not config.has_dependencies:
                # Root Table
                model = self.model_cls(
                    self.metadata, batch_size=batch_size, epochs=epochs, **model_kwargs
                )
                model.fit(
                    target_pdf,
                    table_name=table_name,
                    progress_bar=progress_bar,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=checkpoint_dir,
                    seed=table_seed,
                )
                self.models[table_name] = model
            else:
                # Child Table
                # 1. Identify "Driver" Parent (explicit driver_fk or sorted-first FK)
                pk_map = config.fk
                driver_fk = config.get_driver_fk()
                driver_ref = pk_map[driver_fk]

                driver_parent_table, driver_parent_pk = parse_fk_ref(driver_ref)

                if driver_parent_table not in real_data_paths:
                    raise SchemaError(
                        f"Driver parent table '{driver_parent_table}' of "
                        f"'{table_name}' has no entry in the provided data sources."
                    )
                parent_df = self.io.read_pandas(real_data_paths[driver_parent_table])

                # 2. Train Linkage Model on Driver Parent
                log.info(
                    "training_linkage",
                    table=table_name,
                    driver_parent=driver_parent_table,
                )
                linkage_method = self.metadata.tables[table_name].linkage_method
                linkage = LinkageModel(method=linkage_method)
                linkage.fit(
                    parent_df, target_pdf, fk_col=driver_fk, pk_col=driver_parent_pk
                )
                self.linkage_models[table_name] = linkage

                # 3. Train Conditional CTGAN (Conditioning on Driver Parent Context)
                context_cols = config.parent_context_cols
                if context_cols:
                    missing = [c for c in context_cols if c not in parent_df.columns]
                    if missing:
                        raise SchemaError(
                            f"parent_context_cols {missing} not found in parent table "
                            f"'{driver_parent_table}' columns: {list(parent_df.columns)}"
                        )
                    # Prepare parent data for merge
                    right_side = parent_df[[driver_parent_pk] + context_cols].copy()

                    rename_map = {c: f"__ctx__{c}" for c in context_cols}
                    right_side = right_side.rename(columns=rename_map)

                    joined = target_pdf.merge(
                        right_side,
                        left_on=driver_fk,
                        right_on=driver_parent_pk,
                        how="left",
                    )

                    context_df = joined[list(rename_map.values())].copy()
                    context_df.columns = context_cols
                else:
                    context_df = None

                model = self.model_cls(
                    self.metadata, batch_size=batch_size, epochs=epochs, **model_kwargs
                )
                # Note: We exclude ALL FK columns from CTGAN modeling to avoid them being treated as continuous/categorical features
                # The DataTransformer handles excluding PK/FK if they are marked in metadata.
                # But we must ensure metadata knows about ALL FKs. (It does via config.fk)
                model.fit(
                    target_pdf,
                    context=context_df,
                    table_name=table_name,
                    progress_bar=progress_bar,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=checkpoint_dir,
                    seed=table_seed,
                )
                self.models[table_name] = model

    def generate(
        self,
        num_rows_root: Dict[str, int],
        output_path_base: Optional[str] = None,
        seed: Optional[int] = None,
        output_format: str = "parquet",
    ) -> Dict[str, pd.DataFrame]:
        """Execute the multi-stage generation pipeline.

        Args:
            num_rows_root: Mapping of root table name to number of rows to generate.
                Every root table must have an entry; entries for child tables are
                ignored with a warning (child volumes are driven by the fitted
                cardinality model).
            output_path_base: Base path where generated tables will be stored.
                When set, DataFrames are written to disk and released from memory
                after each table, preventing OOM on large schemas. Child tables
                read parent data from disk via this path. When None, all DataFrames
                are accumulated in memory (original behavior).
            seed: Optional integer seed making the full generation pass
                (model sampling, cardinality draws, FK assignment) reproducible.
            output_format: Storage format for written tables (default ``"parquet"``).

        Raises:
            GenerationError: If called before ``fit_all()``.
            SchemaError: If a root table has no ``num_rows_root`` entry.

        Returns:
            Dictionary of generated DataFrames. When ``output_path_base`` is set,
            the dict contains only tables that could not be released (i.e., an empty
            dict is normal). When ``output_path_base`` is None, all tables are
            returned in memory.
        """
        if not self.models:
            raise GenerationError(
                "Orchestrator has no fitted models. Call fit_all() before generate()."
            )

        generation_order = self.graph.get_generation_order()

        # Validate num_rows entries up front instead of silently defaulting.
        root_tables = [
            t for t in generation_order if not self.metadata.get_table(t).fk
        ]
        missing_roots = [t for t in root_tables if t not in num_rows_root]
        if missing_roots:
            raise SchemaError(
                f"num_rows missing for root table(s) {missing_roots}. "
                f"Every root table needs an explicit row count."
            )
        child_entries = [t for t in num_rows_root if t not in root_tables]
        if child_entries:
            log.warning(
                "child_num_rows_ignored",
                tables=child_entries,
                note="Child table volumes are driven by the fitted cardinality model",
            )

        rng = np.random.default_rng(seed)

        generated_tables = {}
        written_paths: List[str] = []
        parent_cache: Dict[str, pd.DataFrame] = {}

        def _read_parent(parent_table: str) -> pd.DataFrame:
            if parent_table in parent_cache:
                return parent_cache[parent_table]
            if output_path_base:
                pdf = self.io.read_pandas(f"{output_path_base}/{parent_table}")
            else:
                pdf = generated_tables[parent_table]
            parent_cache[parent_table] = pdf
            return pdf

        for table_name in generation_order:
            config = self.metadata.get_table(table_name)
            if config is None:
                raise SchemaError(f"Table '{table_name}' not found in metadata")
            is_root = not config.fk

            model = self.models.get(table_name)
            if model is None:
                raise GenerationError(
                    f"No fitted model for table '{table_name}'. "
                    f"Was fit_all() interrupted?"
                )

            # Derive a per-table sampling seed only when the caller asked for
            # determinism; otherwise let the model sample freely.
            table_seed = int(rng.integers(2**31 - 1)) if seed is not None else None

            generated_pdf = None

            if is_root:
                log.info("generating_root_table", table=table_name)
                n_rows = num_rows_root[table_name]
                generated_pdf = model.sample(n_rows, seed=table_seed)
                # Assign PKs — use actual DataFrame length in case model returns different count
                generated_pdf[config.pk] = range(1, len(generated_pdf) + 1)
            else:
                log.info("generating_child_table", table=table_name)

                # 1. Handle Driver Parent (Cardinality & Context)
                pk_map = config.fk
                driver_fk = config.get_driver_fk()
                driver_ref = pk_map[driver_fk]
                driver_parent_table, driver_parent_pk = parse_fk_ref(driver_ref)

                parent_df = _read_parent(driver_parent_table)

                linkage = self.linkage_models[table_name]

                # Sample Counts
                counts = linkage.sample_counts(parent_df, rng=rng)

                # Construct Context from Driver
                parent_ids_repeated = np.repeat(
                    parent_df[driver_parent_pk].to_numpy(), counts
                )

                context_cols = config.parent_context_cols
                if context_cols:
                    context_repeated_vals = {}
                    for col in context_cols:
                        context_repeated_vals[col] = np.repeat(
                            parent_df[col].to_numpy(), counts
                        )
                    context_df = pd.DataFrame(context_repeated_vals)
                else:
                    context_df = None

                total_child_rows = len(parent_ids_repeated)

                # 2. Generate Data
                if total_child_rows > 0:
                    generated_pdf = model.sample(
                        total_child_rows, context=context_df, seed=table_seed
                    )

                    # Assign Driver FK
                    generated_pdf[driver_fk] = parent_ids_repeated

                    # Assign Secondary FKs (Random Sampling from respective Parents)
                    for fk_col in sorted(pk_map):
                        if fk_col == driver_fk:
                            continue
                        p_table, p_pk = parse_fk_ref(pk_map[fk_col])

                        valid_pks = _read_parent(p_table)[p_pk].to_numpy()

                        # Randomly sample valid PKs for this column
                        generated_pdf[fk_col] = rng.choice(
                            valid_pks, size=total_child_rows
                        )

                    # Assign PKs
                    generated_pdf[config.pk] = range(1, len(generated_pdf) + 1)
                else:
                    # Zero child rows: create empty DataFrame with the real
                    # training schema so downstream grandchild tables (and
                    # consumers expecting PK/FK columns) don't crash.
                    log.info(
                        "zero_child_rows",
                        table=table_name,
                        driver_parent=driver_parent_table,
                    )
                    train_columns = self._table_schemas.get(table_name) or [config.pk]
                    generated_pdf = pd.DataFrame(columns=train_columns)

            if generated_pdf is not None:
                if output_path_base:
                    output_path = f"{output_path_base}/{table_name}"
                    _write_with_failure_policy(
                        io=self.io,
                        pdf=generated_pdf,
                        path=output_path,
                        policy=self.on_write_failure,
                        written_paths=written_paths,
                        format=output_format,
                    )
                    log.debug(
                        "table_released_from_memory", table=table_name, path=output_path
                    )
                    # Do NOT store in generated_tables — child tables read from disk via output_path_base
                else:
                    generated_tables[table_name] = generated_pdf

        return generated_tables
