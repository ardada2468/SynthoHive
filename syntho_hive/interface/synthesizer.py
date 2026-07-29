from typing import Dict, Optional, Any, Union, Tuple, Type
import re
import pandas as pd
import structlog
from syntho_hive.interface.config import Metadata, PrivacyConfig
from syntho_hive.relational.orchestrator import StagedOrchestrator
from syntho_hive.validation.report_generator import ValidationReport
from syntho_hive.exceptions import (
    SynthoHiveError,
    SchemaError,
    TrainingError,
    SerializationError,
)
from syntho_hive.core.models.base import ConditionalGenerativeModel
from syntho_hive.core.models.ctgan import CTGAN

# Allowlist regex for Hive/SQL identifier validation.
# Only letters, digits, and underscores are permitted — everything else is rejected
# before any spark.sql() interpolation occurs, preventing SQL injection via user input.
_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_]+$")

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = Any

log = structlog.get_logger()


class Synthesizer:
    """Main entry point that wires metadata, privacy, and orchestration."""

    def __init__(
        self,
        metadata: Metadata,
        privacy_config: PrivacyConfig,
        spark_session: Optional[SparkSession] = None,
        model: Type[ConditionalGenerativeModel] = CTGAN,
        embedding_threshold: int = 50,
    ):
        """Instantiate the synthesizer façade.

        Args:
            metadata: Dataset schema and relational configuration.
            privacy_config: Privacy guardrail configuration.
            spark_session: Optional SparkSession required for orchestration.
            model: Generative model class to use for synthesis. Must be a class
                (not an instance) that implements ``ConditionalGenerativeModel``.
                The class constructor must accept ``(metadata, batch_size, epochs,
                **kwargs)`` and instances must implement ``fit()``, ``sample()``,
                ``save()``, and ``load()``.

                Supported classes:
                - ``syntho_hive.core.models.ctgan.CTGAN`` (default)
                - Any custom class implementing ``ConditionalGenerativeModel``

                Existing callers that omit this parameter receive CTGAN behavior
                unchanged.
            embedding_threshold: Cardinality threshold for switching to embeddings.
        """
        if not (
            isinstance(model, type) and issubclass(model, ConditionalGenerativeModel)
        ):
            raise TypeError(
                f"model must be a subclass of ConditionalGenerativeModel, "
                f"got {model!r}. Implement fit(), sample(), save(), load() "
                f"and subclass ConditionalGenerativeModel."
            )

        self.metadata = metadata
        self.privacy = privacy_config
        self.spark = spark_session
        self.model_cls = model
        self.embedding_threshold = embedding_threshold

        # Initialize internal components. Without a SparkSession the
        # orchestrator falls back to the pandas-native LocalIO backend, so the
        # full pipeline works on a single machine with no Spark installed.
        self.orchestrator = StagedOrchestrator(
            metadata,
            self.spark,
            model_cls=self.model_cls,
            privacy_config=privacy_config,
        )

    def fit(
        self,
        data: Any,  # Str (database name) or Dict[str, str|pd.DataFrame]
        sampling_strategy: str = "full",
        sample_size: int = 5_000_000,
        validate: bool = False,
        epochs: int = 300,
        batch_size: int = 500,
        progress_bar: bool = True,
        checkpoint_interval: int = 10,
        checkpoint_dir: Optional[str] = None,
        seed: Optional[int] = None,
        **model_kwargs: Union[int, str, Tuple[int, int]],
    ):
        """Fit the generative models on the real database.

        Args:
            data: Database name (str), or a mapping of table name to a
                path/table identifier or an in-memory pandas DataFrame.
            sampling_strategy: Strategy for sampling real data. Only ``"full"``
                (the default) is currently implemented.
            sample_size: Number of rows to sample from real data (approx).
            validate: Whether to run validation after fitting.
            epochs: Number of training epochs for CTGAN.
            batch_size: Batch size for training.
            progress_bar: If True (default), display tqdm progress bar to stderr during training.
                Structured log events always emit regardless of this flag.
            checkpoint_interval: Save a validation checkpoint every N epochs. Default 10.
            checkpoint_dir: Optional directory to save best_checkpoint/ and final_checkpoint/
                during training.
            seed: Optional integer seed for deterministic training across all tables.
            **model_kwargs: Additional args forwarded to the underlying model (e.g., embedding_dim).

        Raises:
            SchemaError: If the data argument is invalid.
            NotImplementedError: If an unimplemented sampling strategy or
                differential privacy is requested.
            TrainingError: If training fails for any reason.
        """
        if sampling_strategy != "full":
            raise NotImplementedError(
                f"sampling_strategy='{sampling_strategy}' is not implemented yet; "
                f"use 'full'."
            )

        if self.privacy is not None and self.privacy.enable_differential_privacy:
            raise NotImplementedError(
                "Differential privacy is not implemented yet. Set "
                "enable_differential_privacy=False (PII sanitization via "
                "pii_strategy still applies)."
            )

        try:
            if validate:
                if (
                    isinstance(data, dict)
                    and data
                    and isinstance(next(iter(data.values())), pd.DataFrame)
                ):
                    # User passed actual DataFrames — data-level FK type checks are possible
                    self.metadata.validate_schema(real_data=data)
                else:
                    # String (DB name) or dict of path strings — structural checks only
                    self.metadata.validate_schema()

            if sample_size <= 0:
                raise ValueError("sample_size must be positive")
            if epochs <= 0:
                raise ValueError("epochs must be positive")
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")

            log.info(
                "fit_start",
                sampling_strategy=sampling_strategy,
                target_rows=sample_size,
                epochs=epochs,
                batch_size=batch_size,
            )

            # Determine paths
            if isinstance(data, str):
                real_paths = {t: f"{data}.{t}" for t in self.metadata.tables}
            elif isinstance(data, dict):
                real_paths = data
            else:
                raise SchemaError(
                    f"fit() argument 'data' must be a database name (str) or path mapping (dict), "
                    f"got {type(data).__name__}."
                )

            self.orchestrator.fit_all(
                real_paths,
                epochs=epochs,
                batch_size=batch_size,
                progress_bar=progress_bar,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
                seed=seed,
                **model_kwargs,
            )
        except (SynthoHiveError, NotImplementedError):
            raise
        except Exception as exc:
            log.error("fit_failed", error=str(exc))
            raise TrainingError(f"fit() failed. Original error: {exc}") from exc

    def sample(
        self,
        num_rows: Dict[str, int],
        output_format: str = "parquet",
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Union[Dict[str, str], Dict[str, pd.DataFrame]]:
        """Generate synthetic data for each table.

        Args:
            num_rows: Mapping of root table name to number of rows to generate.
                Child table volumes are driven by the fitted cardinality model.
            output_format: Storage format for generated datasets (default
                ``"parquet"``). Ignored when ``output_path`` is None.
            output_path: Optional path to write files. If None, returns DataFrames in memory.
            seed: Optional integer seed making generation reproducible.

        Raises:
            GenerationError: If called before fit()/load().
            TrainingError: If generation fails for any other reason.

        Returns:
            Mapping of table name to the output path (if wrote to disk) OR Dictionary of DataFrames (if in-memory).
        """
        try:
            for table, n in num_rows.items():
                if not isinstance(n, int) or n < 0:
                    raise ValueError(
                        f"num_rows['{table}'] must be a non-negative int, got {n!r}"
                    )

            log.info("sample_start", model=self.model_cls.__name__)

            # If output_path is explicitly None, we return DataFrames
            if output_path is None:
                return self.orchestrator.generate(
                    num_rows, output_path_base=None, seed=seed
                )

            output_base = output_path.rstrip("/")
            self.orchestrator.generate(
                num_rows, output_base, seed=seed, output_format=output_format
            )

            # Return paths mapping
            return {t: f"{output_base}/{t}" for t in self.metadata.tables}
        except SynthoHiveError:
            raise
        except Exception as exc:
            log.error("sample_failed", error=str(exc))
            raise TrainingError(f"sample() failed. Original error: {exc}") from exc

    def save(self, path: str) -> None:
        """Persist the synthesizer state to disk.

        Args:
            path: Filesystem path to write the synthesizer checkpoint to.

        Raises:
            SerializationError: If saving fails for any reason.
        """
        try:
            import joblib

            joblib.dump(self, path)
            log.info("synthesizer_saved", path=path)
        except SynthoHiveError:
            raise
        except Exception as exc:
            log.error("save_failed", path=path, error=str(exc))
            raise SerializationError(
                f"save() failed writing synthesizer to '{path}'. Original error: {exc}"
            ) from exc

    def __getstate__(self):
        """Exclude non-serializable attributes (SparkSession, IO) from pickling."""
        state = self.__dict__.copy()
        state.pop("spark", None)
        state.pop("io", None)
        return state

    def __setstate__(self, state):
        """Restore instance from pickled state; Spark handles reset to None."""
        self.__dict__.update(state)
        self.spark = None

    @classmethod
    def load(cls, path: str, spark_session: Optional[SparkSession] = None) -> "Synthesizer":
        """Load a synthesizer from a previously saved checkpoint.

        Args:
            path: Filesystem path to the synthesizer checkpoint.
            spark_session: Optional SparkSession to re-attach to the loaded
                instance. Without one, the loaded synthesizer uses the
                pandas-native LocalIO backend.

        Raises:
            SerializationError: If loading fails for any reason.

        Returns:
            Loaded Synthesizer instance.
        """
        try:
            import joblib

            instance = joblib.load(path)
            instance.spark = spark_session
            if getattr(instance, "orchestrator", None) is not None:
                instance.orchestrator.bind_spark(spark_session)
            log.info("synthesizer_loaded", path=path)
            return instance
        except SynthoHiveError:
            raise
        except Exception as exc:
            log.error("load_failed", path=path, error=str(exc))
            raise SerializationError(
                f"load() failed reading synthesizer from '{path}'. Original error: {exc}"
            ) from exc

    def generate_validation_report(
        self,
        real_data: Dict[str, str],
        synthetic_data: Dict[str, str],
        output_path: str,
    ):
        """Generate a validation report comparing real vs synthetic datasets.

        Args:
            real_data: Map of table name to real dataset path/table.
            synthetic_data: Map of table name to generated dataset path.
            output_path: Filesystem path for the rendered report.

        Raises:
            SynthoHiveError: If the report generation fails for any reason.
        """
        try:
            log.info("validation_report_start", output_path=output_path)
            report_gen = ValidationReport()

            io = self.orchestrator.io

            real_dfs = {}
            synth_dfs = {}

            # 1. Load Real Data (same IO backend/format as the training reads)
            for table, path in real_data.items():
                log.info("loading_real_data", table=table, path=str(path))
                real_dfs[table] = io.read_pandas(path)

            # 2. Load Synthetic Data (same IO backend/format sample() wrote)
            for table, path in synthetic_data.items():
                log.info("loading_synthetic_data", table=table, path=str(path))
                synth_dfs[table] = io.read_pandas(path)

            # 3. Generate Report
            report_gen.generate(real_dfs, synth_dfs, output_path)
        except SynthoHiveError:
            raise
        except Exception as exc:
            log.error(
                "generate_validation_report_failed",
                output_path=output_path,
                error=str(exc),
            )
            raise SynthoHiveError(
                f"generate_validation_report() failed. Original error: {exc}"
            ) from exc

    def save_to_hive(
        self,
        synthetic_data: Dict[str, str],
        target_db: str,
        overwrite: bool = True,
        table_format: str = "parquet",
    ):
        """Register generated datasets as Hive tables.

        Args:
            synthetic_data: Map of table name to generated dataset path.
            target_db: Hive database where tables should be registered.
            overwrite: Whether to drop and recreate existing tables.
            table_format: Storage format the datasets were written in
                (``"parquet"`` — the default written by ``sample()`` — or
                ``"delta"``). Must match the actual on-disk format.

        Raises:
            ValueError: If Spark is unavailable or the format is unsupported.
        """
        if not self.spark:
            raise ValueError("SparkSession required for Hive registration")

        if table_format.lower() not in ("parquet", "delta"):
            raise ValueError(
                f"table_format must be 'parquet' or 'delta', got '{table_format}'"
            )

        # Validate database name against allowlist before any SQL interpolation.
        # Raises SchemaError immediately — no Spark context touched for invalid names.
        if not _SAFE_IDENTIFIER.match(target_db):
            raise SchemaError(
                f"Database name '{target_db}' contains invalid characters. "
                f"Only letters, digits, and underscores [a-zA-Z0-9_] are allowed. "
                f"This validation prevents SQL injection via unsanitized user input."
            )

        # Validate table names from synthetic_data keys
        for table_name in synthetic_data:
            if not _SAFE_IDENTIFIER.match(str(table_name)):
                raise SchemaError(
                    f"Table name '{table_name}' contains invalid characters. "
                    f"Only letters, digits, and underscores [a-zA-Z0-9_] are allowed."
                )

        # Validate paths from synthetic_data values
        for table_name, path in synthetic_data.items():
            if "'" in str(path):
                raise ValueError(
                    f"Path for table '{table_name}' contains invalid characters: {path}"
                )

        log.info("save_to_hive_start", database=target_db)

        # Ensure DB exists
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {target_db}")

        for table, path in synthetic_data.items():
            full_table_name = f"{target_db}.{table}"
            log.info("registering_table", table=full_table_name, path=path)

            if overwrite:
                self.spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")

            # Register External Table (format must match what sample() wrote)
            self.spark.sql(
                f"CREATE TABLE {full_table_name} "
                f"USING {table_format.upper()} LOCATION '{path}'"
            )
