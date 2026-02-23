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
_SAFE_IDENTIFIER = re.compile(r'^[a-zA-Z0-9_]+$')

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
        embedding_threshold: int = 50
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
        if not (isinstance(model, type) and issubclass(model, ConditionalGenerativeModel)):
            raise TypeError(
                f"model_cls must be a subclass of ConditionalGenerativeModel, "
                f"got {model!r}. Implement fit(), sample(), save(), load() "
                f"and subclass ConditionalGenerativeModel."
            )

        self.metadata = metadata
        self.privacy = privacy_config
        self.spark = spark_session
        self.model_cls = model
        self.embedding_threshold = embedding_threshold

        # Initialize internal components
        if self.spark:
            self.orchestrator = StagedOrchestrator(metadata, self.spark, model_cls=self.model_cls)
        else:
            self.orchestrator = None # Mode without Spark (maybe local pandas only in future)

    def fit(
        self,
        data: Any, # Str (database name) or Dict[str, str] (table paths)
        sampling_strategy: str = "relational_stratified",
        sample_size: int = 5_000_000,
        validate: bool = False,
        epochs: int = 300,
        batch_size: int = 500,
        **model_kwargs: Union[int, str, Tuple[int, int]]
    ):
        """Fit the generative models on the real database.

        Args:
            data: Database name (str) or mapping of {table: path} (dict).
            sampling_strategy: Strategy for sampling real data.
            sample_size: Number of rows to sample from real data (approx).
            validate: Whether to run validation after fitting.
            epochs: Number of training epochs for CTGAN.
            batch_size: Batch size for training.
            **model_kwargs: Additional args forwarded to the underlying model (e.g., embedding_dim).

        Raises:
            SchemaError: If the data argument is invalid.
            TrainingError: If training fails for any reason.
        """
        try:
            if validate:
                if isinstance(data, dict) and data and isinstance(next(iter(data.values())), pd.DataFrame):
                    # User passed actual DataFrames — data-level FK type checks are possible
                    self.metadata.validate_schema(real_data=data)
                else:
                    # String (DB name) or dict of path strings — structural checks only
                    self.metadata.validate_schema()

            if not self.orchestrator:
                raise ValueError("SparkSession required for fit()")

            if sample_size <= 0:
                raise ValueError("sample_size must be positive")

            print(f"Fitting on data source with {sampling_strategy} (target: {sample_size} rows)...")
            print(f"Training Config: epochs={epochs}, batch_size={batch_size}")

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

            self.orchestrator.fit_all(real_paths, epochs=epochs, batch_size=batch_size, **model_kwargs)
        except SynthoHiveError:
            raise
        except Exception as exc:
            log.error("fit_failed", error=str(exc))
            raise TrainingError(
                f"fit() failed. Original error: {exc}"
            ) from exc

    def sample(self, num_rows: Dict[str, int], output_format: str = "delta", output_path: Optional[str] = None) -> Union[Dict[str, str], Dict[str, pd.DataFrame]]:
        """Generate synthetic data for each table.

        Args:
            num_rows: Mapping of table name to number of rows to generate.
            output_format: Storage format for generated datasets (default ``"delta"``).
            output_path: Optional path to write files. If None, returns DataFrames in memory.

        Raises:
            TrainingError: If generation fails for any reason.

        Returns:
            Mapping of table name to the output path (if wrote to disk) OR Dictionary of DataFrames (if in-memory).
        """
        try:
            if not self.orchestrator:
                raise ValueError("SparkSession required for sample()")

            print(f"Generating data with {self.model_cls.__name__} backend...")

            # If output_path is explicitly None, we return DataFrames
            if output_path is None:
                 return self.orchestrator.generate(num_rows, output_path_base=None)

            output_base = output_path
            self.orchestrator.generate(num_rows, output_base)

            # Return paths mapping
            return {t: f"{output_base}/{t}" for t in self.metadata.tables}
        except SynthoHiveError:
            raise
        except Exception as exc:
            log.error("sample_failed", error=str(exc))
            raise TrainingError(
                f"sample() failed. Original error: {exc}"
            ) from exc

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

    @classmethod
    def load(cls, path: str) -> "Synthesizer":
        """Load a synthesizer from a previously saved checkpoint.

        Args:
            path: Filesystem path to the synthesizer checkpoint.

        Raises:
            SerializationError: If loading fails for any reason.

        Returns:
            Loaded Synthesizer instance.
        """
        try:
            import joblib
            instance = joblib.load(path)
            log.info("synthesizer_loaded", path=path)
            return instance
        except SynthoHiveError:
            raise
        except Exception as exc:
            log.error("load_failed", path=path, error=str(exc))
            raise SerializationError(
                f"load() failed reading synthesizer from '{path}'. Original error: {exc}"
            ) from exc

    def generate_validation_report(self, real_data: Dict[str, str], synthetic_data: Dict[str, str], output_path: str):
        """Generate a validation report comparing real vs synthetic datasets.

        Args:
            real_data: Map of table name to real dataset path/table.
            synthetic_data: Map of table name to generated dataset path.
            output_path: Filesystem path for the rendered report.

        Raises:
            SynthoHiveError: If the report generation fails for any reason.
        """
        try:
            if not self.spark:
                 raise ValueError("SparkSession required for validation report generation")

            print("Generating validation report...")
            report_gen = ValidationReport()

            real_dfs = {}
            synth_dfs = {}

            # 1. Load Real Data
            for table, path in real_data.items():
                print(f"Loading real data for {table} from {path}...")
                # Try reading as table first, then path
                try:
                    df = self.spark.read.table(path)
                except Exception as exc:
                    log.warning("delta_read_fallback_failed", error=str(exc))
                    raise SerializationError(
                        f"generate_validation_report() failed reading synthetic data. "
                        f"Original error: {exc}"
                    ) from exc

                real_dfs[table] = df.toPandas()

            # 2. Load Synthetic Data
            for table, path in synthetic_data.items():
                print(f"Loading synthetic data for {table} from {path}...")
                df = self.spark.read.format("delta").load(path)
                synth_dfs[table] = df.toPandas()

            # 3. Generate Report
            report_gen.generate(real_dfs, synth_dfs, output_path)
        except SynthoHiveError:
            raise
        except Exception as exc:
            log.error("generate_validation_report_failed", output_path=output_path, error=str(exc))
            raise SynthoHiveError(
                f"generate_validation_report() failed. Original error: {exc}"
            ) from exc

    def save_to_hive(self, synthetic_data: Dict[str, str], target_db: str, overwrite: bool = True):
        """Register generated datasets as Hive tables.

        Args:
            synthetic_data: Map of table name to generated dataset path.
            target_db: Hive database where tables should be registered.
            overwrite: Whether to drop and recreate existing tables.

        Raises:
            ValueError: If Spark is unavailable.
        """
        if not self.spark:
            raise ValueError("SparkSession required for Hive registration")

        # Validate database name against allowlist before any SQL interpolation.
        # Raises SchemaError immediately — no Spark context touched for invalid names.
        if not _SAFE_IDENTIFIER.match(target_db):
            raise SchemaError(
                f"SchemaError: Database name '{target_db}' contains invalid characters. "
                f"Only letters, digits, and underscores [a-zA-Z0-9_] are allowed. "
                f"This validation prevents SQL injection via unsanitized user input."
            )

        # Validate table names from synthetic_data keys
        for table_name in synthetic_data:
            if not _SAFE_IDENTIFIER.match(str(table_name)):
                raise SchemaError(
                    f"SchemaError: Table name '{table_name}' contains invalid characters. "
                    f"Only letters, digits, and underscores [a-zA-Z0-9_] are allowed."
                )

        print(f"Save to Hive database: {target_db}")

        # Ensure DB exists
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {target_db}")

        for table, path in synthetic_data.items():
            full_table_name = f"{target_db}.{table}"
            print(f"Registering table {full_table_name} at {path}")

            if overwrite:
                self.spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")

            # Register External Table
            self.spark.sql(f"CREATE TABLE {full_table_name} USING DELTA LOCATION '{path}'")
