
import pytest
from unittest.mock import MagicMock, patch
from syntho_hive.interface.synthesizer import Synthesizer
from syntho_hive.interface.config import Metadata, PrivacyConfig, TableConfig
from syntho_hive.exceptions import SchemaValidationError

# Mock SparkSession
class MockSparkSession:
    def __init__(self):
        self.read = MagicMock()
        self.sql = MagicMock()

@pytest.fixture
def mock_spark():
    return MockSparkSession()

@pytest.fixture
def metadata():
    m = Metadata()
    m.add_table("users", "user_id")
    m.add_table("orders", "order_id", fk={"user_id": "users.user_id"})
    return m

@pytest.fixture
def privacy_config():
    return PrivacyConfig()

def test_metadata_validation(metadata):
    # Valid schema
    metadata.validate_schema()
    
    # Invalid parent table
    metadata.add_table("items", "item_id", fk={"order_id": "invalid_table.order_id"})
    with pytest.raises(SchemaValidationError, match="references non-existent parent table"):
        metadata.validate_schema()

def test_metadata_invalid_fk_format(metadata):
    with pytest.raises(SchemaValidationError, match="Invalid FK reference"):
        metadata.add_table("logs", "log_id", fk={"user_id": "users_user_id"}) # Missing dot
        metadata.validate_schema()

def test_synthesizer_init_no_spark(metadata, privacy_config):
    syn = Synthesizer(metadata, privacy_config, spark_session=None)
    assert syn.orchestrator is None

def test_synthesizer_fit_requires_spark(metadata, privacy_config):
    syn = Synthesizer(metadata, privacy_config, spark_session=None)
    with pytest.raises(ValueError, match="SparkSession required"):
        syn.fit("test_db")

def test_synthesizer_sample_requires_spark(metadata, privacy_config):
    syn = Synthesizer(metadata, privacy_config, spark_session=None)
    with pytest.raises(ValueError, match="SparkSession required"):
        syn.sample({"users": 100})

def test_synthesizer_fit_call(mock_spark, metadata, privacy_config):
    with patch("syntho_hive.interface.synthesizer.StagedOrchestrator") as MockOrchestrator:
        syn = Synthesizer(metadata, privacy_config, spark_session=mock_spark)
        syn.fit("test_db", sample_size=100)
        
        # Check if orchestrator.fit_all was called
        syn.orchestrator.fit_all.assert_called_once()
        # Check args passed to fit_all are correct
        expected_paths = {'users': 'test_db.users', 'orders': 'test_db.orders'}
        syn.orchestrator.fit_all.assert_called_with(expected_paths)

def test_synthesizer_sample_call(mock_spark, metadata, privacy_config):
    with patch("syntho_hive.interface.synthesizer.StagedOrchestrator") as MockOrchestrator:
        syn = Synthesizer(metadata, privacy_config, spark_session=mock_spark)
        syn.sample({"users": 50})
        
        syn.orchestrator.generate.assert_called_once()
        # Verify output path
        args, _ = syn.orchestrator.generate.call_args
        rows, output_base = args
        assert rows == {"users": 50}
        assert "/tmp/syntho_hive_output/delta" == output_base

def test_save_to_hive(mock_spark, metadata, privacy_config):
    syn = Synthesizer(metadata, privacy_config, spark_session=mock_spark)
    synthetic_data = {"users": "/tmp/users", "orders": "/tmp/orders"}

    syn.save_to_hive(synthetic_data, "synth_db")

    # Verify SQL calls
    calls = mock_spark.sql.call_args_list
    # Should create DB
    assert "CREATE DATABASE IF NOT EXISTS synth_db" in str(calls[0])
    # Should drop/create tables
    # Check for at least one CREATE TABLE call
    create_calls = [c for c in calls if "CREATE TABLE synth_db.users" in str(c) or "CREATE TABLE synth_db.orders" in str(c)]
    assert len(create_calls) >= 2


# ---------------------------------------------------------------------------
# MODEL-03: ConditionalGenerativeModel ABC contract — stub model integration
# ---------------------------------------------------------------------------

import pandas as pd
from syntho_hive.core.models.base import ConditionalGenerativeModel
from syntho_hive.relational.orchestrator import StagedOrchestrator
from syntho_hive.interface.config import Metadata
from unittest.mock import MagicMock


class StubModel(ConditionalGenerativeModel):
    """Minimal stub satisfying ConditionalGenerativeModel without training.

    Stores columns seen during fit(). Returns a zero-filled DataFrame of
    correct shape on sample(). save() and load() are no-ops.
    """

    def __init__(self, metadata, batch_size=500, epochs=300, **kwargs):
        self._columns = []

    def fit(self, data: pd.DataFrame, context=None, table_name=None, **kwargs) -> None:
        self._columns = list(data.columns)

    def sample(self, num_rows: int, context=None, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({col: [0] * num_rows for col in self._columns})

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


class _MockSparkDF:
    """Thin wrapper so toPandas() works in tests without a real Spark session."""
    def __init__(self, pdf: pd.DataFrame):
        self.pdf = pdf

    def toPandas(self) -> pd.DataFrame:
        return self.pdf


def test_stub_model_routes_through_pipeline(tmp_path):
    """MODEL-03: StubModel routes correctly through full orchestration pipeline.

    Verifies that StagedOrchestrator never calls CTGAN() when model_cls=StubModel
    and that the generate() result contains a DataFrame with the expected shape.
    """
    meta = Metadata()
    meta.add_table("users", pk="id")

    users_df = pd.DataFrame({"id": range(5), "age": range(20, 25)})

    mock_io = MagicMock()
    mock_io.read_dataset.return_value = _MockSparkDF(users_df)

    def _write_pandas(pdf, path, **kw):
        out = tmp_path / "users"
        out.mkdir(parents=True, exist_ok=True)
        pdf.to_csv(out / "data.csv", index=False)

    mock_io.write_pandas.side_effect = _write_pandas

    orch = StagedOrchestrator(metadata=meta, io=mock_io, model_cls=StubModel)
    orch.fit_all({"users": "path/users"}, epochs=1, batch_size=5)

    result = orch.generate({"users": 3})

    assert "users" in result, "generate() must return 'users' table"
    assert len(result["users"]) == 3, f"Expected 3 rows, got {len(result['users'])}"

    # MODEL-01 negative check: no CTGAN instance stored — only StubModel
    for table_name, model_instance in orch.models.items():
        assert isinstance(model_instance, StubModel), (
            f"StagedOrchestrator stored {type(model_instance).__name__} "
            f"for table '{table_name}' despite model_cls=StubModel"
        )
        assert type(model_instance).__name__ != "CTGAN", (
            "StagedOrchestrator stored a CTGAN instance despite model_cls=StubModel"
        )

    print("test_stub_model_routes_through_pipeline: PASSED")


def test_synthesizer_accepts_model_parameter(metadata, privacy_config):
    """MODEL-02: Synthesizer accepts model= parameter and stores it as model_cls."""
    from syntho_hive.interface.synthesizer import Synthesizer
    syn = Synthesizer(metadata, privacy_config, spark_session=None, model=StubModel)
    assert syn.model_cls is StubModel, (
        f"Expected syn.model_cls to be StubModel, got {syn.model_cls}"
    )


def test_synthesizer_default_model_is_ctgan(metadata, privacy_config):
    """MODEL-02: Synthesizer with no model= argument defaults to CTGAN."""
    from syntho_hive.interface.synthesizer import Synthesizer
    from syntho_hive.core.models.ctgan import CTGAN
    syn = Synthesizer(metadata, privacy_config, spark_session=None)
    assert syn.model_cls is CTGAN, (
        f"Expected default model_cls=CTGAN, got {syn.model_cls}"
    )


def test_issubclass_guard_rejects_invalid_model_cls():
    """MODEL-01: StagedOrchestrator raises TypeError for non-ConditionalGenerativeModel class."""
    from syntho_hive.relational.orchestrator import StagedOrchestrator
    from syntho_hive.interface.config import Metadata

    class NotAModel:
        pass

    meta = Metadata()
    meta.add_table("t", pk="id")

    with pytest.raises(TypeError, match="ConditionalGenerativeModel"):
        StagedOrchestrator(metadata=meta, model_cls=NotAModel)


def test_synthesizer_rejects_invalid_model_cls_without_spark(metadata, privacy_config):
    """TD-04 fix: Synthesizer raises TypeError at __init__ even when spark_session=None.

    Before the fix, invalid model_cls was silently accepted when spark_session=None
    because the issubclass guard lived only inside StagedOrchestrator.__init__(),
    which is never called when no Spark session is provided.
    """
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="ConditionalGenerativeModel"):
        Synthesizer(metadata, privacy_config, spark_session=None, model=NotAModel)


def test_synthesizer_fit_validate_catches_fk_type_mismatch(metadata, privacy_config):
    """TD-01 fix: fit(validate=True, data=DataFrames) raises SchemaValidationError on FK mismatch.

    The metadata fixture defines users.user_id (PK) and orders.user_id (FK).
    Passing users_df with int user_id and orders_df with str user_id creates a
    dtype mismatch that validate_schema(real_data=dfs) detects.

    Before the fix, validate_schema() was called without real_data, so data-level
    FK type checks were silently skipped through the Synthesizer facade.
    """
    users_df = pd.DataFrame({"user_id": [1, 2, 3], "name": ["alice", "bob", "carol"]})
    orders_df = pd.DataFrame({"order_id": [10, 11], "user_id": ["1", "2"]})  # str FK — mismatch

    syn = Synthesizer(metadata, privacy_config, spark_session=None)
    with pytest.raises(SchemaValidationError):
        syn.fit(data={"users": users_df, "orders": orders_df}, validate=True)
