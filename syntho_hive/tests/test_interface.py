
import pytest
from unittest.mock import MagicMock, patch
from syntho_hive.interface.synthesizer import Synthesizer
from syntho_hive.interface.config import Metadata, PrivacyConfig, TableConfig

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
    with pytest.raises(ValueError, match="references non-existent parent table"):
        metadata.validate_schema()

def test_metadata_invalid_fk_format(metadata):
    with pytest.raises(ValueError, match="Invalid FK reference"):
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
