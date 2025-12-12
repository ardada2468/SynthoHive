import pytest
from syntho_hive import Metadata, PrivacyConfig
from syntho_hive.core.models.base import GenerativeModel, ConditionalGenerativeModel

def test_imports():
    """Test that core components can be imported."""
    assert Metadata is not None
    assert PrivacyConfig is not None
    assert GenerativeModel is not None

def test_metadata_configuration():
    """Test that metadata can be configured programmatically."""
    meta = Metadata()
    meta.add_table(
        name="users",
        pk="user_id",
        pii_cols=["email"],
        high_cardinality_cols=["city"]
    )
    
    table = meta.get_table("users")
    assert table.name == "users"
    assert table.pk == "user_id"
    assert "email" in table.pii_cols
    assert "city" in table.high_cardinality_cols

def test_privacy_config_defaults():
    """Test default values for privacy config."""
    config = PrivacyConfig()
    assert config.enable_differential_privacy is False
    assert config.epsilon == 1.0
    assert config.pii_strategy == "context_aware_faker"
