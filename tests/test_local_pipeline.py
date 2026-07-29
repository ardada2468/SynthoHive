"""End-to-end pipeline tests with no Spark: the LocalIO pandas-native path."""

import numpy as np
import pandas as pd
import pytest

from syntho_hive import Metadata, PrivacyConfig
from syntho_hive.exceptions import SchemaError, SchemaValidationError
from syntho_hive.interface.synthesizer import Synthesizer


@pytest.fixture()
def metadata():
    meta = Metadata()
    meta.add_table("users", pk="user_id", pii_cols=["email"])
    meta.add_table(
        "orders",
        pk="order_id",
        fk={"user_id": "users.user_id"},
        parent_context_cols=["region"],
    )
    return meta


@pytest.fixture()
def real_data():
    rng = np.random.default_rng(0)
    users = pd.DataFrame(
        {
            "user_id": range(1, 51),
            "email": [f"user{i}@example.com" for i in range(50)],
            "age": rng.integers(18, 80, 50),
            "region": rng.choice(["NA", "EU", "APAC"], 50),
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": range(1, 201),
            "user_id": rng.integers(1, 51, 200),
            "amount": rng.normal(100, 25, 200).round(2),
            "region": rng.choice(["NA", "EU", "APAC"], 200),
        }
    )
    return {"users": users, "orders": orders}


def _fit(metadata, real_data, **kwargs):
    synth = Synthesizer(metadata=metadata, privacy_config=PrivacyConfig())
    synth.fit(
        real_data,
        validate=True,
        epochs=2,
        batch_size=32,
        progress_bar=False,
        seed=7,
        **kwargs,
    )
    return synth


def test_fit_sample_without_spark(metadata, real_data):
    synth = _fit(metadata, real_data)
    out = synth.sample({"users": 30}, seed=11)

    assert set(out) == {"users", "orders"}
    assert len(out["users"]) == 30
    # FK integrity: every child FK points at a generated parent PK.
    assert set(out["orders"]["user_id"]).issubset(set(out["users"]["user_id"]))


def test_pii_never_reaches_training_data(metadata, real_data):
    synth = _fit(metadata, real_data)
    out = synth.sample({"users": 30}, seed=11)
    real_emails = set(real_data["users"]["email"])
    assert not real_emails.intersection(set(out["users"]["email"].dropna()))


def test_seeded_sampling_is_reproducible(metadata, real_data):
    synth = _fit(metadata, real_data)
    a = synth.sample({"users": 20}, seed=5)
    b = synth.sample({"users": 20}, seed=5)
    for table in a:
        pd.testing.assert_frame_equal(a[table], b[table])


def test_save_load_roundtrip_without_spark(metadata, real_data, tmp_path):
    synth = _fit(metadata, real_data)
    path = str(tmp_path / "synth.pkl")
    synth.save(path)
    loaded = Synthesizer.load(path)
    out = loaded.sample({"users": 10}, seed=3)
    assert len(out["users"]) == 10


def test_sample_to_disk_parquet_roundtrip(metadata, real_data, tmp_path):
    synth = _fit(metadata, real_data)
    out_dir = str(tmp_path / "synthetic")
    paths = synth.sample({"users": 15}, output_path=out_dir, seed=1)
    assert set(paths) == {"users", "orders"}
    users = pd.read_parquet(paths["users"])
    assert len(users) == 15
    # Validation report over the written files, still without Spark.
    report_path = str(tmp_path / "report.html")
    synth.generate_validation_report(
        real_data={"users": real_data["users"]},
        synthetic_data={"users": paths["users"]},
        output_path=report_path,
    )
    with open(report_path) as f:
        assert "users" in f.read()


def test_missing_root_num_rows_raises(metadata, real_data):
    synth = _fit(metadata, real_data)
    with pytest.raises(SchemaError, match="root table"):
        synth.sample({})


def test_missing_table_data_raises_at_fit(metadata, real_data):
    synth = Synthesizer(metadata=metadata, privacy_config=PrivacyConfig())
    with pytest.raises(SchemaError, match="No data source"):
        synth.fit(
            {"users": real_data["users"]},
            epochs=1,
            batch_size=16,
            progress_bar=False,
        )


def test_unimplemented_sampling_strategy_raises(metadata, real_data):
    synth = Synthesizer(metadata=metadata, privacy_config=PrivacyConfig())
    with pytest.raises(NotImplementedError):
        synth.fit(real_data, sampling_strategy="relational_stratified")


def test_differential_privacy_fails_loudly(metadata, real_data):
    synth = Synthesizer(
        metadata=metadata,
        privacy_config=PrivacyConfig(enable_differential_privacy=True),
    )
    with pytest.raises(NotImplementedError, match="Differential privacy"):
        synth.fit(real_data, epochs=1, batch_size=16, progress_bar=False)


def test_cyclic_schema_rejected():
    meta = Metadata()
    meta.add_table("a", pk="id", fk={"b_id": "b.id"})
    meta.add_table("b", pk="id", fk={"a_id": "a.id"})
    with pytest.raises(SchemaValidationError, match="cycle"):
        meta.validate_schema()


def test_self_referencing_fk_rejected():
    meta = Metadata()
    meta.add_table("emp", pk="id", fk={"manager_id": "emp.id"})
    with pytest.raises(SchemaValidationError, match="[Ss]elf-referenc"):
        meta.validate_schema()


def test_malformed_fk_ref_rejected():
    meta = Metadata()
    meta.add_table("p", pk="id")
    meta.add_table("c", pk="id", fk={"p_id": "no_dot_ref"})
    with pytest.raises(SchemaValidationError, match="Invalid FK reference"):
        meta.validate_schema()
