import numpy as np
import pandas as pd
import pytest

from syntho_hive.exceptions import PrivacyError
from syntho_hive.privacy.faker_contextual import ContextualFaker
from syntho_hive.privacy.sanitizer import PIISanitizer, PiiRule, PrivacyConfig


def test_contextual_faker_locale():
    faker = ContextualFaker()

    # Context with JP country
    jp_names = faker.generate_pii('name', context={'country': 'JP'}, count=5)
    # Check if names look Japanese (simplified check, check if non-latin? Assuming Faker produces romaji or kanji)
    # Faker ja_JP usually produces Kanji/Kana.
    # Let's just check they are strings and not empty.
    assert len(jp_names) == 5
    assert all(isinstance(n, str) for n in jp_names)

    # Context with US
    us_names = faker.generate_pii('name', context={'country': 'US'}, count=1)
    assert len(us_names) == 1


def test_pii_detection():
    data = pd.DataFrame({
        "user_email": ["a@b.com", "foo@bar.org"],
        "random_col": ["a", "b"]
    })

    sanitizer = PIISanitizer()
    detected = sanitizer.analyze(data)

    assert "user_email" in detected
    assert detected["user_email"] == "email"
    assert "random_col" not in detected


def test_sanitize_unknown_rule_raises():
    df = pd.DataFrame({"col": ["secret-value"]})
    sanitizer = PIISanitizer()

    with pytest.raises(PrivacyError, match="unknown rule"):
        sanitizer.sanitize(df, pii_map={"col": "not_a_rule"})


def test_sanitize_unknown_action_raises():
    config = PrivacyConfig(rules=[PiiRule(name="weird", patterns=[], action="scramble")])
    sanitizer = PIISanitizer(config=config)
    df = pd.DataFrame({"col": ["secret-value"]})

    with pytest.raises(PrivacyError, match="Unrecognized action"):
        sanitizer.sanitize(df, pii_map={"col": "weird"})


def test_sanitize_keep_action_leaves_column():
    config = PrivacyConfig(rules=[PiiRule(name="public_id", patterns=[], action="keep")])
    sanitizer = PIISanitizer(config=config)
    df = pd.DataFrame({"col": ["abc", "def"]})

    result = sanitizer.sanitize(df, pii_map={"col": "public_id"})
    assert list(result["col"]) == ["abc", "def"]


def test_date_of_birth_fully_masked():
    sanitizer = PIISanitizer()
    df = pd.DataFrame({"dob": ["01/15/1990", "1985-03-02"]})

    result = sanitizer.sanitize(df, pii_map={"dob": "date_of_birth"})

    assert list(result["dob"]) == ["*" * 10, "*" * 10]
    # No fragment of the original dates may survive.
    assert not result["dob"].str.contains(r"\d").any()


def test_ssn_fully_masked():
    sanitizer = PIISanitizer()
    df = pd.DataFrame({"ssn": ["123-45-6789"]})

    result = sanitizer.sanitize(df, pii_map={"ssn": "ssn"})
    assert list(result["ssn"]) == ["*" * 11]


def test_mask_preserve_suffix_opt_in():
    config = PrivacyConfig(
        rules=[PiiRule(name="phone", patterns=[], action="mask", preserve_suffix=4)]
    )
    sanitizer = PIISanitizer(config=config)
    df = pd.DataFrame({"phone": ["555-123-4567"]})

    result = sanitizer.sanitize(df, pii_map={"phone": "phone"})
    assert list(result["phone"]) == ["*" * 8 + "4567"]


def test_mask_handles_list_cells():
    """pd.isna on list-like cells must not raise ValueError."""
    config = PrivacyConfig(rules=[PiiRule(name="ssn", patterns=[], action="mask")])
    sanitizer = PIISanitizer(config=config)
    df = pd.DataFrame({"ssn": pd.Series([["123", "456"], "123-45-6789"], dtype=object)})

    result = sanitizer.sanitize(df, pii_map={"ssn": "ssn"})
    assert result["ssn"].iloc[1] == "*" * 11
    # The list cell is stringified and fully masked, not left raw.
    assert set(result["ssn"].iloc[0]) == {"*"}


def test_hash_salt_reproducibility():
    config = PrivacyConfig(rules=[PiiRule(name="ssn", patterns=[], action="hash")])
    df = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321"]})

    s1 = PIISanitizer(config=config, salt=b"shared-salt")
    s2 = PIISanitizer(config=config, salt=b"shared-salt")
    s3 = PIISanitizer(config=config, salt=b"other-salt")
    s4 = PIISanitizer(config=config)  # random salt

    h1 = s1.sanitize(df, pii_map={"ssn": "ssn"})
    h2 = s2.sanitize(df, pii_map={"ssn": "ssn"})
    h3 = s3.sanitize(df, pii_map={"ssn": "ssn"})
    h4 = s4.sanitize(df, pii_map={"ssn": "ssn"})

    # Same explicit salt -> identical hashes across instances/runs.
    assert list(h1["ssn"]) == list(h2["ssn"])
    # Different or random salts -> different hashes.
    assert list(h1["ssn"]) != list(h3["ssn"])
    assert list(h1["ssn"]) != list(h4["ssn"])
    # String salts are accepted too.
    s5 = PIISanitizer(config=config, salt="shared-salt")
    h5 = s5.sanitize(df, pii_map={"ssn": "ssn"})
    assert list(h5["ssn"]) == list(h1["ssn"])


def test_fake_preserves_nulls():
    sanitizer = PIISanitizer()
    df = pd.DataFrame({"user_email": ["a@b.com", np.nan, "c@d.com"]})

    result = sanitizer.sanitize(df, pii_map={"user_email": "email"})

    assert pd.isna(result["user_email"].iloc[1])
    assert result["user_email"].iloc[0] != "a@b.com"
    assert result["user_email"].iloc[2] != "c@d.com"


def test_fake_seed_reproducibility():
    df = pd.DataFrame({"user_email": ["a@b.com", "b@c.com", "c@d.com"]})

    s1 = PIISanitizer()
    s2 = PIISanitizer()
    r1 = s1.sanitize(df, pii_map={"user_email": "email"}, seed=42)
    r2 = s2.sanitize(df, pii_map={"user_email": "email"}, seed=42)

    assert list(r1["user_email"]) == list(r2["user_email"])


def test_contextual_faker_seed_reproducibility():
    f1 = ContextualFaker(seed=123)
    f2 = ContextualFaker(seed=123)

    assert f1.generate_pii("name", count=5) == f2.generate_pii("name", count=5)
    # Seeded locale-specific instances are reproducible too.
    assert f1.generate_pii("name", context={"country": "JP"}, count=3) == (
        f2.generate_pii("name", context={"country": "JP"}, count=3)
    )


def test_faker_rejects_unknown_pii_type():
    faker = ContextualFaker()

    with pytest.raises(PrivacyError, match="Unsupported pii_type"):
        faker.generate_pii("os_system", count=1)

    with pytest.raises(PrivacyError, match="Unsupported pii_type"):
        faker.process_dataframe(
            pd.DataFrame({"col": ["x"]}), pii_cols={"col": "shell_command"}
        )


def test_faker_unknown_locale_falls_back():
    faker = ContextualFaker()

    # Unknown code falls back to the default locale without raising.
    values = faker.generate_pii("name", context={"country": "ZZ"}, count=2)
    assert len(values) == 2
    assert all(isinstance(v, str) and v for v in values)

    # Raw Faker locale strings work without a LOCALE_MAP entry.
    values = faker.generate_pii("name", context={"country": "pt_BR"}, count=1)
    assert isinstance(values[0], str)


def test_process_dataframe_region_context_and_nonunique_index():
    faker = ContextualFaker(seed=7)
    df = pd.DataFrame(
        {"region": ["JP", "US", "JP"], "email": ["x", "y", "z"]},
        index=[0, 0, 1],  # non-unique index
    )

    result = faker.process_dataframe(df, pii_cols={"email": "email"})

    assert len(result) == 3
    # Placeholders replaced with generated values, one per row.
    assert all(isinstance(v, str) and v not in {"x", "y", "z"} for v in result["email"])
    # Context columns untouched.
    assert list(result["region"]) == ["JP", "US", "JP"]
