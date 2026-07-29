"""Pre-training PII sanitization applied by the orchestration pipeline.

This is the glue that makes ``PrivacyConfig`` real: before any table reaches a
generative model, its declared ``pii_cols`` are replaced according to the
configured strategy so raw PII never enters training.
"""

import hashlib
from typing import Iterable, Optional

import pandas as pd
import structlog

from syntho_hive.exceptions import PrivacyError

log = structlog.get_logger()

# Column-name fragments mapped to Faker providers for the faker strategies.
_PROVIDER_HINTS = [
    ("email", "email"),
    ("phone", "phone_number"),
    ("ssn", "ssn"),
    ("address", "address"),
    ("city", "city"),
    ("country", "country"),
    ("first_name", "first_name"),
    ("last_name", "last_name"),
    ("name", "name"),
    ("company", "company"),
    ("dob", "date_of_birth"),
    ("birth", "date_of_birth"),
]

_MASK_VALUE = "********"


def _provider_for_column(col_name: str) -> Optional[str]:
    """Infer a Faker provider from the column name, or None if unrecognized."""
    lowered = str(col_name).lower()
    for fragment, provider in _PROVIDER_HINTS:
        if fragment in lowered:
            return provider
    return None


def _pseudonymize(series: pd.Series, salt: str) -> pd.Series:
    """Deterministic salted-hash pseudonyms preserving nulls and duplicates."""

    def _hash(v):
        if pd.isnull(v):
            return v
        digest = hashlib.sha256(f"{salt}{v}".encode()).hexdigest()[:12]
        return f"anon_{digest}"

    return series.map(_hash)


def apply_privacy(
    df: pd.DataFrame,
    pii_cols: Iterable[str],
    strategy: str,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Replace PII columns in a table before it reaches a generative model.

    Args:
        df: Table to sanitize (not mutated).
        pii_cols: Columns declared as PII in the table's metadata.
        strategy: ``"mask"`` replaces values with a fixed mask; ``"faker"`` /
            ``"context_aware_faker"`` substitute realistic fake values (falling
            back to salted-hash pseudonyms for unrecognized column types).
        seed: Optional seed making faker output and pseudonyms reproducible.

    Raises:
        PrivacyError: If a declared PII column is missing or the strategy is unknown.

    Returns:
        A sanitized copy of the input DataFrame.
    """
    pii_cols = list(pii_cols)
    if not pii_cols:
        return df

    missing = [c for c in pii_cols if c not in df.columns]
    if missing:
        raise PrivacyError(
            f"Declared PII column(s) {missing} not found in table columns "
            f"{list(df.columns)}. Refusing to train with unsanitized PII."
        )

    if strategy not in ("mask", "faker", "context_aware_faker"):
        raise PrivacyError(f"Unknown pii_strategy '{strategy}'")

    result = df.copy()
    salt = str(seed) if seed is not None else "synthohive"

    if strategy == "mask":
        for col in pii_cols:
            result[col] = result[col].where(result[col].isnull(), _MASK_VALUE)
        log.info("pii_masked", columns=pii_cols)
        return result

    # faker / context_aware_faker
    from faker import Faker

    faker = Faker()
    if seed is not None:
        Faker.seed(seed)

    for col in pii_cols:
        provider = _provider_for_column(col)
        if provider is None:
            result[col] = _pseudonymize(result[col], salt)
            log.info("pii_pseudonymized", column=col)
            continue

        gen = getattr(faker, provider)
        mask = result[col].notna()
        result.loc[mask, col] = [str(gen()) for _ in range(int(mask.sum()))]
        log.info("pii_faked", column=col, provider=provider)

    return result
