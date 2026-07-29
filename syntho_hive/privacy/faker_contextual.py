from typing import Dict, Any, Optional, List

import pandas as pd
import structlog
from faker import Faker

from ..exceptions import PrivacyError


class ContextualFaker:
    """Context-aware PII generator leveraging Faker locales."""

    LOCALE_MAP = {
        "JP": "ja_JP",
        "US": "en_US",
        "UK": "en_GB",
        "GB": "en_GB",
        "DE": "de_DE",
        "FR": "fr_FR",
        "CN": "zh_CN",
        "IN": "en_IN",
        # Add more as needed
    }

    #: Context columns used to infer locale. Must stay in sync between
    #: ``generate_pii`` and the ``process_dataframe`` fast-path check.
    CONTEXT_COLUMNS = ("country", "locale", "region")

    #: Whitelisted Faker provider names (plus internal aliases handled in
    #: ``_generate_single_value``). Any other ``pii_type`` raises
    #: ``PrivacyError`` instead of silently falling back to random text.
    ALLOWED_PII_TYPES = frozenset(
        {
            "name",
            "first_name",
            "last_name",
            "user_name",
            "email",
            "free_email",
            "company_email",
            "phone",
            "phone_number",
            "address",
            "street_address",
            "city",
            "state",
            "zipcode",
            "postcode",
            "country",
            "company",
            "job",
            "ssn",
            "date_of_birth",
            "credit_card",
            "credit_card_number",
            "iban",
            "ip",
            "ipv4",
            "ipv6",
            "mac_address",
            "url",
            "uuid4",
        }
    )

    def __init__(self, seed: Optional[int] = None):
        """Initialize faker cache and logger.

        Args:
            seed: Optional seed applied to every cached Faker instance for
                reproducible generation. ``None`` (default) keeps Faker's
                non-deterministic behaviour.
        """
        self._seed = seed
        self._fakers: Dict[str, Faker] = {}
        self._failure_count = 0
        self.logger = structlog.get_logger(__name__)
        # Initialize default
        self._fakers["default"] = self._new_faker(None)

    def _new_faker(self, locale: Optional[str]) -> Faker:
        """Create a Faker instance, seeding it if a seed was configured.

        Args:
            locale: Faker locale string, or ``None`` for the default locale.

        Returns:
            New (optionally seeded) Faker instance.

        Raises:
            Exception: Propagates Faker construction errors for unknown locales.
        """
        fake = Faker(locale) if locale else Faker()
        if self._seed is not None:
            fake.seed_instance(self._seed)
        return fake

    def reseed(self, seed: int) -> None:
        """Reset the seed on all cached Faker instances.

        Args:
            seed: Seed applied to every cached (and future) Faker instance.
        """
        self._seed = seed
        for fake in self._fakers.values():
            fake.seed_instance(seed)

    @property
    def failure_count(self) -> int:
        """Total number of generation failures that fell back to ``"REDACTED"``."""
        return self._failure_count

    def warn_failures(self, column: str, failures_before: int) -> None:
        """Emit a single warning for generation failures accumulated for a column.

        Args:
            column: Column name the failures belong to.
            failures_before: Snapshot of ``failure_count`` taken before the
                column was processed.
        """
        failures = self._failure_count - failures_before
        if failures:
            self.logger.warning(
                "pii_generation_failures",
                column=column,
                count=failures,
                note="values were replaced with 'REDACTED'",
            )

    def _get_faker(self, locale: Optional[str]) -> Faker:
        """Get or create a Faker instance for a locale.

        Unknown locales are first tried verbatim against Faker (so full locale
        strings such as ``"pt_BR"`` work without an entry in ``LOCALE_MAP``);
        only if that fails does the lookup fall back to the default locale,
        with a warning.

        Args:
            locale: Optional locale string (e.g., ``"JP"`` or ``"en_US"``).

        Returns:
            Faker instance configured for the requested locale.
        """
        if not locale:
            return self._fakers["default"]

        cache_key = locale.upper()
        if cache_key in self._fakers:
            return self._fakers[cache_key]

        mapped_locale = self.LOCALE_MAP.get(cache_key)
        # Try the mapped locale if known, otherwise the raw locale string.
        candidate = mapped_locale or locale

        try:
            fake = self._new_faker(candidate)
        except Exception as e:
            self.logger.warning(
                "locale_fallback",
                requested=locale,
                attempted=candidate,
                fallback="default (en_US)",
                error=str(e),
            )
            fake = self._fakers["default"]

        self._fakers[cache_key] = fake
        return fake

    def generate_pii(
        self, pii_type: str, context: Optional[Dict[str, Any]] = None, count: int = 1
    ) -> List[str]:
        """Generate PII values with optional contextual locale.

        Args:
            pii_type: Faker provider name (e.g., ``"email"`` or ``"phone"``).
                Must be one of ``ALLOWED_PII_TYPES``.
            context: Optional row context used to infer locale
                (country/locale/region keys).
            count: Number of values to generate.

        Returns:
            List of generated PII strings.

        Raises:
            PrivacyError: If ``pii_type`` is not a whitelisted provider name.
        """
        if context is None:
            context = {}

        # Attempt to infer locale from context.
        # Heuristic: look for the keys in CONTEXT_COLUMNS, in order.
        locale = next(
            (context.get(key) for key in self.CONTEXT_COLUMNS if context.get(key)),
            None,
        )

        fake = self._get_faker(locale if isinstance(locale, str) else None)

        results = []
        for _ in range(count):
            val = self._generate_single_value(fake, pii_type)
            results.append(val)

        return results

    def _generate_single_value(self, fake: Faker, pii_type: str) -> str:
        """Generate a single PII value using a Faker instance.

        Args:
            fake: Faker instance to use.
            pii_type: Whitelisted provider name or alias.

        Returns:
            Generated value, or ``"REDACTED"`` if generation failed.

        Raises:
            PrivacyError: If ``pii_type`` is not a whitelisted provider name.
        """
        if pii_type not in self.ALLOWED_PII_TYPES:
            raise PrivacyError(
                f"Unsupported pii_type '{pii_type}'. Allowed types: "
                f"{sorted(self.ALLOWED_PII_TYPES)}"
            )

        try:
            # Aliases for common PII types whose Faker provider name differs.
            if pii_type == "phone":
                return fake.phone_number()
            elif pii_type in ("ip", "ipv4"):
                return fake.ipv4()
            elif pii_type == "credit_card":
                return fake.credit_card_number()
            elif pii_type == "date_of_birth":
                return str(fake.date_of_birth())
            elif pii_type == "address":
                return fake.address()

            # Dynamic method call on the Faker instance.
            return str(getattr(fake, pii_type)())
        except Exception as e:
            self._failure_count += 1
            self.logger.debug(
                "pii_generation_error", pii_type=pii_type, error=str(e)
            )
            return "REDACTED"

    def process_dataframe(
        self, df: pd.DataFrame, pii_cols: Dict[str, str]
    ) -> pd.DataFrame:
        """Replace placeholders with generated PII in a dataframe.

        Args:
            df: Input dataframe containing placeholder columns.
            pii_cols: Mapping of column name to PII type (e.g., ``{"user_email": "email"}``).

        Returns:
            DataFrame with specified columns replaced by generated PII.

        Raises:
            PrivacyError: If a requested PII type is not whitelisted.
        """
        output_df = df.copy()

        # Check if we have context columns (must match generate_pii's heuristic).
        has_context = any(col in df.columns for col in self.CONTEXT_COLUMNS)

        if not has_context:
            # Fast path: no per-row context needed, use the default locale.
            fake = self._get_faker(None)
            for col, pii_type in pii_cols.items():
                failures_before = self._failure_count
                values = [
                    self._generate_single_value(fake, pii_type)
                    for _ in range(len(df))
                ]
                # Whole-column positional assignment: safe under non-unique
                # indexes, unlike per-cell ``.at`` writes.
                output_df[col] = values
                self.warn_failures(col, failures_before)
        else:
            # Context-aware path: build each column positionally from the
            # original row contexts, then assign the whole column at once.
            records = df.to_dict(orient="records")
            for col, pii_type in pii_cols.items():
                failures_before = self._failure_count
                values = [
                    self.generate_pii(pii_type, context=record, count=1)[0]
                    for record in records
                ]
                output_df[col] = values
                self.warn_failures(col, failures_before)

        return output_df
