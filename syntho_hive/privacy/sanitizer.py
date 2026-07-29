from typing import List, Dict, Optional, Any, Union, Callable
import re
import pandas as pd
import numpy as np
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field

from .faker_contextual import ContextualFaker
from ..exceptions import PrivacyError


@dataclass
class PiiRule:
    """Configuration for a single PII type and handling strategy."""

    name: str
    patterns: List[str]  # List of regex patterns to match
    action: str = "drop"  # Options: "drop", "mask", "hash", "fake", "custom", "keep"
    context_key: Optional[str] = (
        None  # Key to look for in context (e.g. 'country' for locale)
    )
    custom_generator: Optional[Callable[[Dict[str, Any]], Any]] = (
        None  # Custom lambda for generation
    )
    preserve_suffix: int = (
        0  # Number of trailing characters kept visible when masking (0 = full mask)
    )


@dataclass
class PrivacyConfig:
    """Collection of rules for PII detection and handling."""

    rules: List[PiiRule] = field(default_factory=list)

    @classmethod
    def default(cls) -> "PrivacyConfig":
        """Create a default privacy configuration with common PII rules.

        Masking rules fully mask by default; opt in to keeping a trailing
        fragment (e.g. phone last-4) via ``preserve_suffix`` on a rule.
        SSN and date_of_birth are always fully masked.
        """
        return cls(
            rules=[
                PiiRule(
                    name="email",
                    patterns=[r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"],
                    action="fake",
                ),
                PiiRule(
                    name="ssn",
                    patterns=[r"^\d{3}-\d{2}-\d{4}$", r"^\d{9}$"],
                    action="mask",
                    preserve_suffix=0,
                ),
                PiiRule(
                    name="phone",
                    patterns=[
                        r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
                        r"^\+\d{1,3}[-.\s]?\d{1,14}$",
                    ],
                    action="fake",
                ),
                PiiRule(
                    name="credit_card",
                    patterns=[
                        r"^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$",
                        r"^\d{4}[-\s]?\d{6}[-\s]?\d{5}$",
                    ],
                    action="mask",
                    preserve_suffix=0,
                ),
                PiiRule(
                    name="ipv4",
                    patterns=[
                        r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
                    ],
                    action="fake",
                ),
                PiiRule(name="name", patterns=[], action="fake"),
                PiiRule(name="address", patterns=[], action="fake"),
                PiiRule(
                    name="date_of_birth",
                    patterns=[r"^\d{4}-\d{2}-\d{2}$", r"^\d{2}/\d{2}/\d{4}$"],
                    action="mask",
                    preserve_suffix=0,
                ),
            ]
        )


class PIISanitizer:
    """Detect and sanitize PII columns based on configurable rules."""

    COLUMN_NAME_ALIASES: Dict[str, List[str]] = {
        "email": ["email", "e_mail", "email_address"],
        "ssn": ["ssn", "social_security", "social_sec"],
        "phone": ["phone", "mobile", "cell", "tel", "telephone", "phone_number"],
        "name": ["first_name", "last_name", "full_name", "firstname", "lastname"],
        "address": ["address", "street", "city", "zip", "zipcode", "postal"],
        "date_of_birth": ["dob", "date_of_birth", "birth_date", "birthday"],
        "credit_card": ["credit_card", "card_number", "cc_num", "card_num"],
    }

    def __init__(
        self,
        config: Optional[PrivacyConfig] = None,
        salt: Optional[Union[bytes, str]] = None,
    ):
        """Create a sanitizer with contextual faker support.

        Args:
            config: Optional privacy configuration; defaults to ``PrivacyConfig.default``.
            salt: Optional salt for the ``hash`` action. When ``None`` (default)
                a random per-instance salt is generated, so hashes are only
                consistent within one sanitizer instance. Supply an explicit
                salt to make hashed values consistent across tables, instances
                and runs (e.g. to preserve join keys).
        """
        self.config = config or PrivacyConfig.default()
        self.faker = ContextualFaker()
        if salt is None:
            self._hash_salt = secrets.token_bytes(32)
        elif isinstance(salt, str):
            self._hash_salt = salt.encode("utf-8")
        else:
            self._hash_salt = salt

    def analyze(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect potential PII columns using configured rules.

        Args:
            df: DataFrame to inspect for PII.

        Returns:
            Mapping of column name to matched PII rule name.
        """
        detected = {}

        # 1. Check column names using alias-based heuristics
        for col in df.columns:
            col_lower = col.lower()
            for rule in self.config.rules:
                aliases = self.COLUMN_NAME_ALIASES.get(rule.name, [rule.name])
                for alias in aliases:
                    # Use word-boundary-like matching: check if alias matches the
                    # full column name or appears as a delimited token within it.
                    if col_lower == alias or re.search(
                        r"(?:^|[_\-\s])" + re.escape(alias) + r"(?:$|[_\-\s])",
                        col_lower,
                    ):
                        detected[col] = rule.name
                        break
                if col in detected:
                    break

        # 2. Check content for remaining columns
        # Sample up to 100 random rows to avoid positional bias
        sample = df.sample(min(100, len(df)), random_state=42)

        for col in df.columns:
            if col in detected:
                continue

            # Skip non-string columns for regex matching
            if not pd.api.types.is_string_dtype(sample[col]):
                continue

            valid_rows = sample[col].dropna().astype(str)
            if len(valid_rows) == 0:
                continue

            # Check each rule
            best_rule = None
            max_matches = 0

            for rule in self.config.rules:
                match_count = 0
                for val in valid_rows:
                    # Check any pattern for this rule
                    for pat in rule.patterns:
                        if re.search(pat, val):
                            match_count += 1
                            break  # Match found for this value

                # If > 50% match, consider it a candidate
                if match_count > len(valid_rows) * 0.5:
                    if match_count > max_matches:
                        max_matches = match_count
                        best_rule = rule.name

            if best_rule:
                detected[col] = best_rule

        return detected

    def sanitize(
        self,
        df: pd.DataFrame,
        pii_map: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Apply sanitization rules to a dataframe.

        Args:
            df: Input dataframe containing potential PII.
            pii_map: Optional precomputed map of column name to PII rule name.
            seed: Optional seed for the ``fake`` action, making generated
                replacements reproducible across calls.

        Returns:
            Sanitized dataframe with PII handled according to configured actions.

        Raises:
            ValueError: If ``pii_map`` references columns absent from ``df``.
            PrivacyError: If ``pii_map`` references a rule name that is not
                configured, or a rule has an unrecognized action.
        """
        if pii_map is None:
            pii_map = self.analyze(df)
        else:
            invalid_cols = [col for col in pii_map if col not in df.columns]
            if invalid_cols:
                raise ValueError(
                    f"pii_map contains columns not in DataFrame: {invalid_cols}"
                )

        output_df = df.copy()

        for col, rule_name in pii_map.items():
            rule = next((r for r in self.config.rules if r.name == rule_name), None)
            if not rule:
                raise PrivacyError(
                    f"pii_map references unknown rule '{rule_name}' for column "
                    f"'{col}'. Known rules: "
                    f"{sorted(r.name for r in self.config.rules)}. Refusing to "
                    f"leave the column unsanitized."
                )

            if rule.action == "drop":
                output_df.drop(columns=[col], inplace=True)

            elif rule.action == "mask":
                output_df[col] = output_df[col].apply(
                    lambda x: self._mask_value(x, rule.preserve_suffix)
                )

            elif rule.action == "hash":
                output_df[col] = output_df[col].apply(lambda x: self._hash_value(x))

            elif rule.action == "fake":
                output_df[col] = self._fake_column(output_df, col, rule, seed=seed)

            elif rule.action == "custom":
                if rule.custom_generator:
                    # Use custom generator, passing row context
                    # Note: This checks frame line by line, slower but powerful
                    output_df[col] = output_df.apply(
                        lambda row: rule.custom_generator(row.to_dict()), axis=1
                    )
                else:
                    # Fallback if no generator provided
                    output_df[col] = output_df[col].apply(
                        lambda x: self._mask_value(x, rule.preserve_suffix)
                    )

            elif rule.action == "keep":
                # Explicitly leave the column untouched.
                pass

            else:
                raise PrivacyError(
                    f"Unrecognized action '{rule.action}' for rule '{rule.name}' "
                    f"on column '{col}'. Valid actions: drop, mask, hash, fake, "
                    f"custom, keep."
                )

        return output_df

    @staticmethod
    def _is_missing(val: Any) -> bool:
        """Return True for scalar null values; False for lists/arrays and non-nulls."""
        if val is None:
            return True
        # pd.isna on list-like values returns an array, which is ambiguous in
        # a boolean context — treat list-like cells as non-missing.
        if isinstance(val, (list, tuple, set, dict, np.ndarray)):
            return False
        try:
            return bool(pd.isna(val))
        except (TypeError, ValueError):
            return False

    def _mask_value(self, val: Any, preserve_suffix: int = 0) -> Any:
        """Mask a value, fully by default.

        Args:
            val: Value to mask.
            preserve_suffix: Number of trailing characters left visible.
                ``0`` (default) masks the entire value; only opt in per rule
                for low-risk fragments such as phone last-4.

        Returns:
            Masked string, or the original value if it is null.
        """
        if self._is_missing(val):
            return val
        s = str(val)
        if preserve_suffix > 0 and len(s) > preserve_suffix:
            return "*" * (len(s) - preserve_suffix) + s[-preserve_suffix:]
        return "*" * len(s)

    def _hash_value(self, val: Any) -> Any:
        """Return an HMAC-SHA256 hash representation of a value using the configured salt."""
        if self._is_missing(val):
            return val
        return hmac.new(self._hash_salt, str(val).encode(), hashlib.sha256).hexdigest()

    def _fake_column(
        self,
        df: pd.DataFrame,
        col: str,
        rule: PiiRule,
        seed: Optional[int] = None,
    ) -> pd.Series:
        """Generate fake data for a column using contextual faker.

        Null cells are preserved as-is rather than replaced with fake values.

        Args:
            df: DataFrame containing the column to fake.
            col: Column name.
            rule: PII rule describing the type being faked.
            seed: Optional seed; when given, the underlying Faker instances are
                re-seeded via ``Faker.seed_instance`` so output is reproducible.

        Returns:
            Series of fake values aligned to ``df``.
        """
        if seed is not None:
            self.faker.reseed(seed)

        failures_before = self.faker.failure_count
        context_records = df.to_dict(orient="records")
        original_values = df[col].tolist()

        values: List[Any] = []
        for record, original in zip(context_records, original_values):
            if self._is_missing(original):
                values.append(original)
            else:
                values.append(
                    self.faker.generate_pii(rule.name, context=record)[0]
                )

        self.faker.warn_failures(col, failures_before)
        return pd.Series(values, index=df.index, dtype=object)
