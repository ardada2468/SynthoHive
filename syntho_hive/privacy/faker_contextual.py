from typing import Dict, Any, Optional, List, Union
from faker import Faker
import pandas as pd
import numpy as np
import logging

class ContextualFaker:
    """
    Direct PII generation with context awareness.
    Example: Country='JP' -> Faker('ja_JP').name()
    """
    
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
    
    def __init__(self):
        self._fakers: Dict[str, Faker] = {}
        # Initialize default
        self._fakers["default"] = Faker()
        self.logger = logging.getLogger(__name__)
        
    def _get_faker(self, locale: Optional[str]) -> Faker:
        if not locale:
            return self._fakers["default"]
            
        mapped_locale = self.LOCALE_MAP.get(locale.upper(), "en_US")
        
        if mapped_locale not in self._fakers:
            try:
                self._fakers[mapped_locale] = Faker(mapped_locale)
            except Exception as e:
                self.logger.warning(f"Could not load locale {mapped_locale}, falling back to default. Error: {e}")
                self._fakers[mapped_locale] = self._fakers["default"]
            
        return self._fakers[mapped_locale]

    def generate_pii(self, pii_type: str, context: Optional[Dict[str, Any]] = None, count: int = 1) -> List[str]:
        """
        Generate PII based on context.
        Context is expected to be a row from the dataset (e.g., {'country': 'JP'}).
        """
        if context is None:
            context = {}
            
        # Attempt to infer locale from context
        # Heuristic: Look for 'country', 'region', 'locale' keys
        locale = context.get('country') or context.get('locale') or context.get('region')
        
        fake = self._get_faker(locale if isinstance(locale, str) else None)
        
        results = []
        for _ in range(count):
            val = self._generate_single_value(fake, pii_type)
            results.append(val)
                
        return results

    def _generate_single_value(self, fake: Faker, pii_type: str) -> str:
        """Helper to generate a single value safely."""
        try:
            if hasattr(fake, pii_type):
                 # Dynamic method call on Faker instance
                return str(getattr(fake, pii_type)())
            
            # Custom mappings for common PII types if name mismatch or special logic
            if pii_type == 'phone':
                 return fake.phone_number()
            elif pii_type == 'ip' or pii_type == 'ipv4':
                return fake.ipv4()
            elif pii_type == 'credit_card':
                return fake.credit_card_number()
            
            # Fallback
            return str(fake.text(max_nb_chars=20))
        except Exception as e:
            self.logger.error(f"Error generating {pii_type}: {e}")
            return "REDACTED"

    def process_dataframe(self, df: pd.DataFrame, pii_cols: Dict[str, str]) -> pd.DataFrame:
        """
        Replace synthetic placeholders in DF with context-aware PII.
        pii_cols: {col_name: pii_type} e.g. {'user_email': 'email'}
        """
        output_df = df.copy()
        
        # Check if we have context columns
        has_country_context = 'country' in df.columns or 'locale' in df.columns
        
        if not has_country_context:
            # Fast path: Vectorized apply (Fake doesn't vectorize well but we avoid row iteration overhead if possible)
            # Actually simpler: Just generate N fake values using default locale
            for col, pii_type in pii_cols.items():
                fake = self._get_faker(None)
                # Generate list
                values = [self._generate_single_value(fake, pii_type) for _ in range(len(df))]
                output_df[col] = values
        else:
            # Slow path: Row-by-row for context awareness
            for idx, row in output_df.iterrows():
                context = row.to_dict()
                for col, pii_type in pii_cols.items():
                    val = self.generate_pii(pii_type, context=context, count=1)[0]
                    output_df.at[idx, col] = val
                
        return output_df
