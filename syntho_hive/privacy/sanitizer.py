from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import re
import pandas as pd
import hashlib
from dataclasses import dataclass, field
from .faker_contextual import ContextualFaker

@dataclass
class PiiRule:
    """
    Configuration for a single PII type.
    """
    name: str
    patterns: List[str]  # List of regex patterns to match
    action: str = "drop"  # Options: "drop", "mask", "hash", "fake", "custom", "keep"
    context_key: Optional[str] = None  # Key to look for in context (e.g. 'country' for locale)
    custom_generator: Optional[Callable[[Dict[str, Any]], Any]] = None  # Custom lambda for generation

@dataclass
class PrivacyConfig:
    """
    Collection of rules for PII detection and handling.
    """
    rules: List[PiiRule] = field(default_factory=list)

    @classmethod
    def default(cls) -> 'PrivacyConfig':
        return cls(rules=[
            PiiRule(name="email", patterns=[r"[^@]+@[^@]+\.[^@]+"], action="fake"),
            PiiRule(name="ssn", patterns=[r"\d{3}-\d{2}-\d{4}"], action="mask"),
            PiiRule(name="phone", patterns=[r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"], action="fake"),
            PiiRule(name="credit_card", patterns=[r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"], action="mask"),
            PiiRule(name="ipv4", patterns=[r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"], action="fake"),
        ])

class PIISanitizer:
    """
    Detects and sanitizes PII columns based on configurable rules.
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig.default()
        self.faker = ContextualFaker()
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect potential PII columns using defined rules.
        Returns map: col_name -> pii_rule_name
        """
        detected = {}
        
        # 1. Check column names (heuristics)
        for col in df.columns:
            col_lower = col.lower()
            for rule in self.config.rules:
                if rule.name in col_lower:
                    detected[col] = rule.name
                    break
        
        # 2. Check content for remaining columns
        # Sample first 100 rows (or all if small) to speed up
        sample = df.head(100)
        
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
                            break # Match found for this value
                
                # If > 50% match, consider it a candidate
                if match_count > len(valid_rows) * 0.5:
                    if match_count > max_matches:
                        max_matches = match_count
                        best_rule = rule.name
            
            if best_rule:
                detected[col] = best_rule
                    
        return detected

    def sanitize(self, df: pd.DataFrame, pii_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Apply sanitization rules to the dataframe.
        If pii_map is not provided, it will be generated via analyze().
        """
        if pii_map is None:
            pii_map = self.analyze(df)
            
        output_df = df.copy()
        
        for col, rule_name in pii_map.items():
            rule = next((r for r in self.config.rules if r.name == rule_name), None)
            if not rule:
                continue
                
            if rule.action == "drop":
                output_df.drop(columns=[col], inplace=True)
                
            elif rule.action == "mask":
                output_df[col] = output_df[col].apply(lambda x: self._mask_value(x))
                
            elif rule.action == "hash":
                output_df[col] = output_df[col].apply(lambda x: self._hash_value(x))
                
            elif rule.action == "fake":
                output_df[col] = self._fake_column(output_df, col, rule)
                
            elif rule.action == "custom":
                if rule.custom_generator:
                    # Use custom generator, passing row context
                    # Note: This checks frame line by line, slower but powerful
                    output_df[col] = output_df.apply(lambda row: rule.custom_generator(row.to_dict()), axis=1)
                else:
                    # Fallback if no generator provided
                    output_df[col] = self._mask_value(output_df[col])

        return output_df

    def _mask_value(self, val: Any) -> str:
        s = str(val)
        if len(s) <= 4:
            return "*" * len(s)
        return "*" * (len(s) - 4) + s[-4:]
    
    def _hash_value(self, val: Any) -> str:
        return hashlib.sha256(str(val).encode()).hexdigest()

    def _fake_column(self, df: pd.DataFrame, col: str, rule: PiiRule) -> pd.Series:
        """
        Generate fake data for a column, potentially using context from other columns.
        """
        # Context strategy: 
        # If the rule has a context_key (not yet fully implemented in config, but good for future), use it.
        # Fallback to simple random generation.
        
        # We can pass the dataframe to the faker to handle this column
        # But our FakerContextual currently handles whole DF. 
        # Let's call generate_pii for the length of DF.
        
        # Optimization: fast path if no context needed
        return df.apply(lambda row: self.faker.generate_pii(rule.name, context=row.to_dict())[0], axis=1)
