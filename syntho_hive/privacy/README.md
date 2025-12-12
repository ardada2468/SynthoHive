# SynthoHive Privacy Module

The **SynthoHive Privacy Module** provides robust tools for detecting and sanitizing Personally Identifiable Information (PII) in datasets. It is designed to be highly configurable, supporting custom detection rules and various sanitization strategies (masking, hashing, faking, dropping).

## Table of Contents
- [Overview](#overview)
- [Key Components](#key-components)
- [How It Works](#how-it-works)
    - [Detection](#detection)
    - [Sanitization](#sanitization)
    - [Context Awareness](#context-awareness)
- [API Reference](#api-reference)
- [Usage Guide](#usage-guide)

## Overview

Synthetic data generation requires strict privacy guardrails. This module ensures that sensitive data from source datasets is identified and transformed *before* it matches any training process or downstream usage. It combines regex-based detection, heuristic column naming analysis, and context-aware synthetic data generation.

## Key Components

### 1. `PIISanitizer`
The main orchestrator class. It takes a configuration, analyzes a pandas DataFrame to find PII, and applies sanitization rules.

### 2. `PrivacyConfig` & `PiiRule`
Data structures for configuration.
- **`PiiRule`**: Defines a single PII type (e.g., "Email"), the regex patterns to detect it, and the action to take (e.g., "fake").
- **`PrivacyConfig`**: A collection of `PiiRule` objects.

### 3. `ContextualFaker`
A wrapper around the `faker` library that handles localization and consistency. It ensures that if a row has `country='JP'`, the generated fake data (like names or phones) respects that locale.

## How It Works

### Detection
The `PIISanitizer.analyze(df)` method operates in two stages:
1.  **Column Name Heuristics**: It checks if column names contain keywords associated with defined rules (e.g., a column named `user_email` matches the `email` rule).
2.  **Content Content Scanning**: For columns not identified by name, it samples the data (first 100 rows). It checks values against regex patterns defined in `PiiRule`. If >50% of non-null values match a pattern, the column is flagged.

### Sanitization
The `PIISanitizer.sanitize(df, pii_map)` method transforms the data based on the `action` defined in the matching `PiiRule`. Supported actions:
- **`drop`**: Removed the column entirely.
- **`mask`**: Replaces all but the last 4 characters with `*` (e.g., `***-**-6789`).
- **`hash`**: Replaces values with their SHA-256 hash.
- **`fake`**: Replaces values with realistic synthetic data using `ContextualFaker`.
- **`keep`**: Retains the original data (useful for whitelisting).

### Context Awareness
When using the `fake` action, the system attempts to generate culturally relevant data.
- **Input**: A row with `{'country': 'FR', 'phone': '...'}`
- **Process**: `ContextualFaker` detects the `FR` context.
- **Output**: Generates a valid French phone number format.

## API Reference

### `syntho_hive.privacy.sanitizer`

#### `class PiiRule`
- `name`: `str` (Unique identifier, e.g., 'ssn')
- `patterns`: `List[str]` (Regex patterns)
- `action`: `str` ('drop', 'mask', 'hash', 'fake')
- `context_key`: `Optional[str]` (Context column to guide generation)

#### `class PrivacyConfig`
- `rules`: `List[PiiRule]`
- `default()`: Returns a standard configuration for common PII (Email, SSN, Phone, IP, Credit Card).

#### `class PIISanitizer(config: PrivacyConfig = None)`
- **`analyze(df: pd.DataFrame) -> Dict[str, str]`**
    - Scans dataframe and returns a map of `{column_name: rule_name}`.
- **`sanitize(df: pd.DataFrame, pii_map: Optional[Dict[str, str]] = None) -> pd.DataFrame`**
    - Returns a new DataFrame with PII handled. If `pii_map` is omitted, `analyze` is called internally.

### `syntho_hive.privacy.faker_contextual`

#### `class ContextualFaker()`
- **`generate_pii(pii_type: str, context: Dict, count: int) -> List[str]`**
    - Generates `count` items of `pii_type` using `context` (like country/locale) to select the correct Faker provider.
- **`process_dataframe(df, pii_cols)`**
    - Batch helper to process a full dataframe.

## Usage Guide

### Basic Usage with Defaults
```python
import pandas as pd
from syntho_hive.privacy.sanitizer import PIISanitizer

# Load data
df = pd.read_csv("users.csv")

# Initialize sanitizer with default rules
sanitizer = PIISanitizer()

# specific analysis (optional, helpful for auditing)
pii_map = sanitizer.analyze(df)
print(f"Detected PII: {pii_map}")

# Sanitize
clean_df = sanitizer.sanitize(df)
```

### Custom Configuration
```python
from syntho_hive.privacy.sanitizer import PIISanitizer, PrivacyConfig, PiiRule

# Define custom rules
config = PrivacyConfig(rules=[
    # Fake emails
    PiiRule(name="email", patterns=[r"@"], action="fake"),
    # Hash internal IDs
    PiiRule(name="internal_id", patterns=[r"^ID-\d+"], action="hash"),
    # Drop raw notes
    PiiRule(name="notes", patterns=[], action="drop") 
])


sanitizer = PIISanitizer(config=config)
clean_df = sanitizer.sanitize(df)
```

### Custom Generators (Lambdas)
For complex or "weird" columns, you can define a custom `custom_generator` lambda (or function). This function receives the entire row context as a dictionary.

```python
# Custom rule for a "Special Code" that combines an ID and a generic string
# Rule: "generate a string like 'SECURE-{id}'"

def my_generator(context):
    # Context contains the full row data
    # We can use other columns to generate consistent data
    user_id = context.get('id', '000')
    return f"SECURE-{user_id}-X"

config = PrivacyConfig(rules=[
    PiiRule(
        name="special_code", 
        patterns=[r"SECRET-\d+"], 
        action="custom", 
        custom_generator=my_generator
    )
])

# Or using a simple lambda
PiiRule(
    name="weird_col",
    patterns=[],
    action="custom",
    custom_generator=lambda row: "FIXED_VALUE"
)
```
