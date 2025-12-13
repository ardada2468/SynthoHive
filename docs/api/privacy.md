---
title: Privacy
---

# Privacy & Sanitization

The privacy module ensures that sensitive information is detected and handled correctly before any modeling takes place.

## Sanitizer

::: syntho_hive.privacy.sanitizer.PIISanitizer
    options:
      members:
        - analyze
        - sanitize

## Configuration

::: syntho_hive.privacy.sanitizer.PiiRule

::: syntho_hive.privacy.sanitizer.PrivacyConfig

## Faking Strategy

::: syntho_hive.privacy.faker_contextual.ContextualFaker
