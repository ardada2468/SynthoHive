---
title: Privacy Sanitization Demo
---

# Privacy Sanitization Demo

Path: `examples/demos/02_privacy_sanitization`

## Goal
Detect PII, apply sanitization (mask/hash/fake), and compare raw vs sanitized CSV outputs.

## Run
```bash
python examples/demos/02_privacy_sanitization/run.py
```

## Outputs
- `outputs/raw_users.csv`
- `outputs/sanitized_users.csv`

## Notes
- Uses `PIISanitizer` + `ContextualFaker`.
- Adjust rules or actions in the demo script to experiment with masking vs hashing vs faking. 
