## Privacy Guardrails & PII Sanitization

- **What it shows:** Automatic PII detection plus custom rules to fake or mask sensitive columns before any modeling.
- **Key APIs:** `PrivacyConfig.default`, `PiiRule`, `PIISanitizer.analyze`, `PIISanitizer.sanitize`.
- **Outputs:** `outputs/raw_users.csv` and `outputs/sanitized_users.csv`.

### Run

```bash
python examples/demos/02_privacy_sanitization/run.py
```

Optional flags:
- `--rows`: Number of fake raw users to generate (default 50).
- `--output-dir`: Where to write CSV outputs.

### What to look at
- Console printout of detected PII map per column.
- The sanitized CSV should have masked SSNs/credit cards, faked emails/phones, and a custom-hashed `loyalty_id`.

