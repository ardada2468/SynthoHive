from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from syntho_hive.privacy.sanitizer import PIISanitizer, PrivacyConfig, PiiRule


def make_raw_users(num_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    first_names = ["Alex", "Jordan", "Sam", "Taylor", "Jamie", "Riley", "Casey", "Drew"]
    last_names = ["Lee", "Patel", "Garcia", "Chen", "Olsen", "Diaz", "Nguyen", "Brown"]
    cities = ["NY", "SF", "SEA", "DAL", "BOS"]

    rows = []
    for i in range(num_rows):
        first = rng.choice(first_names)
        last = rng.choice(last_names)
        city = rng.choice(cities)
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        phone = f"({rng.integers(200, 999)})-{rng.integers(200, 999)}-{rng.integers(1000, 9999)}"
        ssn = f"{rng.integers(100, 999):03d}-{rng.integers(10, 99):02d}-{rng.integers(1000, 9999):04d}"
        credit_card = f"{rng.integers(1000, 9999):04d}-{rng.integers(1000, 9999):04d}-{rng.integers(1000, 9999):04d}-{rng.integers(1000, 9999):04d}"
        loyalty_id = f"L-{rng.integers(10_000, 99_999)}"

        rows.append(
            {
                "user_id": i + 1,
                "first_name": first,
                "last_name": last,
                "city": city,
                "email": email,
                "phone": phone,
                "ssn": ssn,
                "credit_card": credit_card,
                "loyalty_id": loyalty_id,
                "notes": f"Called support on ticket {rng.integers(1000, 9999)}",
            }
        )

    return pd.DataFrame(rows)


def build_config() -> PrivacyConfig:
    """
    Extend the defaults with a custom rule to hash loyalty IDs
    instead of masking or faking them.
    """
    config = PrivacyConfig.default()
    config.rules.append(
        PiiRule(
            name="loyalty_id",
            patterns=[r"L-\d{5}"],
            action="hash",
        )
    )
    return config


def main():
    parser = argparse.ArgumentParser(description="Run the PII sanitization demo.")
    parser.add_argument("--rows", type=int, default=50, help="How many raw rows to generate.")
    parser.add_argument(
        "--output-dir",
        default="examples/demos/02_privacy_sanitization/outputs",
        help="Directory to place raw and sanitized CSVs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = make_raw_users(args.rows)
    raw_path = output_dir / "raw_users.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Wrote raw data to {raw_path}")

    config = build_config()
    sanitizer = PIISanitizer(config=config)
    detected = sanitizer.analyze(raw_df)
    print("Detected PII columns:", detected)

    sanitized_df = sanitizer.sanitize(raw_df, pii_map=detected)
    sanitized_path = output_dir / "sanitized_users.csv"
    sanitized_df.to_csv(sanitized_path, index=False)
    print(f"Wrote sanitized data to {sanitized_path}")
    print(sanitized_df.head())


if __name__ == "__main__":
    main()

