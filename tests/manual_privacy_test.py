
import pandas as pd
from syntho_hive.privacy.sanitizer import PIISanitizer, PrivacyConfig, PiiRule

def test_privacy_robustness():
    # 1. Setup Data
    data = {
        'id': [1, 2, 3, 4],
        'user_email': ['alice@example.com', 'bob@test.org', 'charlie@gmail.com', 'david@yahoo.com'],
        'ssn_col': ['123-45-6789', '987-65-4321', '000-00-0000', '111-22-3333'],
        'phone_num': ['555-0199', '555-0100', '555-0123', '555-9999'], # Simple format
        'notes': ['Just some text', 'Another note', 'Nothing private here', 'Hello world'],

        'country': ['US', 'JP', 'DE', 'FR'], # For context
        'custom_col': ['PROPRIETARY-A', 'PROPRIETARY-B', 'PROPRIETARY-C', 'PROPRIETARY-D']
    }
    df = pd.DataFrame(data)
    
    print("Original Data:")
    print(df.head())
    print("-" * 20)
    
    # Custom lambda generator
    def my_custom_generator(context):
        # Generate something using the ID from the row
        return f"REDACTED-ID-{context['country']}"

    # 2. Configure Rules
    config = PrivacyConfig(rules=[
        PiiRule(name="email", patterns=[r"[^@]+@[^@]+\.[^@]+"], action="fake"), # Fake emails
        PiiRule(name="ssn", patterns=[r"\d{3}-\d{2}-\d{4}"], action="mask"),   # Mask SSNs
        PiiRule(name="phone_num", patterns=[r"\d{3}-\d{4}"], action="hash"), # Hash phones (custom name match)
        PiiRule(name="notes", patterns=[], action="drop"),  # Drop notes (name match)
        PiiRule(
            name="custom_col", 
            patterns=[r"PROPRIETARY"], 
            action="custom", 
            custom_generator=my_custom_generator
        )
    ])
    
    sanitizer = PIISanitizer(config=config)
    
    # 3. Analyze
    detected = sanitizer.analyze(df)
    print("Detected PII Map:", detected)
    
    # 4. Sanitize
    clean_df = sanitizer.sanitize(df, detected)
    print("-" * 20)
    print("Sanitized Data:")
    print(clean_df.head())
    
    # 5. Verify
    # Email should be changed
    assert not clean_df['user_email'].equals(df['user_email']), "Emails should be faked"
    # SSN should be masked
    assert clean_df['ssn_col'].iloc[0].endswith("6789"), "SSN should be masked but show last 4"
    assert clean_df['ssn_col'].iloc[0].startswith("***"), "SSN should be masked"
    # Notes should be dropped
    assert 'notes' not in clean_df.columns, "Notes column should be dropped"
    # Custom col should be transformed
    assert clean_df['custom_col'].iloc[0] == "REDACTED-ID-US", "Custom lambda failed"
    assert clean_df['custom_col'].iloc[1] == "REDACTED-ID-JP", "Custom lambda failed"
    
    print("\nSUCCESS: Privacy module verification passed!")

if __name__ == "__main__":
    test_privacy_robustness()
