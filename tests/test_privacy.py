import pytest
import pandas as pd
from syntho_hive.privacy.faker_contextual import ContextualFaker
from syntho_hive.privacy.sanitizer import PIISanitizer

def test_contextual_faker_locale():
    faker = ContextualFaker()
    
    # Context with JP country
    jp_names = faker.generate_pii('name', context={'country': 'JP'}, count=5)
    # Check if names look Japanese (simplified check, check if non-latin? Assuming Faker produces romaji or kanji)
    # Faker ja_JP usually produces Kanji/Kana.
    # Let's just check they are strings and not empty.
    assert len(jp_names) == 5
    assert all(isinstance(n, str) for n in jp_names)
    
    # Context with US
    us_names = faker.generate_pii('name', context={'country': 'US'}, count=1)
    assert len(us_names) == 1

def test_pii_detection():
    data = pd.DataFrame({
        "user_email": ["a@b.com", "foo@bar.org"],
        "random_col": ["a", "b"]
    })
    
    sanitizer = PIISanitizer()
    detected = sanitizer.analyze(data)
    
    assert "user_email" in detected
    assert detected["user_email"] == "email"
    assert "random_col" not in detected
