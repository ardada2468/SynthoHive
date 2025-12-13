import pandas as pd
from pyspark.sql import SparkSession
from syntho_hive.interface.config import Metadata, PrivacyConfig
from syntho_hive.interface.synthesizer import Synthesizer
import shutil

# Initialize Spark (Local Mode)
spark = SparkSession.builder \
    .appName("SynthoHive_QuickStart") \
    .master("local[1]") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()
    
# Create some dummy data to simulate a real "users" table
raw_data = pd.DataFrame({
    "user_id": range(1, 101),
    "age": [20, 30, 40, 50] * 25,
    "city": ["New York", "London", "Tokyo", "Paris"] * 25,
    "income": [50000.0, 60000.0, 75000.0, 90000.0] * 25
})

# Save as Parquet
pd_path = "/tmp/users_data.parquet"
raw_data.to_parquet(pd_path)
print(f"Dummy data created at {pd_path}")

# Define Metadata
metadata = Metadata()
metadata.add_table("users", pk="user_id")

# Configure Privacy
privacy = PrivacyConfig()

# Initialize Synthesizer
synth = Synthesizer(
    metadata=metadata,
    privacy_config=privacy,
    spark_session=spark
)

# Fit the model
data_paths = {
    "users": pd_path
}

print("Training model...")
synth.fit(
    data=data_paths, 
    epochs=1,        # 1 epoch for testing
    batch_size=50
)

# Generate Data
print("Sampling data...")
# Clean up output dir first
shutil.rmtree("/tmp/syntho_hive_output", ignore_errors=True)

output_paths = synth.sample(
    num_rows={"users": 10},
    output_format="parquet"
)

# Read and inspect results
synth_df = pd.read_parquet(output_paths["users"])
print(f"\nGenerated {len(synth_df)} synthetic records:")
print(synth_df.head())
