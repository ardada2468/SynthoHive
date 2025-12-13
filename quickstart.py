import os
import shutil
import pandas as pd
from pyspark.sql import SparkSession
from syntho_hive.interface.config import Metadata, PrivacyConfig
from syntho_hive.interface.synthesizer import Synthesizer

# --- 1. SETUP SPARKSESSION ---
# Initialize a local Spark session. In production, this would connect to your cluster.
spark = SparkSession.builder \
    .appName("SynthoHive_QuickStart") \
    .master("local[1]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

# --- 2. CREATE DUMMY DATA ---
# For this demo, we create a small dataset and save it to disk. 
# SynthoHive normally reads from data lakes (Delta/Parquet) or Hive tables.
raw_data = pd.DataFrame({
    "user_id": range(1, 101),
    "age": [20, 30, 40, 50] * 25,
    "city": ["New York", "London", "Tokyo", "Paris"] * 25,
    "income": [50000.0, 60000.0, 75000.0, 90000.0] * 25
})

# Setup local data directory
data_dir = "./quickstart_data"
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir)

input_path = f"{data_dir}/users_input.parquet"
raw_data.to_parquet(input_path)
print(f"✅ Dummy data created at {input_path}")

# --- 3. DEFINE METADATA ---
# Tell SynthoHive about the schema.
# We explicitly mark the Primary Key (pk).
metadata = Metadata()
metadata.add_table("users", pk="user_id")

# --- 4. CONFIGURE & TRAIN ---
# PrivacyConfig controls sanitization (e.g. masking PII), default is safe.
privacy = PrivacyConfig()

# Initialize the Synthesizer
# It acts as the coordinator for reading data, privacy enforcement, and training.
synth = Synthesizer(
    metadata=metadata,
    privacy_config=privacy,
    spark_session=spark
)

# Fit the model
# We point 'users' to the parquet file we just made.
print("🚀 Training model...")
synth.fit(
    data={"users": input_path}, 
    epochs=10,        # Use 300+ for production quality
    batch_size=50
)

# --- 5. GENERATE DATA ---
# Sample new synthetic records from the learned distribution.
print("✨ Generating data...")
output_base_path = f"{data_dir}/output"
output_paths = synth.sample(
    num_rows={"users": 50},
    output_format="parquet",  # Using parquet for easy local reading. Default is 'delta'.
    output_path=output_base_path
)

# --- 6. INSPECT RESULTS ---
synth_df = pd.read_parquet(output_paths["users"])
print(f"\n📊 Generated {len(synth_df)} synthetic records:")
print(synth_df.head())

# Clean up
spark.stop()
