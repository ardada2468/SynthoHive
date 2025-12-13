import os
import shutil
import pandas as pd
from pyspark.sql import SparkSession
from syntho_hive.interface.config import Metadata, PrivacyConfig
from syntho_hive.interface.synthesizer import Synthesizer

# --- 1. SETUP SPARKSESSION ---
spark = SparkSession.builder \
    .appName("SynthoHive_QuickStart") \
    .master("local[1]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

# --- 2. CREATE DUMMY DATA ---
data_dir = "./quickstart_data"
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir)

raw_data = pd.DataFrame({
    "user_id": range(1, 101),
    "age": [20, 30, 40, 50] * 25,
    "city": ["New York", "London", "Tokyo", "Paris"] * 25,
    "income": [50000.0, 60000.0, 75000.0, 90000.0] * 25
})

input_path = f"{data_dir}/users_input.parquet"
raw_data.to_parquet(input_path)
print(f"✅ Dummy data created at {input_path}")

# --- 3. DEFINE METADATA ---
metadata = Metadata()
metadata.add_table("users", pk="user_id")

# --- 4. CONFIGURE & TRAIN ---
privacy = PrivacyConfig()

synth = Synthesizer(
    metadata=metadata,
    privacy_config=privacy,
    spark_session=spark
)

print("🚀 Training model...")
synth.fit(
    data={"users": input_path}, 
    epochs=1,        # Logic check: keeping it fast for verification
    batch_size=50
)

# --- 5. GENERATE DATA ---
print("✨ Generating data...")
output_base_path = f"{data_dir}/output"
output_paths = synth.sample(
    num_rows={"users": 50},
    output_path=output_base_path,
    output_format="parquet"
)

# --- 6. INSPECT RESULTS ---
# The return value of sample with output_path set is a dict table_name -> directory_path
# e.g. {'users': './quickstart_data/output/users'}
# Parquet files are inside that directory.
print(f"Output paths: {output_paths}")

synth_df = pd.read_parquet(output_paths["users"])
print(f"\n📊 Generated {len(synth_df)} synthetic records:")
print(synth_df.head())

# Clean up
try:
    spark.stop()
except:
    pass
