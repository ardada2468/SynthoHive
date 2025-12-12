
import os
import sys
import shutil
import pandas as pd
import numpy as np
from faker import Faker
import random
from typing import Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from pyspark.sql import SparkSession
except ImportError:
    print("PySpark not installed. Skipping Spark-dependent parts.")
    sys.exit(0)

from syntho_hive.interface.config import Metadata, Constraint
from syntho_hive.privacy.sanitizer import PIISanitizer, PrivacyConfig, PiiRule
from syntho_hive.relational.orchestrator import StagedOrchestrator
from syntho_hive.validation.report_generator import ValidationReport

# --- Configuration ---
OUTPUT_DIR = "output/test_retail"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean")
SYNTH_DIR = os.path.join(OUTPUT_DIR, "synthetic")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.html")

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

def clean_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(RAW_DIR)
    os.makedirs(CLEAN_DIR)
    os.makedirs(SYNTH_DIR)

# --- Phase 1: Ground Truth Generation ---
def generate_ground_truth():
    print(">>> Phase 1: Generating Ground Truth Data...")
    
    # 1. Regions (Root)
    regions = []
    for i in range(1, 6):
        regions.append({
            "region_id": i,
            "region_code": fake.state_abbr(),
            "country": "USA"
        })
    df_regions = pd.DataFrame(regions)
    df_regions.to_csv(f"{RAW_DIR}/regions.csv", index=False)
    
    # 2. Products (Root)
    products = []
    categories = ["Electronics", "Clothing", "Home", "Books"]
    for i in range(1, 51):
        products.append({
            "product_id": i,
            "category": random.choice(categories),
            "price": round(random.uniform(10.0, 500.0), 2),
            "stock_level": random.randint(0, 1000)
        })
    df_products = pd.DataFrame(products)
    df_products.to_csv(f"{RAW_DIR}/products.csv", index=False)
    
    # 3. Users (Child of Regions)
    users = []
    for i in range(1, 101):
        users.append({
            "user_id": i,
            "region_id": random.choice(regions)["region_id"],
            "email": fake.email(),
            "phone": fake.phone_number(),
            "age": random.randint(18, 80),
            "signup_date": fake.date_between(start_date='-2y', end_date='today').isoformat()
        })
    df_users = pd.DataFrame(users)
    df_users.to_csv(f"{RAW_DIR}/users.csv", index=False)
    
    # 4. Orders (Child of Users)
    orders = []
    order_id_counter = 1
    for user in users:
        # Each user makes 0-5 orders
        num_orders = random.randint(0, 5)
        for _ in range(num_orders):
            orders.append({
                "order_id": order_id_counter,
                "user_id": user["user_id"],
                "total": round(random.uniform(20.0, 1000.0), 2),
                "status": random.choice(["Pending", "Shipped", "Delivered", "Cancelled"])
            })
            order_id_counter += 1
    df_orders = pd.DataFrame(orders)
    df_orders.to_csv(f"{RAW_DIR}/orders.csv", index=False)
    
    # 5. OrderItems (Child of Orders)
    # Note: To simplify, we treat OrderItems as child of Orders. 
    # Product_id is just an integer feature here (Loose FK).
    items = []
    item_id_counter = 1
    for order in orders:
        if order["status"] == "Cancelled":
            continue
        num_items = random.randint(1, 5)
        for _ in range(num_items):
            items.append({
                "item_id": item_id_counter,
                "order_id": order["order_id"],
                "product_id": random.choice(products)["product_id"],
                "quantity": random.randint(1, 10)
            })
            item_id_counter += 1
    df_items = pd.DataFrame(items)
    df_items.to_csv(f"{RAW_DIR}/order_items.csv", index=False)
    
    # 6. Shipments (Child of Orders 1:1ish)
    shipments = []
    shipment_id_counter = 1
    for order in orders:
        if order["status"] in ["Shipped", "Delivered"]:
            shipments.append({
                "shipment_id": shipment_id_counter,
                "order_id": order["order_id"],
                "address": fake.address().replace("\n", ", "),
                "tracking_num": fake.uuid4(),
                "delivery_days": random.randint(1, 7)
            })
            shipment_id_counter += 1
    df_shipments = pd.DataFrame(shipments)
    df_shipments.to_csv(f"{RAW_DIR}/shipments.csv", index=False)
    
    print(f"Generated {len(df_users)} users, {len(df_orders)} orders.")

# --- Phase 2: Privacy Sanitization ---
def sanitize_data():
    print(">>> Phase 2: Sanitizing PII Data...")
    
    # Define Privacy Config
    # Custom rule for address just to show flexibility
    privacy_conf = PrivacyConfig(rules=[
        PiiRule(name="email", patterns=[r"[^@]+@[^@]+\.[^@]+"], action="fake"),
        PiiRule(name="phone", patterns=[r".*"], action="fake"), # Simple catch-all for phone
        PiiRule(name="address", patterns=[r"\d+ .+,.*"], action="fake"), # Heuristic address
        PiiRule(name="tracking_num", patterns=[r".*"], action="mask"), 
    ])
    
    sanitizer = PIISanitizer(config=privacy_conf)
    
    # Process Users
    df_users = pd.read_csv(f"{RAW_DIR}/users.csv")
    clean_users = sanitizer.sanitize(df_users)
    clean_users.to_csv(f"{CLEAN_DIR}/users.csv", index=False)
    
    # Process Shipments
    df_shipments = pd.read_csv(f"{RAW_DIR}/shipments.csv")
    # Manually map columns if detection fails or just force it
    # We'll rely on detection or explicit mapping. Let's force mapping for safety in test.
    pii_map = {"address": "address", "tracking_num": "tracking_num"}
    clean_shipments = sanitizer.sanitize(df_shipments, pii_map=pii_map)
    clean_shipments.to_csv(f"{CLEAN_DIR}/shipments.csv", index=False)
    
    # Copy others
    shutil.copy(f"{RAW_DIR}/regions.csv", f"{CLEAN_DIR}/regions.csv")
    shutil.copy(f"{RAW_DIR}/products.csv", f"{CLEAN_DIR}/products.csv")
    shutil.copy(f"{RAW_DIR}/orders.csv", f"{CLEAN_DIR}/orders.csv")
    shutil.copy(f"{RAW_DIR}/order_items.csv", f"{CLEAN_DIR}/order_items.csv")

# --- Phase 3: Relational Modeling ---
def train_and_generate():
    print(">>> Phase 3: Training & Generating Synthetic Data...")
    
    spark = SparkSession.builder \
        .appName("SynthoHiveRetailTest") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Define Metadata
    meta = Metadata()
    
    # Roots
    meta.add_table("regions", pk="region_id", pii_cols=[])
    meta.add_table("products", pk="product_id", pii_cols=[])
    
    # Children
    meta.add_table("users", pk="user_id", 
                   fk={"region_id": "regions.region_id"},
                   constraints={
                       "age": Constraint(dtype="int", min=18, max=90)
                   },
                   parent_context_cols=["region_code"]) # Condition on region code
                   
    meta.add_table("orders", pk="order_id", 
                   fk={"user_id": "users.user_id"},
                   constraints={
                       "total": Constraint(dtype="float", min=0.0, max=10000.0)
                   },
                   parent_context_cols=["age"]) # Spending might depend on age

    meta.add_table("order_items", pk="item_id", 
                   fk={
                       "order_id": "orders.order_id",
                       "product_id": "products.product_id"
                   },
                   constraints={
                       "quantity": Constraint(dtype="int", min=1, max=3000)
                   },
                   parent_context_cols=["total"]) # Quantity/Products might relate to total is better -> order_total in csv is "total"
                   
    meta.add_table("shipments", pk="shipment_id", 
                   fk={"order_id": "orders.order_id"},
                   parent_context_cols=["status"],
                   constraints={
                       "delivery_days": Constraint(dtype="int", min=1, max=10)
                   }) # Only shipped orders have shipments
    
    try:
        meta.validate_schema()
        print("Schema Validated.")
    except Exception as e:
        print(f"Schema Validation Failed: {e}")
        return

    # 2. Orchestrator
    orchestrator = StagedOrchestrator(metadata=meta, spark=spark)
    
    # Data Paths (Absolute for Spark)
    abs_clean_dir = os.path.abspath(CLEAN_DIR)
    data_paths = {
        "regions": f"file://{abs_clean_dir}/regions.csv",
        "products": f"file://{abs_clean_dir}/products.csv",
        "users": f"file://{abs_clean_dir}/users.csv",
        "orders": f"file://{abs_clean_dir}/orders.csv",
        "order_items": f"file://{abs_clean_dir}/order_items.csv",
        "shipments": f"file://{abs_clean_dir}/shipments.csv",
    }
    
    orchestrator.fit_all(data_paths, embedding_threshold=80, epochs=100)
    
    # 3. Generate
    # Scale up slightly: 200 users instead of 100? No let's keep it quick for test.
    # Logic: Regions is root. Products is root.
    # We need to tell it how many rows to generate for ROOTS.
    orchestrator.generate(
        num_rows_root={"regions": 5, "products": 50}, 
        output_path_base=os.path.abspath(SYNTH_DIR)
    )
    
    spark.stop()

# --- Phase 4: Validation ---
def validate_results():
    print(">>> Phase 4: Validating...")
    
    report = ValidationReport()
    
    real_data = {}
    synth_data = {}
    
    # Load Real (Clean)
    for f in os.listdir(CLEAN_DIR):
        if f.endswith(".csv"):
            name = f.replace(".csv", "")
            real_data[name] = pd.read_csv(os.path.join(CLEAN_DIR, f))
            
    # Load Synth
    for table in real_data.keys():
        path = os.path.join(SYNTH_DIR, table)
        if os.path.exists(path):
            try:
                # Try reading parquet first
                synth_data[table] = pd.read_parquet(path)
            except Exception as e:
                # print(f"Direct parquet read failed: {e}")
                try:
                    # Try manual walk for parquet parts (Spark style)
                    import glob
                    parquet_files = glob.glob(os.path.join(path, "*.parquet"))
                    if parquet_files:
                        synth_data[table] = pd.concat([pd.read_parquet(f) for f in parquet_files])
                    else:
                        # Try reading as directory of CSVs
                        all_files = glob.glob(os.path.join(path, "*.csv"))
                        if all_files:
                            synth_data[table] = pd.concat((pd.read_csv(f) for f in all_files))
                        else:
                            # Maybe it is a single file?
                            synth_data[table] = pd.read_csv(path)
                except Exception as e2:
                    print(f"Could not read synthetic data for {table}: {e2}")

    report.generate(
        real_data=real_data,
        synth_data=synth_data,
        output_path=REPORT_PATH
    )

if __name__ == "__main__":
    clean_dirs()
    generate_ground_truth()
    sanitize_data()
    train_and_generate()
    validate_results()
    print(">>> E2E Test Complete.")
