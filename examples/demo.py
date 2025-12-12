from syntho_hive import Metadata, PrivacyConfig
from syntho_hive.relational.orchestrator import StagedOrchestrator
from syntho_hive.core.models.ctgan import CTGAN

def main():
    # 1. Define Schema
    meta = Metadata()
    meta.add_table(
        name="users", 
        pk="user_id", 
        pii_cols=["email", "name"],
        high_cardinality_cols=["city"]
    )
    meta.add_table(
        name="orders", 
        pk="order_id", 
        fk={"user_id": "users.user_id"},
        parent_context_cols=["users.region"]
    )

    # 2. Configure Privacy
    privacy = PrivacyConfig(
        enable_differential_privacy=True,
        epsilon=1.0,
        pii_strategy="context_aware_faker"
    )

    print("Metadata and Privacy Configured.")
    
    # 3. Initialize Orchestrator (Mock Spark)
    # real_spark = SparkSession.builder.getOrCreate()
    orchestrator = StagedOrchestrator(metadata=meta, spark=None)
    
    print("Orchestrator Initialized.")
    
    # 4. Generate Data (Dry Run)
    # orchestrator.fit_all(...)
    # orchestrator.generate(...)
    print("Example workflow setup complete.")

if __name__ == "__main__":
    main()
