
import os
import shutil
import pandas as pd
import torch
import numpy as np
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata

def test_checkpointing():
    # Setup
    checkpoint_dir = "./test_checkpoints"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    
    # Dummy Data
    df = pd.DataFrame({
        "A": np.random.randn(100),
        "B": np.random.randint(0, 10, 100)
    })
    
    # Valid Metadata
    meta = Metadata()
    meta.add_table("test_table", pk="id")
    
    # Add ID column to df to match PK
    df["id"] = range(len(df))
    
    print("Initializing CTGAN...")
    model = CTGAN(
        metadata=meta,
        batch_size=10,
        epochs=5,
        device="cpu"
    )
    
    print("Training with checkpointing...")
    # fit expect data as DataFrame
    model.fit(
        df, 
        table_name="test_table",
        checkpoint_dir=checkpoint_dir,
        log_metrics=True
    )
    
    # Verification
    print("Verifying artifacts...")
    
    files = os.listdir(checkpoint_dir)
    print(f"Files in {checkpoint_dir}: {files}")
    
    assert "best_model.pt" in files, "best_model.pt hidden"
    assert "last_model.pt" in files, "last_model.pt missing"
    assert "training_metrics.csv" in files, "training_metrics.csv missing"
    
    # Check Metric Content
    metrics_df = pd.read_csv(os.path.join(checkpoint_dir, "training_metrics.csv"))
    print(f"Metrics head:\n{metrics_df.head()}")
    assert not metrics_df.empty, "Metrics CSV is empty"
    assert "epoch" in metrics_df.columns
    assert "loss_g" in metrics_df.columns
    assert "loss_d" in metrics_df.columns
    
    print("Verification Successful!")
    
    # Cleanup
    shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    test_checkpointing()
