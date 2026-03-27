import os
import pandas as pd
import torch
import numpy as np
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata


def test_checkpointing(tmp_path):
    # Setup
    checkpoint_dir = str(tmp_path / "test_checkpoints")

    # Dummy Data
    df = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randint(0, 10, 100)})

    # Valid Metadata
    meta = Metadata()
    meta.add_table("test_table", pk="id")

    # Add ID column to df to match PK
    df["id"] = range(len(df))

    print("Initializing CTGAN...")
    model = CTGAN(metadata=meta, batch_size=10, epochs=5, device="cpu")

    print("Training with checkpointing...")
    # fit expect data as DataFrame
    model.fit(
        df,
        table_name="test_table",
        checkpoint_dir=checkpoint_dir,
        log_metrics=True,
        checkpoint_interval=1,
        progress_bar=False,
    )

    # Verification
    print("Verifying artifacts...")

    files = os.listdir(checkpoint_dir)
    print(f"Files in {checkpoint_dir}: {files}")

    assert "best_checkpoint" in files, "best_checkpoint directory missing"
    assert os.path.isdir(os.path.join(checkpoint_dir, "best_checkpoint")), (
        "best_checkpoint must be a directory"
    )
    assert "final_checkpoint" in files, "final_checkpoint directory missing"
    assert os.path.isdir(os.path.join(checkpoint_dir, "final_checkpoint")), (
        "final_checkpoint must be a directory"
    )
    assert "training_metrics.csv" in files, "training_metrics.csv missing"

    # Check Metric Content
    metrics_df = pd.read_csv(os.path.join(checkpoint_dir, "training_metrics.csv"))
    print(f"Metrics head:\n{metrics_df.head()}")
    assert not metrics_df.empty, "Metrics CSV is empty"
    assert "epoch" in metrics_df.columns
    assert "loss_g" in metrics_df.columns
    assert "loss_d" in metrics_df.columns

    print("Verification Successful!")


if __name__ == "__main__":
    import tempfile, pathlib

    test_checkpointing(pathlib.Path(tempfile.mkdtemp()))
