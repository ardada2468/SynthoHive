"""Tests for Phase 8: Training Observability (CORE-05, QUAL-03)."""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata


def _make_model_and_data(epochs: int = 5, n_rows: int = 60):
    """Helper: return (model, dataframe, metadata) for testing."""
    np.random.seed(42)
    meta = Metadata()
    meta.add_table("obs_table", pk="id")
    df = pd.DataFrame(
        {
            "id": range(n_rows),
            "age": np.random.randint(18, 80, n_rows),
            "score": np.random.randn(n_rows).astype(float),
            "category": np.random.choice(["A", "B", "C"], n_rows),
        }
    )
    model = CTGAN(metadata=meta, batch_size=20, epochs=epochs, device="cpu")
    return model, df, meta


# -- CORE-05: Structured log events ------------------------------------------


def test_epoch_log_events():
    """Every epoch emits an epoch_end event with required fields."""
    epochs = 4
    model, df, _ = _make_model_and_data(epochs=epochs)

    with capture_logs() as logs:
        model.fit(df, table_name="obs_table", progress_bar=False)

    epoch_events = [e for e in logs if e.get("event") == "epoch_end"]
    assert len(epoch_events) == epochs, (
        f"Expected {epochs} epoch_end events; got {len(epoch_events)}"
    )

    required_fields = {"epoch", "g_loss", "d_loss", "eta_seconds"}
    for ev in epoch_events:
        missing = required_fields - ev.keys()
        assert not missing, f"epoch_end missing fields {missing}: {ev}"


def test_training_start_event():
    """training_start event is emitted exactly once before training begins."""
    model, df, _ = _make_model_and_data(epochs=3)

    with capture_logs() as logs:
        model.fit(
            df, table_name="obs_table", progress_bar=False, checkpoint_interval=10
        )

    start_events = [e for e in logs if e.get("event") == "training_start"]
    assert len(start_events) == 1, f"Expected 1 training_start; got {len(start_events)}"

    ev = start_events[0]
    assert ev["total_epochs"] == 3
    assert ev["checkpoint_interval"] == 10
    assert "batch_size" in ev
    assert "embedding_dim" in ev


def test_training_complete_event():
    """training_complete event is emitted exactly once after training ends."""
    model, df, _ = _make_model_and_data(epochs=3)

    with capture_logs() as logs:
        model.fit(df, table_name="obs_table", progress_bar=False)

    complete_events = [e for e in logs if e.get("event") == "training_complete"]
    assert len(complete_events) == 1, (
        f"Expected 1 training_complete; got {len(complete_events)}"
    )

    ev = complete_events[0]
    assert ev["total_epochs"] == 3
    assert "best_epoch" in ev
    assert "best_val_metric" in ev
    assert "checkpoint_path" in ev


def test_eta_seconds_non_zero_after_first_epoch():
    """eta_seconds is non-zero for all epochs except the final one."""
    epochs = 5
    model, df, _ = _make_model_and_data(epochs=epochs)

    with capture_logs() as logs:
        model.fit(df, table_name="obs_table", progress_bar=False)

    epoch_events = [e for e in logs if e.get("event") == "epoch_end"]
    assert len(epoch_events) == epochs

    # All epochs except the last must have non-zero eta
    for ev in epoch_events[:-1]:
        assert ev["eta_seconds"] >= 0, (
            f"eta_seconds must be >= 0; got {ev['eta_seconds']} at epoch {ev['epoch']}"
        )
    # The last epoch should have eta_seconds == 0.0 (no remaining epochs)
    assert epoch_events[-1]["eta_seconds"] == 0.0, (
        f"Final epoch should have eta_seconds=0.0; got {epoch_events[-1]['eta_seconds']}"
    )


def test_progress_bar_false_does_not_suppress_log_events():
    """progress_bar=False suppresses tqdm bar but log events still fire."""
    model, df, _ = _make_model_and_data(epochs=3)

    with capture_logs() as logs:
        model.fit(df, table_name="obs_table", progress_bar=False)

    epoch_events = [e for e in logs if e.get("event") == "epoch_end"]
    assert len(epoch_events) == 3, "Log events must fire even when progress_bar=False"


# -- QUAL-03: Validation-metric checkpointing --------------------------------


def test_best_checkpoint_is_best_val_epoch():
    """best_checkpoint corresponds to the epoch with lowest val_metric."""
    with tempfile.TemporaryDirectory() as ckpt_dir:
        epochs = 10
        model, df, meta = _make_model_and_data(epochs=epochs)

        with capture_logs() as logs:
            model.fit(
                df,
                table_name="obs_table",
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=5,
                progress_bar=False,
            )

        files = os.listdir(ckpt_dir)

        # Both checkpoint dirs must exist
        assert "best_checkpoint" in files, f"best_checkpoint missing; dir={files}"
        assert "final_checkpoint" in files, f"final_checkpoint missing; dir={files}"
        assert os.path.isdir(os.path.join(ckpt_dir, "best_checkpoint"))
        assert os.path.isdir(os.path.join(ckpt_dir, "final_checkpoint"))

        # training_complete has best_epoch >= 0 and finite best_val_metric
        complete_events = [e for e in logs if e.get("event") == "training_complete"]
        assert len(complete_events) == 1
        ev = complete_events[0]
        assert ev["best_epoch"] >= 0, f"best_epoch={ev['best_epoch']}"
        assert ev["best_val_metric"] < float("inf"), (
            f"best_val_metric should be finite; got {ev['best_val_metric']}"
        )

        # checkpoint_path points to the best_checkpoint directory
        assert ev["checkpoint_path"] is not None
        assert "best_checkpoint" in ev["checkpoint_path"]

        # val_metric present only on checkpoint epoch_end events
        ckpt_epoch_events = [
            e for e in logs if e.get("event") == "epoch_end" and "val_metric" in e
        ]
        # checkpoint_interval=5, epochs=10 -> 2 checkpoint epochs (4 and 9)
        assert len(ckpt_epoch_events) == 2, (
            f"Expected 2 checkpoint epoch_end events; got {len(ckpt_epoch_events)}"
        )

        # Non-checkpoint epoch_end events must NOT have val_metric
        non_ckpt_epoch_events = [
            e for e in logs if e.get("event") == "epoch_end" and "val_metric" not in e
        ]
        assert len(non_ckpt_epoch_events) == 8, (
            f"Expected 8 non-checkpoint events; got {len(non_ckpt_epoch_events)}"
        )


def test_cold_load_uses_best_checkpoint():
    """Cold CTGAN.load(best_checkpoint) + sample() succeeds without error."""
    with tempfile.TemporaryDirectory() as ckpt_dir:
        epochs = 6
        model, df, meta = _make_model_and_data(epochs=epochs)
        model.fit(
            df,
            table_name="obs_table",
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=3,
            progress_bar=False,
        )

        best_path = os.path.join(ckpt_dir, "best_checkpoint")
        assert os.path.isdir(best_path), "best_checkpoint directory must exist"

        # Cold load into a fresh model instance
        fresh = CTGAN(metadata=meta, batch_size=20, epochs=epochs, device="cpu")
        fresh.load(best_path)
        synth = fresh.sample(10)

        assert len(synth) == 10, f"Expected 10 rows; got {len(synth)}"
        # Generated columns should match data schema (minus pk)
        expected_cols = {"age", "score", "category"}
        actual_cols = set(synth.columns)
        assert expected_cols.issubset(actual_cols), (
            f"Expected columns {expected_cols}; got {actual_cols}"
        )
