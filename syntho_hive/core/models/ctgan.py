import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import os
import csv
import sys
import time
import structlog
from tqdm import trange
from typing import Optional, List, Any, Tuple
from .base import ConditionalGenerativeModel
from .layers import ResidualLayer, Discriminator, EntityEmbeddingLayer
from syntho_hive.core.data.transformer import DataTransformer
from syntho_hive.exceptions import (
    SerializationError,
    TrainingError,
    GenerationError,
    ConstraintViolationError,
)

log = structlog.get_logger()


def _set_seed(seed: int) -> None:
    """Set all RNG seeds for deterministic behavior.

    Covers PyTorch, NumPy, and Python's random module. Spark code paths are
    explicitly excluded — Spark's distributed shuffle is inherently
    non-deterministic and cannot be seeded via this function.

    Args:
        seed: Integer seed value to apply across all supported RNG backends.
    """
    import random
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # CuDNN determinism (no-op on CPU, required for GPU reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculate the WGAN-GP gradient penalty term.

    Args:
        discriminator: Discriminator network used to score samples.
        real_samples: Tensor of real samples after preprocessing.
        fake_samples: Tensor of generated samples.
        device: Torch device for computation.

    Returns:
        Scalar gradient penalty encouraging Lipschitz continuity.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = discriminator(interpolates)
    fake = torch.ones((real_samples.size(0), 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class CTGAN(ConditionalGenerativeModel):
    """Conditional Tabular GAN with entity embeddings and parent context."""

    def __init__(
        self,
        metadata: Any,
        embedding_dim: int = 128,
        generator_dim: Tuple[int, int] = (256, 256),
        discriminator_dim: Tuple[int, int] = (256, 256),
        batch_size: int = 500,
        epochs: int = 300,
        device: str = "cpu",
        embedding_threshold: int = 50,
        discriminator_steps: int = 5,
        legacy_context_conditioning: bool = False,
    ):
        """Create a CTGAN instance configured for tabular synthesis.

        Args:
            metadata: Table metadata describing columns and constraints.
            embedding_dim: Dimension of input noise vector.
            generator_dim: Hidden layer widths for the generator.
            discriminator_dim: Hidden layer widths for the discriminator.
            batch_size: Training batch size.
            epochs: Number of training epochs.
            device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
            embedding_threshold: Cardinality threshold for switching to embeddings.
            discriminator_steps: Number of discriminator steps per generator step.
            legacy_context_conditioning: If True, reuses discriminator batch context
                in generator step (legacy behavior). Default False applies correct
                independent resample, which prevents FK cardinality drift.

        Raises:
            ValueError: If any numeric hyperparameter is out of range.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if discriminator_steps < 1:
            raise ValueError(
                f"discriminator_steps must be >= 1, got {discriminator_steps}"
            )
        self.metadata = metadata
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)
        self.discriminator_steps = discriminator_steps
        # Prioritize init arg, fallback to metadata if available, else default (already 50)
        self.embedding_threshold = embedding_threshold
        self.legacy_context_conditioning = legacy_context_conditioning

        self.generator = None
        self.discriminator = None
        self.transformer = DataTransformer(
            metadata, embedding_threshold=self.embedding_threshold
        )
        self.context_transformer = DataTransformer(
            metadata, embedding_threshold=self.embedding_threshold
        )

        # Embedding Layers
        self.embedding_layers = nn.ModuleDict()
        self.data_column_info = []  # List of tuples: (dim, type, related_info)

    @staticmethod
    def _activation_spec(col_info) -> list:
        """Build the per-block activation layout for a non-embedding column.

        Continuous columns are laid out as ``[mode one-hot, scalar, (null flag)]``
        and one-hot categoricals as a single softmax block, mirroring
        ``DataTransformer``'s output layout.
        """
        if col_info["type"] == "continuous":
            t = col_info["transformer"]
            spec = [("softmax", t.n_components), ("tanh", 1)]
            if t.output_dim == t.n_components + 2:  # null indicator present
                spec.append(("sigmoid", 1))
            return spec
        # One-hot categorical: single softmax block.
        return [("softmax", col_info["dim"])]

    def _compile_layout(self, transformer):
        """Analyze transformer output to map column indices and types.

        Args:
            transformer: Fitted ``DataTransformer`` for the child table.
        """
        self.data_column_info = []
        self.embedding_layers = nn.ModuleDict()

        current_idx = 0
        for idx, (col, info) in enumerate(transformer._column_info.items()):
            if info["type"] == "categorical_embedding":
                # Create Embedding Layer
                num_categories = info["num_categories"]
                # Heuristic for embedding dimension: min(50, num_categories/2)
                emb_dim = min(50, (num_categories + 1) // 2)

                # ModuleDict keys must be '.'-free strings; index-based keys
                # support arbitrary column names (ints, dotted names, ...).
                emb_key = f"emb_{idx}"
                self.embedding_layers[emb_key] = EntityEmbeddingLayer(
                    num_categories, emb_dim
                ).to(self.device)

                self.data_column_info.append(
                    {
                        "name": col,
                        "type": "embedding",
                        "emb_key": emb_key,
                        "input_idx": current_idx,
                        "input_dim": 1,
                        "output_dim": emb_dim,
                        "num_categories": num_categories,
                    }
                )
                current_idx += 1
            else:
                self.data_column_info.append(
                    {
                        "name": col,
                        "type": "normal",
                        "input_idx": current_idx,
                        "input_dim": info["dim"],
                        "output_dim": info["dim"],
                        "activation_spec": self._activation_spec(info),
                    }
                )
                current_idx += info["dim"]

    def _embedding_layer(self, info) -> EntityEmbeddingLayer:
        """Resolve the embedding layer for a column-info entry (legacy-key aware)."""
        key = info.get("emb_key", str(info["name"]))
        return self.embedding_layers[key]

    def _activate(self, fake_raw: torch.Tensor) -> torch.Tensor:
        """Apply per-block output activations to raw generator output.

        The generator ends in a bare Linear layer; per the CTGAN design each
        block needs its own activation before it can be compared with real
        (exactly one-hot) data: gumbel-softmax for one-hot/mode/categorical
        blocks, tanh for the normalized scalar, sigmoid for null indicators.
        """
        parts = []
        ptr = 0
        for info in self.data_column_info:
            if info["type"] == "embedding":
                dim = info["num_categories"]
                parts.append(F.gumbel_softmax(fake_raw[:, ptr : ptr + dim], tau=0.2))
                ptr += dim
            else:
                spec = info.get("activation_spec")
                if spec is None:
                    # Legacy checkpoint without a spec — pass through unchanged.
                    dim = info["output_dim"]
                    parts.append(fake_raw[:, ptr : ptr + dim])
                    ptr += dim
                    continue
                for kind, dim in spec:
                    blk = fake_raw[:, ptr : ptr + dim]
                    if kind == "softmax":
                        blk = F.gumbel_softmax(blk, tau=0.2)
                    elif kind == "tanh":
                        blk = torch.tanh(blk)
                    elif kind == "sigmoid":
                        blk = torch.sigmoid(blk)
                    parts.append(blk)
                    ptr += dim
        return torch.cat(parts, dim=1)

    def _fake_to_disc_input(self, activated: torch.Tensor) -> torch.Tensor:
        """Map activated generator output into discriminator space (soft embeddings)."""
        parts = []
        ptr = 0
        for info in self.data_column_info:
            if info["type"] == "embedding":
                dim = info["num_categories"]
                probs = activated[:, ptr : ptr + dim]
                ptr += dim
                parts.append(self._embedding_layer(info).forward_soft(probs))
            else:
                dim = info["output_dim"]
                parts.append(activated[:, ptr : ptr + dim])
                ptr += dim
        return torch.cat(parts, dim=1)

    def _real_to_disc_input(self, real_batch: torch.Tensor) -> torch.Tensor:
        """Map transformed real data (indices + one-hots) into discriminator space."""
        parts = []
        ptr = 0
        for info in self.data_column_info:
            dim = info["input_dim"]
            col_data = real_batch[:, ptr : ptr + dim]
            ptr += dim
            if info["type"] == "embedding":
                parts.append(self._embedding_layer(info)(col_data.long().squeeze(1)))
            else:
                parts.append(col_data)
        return torch.cat(parts, dim=1)

    def _build_model(self, transformer_output_dim: int, context_dim: int = 0):
        """Instantiate generator and discriminator modules.

        Args:
            transformer_output_dim: Flattened dimension of transformed child data.
            context_dim: Flattened dimension of transformed context (if any).
        """
        # 1. Compile Layout first
        self._compile_layout(self.transformer)

        # 2. Calculate Generator Output Dim & Discriminator Input Dim
        gen_output_dim = 0
        disc_input_dim_base = 0

        for info in self.data_column_info:
            if info["type"] == "embedding":
                gen_output_dim += info["num_categories"]  # Generator outputs logits
                disc_input_dim_base += info["output_dim"]  # D sees embeddings
            else:
                gen_output_dim += info["output_dim"]
                disc_input_dim_base += info["output_dim"]

        # Generator: Noise + Context -> Data (Logits/Values)
        gen_input_dim = self.embedding_dim + context_dim

        self.generator = nn.Sequential(
            ResidualLayer(gen_input_dim, self.generator_dim[0]),
            ResidualLayer(self.generator_dim[0], self.generator_dim[1]),
            nn.Linear(self.generator_dim[1], gen_output_dim),
        ).to(self.device)

        # Discriminator: Data(Embeddings) + Context -> Score
        disc_input_dim = disc_input_dim_base + context_dim

        self.discriminator = Discriminator(
            disc_input_dim, self.discriminator_dim[0], self.discriminator_dim[1]
        ).to(self.device)

    def fit(
        self,
        data: pd.DataFrame,
        context: Optional[pd.DataFrame] = None,
        table_name: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        log_metrics: bool = True,
        seed: Optional[int] = None,
        progress_bar: bool = True,
        checkpoint_interval: int = 10,
        **kwargs: Any,
    ) -> None:
        """Train the CTGAN model on tabular data.

        Args:
            data: Child table data (target) to model.
            context: Parent attributes to condition on (aligned row-wise).
            table_name: Table name for metadata lookup and constraint handling.
            checkpoint_dir: Directory to save checkpoints (best model, metrics). Defaults to None.
            log_metrics: Whether to save training metrics to a CSV file. Defaults to True.
            seed: Integer seed for deterministic training. When None, an integer is
                  auto-generated and logged so the run can be reproduced later.
            progress_bar: If True (default), display a tqdm progress bar to stderr during
                  training. Structured log events always emit regardless of this flag.
            checkpoint_interval: Save a validation checkpoint every N epochs. Default 10.
            **kwargs: Extra training options (unused placeholder for compatibility).
        """
        import random as _random

        # Seed handling — auto-generate when not provided so every run is reproducible.
        if seed is None:
            seed = _random.randint(0, 2**31 - 1)
            log.info(
                "training_seed",
                seed=seed,
                message="No seed provided — auto-generated. Log this value to reproduce this run.",
            )
        else:
            log.info("training_seed", seed=seed)

        _set_seed(seed)

        # 0. Setup Checkpointing
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        history = []

        # Validation-metric checkpoint state (QUAL-03)
        _validator = None
        best_val_metric = float("inf")
        best_epoch = -1
        best_checkpoint_path = None

        if checkpoint_dir:
            from syntho_hive.validation.statistical import StatisticalValidator

            _validator = StatisticalValidator()
        # 1. Fit and Transform Data
        self.transformer.fit(data, table_name=table_name, seed=seed)
        train_data = self.transformer.transform(data)
        train_data = torch.from_numpy(train_data).float().to(self.device)

        # 2. Handle Context
        if context is not None:
            if len(data) != len(context):
                raise ValueError(
                    f"Data and context must have same number of rows, "
                    f"got {len(data)} and {len(context)}"
                )

            # Use dedicated transformer for context
            # NOTE: We abuse metdata here slightly. Ideally context comes from a known table (Parent).
            # But context might be a mix of parent columns.
            # For fit, we pass table_name=None to fit on just the columns present in context df.
            self.context_transformer.fit(context, seed=seed)
            context_transformed = self.context_transformer.transform(context)
            context_data = torch.from_numpy(context_transformed).float().to(self.device)
            context_dim = context_data.shape[1]
        else:
            context_data = None
            context_dim = 0

        data_dim = train_data.shape[1]

        # 3. Build Model — always rebuild so a refit with a different schema
        # cannot silently reuse a network with mismatched dimensions.
        self._build_model(data_dim, context_dim)

        all_gen_params = list(self.generator.parameters()) + list(
            self.embedding_layers.parameters()
        )
        optimizer_G = optim.Adam(all_gen_params, lr=2e-4, betas=(0.5, 0.9))
        optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9)
        )

        # 4. Training Loop (WGAN-GP)
        steps_per_epoch = max(len(train_data) // self.batch_size, 1)

        # Emit training_start event
        log.info(
            "training_start",
            total_epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            checkpoint_interval=checkpoint_interval,
        )
        _start_time = time.time()

        # Replace bare for-loop with trange (disable=True suppresses bar; log events always fire)
        pbar = trange(
            self.epochs,
            desc="Training",
            file=sys.stderr,
            leave=True,
            disable=not progress_bar,
        )

        for epoch in pbar:
            for i in range(steps_per_epoch):
                # --- Train Discriminator ---
                for _ in range(self.discriminator_steps):
                    optimizer_D.zero_grad()

                    # Sample real data
                    idx = np.random.randint(0, len(train_data), self.batch_size)
                    real_data_batch = train_data[idx]
                    if context_data is not None:
                        real_context_batch = context_data[idx]
                        real_input = torch.cat(
                            [real_data_batch, real_context_batch], dim=1
                        )
                    else:
                        real_context_batch = None
                        real_input = real_data_batch

                    # Generate fake data. No grads through the generator here —
                    # the D step only updates the discriminator, and backprop
                    # through G in every critic step wastes ~5x compute/memory.
                    noise = torch.randn(
                        self.batch_size, self.embedding_dim, device=self.device
                    )
                    if real_context_batch is not None:
                        gen_input = torch.cat([noise, real_context_batch], dim=1)
                    else:
                        gen_input = noise

                    with torch.no_grad():
                        fake_raw = self.generator(gen_input)
                        fake_data_batch = self._fake_to_disc_input(
                            self._activate(fake_raw)
                        )
                        real_data_processed = self._real_to_disc_input(real_data_batch)

                    if real_context_batch is not None:
                        fake_input = torch.cat(
                            [fake_data_batch, real_context_batch], dim=1
                        )
                        real_input_processed = torch.cat(
                            [real_data_processed, real_context_batch], dim=1
                        )
                    else:
                        fake_input = fake_data_batch
                        real_input_processed = real_data_processed

                    # Compute WGAN loss
                    d_real = self.discriminator(real_input_processed)
                    d_fake = self.discriminator(fake_input)

                    # Gradient Penalty
                    gp = compute_gradient_penalty(
                        self.discriminator,
                        real_input_processed,
                        fake_input,
                        self.device,
                    )

                    loss_D = -torch.mean(d_real) + torch.mean(d_fake) + 10.0 * gp

                    loss_D.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), max_norm=1.0
                    )
                    optimizer_D.step()

                # --- Train Generator ---
                # Train generator once after n_critic discriminator steps
                noise = torch.randn(
                    self.batch_size, self.embedding_dim, device=self.device
                )
                if context_data is not None:
                    if self.legacy_context_conditioning:
                        # Backwards-compatible: reuse last discriminator batch context
                        gen_context_batch = real_context_batch
                    else:
                        # Correct: independently sample fresh context for generator step
                        gen_ctx_idx = np.random.randint(
                            0, len(context_data), self.batch_size
                        )
                        gen_context_batch = context_data[gen_ctx_idx]
                    gen_input = torch.cat([noise, gen_context_batch], dim=1)
                else:
                    gen_input = noise

                fake_raw = self.generator(gen_input)
                fake_data_batch = self._fake_to_disc_input(self._activate(fake_raw))

                if context_data is not None:
                    fake_input = torch.cat([fake_data_batch, gen_context_batch], dim=1)
                else:
                    fake_input = fake_data_batch

                d_fake = self.discriminator(fake_input)
                loss_G = -torch.mean(d_fake)

                optimizer_G.zero_grad()
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.generator.parameters())
                    + list(self.embedding_layers.parameters()),
                    max_norm=1.0,
                )
                optimizer_G.step()

            # --- Checkpointing & Logging ---
            current_loss_g = loss_G.item()
            current_loss_d = loss_D.item()

            # Divergence detection — a NaN loss silently trains to garbage.
            if not (math.isfinite(current_loss_g) and math.isfinite(current_loss_d)):
                raise TrainingError(
                    f"Training diverged at epoch {epoch}: "
                    f"g_loss={current_loss_g}, d_loss={current_loss_d}. "
                    f"Try a smaller learning task, more data, or a different seed."
                )

            # ETA calculation (linear extrapolation)
            _elapsed = time.time() - _start_time
            _epochs_done = epoch + 1
            _elapsed_per_epoch = _elapsed / _epochs_done
            _remaining_epochs = self.epochs - _epochs_done
            eta_seconds = (
                _elapsed_per_epoch * _remaining_epochs
            )  # 0.0 on final epoch — correct

            # Update tqdm postfix (visual only; fires regardless of disable state)
            pbar.set_postfix(
                {
                    "g_loss": f"{current_loss_g:.4f}",
                    "d_loss": f"{current_loss_d:.4f}",
                    "eta": f"{int(eta_seconds)}s",
                }
            )

            # Prepare epoch_end log fields (val_metric added below on checkpoint epochs)
            epoch_log_fields = dict(
                epoch=epoch,
                g_loss=current_loss_g,
                d_loss=current_loss_d,
                eta_seconds=eta_seconds,
            )

            # Checkpoint validation (every checkpoint_interval epochs, only when checkpoint_dir set)
            _is_checkpoint_epoch = (
                checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0
            )

            if _is_checkpoint_epoch:
                if context_dim > 0:
                    log.debug(
                        "checkpoint_validation_skipped",
                        epoch=epoch,
                        note="Skipping checkpoint validation for context-conditioned model",
                    )
                    val_metric = float("inf")
                else:
                    # Generate a small validation sample from current generator state.
                    # Save/restore RNG state so checkpoint validation does not
                    # consume the training RNG stream (same seed must produce the
                    # same model regardless of checkpointing settings).
                    _torch_state = torch.get_rng_state()
                    _np_state = np.random.get_state()
                    self.generator.eval()
                    self.discriminator.eval()
                    with torch.no_grad():
                        val_synth = self.sample(min(len(data), 500))
                    self.generator.train()
                    self.discriminator.train()
                    torch.set_rng_state(_torch_state)
                    np.random.set_state(_np_state)

                    # Align columns: drop columns not present in synthetic output (FK/PK)
                    real_for_val = data[
                        [c for c in data.columns if c in val_synth.columns]
                    ].copy()

                    results = _validator.compare_columns(real_for_val, val_synth)
                    stats = [
                        v["statistic"]
                        for v in results.values()
                        if isinstance(v, dict) and "statistic" in v
                    ]

                    if stats:
                        val_metric = sum(stats) / len(stats)
                        # Include val_metric in epoch_end only on checkpoint epochs
                        epoch_log_fields["val_metric"] = val_metric
                    else:
                        log.warning(
                            "checkpoint_validation_empty",
                            epoch=epoch,
                            note="compare_columns returned no valid stats — skipping checkpoint",
                        )
                        val_metric = float("inf")

                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    best_epoch = epoch
                    best_cp = os.path.join(checkpoint_dir, "best_checkpoint")
                    self.save(best_cp, overwrite=True)
                    best_checkpoint_path = best_cp
                    log.info(
                        "new_best_checkpoint",
                        epoch=epoch,
                        val_metric=val_metric,
                        path=best_cp,
                    )

            # Emit epoch_end event (always, independent of progress_bar flag)
            # val_metric included in epoch_log_fields only on checkpoint epochs
            log.info("epoch_end", **epoch_log_fields)

            if log_metrics:
                history.append(
                    {"epoch": epoch, "loss_g": current_loss_g, "loss_d": current_loss_d}
                )

        # End of training: Save final checkpoint and metrics
        if checkpoint_dir:
            final_cp = os.path.join(checkpoint_dir, "final_checkpoint")
            self.save(final_cp, overwrite=True)
            # If no validation checkpoint was ever saved (no checkpoint epoch ran),
            # fall back: treat final as best
            if best_checkpoint_path is None:
                best_checkpoint_path = final_cp
                best_epoch = self.epochs - 1
                best_val_metric = float("inf")

        # Emit training_complete with real validation-metric values (QUAL-03)
        log.info(
            "training_complete",
            best_epoch=best_epoch,
            best_val_metric=best_val_metric,
            total_epochs=self.epochs,
            checkpoint_path=str(best_checkpoint_path) if best_checkpoint_path else None,
        )

        if checkpoint_dir and log_metrics and history:
            metrics_path = os.path.join(checkpoint_dir, "training_metrics.csv")
            keys = history[0].keys()
            with open(metrics_path, "w", newline="") as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(history)
            log.info("metrics_saved", path=metrics_path)

    def sample(
        self,
        num_rows: int,
        context: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
        enforce_constraints: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate synthetic samples, optionally conditioned on parent context.

        Args:
            num_rows: Number of rows to generate.
            context: Optional parent attributes aligned to the requested rows.
            seed: Optional integer seed for deterministic sampling. Only applied
                  when provided; no auto-generation (fits and samples may use
                  independent seeds per CONTEXT.md decision).
            enforce_constraints: When True, inspects generated rows against column
                  constraints defined in the table's Metadata config and raises
                  ``ConstraintViolationError`` listing each violated column and
                  the observed value.  When False (default), constraint checking
                  is skipped entirely — inverse_transform() already clips values
                  within each column's defined range.
            **kwargs: Additional sampling controls (unused placeholder).

        Raises:
            GenerationError: If the model has not been fitted or loaded.
            ConstraintViolationError: If ``enforce_constraints=True`` and any
                generated value violates a configured constraint.

        Returns:
            DataFrame of synthetic rows mapped back to original schema.
        """
        if self.generator is None:
            raise GenerationError(
                "Model is not fitted. Call fit() or load() before sample()."
            )

        if seed is not None:
            _set_seed(seed)

        if num_rows <= 0:
            raise ValueError(f"num_rows must be positive, got {num_rows}")

        was_training = self.generator.training
        self.generator.eval()
        try:
            with torch.no_grad():
                noise = torch.randn(num_rows, self.embedding_dim, device=self.device)

                if context is not None:
                    if len(context) != num_rows:
                        raise ValueError(
                            f"context must have exactly num_rows={num_rows} rows, "
                            f"got {len(context)}"
                        )

                    # Transform context using the fitted context transformer
                    context_transformed = self.context_transformer.transform(context)
                    context_data = (
                        torch.from_numpy(context_transformed).float().to(self.device)
                    )

                    gen_input = torch.cat([noise, context_data], dim=1)
                else:
                    gen_input = noise

                fake_raw = self.generator(gen_input)
                fake_act = self._activate(fake_raw)

                # Post-process to transformer layout: argmax embedding blocks to
                # indices, keep activated blocks for inverse_transform.
                output_parts = []
                fake_ptr = 0
                for info in self.data_column_info:
                    if info["type"] == "embedding":
                        dim = info["num_categories"]
                        probs = fake_act[:, fake_ptr : fake_ptr + dim]
                        fake_ptr += dim
                        indices = torch.argmax(probs, dim=1, keepdim=True)
                        output_parts.append(indices.cpu().numpy())
                    else:
                        dim = info["output_dim"]
                        val = fake_act[:, fake_ptr : fake_ptr + dim]
                        fake_ptr += dim
                        output_parts.append(val.cpu().numpy())

                fake_data_np = np.concatenate(output_parts, axis=1)

            result_df = self.transformer.inverse_transform(fake_data_np)

            # Constraint violation checking (opt-in via enforce_constraints=True).
            # Note: inverse_transform() already clips values within defined column
            # ranges, so this is primarily useful for post-hoc auditing.
            if enforce_constraints:
                self._check_constraints(result_df)

            return result_df
        finally:
            # Restore training mode even if transform/inverse/constraint
            # checking raised — otherwise a failed sample() leaves the model
            # stuck in eval mode.
            if was_training:
                self.generator.train()

    def _check_constraints(self, result_df: pd.DataFrame) -> None:
        """Raise ConstraintViolationError if any configured constraint is violated."""
        table_config = None
        table_name = getattr(self.transformer, "table_name", None)
        if hasattr(self, "metadata") and table_name:
            try:
                table_config = self.metadata.get_table(table_name)
            except Exception as exc:
                log.warning(
                    "constraint_config_lookup_failed",
                    table_name=table_name,
                    error=str(exc),
                    note="Skipping constraint enforcement — table config could not be retrieved",
                )
                table_config = None

        if table_config is None or not table_config.constraints:
            return

        violations = []
        for col_name, constraint in table_config.constraints.items():
            if col_name not in result_df.columns:
                continue
            col_numeric = pd.to_numeric(result_df[col_name], errors="coerce")

            if constraint.min is not None:
                bad = col_numeric < constraint.min
                if bad.any():
                    violations.append(
                        f"{col_name}: got {col_numeric[bad].min():.4g} "
                        f"(min={constraint.min})"
                    )

            if constraint.max is not None:
                bad = col_numeric > constraint.max
                if bad.any():
                    violations.append(
                        f"{col_name}: got {col_numeric[bad].max():.4g} "
                        f"(max={constraint.max})"
                    )

        if violations:
            raise ConstraintViolationError(
                f"{len(violations)} constraint violation(s) found — "
                + "; ".join(violations)
            )

    def save(self, path: str, *, overwrite: bool = False) -> None:
        """Persist full model state to a directory checkpoint.

        Saves all components required for a cold load-and-sample without the
        original training data: network weights, DataTransformer state,
        context_transformer state, embedding layer weights, column layout, and
        human-readable metadata.

        The directory contains:
            - generator.pt — generator state_dict
            - discriminator.pt — discriminator state_dict
            - transformer.joblib — fitted DataTransformer for child table
            - context_transformer.joblib — fitted DataTransformer for context
            - embedding_layers.pt — entity embedding weights (state_dict, safe format)
            - data_column_info.joblib — column layout list
            - metadata.json — hyperparameters and version info

        Args:
            path: Directory path to save into.
            overwrite: If False (default), raises SerializationError if path already exists.

        Raises:
            SerializationError: If path exists and overwrite=False, or if any
                component fails to serialize.
        """
        import joblib
        import json
        from pathlib import Path
        from datetime import datetime, timezone

        p = Path(path)
        if p.exists() and not overwrite:
            raise SerializationError(
                f"Save path '{path}' already exists. "
                f"Pass overwrite=True to replace it."
            )

        try:
            p.mkdir(parents=True, exist_ok=True)

            # Network weights — torch native format
            torch.save(self.generator.state_dict(), p / "generator.pt")
            torch.save(self.discriminator.state_dict(), p / "discriminator.pt")

            # sklearn and numpy-heavy objects — joblib for efficient NumPy serialization
            joblib.dump(self.transformer, p / "transformer.joblib")
            joblib.dump(self.context_transformer, p / "context_transformer.joblib")

            # Embedding layers — plain state_dict so load() can use the safe
            # weights_only path (the ModuleDict is rebuilt from data_column_info).
            torch.save(self.embedding_layers.state_dict(), p / "embedding_layers.pt")

            # Column layout list (list of dicts describing each column)
            joblib.dump(self.data_column_info, p / "data_column_info.joblib")

            # Metadata — human-readable, enables version mismatch detection on load
            try:
                from syntho_hive import __version__

                current_version = __version__
            except Exception as exc:
                log.warning(
                    "version_lookup_failed",
                    error=str(exc),
                    note="Could not determine SynthoHive version — using 'unknown'",
                )
                current_version = "unknown"

            meta = {
                "synthohive_version": current_version,
                "embedding_dim": self.embedding_dim,
                "generator_dim": list(self.generator_dim),
                "discriminator_dim": list(self.discriminator_dim),
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "discriminator_steps": self.discriminator_steps,
                "embedding_threshold": self.embedding_threshold,
                "legacy_context_conditioning": self.legacy_context_conditioning,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(p / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            log.info("model_saved", path=str(p))

        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(
                f"Failed to save model to '{path}'. "
                f"Original error: {exc}"
            ) from exc

    def load(self, path: str) -> None:
        """Load full model state from a directory checkpoint.

        Reconstructs the complete model — DataTransformer, context_transformer,
        embedding_layers, column layout, and network weights — without requiring
        the original training data.

        Args:
            path: Directory path produced by save().

        Raises:
            SerializationError: If path does not exist, is missing required files,
                or if any component fails to deserialize.
        """
        import joblib
        import json
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            raise SerializationError(
                f"Checkpoint path '{path}' does not exist."
            )

        required_files = [
            "generator.pt",
            "discriminator.pt",
            "transformer.joblib",
            "context_transformer.joblib",
            "data_column_info.joblib",
        ]
        missing = [f for f in required_files if not (p / f).exists()]
        # Embedding weights: new checkpoints use the safe state_dict format,
        # old ones a pickled ModuleDict — accept either.
        if not (p / "embedding_layers.pt").exists() and not (
            p / "embedding_layers.joblib"
        ).exists():
            missing.append("embedding_layers.pt")
        if missing:
            raise SerializationError(
                f"Checkpoint at '{path}' is incomplete. "
                f"Missing files: {', '.join(missing)}. "
                f"The checkpoint may have been saved by an older version or is corrupt."
            )

        saved_version = "unknown"
        try:
            # Version check — warn but do not fail
            meta_path = p / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                try:
                    from syntho_hive import __version__

                    current_version = __version__
                except Exception as exc:
                    log.warning(
                        "version_lookup_failed",
                        error=str(exc),
                        note="Could not determine SynthoHive version — using 'unknown'",
                    )
                    current_version = "unknown"
                saved_version = meta.get("synthohive_version", "unknown")
                if saved_version != current_version:
                    log.warning(
                        "checkpoint_version_mismatch",
                        saved_version=saved_version,
                        current_version=current_version,
                        path=str(p),
                        note="Attempting load — schema changes between versions may cause failures",
                    )
                # Restore hyperparams from metadata so _build_model() uses correct dims
                if "embedding_dim" in meta:
                    self.embedding_dim = meta["embedding_dim"]
                if "generator_dim" in meta:
                    self.generator_dim = tuple(meta["generator_dim"])
                if "discriminator_dim" in meta:
                    self.discriminator_dim = tuple(meta["discriminator_dim"])
                if "batch_size" in meta:
                    self.batch_size = meta["batch_size"]
                if "epochs" in meta:
                    self.epochs = meta["epochs"]
                if "discriminator_steps" in meta:
                    self.discriminator_steps = meta["discriminator_steps"]
                if "embedding_threshold" in meta:
                    self.embedding_threshold = meta["embedding_threshold"]
                # Default False for forward compatibility with old checkpoints that lack this key
                self.legacy_context_conditioning = meta.get(
                    "legacy_context_conditioning", False
                )

            # Load sklearn objects first — transformer must be in place before _build_model()
            self.transformer = joblib.load(p / "transformer.joblib")
            self.context_transformer = joblib.load(p / "context_transformer.joblib")

            # Load saved column layout (embedding layers restored after _build_model)
            saved_data_column_info = joblib.load(p / "data_column_info.joblib")

            # Validate transformer round-trip integrity
            if (
                not hasattr(self.transformer, "output_dim")
                or self.transformer.output_dim <= 0
            ):
                raise SerializationError(
                    f"Loaded transformer has invalid output_dim "
                    f"({getattr(self.transformer, 'output_dim', 'missing')}). "
                    f"The checkpoint may be corrupt."
                )

            # Derive dimensions needed to reconstruct the generator/discriminator architecture.
            # context_transformer.output_dim is 0 when no context was used during training.
            data_dim = self.transformer.output_dim
            context_dim = getattr(self.context_transformer, "output_dim", 0)

            # Reconstruct generator/discriminator architecture.
            # _build_model() internally calls _compile_layout(self.transformer) which overwrites
            # self.data_column_info and self.embedding_layers with freshly-initialised layers.
            # We restore the saved values immediately after so weights can be loaded correctly.
            self._build_model(data_dim, context_dim)

            # Restore saved column layout (overwrite the freshly compiled one)
            self.data_column_info = saved_data_column_info

            # Restore embedding weights. New checkpoints store a plain
            # state_dict (safe weights_only load); the ModuleDict itself is
            # rebuilt from the saved column layout. Legacy checkpoints stored a
            # pickled ModuleDict — pickle can execute arbitrary code, so only
            # load legacy checkpoints from trusted sources.
            if (p / "embedding_layers.pt").exists():
                rebuilt = nn.ModuleDict()
                for info in self.data_column_info:
                    if info["type"] == "embedding":
                        key = info.get("emb_key", str(info["name"]))
                        rebuilt[key] = EntityEmbeddingLayer(
                            info["num_categories"], info["output_dim"]
                        )
                rebuilt.load_state_dict(
                    torch.load(
                        p / "embedding_layers.pt",
                        map_location=self.device,
                        weights_only=True,
                    )
                )
                self.embedding_layers = rebuilt.to(self.device)
            else:
                self.embedding_layers = joblib.load(
                    p / "embedding_layers.joblib"
                ).to(self.device)

            # Network weights are plain state_dicts (tensors only), so the safe
            # weights_only path works; map_location makes CUDA-saved checkpoints
            # loadable on CPU-only hosts.
            self.generator.load_state_dict(
                torch.load(
                    p / "generator.pt", map_location=self.device, weights_only=True
                )
            )
            self.discriminator.load_state_dict(
                torch.load(
                    p / "discriminator.pt",
                    map_location=self.device,
                    weights_only=True,
                )
            )

            # Set model to eval mode for inference
            self.generator.eval()
            self.discriminator.eval()

            log.info("model_loaded", path=str(p), version=saved_version)

        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(
                f"Failed to load model from '{path}'. "
                f"Original error: {exc}"
            ) from exc
