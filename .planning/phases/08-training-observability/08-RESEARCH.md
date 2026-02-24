# Phase 8: Training Observability - Research

**Researched:** 2026-02-24
**Domain:** Structured logging (structlog), progress bars (tqdm), checkpoint policy, validation metrics
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Output mode:**
- Both: tqdm-style progress bar printed to stderr for visual watching AND structlog JSON events emitted via Python's stdlib `logging` module
- Engineers configure log handlers (file, Datadog, CloudWatch, etc.) through their existing `logging.basicConfig` or handler setup — SynthoHive does not dictate the sink
- `progress_bar` is a separate `fit()` parameter (e.g., `fit(..., progress_bar=False)`) to suppress the visual bar without touching logging
- Structured log events always emit regardless of the `progress_bar` flag — they are purely independent

**Checkpoint policy:**
- Checkpoint frequency: every N epochs, configurable via a `checkpoint_interval` parameter on `fit()`, defaulting to 10 epochs
- Which validation metric wins: Claude's discretion — pick whichever metric (TVD, KS, or composite) is already computed in the existing validation path, to minimise new computation
- File retention: keep two files — `best_checkpoint` (overwritten whenever a new best is found) and `final_checkpoint` (written once at the end of training)

**Log event structure:**
- Epoch event (event name: Claude's discretion, must be greppable/filterable): emitted after every epoch, always contains:
  - `epoch` (int)
  - `g_loss` (float, generator loss)
  - `d_loss` (float, discriminator loss)
  - `eta_seconds` (float, non-zero after first epoch)
  - `val_metric` (float, only present on checkpoint epochs — i.e., when validation actually ran)
- `training_start` event: emitted once before training begins (Claude decides fields — at minimum model config / epoch count)
- `training_complete` event: emitted once after training ends, containing:
  - `best_epoch` (int)
  - `best_val_metric` (float)
  - `total_epochs` (int)
  - `checkpoint_path` (str, path to best checkpoint file)

### Claude's Discretion
- Exact structlog event name strings (must be consistent and greppable)
- Which validation metric to use for best-epoch selection (pick from existing code path)
- Fields included in `training_start` event
- ETA calculation method (linear extrapolation is fine; warm-up period handling is at Claude's judgement)
- tqdm bar format/columns

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CORE-05 | Data engineer can observe training progress — epoch number, generator loss, discriminator loss, and ETA — emitted as structured log events during `fit()` | structlog 25.5.0 already installed; `log.info('epoch_end', ...)` pattern is the right call site; structlog.testing.capture_logs enables test assertions without extra infrastructure |
| QUAL-03 | Data engineer can trust that the saved checkpoint from `fit()` corresponds to the epoch with the best validation TVD/KS score — not generator loss | StatisticalValidator.compare_columns() already returns per-column TVD and KS statistics; mean stat across columns is the natural composite; CTGAN.save() directory format already supports overwrite; checkpoint_interval controls validation overhead |
</phase_requirements>

## Summary

Phase 8 adds two capabilities to `CTGAN.fit()`: real-time observability (structured log events + tqdm progress bar) and best-quality checkpointing (replace generator-loss criterion with validation metric). Both capabilities are surgical additions to `ctgan.py` — no new modules are required.

Structlog 25.5.0 is already installed and used consistently throughout the project (`log = structlog.get_logger()` at the module level). The project's existing pattern emits events via `log.info(event_name, **fields)`, and structlog's default dev renderer prints human-readable output; engineers override with JSON via their own `logging.basicConfig` and handler configuration. The `structlog.testing.capture_logs()` context manager enables test assertions on emitted events without any test-specific configuration.

Tqdm 4.67.3 is NOT currently in the project's `pyproject.toml` dependencies. It must be added. The `trange()` function wrapping the epoch loop accepts `disable=True` to suppress the bar (for `progress_bar=False`) while the rest of the code path remains unchanged. The progress bar and structured log events are strictly independent — the bar is controlled by `disable`, the log events always fire.

For best-quality checkpointing, `StatisticalValidator.compare_columns()` already exists at `syntho_hive/validation/statistical.py`. It returns a dict of per-column results with `'statistic'` keys. The mean statistic across columns (lower = better quality) is the natural composite metric that requires zero new computation — it uses only what `compare_columns()` already returns. Validation runs on checkpoint epochs only (every `checkpoint_interval` epochs); between checkpoints, only losses are logged. The existing `CTGAN.save(path, overwrite=True)` already supports the overwrite-on-improvement pattern needed for `best_checkpoint`.

**Primary recommendation:** Instrument `CTGAN.fit()` directly — add tqdm epoch loop, emit structlog events, add validation on checkpoint epochs, save best checkpoint by val_metric. Then expose `progress_bar` and `checkpoint_interval` as new `fit()` parameters. Add tqdm to `pyproject.toml`. Update the one existing test in `test_checkpointing.py` that asserts `best_model.pt` (now becomes `best_checkpoint/`) and write two new test files covering CORE-05 and QUAL-03.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| structlog | 25.5.0 (installed) | Structured log event emission | Already used project-wide; `log.info(event, **fields)` is the established pattern |
| tqdm | 4.67.3 (needs adding) | Terminal progress bar to stderr | De facto standard for ML training loops; `trange(n, disable=bool)` integrates cleanly with existing for-loop |
| StatisticalValidator | project internal | TVD/KS metric for best-epoch selection | Already implemented at `syntho_hive/validation/statistical.py`; zero new code required for computation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| structlog.testing.capture_logs | 25.5.0 | Capture emitted events in tests | Every test that asserts on log events — provides a list of dicts, no handler setup needed |
| time (stdlib) | - | ETA calculation (linear extrapolation) | `time.time()` calls around epoch loop; no dependency needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| tqdm | rich.progress | rich is heavier dep, not in project; tqdm is ML-standard, lightweight |
| tqdm | manual `\r` stderr print | Already prototyped; works but loses postfix dict, ETA display, and TTY detection |
| mean(TVD+KS stats) composite | max(stats) | Max penalizes single bad column; mean better reflects overall quality |
| StatisticalValidator | compute TVD inline | Duplication; existing validator handles type dispatch, NaN, empty-column edge cases |

**Installation:**
```bash
# Add to pyproject.toml [project] dependencies:
"tqdm>=4.64.0"
# Install in dev:
pip install tqdm
```

## Architecture Patterns

### Recommended Project Structure
No new files are needed. All changes are confined to:
```
syntho_hive/
├── core/
│   └── models/
│       └── ctgan.py          # Primary: add tqdm, structlog events, val-metric checkpointing
├── interface/
│   └── synthesizer.py        # Add progress_bar, checkpoint_interval to fit() signature
pyproject.toml                # Add tqdm dependency
tests/
├── test_checkpointing.py     # Update: assert best_checkpoint/ not best_model.pt
└── test_training_observability.py  # New: CORE-05 and QUAL-03 tests
```

### Pattern 1: Epoch Loop with tqdm + structlog

**What:** Replace the bare `for epoch in range(self.epochs):` loop with `trange()` and emit structlog events after each epoch.

**When to use:** This is the only loop in `CTGAN.fit()` — applies unconditionally.

**Example (verified against installed APIs):**
```python
import time
from tqdm import trange
import structlog

log = structlog.get_logger()

# Before loop — emit training_start
log.info(
    "training_start",
    total_epochs=self.epochs,
    batch_size=self.batch_size,
    embedding_dim=self.embedding_dim,
    checkpoint_interval=checkpoint_interval,
)

_start_time = time.time()
_epoch_times = []

pbar = trange(
    self.epochs,
    desc="Training",
    file=sys.stderr,
    leave=True,
    disable=not progress_bar,  # progress_bar=False → disable=True, no bar output
)

for epoch in pbar:
    # ... existing training step code (unchanged) ...

    # ETA calculation (linear extrapolation, non-zero after first epoch)
    _elapsed = time.time() - _start_time
    _epochs_done = epoch + 1
    _elapsed_per_epoch = _elapsed / _epochs_done
    _remaining = self.epochs - _epochs_done
    eta_seconds = _elapsed_per_epoch * _remaining  # 0.0 on last epoch; non-zero otherwise

    # Update tqdm postfix (visual bar)
    pbar.set_postfix({
        "g_loss": f"{current_loss_g:.4f}",
        "d_loss": f"{current_loss_d:.4f}",
        "eta": f"{int(eta_seconds)}s",
    })

    # Emit structlog epoch_end event (always, regardless of progress_bar)
    epoch_log_fields = dict(
        epoch=epoch,
        g_loss=current_loss_g,
        d_loss=current_loss_d,
        eta_seconds=eta_seconds,
    )
    if _is_checkpoint_epoch:
        epoch_log_fields["val_metric"] = val_metric
    log.info("epoch_end", **epoch_log_fields)

# After loop — emit training_complete
log.info(
    "training_complete",
    best_epoch=best_epoch,
    best_val_metric=best_val_metric,
    total_epochs=self.epochs,
    checkpoint_path=str(best_checkpoint_path),
)
```

### Pattern 2: Validation-Metric Checkpointing

**What:** Every `checkpoint_interval` epochs, generate a small synthetic sample from the current generator state, compute mean TVD/KS via `StatisticalValidator`, and save if it's the best seen.

**When to use:** Only when `checkpoint_dir` is provided; validation adds latency so it runs only on checkpoint epochs.

**Important constraint:** Validation requires `real_data` (the original DataFrame) to be accessible inside `fit()`. It already is — `data` is the `pd.DataFrame` argument to `fit()`. The non-FK columns can be passed directly to `compare_columns()`.

**Example (verified against StatisticalValidator API):**
```python
from syntho_hive.validation.statistical import StatisticalValidator

_validator = StatisticalValidator()
best_val_metric = float('inf')
best_epoch = -1
best_checkpoint_path = None

# Inside epoch loop, after optimizer steps:
_is_checkpoint_epoch = (
    checkpoint_dir is not None
    and (epoch + 1) % checkpoint_interval == 0
)

if _is_checkpoint_epoch:
    # Generate a small sample for validation (100-500 rows is sufficient)
    self.generator.eval()
    with torch.no_grad():
        val_synth = self.sample(min(len(data), 500))
    self.generator.train()

    # Compare with real data (drop PK/FK columns that aren't in synth output)
    real_for_val = data.drop(columns=[c for c in data.columns if c not in val_synth.columns], errors='ignore')

    results = _validator.compare_columns(real_for_val, val_synth)
    stats = [v['statistic'] for v in results.values() if isinstance(v, dict) and 'statistic' in v]
    val_metric = sum(stats) / len(stats) if stats else float('inf')

    if val_metric < best_val_metric:
        best_val_metric = val_metric
        best_epoch = epoch
        best_cp = os.path.join(checkpoint_dir, "best_checkpoint")
        self.save(best_cp, overwrite=True)
        best_checkpoint_path = best_cp
        log.info("new_best_checkpoint", epoch=epoch, val_metric=val_metric, path=best_cp)
```

**Final checkpoint (always written at end):**
```python
if checkpoint_dir:
    final_cp = os.path.join(checkpoint_dir, "final_checkpoint")
    self.save(final_cp, overwrite=True)
```

### Pattern 3: Synthesizer.fit() Parameter Threading

**What:** `Synthesizer.fit()` must accept `progress_bar` and `checkpoint_interval` and forward them to the model layer.

**Current gap:** `CTGAN.fit()` accepts `checkpoint_dir` and `log_metrics` but not `progress_bar` or `checkpoint_interval`. The `StagedOrchestrator.fit_all()` calls `model.fit()` without these parameters — they need threading through.

**Threading path:**
```
Synthesizer.fit(progress_bar=True, checkpoint_interval=10, checkpoint_dir=None)
  → StagedOrchestrator.fit_all(..., progress_bar=True, checkpoint_interval=10, checkpoint_dir=...)
    → model.fit(..., progress_bar=True, checkpoint_interval=10, checkpoint_dir=...)
```

The `**model_kwargs` forwarding already exists in `fit_all()` — `progress_bar` and `checkpoint_interval` can ride through it, but it's cleaner to add them as explicit parameters.

### Anti-Patterns to Avoid

- **Emitting log events inside `progress_bar=False` guard:** Log events must ALWAYS fire, independent of `progress_bar`. The CONTEXT.md decision is explicit. Only `disable=not progress_bar` in tqdm controls the bar.
- **Using generator loss for best-checkpoint selection:** This is the current behavior (line 466 in ctgan.py: `if checkpoint_dir and current_loss_g < best_loss:`). Phase 8 replaces it entirely with val_metric. After this phase, `best_loss`/`loss_G` tracking is removed from checkpoint logic.
- **Running validation every epoch:** Sampling + comparing distributions adds latency. Only run on checkpoint epochs (`(epoch + 1) % checkpoint_interval == 0`).
- **Using `self.save()` without `overwrite=True` for best_checkpoint:** `best_checkpoint` is overwritten each time a new best is found. Without `overwrite=True`, `SerializationError: already exists` fires on the second improvement.
- **Setting eta_seconds = 0 on epoch 0:** After epoch 0 completes, there IS timing data — `elapsed_per_epoch * remaining_epochs` gives a valid non-zero ETA for epochs 1 through N-2. Only the final epoch yields ETA = 0.0 (which is correct). The success criterion says "non-zero ETA estimate" — this is satisfied for all epochs except the last.
- **Importing tqdm at module level without a guard:** tqdm is a new dependency. Import it at the top of ctgan.py once it is in pyproject.toml. No lazy import needed — it will be available after the pyproject.toml update.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Progress bar rendering | Custom `\r`-based stderr printer | `tqdm.trange(disable=bool)` | tqdm handles TTY detection, width calculation, ETA display, postfix formatting; custom version breaks in notebooks and CI |
| Structured log capture in tests | Custom logging handler setup | `structlog.testing.capture_logs()` | Returns plain list of dicts; zero config; already verified working in this project |
| TVD/KS computation | Inline numpy calculation | `StatisticalValidator.compare_columns()` | Handles type dispatch (numeric → KS, categorical → TVD), NaN dropping, empty-column errors — all edge cases already covered |

**Key insight:** The most expensive part (checkpoint validation) is already implemented. The new work is wiring it into the training loop at the right frequency.

## Common Pitfalls

### Pitfall 1: tqdm + structlog Output Interference
**What goes wrong:** tqdm writes to stderr; if structlog also prints to stderr (default dev config), the two interleave and corrupt each other's output.
**Why it happens:** Default structlog uses `PrintLogger` which writes to stdout. Default tqdm writes to stderr. These are different streams — they do NOT interleave in normal use. If an engineer configures structlog to stderr (e.g., `stream=sys.stderr`), the bar can flicker. This is the engineer's problem, not ours — they're overriding the default.
**How to avoid:** Document that the default configuration (structlog to stdout/root logger, tqdm to stderr) does not interfere. No code changes needed.
**Warning signs:** Output looks garbled in some test environments → they've redirected stderr.

### Pitfall 2: Checkpoint Validation Cost Blows Up Training Time
**What goes wrong:** Running `StatisticalValidator.compare_columns()` on the full training dataset every epoch makes 300-epoch training 2-3x slower.
**Why it happens:** Sampling 500 rows takes a forward pass through the generator (~5ms) plus KS/TVD computation (~10ms). Acceptable at `checkpoint_interval=10` (30 validation runs), not at every epoch (300 runs).
**How to avoid:** Always gate validation on `(epoch + 1) % checkpoint_interval == 0`. Default `checkpoint_interval=10` keeps overhead to <10% of training time.
**Warning signs:** Training with `checkpoint_dir` set takes noticeably longer than without.

### Pitfall 3: val_metric is `float('inf')` When compare_columns Returns Errors
**What goes wrong:** If all columns return `{"error": "..."}` entries (e.g., both DataFrames empty), `stats` is an empty list and `val_metric = float('inf')`. This is correct behavior — no checkpoint saved. But if this happens on every epoch, best_epoch stays -1 and `training_complete` emits `best_epoch=-1` and `best_val_metric=inf`.
**Why it happens:** Validation sample might be empty if `min(len(data), 500)` is 0 (impossible — fit requires non-empty data) or if all columns are dropped as FK/PK.
**How to avoid:** Log a warning when `stats` is empty. Include a fallback: if `checkpoint_dir` is set but no validation run ever succeeded, write `final_checkpoint` only and set `best_epoch=-1` and `best_checkpoint_path=final_checkpoint`.
**Warning signs:** `best_val_metric=inf` in `training_complete` event.

### Pitfall 4: Breaking the Old `test_checkpointing.py` Test
**What goes wrong:** The existing test at `tests/test_checkpointing.py` asserts `"best_model.pt" in files` (line 52). Phase 8 renames the checkpoint from `best_model.pt` to `best_checkpoint/` (a directory). The test must be updated or it will fail.
**Why it happens:** The checkpoint naming changes from single-file `.pt` to directory (matching CTGAN.save() which already writes a directory).
**How to avoid:** Update `test_checkpointing.py` in the same plan that changes ctgan.py. Assert `"best_checkpoint" in files` and `os.path.isdir(...)`.

### Pitfall 5: eta_seconds on Epoch 0 Is Zero Without Special Handling
**What goes wrong:** On epoch 0, `elapsed_per_epoch * remaining_epochs` = `elapsed * (total - 1)`. This IS non-zero (unless total=1). The success criterion says eta must be "non-zero after first epoch" — meaning after epoch 0 completes, epoch 1 onward must have non-zero ETA. This is naturally satisfied.
**Why it happens:** Confusion about "first epoch." Epoch 0 IS the first epoch; its ETA is computed correctly.
**How to avoid:** No special handling needed. Emit `eta_seconds` unconditionally using linear extrapolation. The only case where it's 0.0 is the last epoch (`remaining=0`), which is correct.

### Pitfall 6: `self.sample()` Inside fit() Puts Generator in eval() Mode
**What goes wrong:** `self.sample()` calls `self.generator.eval()` internally. After validation sampling, the generator remains in eval mode for the next training step.
**Why it happens:** `sample()` has `self.generator.eval()` + `torch.no_grad()` but no cleanup to restore training mode.
**How to avoid:** After the validation block, explicitly call `self.generator.train()` and `self.discriminator.train()`. Wrap the sample call carefully:
```python
self.generator.eval()
with torch.no_grad():
    val_synth = self.sample(min(len(data), 500))
self.generator.train()
self.discriminator.train()
```

## Code Examples

Verified patterns from installed libraries:

### Structlog Event Emission (project pattern)
```python
# Source: existing project code in ctgan.py, synthesizer.py, orchestrator.py
import structlog
log = structlog.get_logger()

# Events are emitted identically to existing project code:
log.info("training_start", total_epochs=300, batch_size=500, embedding_dim=128, checkpoint_interval=10)
log.info("epoch_end", epoch=5, g_loss=0.42, d_loss=0.31, eta_seconds=95.3)
log.info("epoch_end", epoch=10, g_loss=0.38, d_loss=0.29, eta_seconds=85.0, val_metric=0.08)
log.info("training_complete", best_epoch=10, best_val_metric=0.08, total_epochs=300, checkpoint_path="/tmp/ckpt/best_checkpoint")
# Default dev output: "2026-02-24 [info] epoch_end  d_loss=0.31 epoch=5 eta_seconds=95.3 g_loss=0.42"
# JSON output: {"epoch": 5, "g_loss": 0.42, "d_loss": 0.31, "eta_seconds": 95.3, "event": "epoch_end", ...}
```

### structlog.testing.capture_logs in Tests
```python
# Source: verified against structlog 25.5.0 in project venv
from structlog.testing import capture_logs

with capture_logs() as cap_logs:
    # ... call model.fit() ...
    pass

epoch_events = [e for e in cap_logs if e.get("event") == "epoch_end"]
assert len(epoch_events) == total_epochs, "must emit one event per epoch"
assert epoch_events[0]["eta_seconds"] > 0, "eta must be non-zero after first epoch"
assert "g_loss" in epoch_events[0]
assert "d_loss" in epoch_events[0]

complete_events = [e for e in cap_logs if e.get("event") == "training_complete"]
assert len(complete_events) == 1
assert complete_events[0]["best_epoch"] >= 0
assert complete_events[0]["best_val_metric"] < float('inf')
```

### tqdm Progress Bar with disable Parameter
```python
# Source: tqdm 4.67.3 docs + verified in project venv
import sys
from tqdm import trange

# progress_bar=True → disable=False (bar shown on stderr)
# progress_bar=False → disable=True (no bar; no output to stderr)
pbar = trange(
    self.epochs,
    desc="Training",
    file=sys.stderr,
    leave=True,
    disable=not progress_bar,
)
for epoch in pbar:
    # ... training step ...
    pbar.set_postfix({
        "g_loss": f"{current_loss_g:.4f}",
        "d_loss": f"{current_loss_d:.4f}",
        "eta": f"{int(eta_seconds)}s",
    })
```

### Linear ETA Calculation
```python
# Source: standard ML training practice, no library needed
import time

_start_time = time.time()

for epoch in pbar:
    # ... training ...
    _elapsed = time.time() - _start_time
    _epochs_done = epoch + 1
    _elapsed_per_epoch = _elapsed / _epochs_done
    _remaining_epochs = self.epochs - _epochs_done
    eta_seconds = _elapsed_per_epoch * _remaining_epochs  # 0.0 on final epoch
```

### Best-Checkpoint Save Pattern
```python
# Uses CTGAN.save(path, overwrite=True) — already supported
best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint")
self.save(best_checkpoint_path, overwrite=True)

# Final checkpoint — always written at end
final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint")
self.save(final_checkpoint_path, overwrite=True)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `print(f"Epoch {epoch}: Loss D=...")` every 10 epochs | structlog events every epoch + tqdm bar | Phase 8 | Machine-readable, filterable, sink-agnostic |
| Checkpoint on lowest generator loss (`current_loss_g < best_loss`) | Checkpoint on lowest val_metric (mean TVD/KS) | Phase 8 | Saved model represents statistical quality, not GAN convergence proxy |
| `best_model.pt` (single file, uses old torch.save format) | `best_checkpoint/` (directory, uses CTGAN.save() format) | Phase 8 | Consistent with phase 3's directory checkpoint format; cold load works |
| `last_model.pt` | `final_checkpoint/` | Phase 8 | Naming aligns with user decision document |

**Deprecated/outdated:**
- `best_loss = float('inf')` + `if current_loss_g < best_loss` checkpoint criterion: replaced by `best_val_metric` + `val_metric` computed from `StatisticalValidator`.
- `self.save(os.path.join(checkpoint_dir, "best_model.pt"))` — replaced by `self.save(os.path.join(checkpoint_dir, "best_checkpoint"), overwrite=True)`.

## Open Questions

1. **How to handle cold `load()` + `sample()` pointing at `best_checkpoint/`**
   - What we know: `CTGAN.load(path)` already works for any directory checkpoint. The user who calls `Synthesizer.load()` loads the whole synthesizer object. The `training_complete` log event tells them `checkpoint_path`.
   - What's unclear: Should `Synthesizer.fit()` also expose the `checkpoint_dir` parameter in its own signature, so that the best checkpoint path is deterministic for callers? Currently `Synthesizer.fit()` does NOT accept `checkpoint_dir` — it delegates everything to the orchestrator.
   - Recommendation: Add `checkpoint_dir` to `Synthesizer.fit()` in the same plan as the CTGAN changes. The planner should decide whether this is plan 1 or plan 2.

2. **Validation sample size for checkpoint evaluation**
   - What we know: `min(len(data), 500)` rows is a reasonable proxy. More rows = better metric stability.
   - What's unclear: Very small training sets (< 100 rows) may produce noisy val_metric. With 200 rows of training data, a 200-row validation sample is the same data — not technically a held-out set, but acceptable for this use case (checkpoint selection, not final reporting).
   - Recommendation: Use `min(len(data), 500)` unconditionally. No held-out split needed at this phase.

3. **Behavior when `checkpoint_dir=None` but `progress_bar=True`**
   - What we know: Log events and tqdm bar fire regardless of `checkpoint_dir`. Validation only runs when `checkpoint_dir` is set.
   - What's unclear: `training_complete` currently logs `checkpoint_path`. If `checkpoint_dir=None`, what should `checkpoint_path` be?
   - Recommendation: When `checkpoint_dir=None`, set `checkpoint_path=None` in the `training_complete` event and `best_epoch=-1` / `best_val_metric=None`. The event still fires (it's a training summary).

## Sources

### Primary (HIGH confidence)
- structlog 25.5.0 installed in `.venv` — verified `log.info(event, **fields)` and `capture_logs()` patterns directly
- tqdm 4.67.3 installed in `.venv` for testing — verified `trange(n, disable=bool, file=sys.stderr)` and `set_postfix()` patterns
- `syntho_hive/core/models/ctgan.py` — direct read; confirmed current checkpoint logic (lines 452–481), existing `log = structlog.get_logger()`, and `save(path, overwrite=True)` signature
- `syntho_hive/validation/statistical.py` — direct read; confirmed `compare_columns()` returns dict with `'statistic'` keys for both KS and TVD
- `pyproject.toml` — confirmed `structlog>=21.1.0` present, tqdm absent

### Secondary (MEDIUM confidence)
- [structlog standard library logging docs](https://www.structlog.org/en/stable/standard-library.html) — confirmed stdlib integration pattern via `LoggerFactory()` and `JSONRenderer()`
- [tqdm documentation](https://tqdm.github.io/docs/tqdm/) — confirmed `file`, `disable`, `set_postfix`, `desc` parameter behavior
- [tqdm GitHub](https://github.com/tqdm/tqdm) — confirmed 4.67.1 latest stable as of late 2024; 4.67.3 installed in venv confirms currency

### Tertiary (LOW confidence)
- None — all critical claims verified against installed library or codebase directly.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — structlog 25.5.0 already installed and tested; tqdm 4.67.3 tested in venv; StatisticalValidator read and tested
- Architecture: HIGH — all patterns executed against real installed libraries; no hypothetical claims
- Pitfalls: HIGH — discovered by reading ctgan.py carefully (eval() mode issue, overwrite issue, test_checkpointing.py naming issue)

**Research date:** 2026-02-24
**Valid until:** 2026-03-24 (stable libraries; structlog and tqdm APIs are stable)
