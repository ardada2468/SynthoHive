# Phase 8: Training Observability - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire structured log events into the `fit()` training loop so engineers can observe epoch number, loss values, and ETA in real time. Replace the generator-loss checkpoint criterion with validation-metric checkpointing so the saved checkpoint represents the best-quality epoch, not the lowest generator loss. Restore path (`load()` + `sample()`) is already in scope via cold-load requirement. Batch-level metrics, dashboards, and distributed training observability are out of scope.

</domain>

<decisions>
## Implementation Decisions

### Output mode
- Both: tqdm-style progress bar printed to stderr for visual watching **and** structlog JSON events emitted via Python's stdlib `logging` module
- Engineers configure log handlers (file, Datadog, CloudWatch, etc.) through their existing `logging.basicConfig` or handler setup — SynthoHive does not dictate the sink
- `progress_bar` is a separate `fit()` parameter (e.g., `fit(..., progress_bar=False)`) to suppress the visual bar without touching logging
- Structured log events **always emit** regardless of the `progress_bar` flag — they are purely independent

### Checkpoint policy
- Checkpoint frequency: every N epochs, configurable via a `checkpoint_interval` parameter on `fit()`, defaulting to **10 epochs**
- Which validation metric wins: **Claude's discretion** — pick whichever metric (TVD, KS, or composite) is already computed in the existing validation path, to minimise new computation
- File retention: keep **two files** — `best_checkpoint` (overwritten whenever a new best is found) and `final_checkpoint` (written once at the end of training)

### Log event structure
- **Epoch event** (`event` name: Claude's discretion, must be greppable/filterable): emitted after every epoch, always contains:
  - `epoch` (int)
  - `g_loss` (float, generator loss)
  - `d_loss` (float, discriminator loss)
  - `eta_seconds` (float, non-zero after first epoch)
  - `val_metric` (float, only present on checkpoint epochs — i.e., when validation actually ran)
- **`training_start` event**: emitted once before training begins (Claude decides fields — at minimum model config / epoch count)
- **`training_complete` event**: emitted once after training ends, containing:
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

</decisions>

<specifics>
## Specific Ideas

- No specific references provided — open to standard structlog + tqdm patterns
- The `progress_bar` parameter design is intentional: visual output and log output must be independently controllable

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 08-training-observability*
*Context gathered: 2026-02-24*
