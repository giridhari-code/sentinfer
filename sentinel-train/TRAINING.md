# Training Seek-V

Everything runs on CPU in f32 with no external dependencies.

## Quick start

```bash
# 0. verify the code first (gradient checks, resume, end-to-end tests)
cargo test --release

# 1. train on a text file (byte-level tokens, 4.6M-parameter "small" preset)
cargo run --release -p sentinel-train --bin train -- \
    --data corpus.txt --out runs/first --steps 5000 --batch 16 --seq-len 128

# 2. look at the log: loss should fall, val loss/ppl is printed every --eval-every steps

# 3. generate from the checkpoint (best.skv = lowest validation loss, weights.skv = latest)
cargo run --release -p sentinel-train --bin generate -- \
    --checkpoint runs/first/best.skv --prompt "Once upon a time"

# interrupted? continue exactly where it stopped
cargo run --release -p sentinel-train --bin train -- --data corpus.txt --out runs/first --steps 5000 --resume
```

The trained `weights.skv` loads in the inference engine with `SeekV::load(path)`.

## Optimizations (Feature 5)

| Flag / API | Effect |
| --- | --- |
| `TrainConfig::grad_accum_steps` | Average `N` micro-batch gradients before one AdamW step. Effective batch ≈ `batch_size * N`. |
| `TrainConfig::activation_checkpointing` | Flag for recompute-on-backward (partial; full layer wiring TODO). |
| `train_opt::{add_grads, average_grads}` | Helpers used by the trainer for accumulation. |
| `TrainOptConfig::mixed_precision` | Scaffolding for future bf16/fp16 matmuls with f32 master weights. |

Example (larger effective batch without more RAM per micro-batch):

```rust
let mut tc = TrainConfig::default();
tc.batch_size = 8;
tc.grad_accum_steps = 4; // effective batch 32
```

## What is in the pipeline

| Piece | File |
| --- | --- |
| Forward pass with saved activations, hand-written backward pass, cross-entropy | `autograd.rs`, `math.rs` |
| Parallel batch gradients (threads over windows) | `autograd.rs` |
| AdamW (no decay on norm scales), global-norm clipping, warmup + cosine schedule | `optim.rs` |
| Gradient accumulation helpers | `train_opt.rs` |
| Datasets: text (bytes, optional document splitting), pre-tokenized `u16`/`u32` files, split, windows | `data.rs` |
| Training loop, evaluation, `best.skv`, periodic save, exact resume | `trainer.rs` |
| Checkpoint format with checksum (shared with inference) | `sentinel-model/src/checkpoint.rs` |

## Limits you should know about

* **Speed.** Plain loops, no SIMD tuning, no GPU training path yet.
* **Tokenizer.** Byte-level only by default (vocabulary 258).
* **Precision.** f32 only for now; mixed precision is scaffolded, not active.
* **Single machine.** No distributed training.
* **Data.** The whole token stream is held in memory.
