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
Use `train --help`-style usage (run with a bad argument) for all options.

## What is in the pipeline

| Piece | File |
| --- | --- |
| Forward pass with saved activations, hand-written backward pass, cross-entropy | `autograd.rs`, `math.rs` |
| Parallel batch gradients (threads over windows) | `autograd.rs` |
| AdamW (no decay on norm scales), global-norm clipping, warmup + cosine schedule | `optim.rs` |
| Datasets: text (bytes, optional document splitting), pre-tokenized `u16`/`u32` files, split, windows | `data.rs` |
| Training loop, evaluation, `best.skv`, periodic save, exact resume | `trainer.rs` |
| Checkpoint format with checksum (shared with inference) | `sentinel-model/src/checkpoint.rs` |

## How correctness is checked (run `cargo test`)

* Backward pass vs finite differences, for every tensor, with grouped-query and multi-head attention.
* Training forward vs the inference model's logits (a trained model behaves the same when served).
* Gradients independent of the thread count; batch gradient equals the mean of per-window gradients.
* A gradient step lowers the loss; a tiny model learns a repeating pattern and then continues it
  correctly through the inference engine.
* Resume gives bit-identical weights to uninterrupted training (single thread).
* Checkpoints: roundtrip, corruption, truncation, wrong architecture.

These tests were written without a Rust toolchain, so **none of this has been compiled or run yet**.
Expect to fix a few compile errors on the first `cargo test`.

## Limits you should know about

* **Speed.** Plain loops, no SIMD tuning, no GPU. I have not benchmarked it. Treat it as practical
  for models of a few million parameters; billion-parameter training needs the GPU backend
  (see `KERNELS.md`), which is the biggest missing piece.
* **Tokenizer.** Byte-level only (vocabulary 258). For a real vocabulary, tokenize your corpus
  with your own tokenizer, write ids as `u16`/`u32` files, and pass `--tokens-u16 FILE --vocab-size N
  --eos-id N`. `generate` supports byte-level models only.
* **Precision.** f32 only: no bf16, no mixed precision, no loss scaling.
* **Single machine.** No distributed or multi-GPU training; no gradient accumulation across steps
  (raise `--batch` instead).
* **Data.** The whole token stream is held in memory. Windows are sampled with replacement, so
  "epochs" are not tracked.
* **Architecture.** Standard decoder (RMSNorm, RoPE, GQA, SwiGLU). A different architecture needs
  its own forward and backward pass.
