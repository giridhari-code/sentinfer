# Sentinel — Seek-V (Rust)

CPU reference stack for **Seek-V**: a decoder-only transformer (RMSNorm, RoPE, GQA, SwiGLU)
with a full inference engine and a training pipeline. No external ML frameworks.

## Crates

| Crate | Role |
| --- | --- |
| `sentinel-core` | Block manager, sequences, scheduler, MTP planner, FLOPs, sampler metadata |
| `sentinel-kernels` | Backend trait + CPU reference ops + conformance harness |
| `sentinel-model` | Seek-V model, sampler, engine loop, checkpoints |
| `sentinel-train` | Autograd, AdamW, data loading, `train` / `generate` CLIs |

## Quick start

```bash
# tests
./check.sh          # or: cargo test --workspace

# train (byte-level, ~4.6M-param "small" preset)
cargo run --release -p sentinel-train --bin train -- \
  --data corpus.txt --out runs/first --preset small --steps 2000 --batch 16 --seq-len 128

# generate from the best checkpoint
cargo run --release -p sentinel-train --bin generate -- \
  --checkpoint runs/first/best.skv --prompt "Once upon a time"
```

Presets: `tiny` (tests), `small` (~4.6M params). Both use vocabulary 258 (raw bytes).
For a custom tokenizer, write `u16`/`u32` id files and pass `--tokens-u16` / `--tokens-u32`
with `--vocab-size` and `--eos-id`.

## Inference API

```rust
use sentinel_model::{Engine, GenerationParams, SeekV, SeekVConfig};

let model = SeekV::load("runs/first/best.skv")?;
// or: SeekV::random(SeekVConfig::small(), seed)?
let blocks = model.config().max_positions / 16 + 8;
let mut engine = Engine::for_seekv(Default::default(), blocks, model)?;
engine.add_request("r1", prompt_ids, GenerationParams::default())?;
let finished = engine.run_to_completion(100_000)?;
```

`Engine::for_seekv` attaches a matching FLOPs estimator so schedule outputs report TFLOPs.

## Docs

- `sentinel-model/MODEL.md` — model & engine limits
- `sentinel-train/TRAINING.md` — training details & correctness checks
- `PORTING.md` — scheduler/core mapping notes
- `KERNELS.md` — future GPU kernel plan
- `RUNBOOK.md` — first build / test checklist

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

## Phase 3a (current)

See `PHASE3A.md`. Backend now includes matmul / silu / softmax / rope; training writes
`metrics.csv` under `--out`. Tokenizer trait is ready for non-byte vocabularies.

## Inference framework

See `INFERENCE.md`. High-level API:

```rust
use sentinel_model::{EngineArgs, Llm, SamplingParams};
let mut llm = Llm::from_checkpoint("best.skv", EngineArgs::default())?;
let outs = llm.generate(vec!["Hello".into()], SamplingParams::greedy(32))?;
```

## kLLM server

**kLLM** — full-Rust native server (`kllm` binary). See `KLLM.md`.
```bash
cargo run --release -p sentinel-serve --bin kllm -- --checkpoint best.skv --port 8080
```
