# Seek-V inference framework

Maps the useful surface of DeepInfer’s `LLM` / `LLMEngine` onto a single-process
Rust stack for Seek-V.

## Layers

| Layer | Module | DeepInfer analogue |
| --- | --- | --- |
| Config | `config::EngineArgs` | `EngineArgs` / `SchedulerConfig` |
| Core loop | `engine::Engine` | worker + `LLMEngine.step` |
| Scheduler | `sentinel-core` | `csrc/engine/scheduler` |
| High-level API | `llm::Llm` | `entrypoints.llm.LLM` |
| Request I/O | `request::*` | `RequestOutput`, `SamplingParams` |
| Stats | `stats::*` | `StatsLogger` / `StepStats` |
| CLI | `generate` | offline batch generate |

## Not ported (on purpose for Seek-V)

- Multi-GPU / DP voting / process groups
- Multimodal / image tokens
- TileLang GPU kernels / async ZOS workers
- Structured-output matchers
- HuggingFace weight loaders (use `.skv` from `sentinel-train`)

## API

```rust
use sentinel_model::{EngineArgs, Llm, SamplingParams};

let mut llm = Llm::from_checkpoint("runs/best.skv", EngineArgs::default())?;
let outs = llm.generate(
    vec!["Hello".into(), "Once upon a time".into()],
    SamplingParams { max_tokens: 32, temperature: 0.8, ..Default::default() },
)?;
for o in outs {
    println!("{}  [{}]", o.text, o.finish_reason);
}
```

Continuous batching:

```rust
llm.add_prompt("r1", "Hi", SamplingParams::greedy(16))?;
llm.add_prompt("r2", "Yo", SamplingParams::greedy(16))?;
while llm.has_unfinished() {
    let (finished, _stats) = llm.step()?;
    // stream finished requests
}
```

## CLI

```bash
cargo run --release -p sentinel-train --bin generate -- \
  --checkpoint runs/best.skv \
  --prompt "Once upon a time" \
  --prompt "Hello world" \
  --max-tokens 64 --temperature 0.8 --log-stats
```


## HTTP server (kLLM)

```bash
cargo run --release -p kllm (package sentinel-serve) --   --checkpoint runs/best.skv --host 0.0.0.0 --port 8000

curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":32,"temperature":0.8}'
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions`.

## Backends

| Backend | Crate path | Notes |
| --- | --- | --- |
| `CpuBackend` | `sentinel_kernels::CpuBackend` | reference, single-thread |
| `ParallelCpuBackend` | `sentinel_kernels::ParallelCpuBackend` | multi-thread matmul |
| `GpuBackend` | `sentinel_kernels::GpuBackend` | slot for device; currently parallel-CPU fallback |

Seek-V forward uses parallel matmul on CPU automatically.


Prefer the native kLLM API documented in `KLLM.md` (`POST /generate`).
