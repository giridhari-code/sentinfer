# kLLM

Full-Rust, cross-platform Seek-V inference server.

## Phase D (current)

- `PieceTokenizer` + `tokenize` CLI for custom vocab training
- Release: `./scripts/release.sh` / `make release`

## Phase C



- Fused causal **attention** in kernel layer (CPU; Backend-ready for GPU)
- Parallel matmul on Seek-V forward
- Bench: `cargo run -p sentinel-model --example bench --release`

## Phase B



- **Batch:** `"prompt": ["a","b",...]` (up to 32)
- **Stream:** `POST /generate/stream` or `"stream": true` (SSE)
- **Limits:** body ≤ 4 MiB, prompt ≤ 256 KiB, max_tokens ≤ 4096

## Run

```bash
cargo run --release -p sentinel-serve --bin kllm -- \
  --checkpoint runs/best.skv --port 8080
```

## API

### Batch

```bash
curl -s http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":["Hello","Once upon a time"],"max_tokens":24,"temperature":0.0}'
```

### Stream (SSE)

```bash
curl -N http://127.0.0.1:8080/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":32,"temperature":0.0}'
```

Events: `token` (partial / final), then `done`.

## Stack

All Rust: `kllm` → `Llm` → `Engine` / `Scheduler` → Seek-V → CPU | parallel | OpenCL.


## Production

See `PRODUCTION.md` — API key, concurrency limits, systemd, install script.
