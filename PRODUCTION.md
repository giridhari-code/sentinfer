# Production polish

## Server flags

```bash
kllm --checkpoint model.skv \
  --host 0.0.0.0 --port 8080 \
  --api-key "$KLLM_API_KEY" \
  --max-concurrent 32 \
  --log-requests
```

| Flag | Purpose |
| --- | --- |
| `--api-key` | require `Authorization: Bearer <key>` (except `/health` `/ready`) |
| `--max-concurrent` | HTTP handler cap → HTTP 429 when full |
| `--log-requests` | method path status latency |
| SIGTERM / SIGINT | drain in-flight, then exit |

## Endpoints

| Path | Auth |
| --- | --- |
| `GET /health` `/ready` | public |
| `GET /version` `/models` | API key if configured |
| `POST /generate` `/generate/stream` | API key if configured |

## Install

```bash
./scripts/install.sh          # PREFIX=/usr/local
# or
PREFIX=/opt/kllm ./scripts/install.sh
```

## systemd

```bash
sudo cp deploy/kllm.service /etc/systemd/system/
sudo mkdir -p /etc/kllm /var/lib/kllm
sudo cp deploy/env.example /etc/kllm/env   # set KLLM_API_KEY
sudo cp model.skv /var/lib/kllm/model.skv
sudo useradd -r -s /usr/sbin/nologin kllm || true
sudo systemctl daemon-reload
sudo systemctl enable --now kllm
```

## Client

```bash
curl -s http://127.0.0.1:8080/generate \
  -H "Authorization: Bearer $KLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":32}'
```


## Full product checklist

| Feature | Endpoint / flag |
| --- | --- |
| Batch | `POST /generate` with `"prompt":["a","b"]` |
| Stream | `POST /generate/stream` (SSE: `start` → `token` → `done`) |
| Stream + batch | same stream endpoint with prompt array (`index` on each event) |
| Auth | `--api-key` / `KLLM_API_KEY` |
| GPU verify | `GET /gpu` (backend name, accelerated, 2x2 matmul smoke) |
| Concurrency | `--max-concurrent` |
| Graceful stop | SIGTERM / SIGINT |

### Stream multi-prompt example

```bash
curl -N http://127.0.0.1:8080/generate/stream \
  -H "Authorization: Bearer $KLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":["Hello","Once upon a time"],"max_tokens":24,"temperature":0.0}'
```

Events include `"index":0|1|...` so clients can demux the batch.
