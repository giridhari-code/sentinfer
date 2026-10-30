# Gaps filled (this pass)

| Item | Status |
| --- | --- |
| Llm + custom vocab | `from_checkpoint_with_vocab` · kllm `--vocab` |
| Prometheus metrics | `GET /metrics` |
| OpenCL fused attention | device kernel + `GpuBackend` |
| GPU verify | `GET /gpu` (earlier) |
| CI | `.github/workflows/ci.yml` |
| TLS example | `deploy/nginx-tls.example.conf` |
| Stream multi-prompt | earlier + `start`/`done` events |

Still not full tensor-parallel / CUDA-native / true packed int4 checkpoints.
