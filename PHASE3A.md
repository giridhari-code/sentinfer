# Phase 3a — Seek-V production path

## Done

| Item | Status |
| --- | --- |
| Backend ops: matmul, silu, softmax, rope | done |
| Blocked matmul in kernel reference | done |
| Seek-V forward uses `sentinel-kernels::reference` (CpuBackend kernels) | done |
| `Tokenizer` trait + Byte + Id | done |
| Training `metrics.csv` | done |
| `--sample-every` / `--sample-prompt` / `--sample-tokens` | done |
| `Engine::for_seekv` + FLOPs | done |

## Next

| Item | Notes |
| --- | --- |
| Attention op on Backend | fuse QK/softmax/AV |
| Generic `SeekV<B: Backend>` | keep weights on device |
| SIMD / rayon matmul rows | further speed |
| Benchmarks tok/s | before/after table in README |

## Sample during train

```bash
cargo run --release -p sentinel-train --bin train -- \
  --data corpus.txt --out runs/p3a --preset small --steps 2000 \
  --sample-every 100 --sample-prompt "Once upon a time" --sample-tokens 40
```
