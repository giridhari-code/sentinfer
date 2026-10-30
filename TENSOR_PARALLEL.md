# Tensor parallel / multi-GPU

## What shipped

In-process tensor parallelism in `sentinel_kernels::tensor_parallel`:

| API | Role |
| --- | --- |
| `InProcessTp::mesh(N)` | N logical ranks, shared barrier collectives |
| `column_parallel_*` | Shard **output** dim of `W`, all-gather `Y` |
| `row_parallel_*` | Shard **K** dim, all-reduce sum |
| `TpCollective` | Pluggable all-reduce / all-gather |

```rust
use sentinel_kernels::{InProcessTp, column_parallel_from_full};
let mesh = InProcessTp::mesh(4)?;
// each rank thread:
let y = column_parallel_from_full(&mesh[rank], &x, &w, m, k, n)?;
```

## Multi-GPU today

| Mode | Status |
| --- | --- |
| Logical TP on CPU threads | **Yes** |
| OpenCL multi-device TP | Not yet (single GpuBackend) |
| NCCL / CUDA TP | Use `TpCollective` + your runtime |
| Data-parallel multi-node | Existing `kllm-worker` / `kllm-coord` |

## Typical layer mapping

| Weight | Parallel style |
| --- | --- |
| `wq/wk/wv`, `w_gate/w_up` | Column-parallel |
| `wo`, `w_down` | Row-parallel |

## Not included

- Automatic Seek-V weight sharding in Engine (manual / next step)
- NVLink / NCCL bindings
