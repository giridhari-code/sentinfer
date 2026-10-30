# Phase C — attention, copies, benches, harden

## Done

| Item | Detail |
| --- | --- |
| Fused `attention_causal` | `sentinel-kernels::reference` + `Backend` |
| Seek-V forward | uses kernel attention (no hand-rolled scores loop) |
| Matmul | already blocked + parallel |
| Bench | `cargo run -p sentinel-model --example bench --release` |
| Serve limits | Phase B body/prompt/token caps |

## Still open (next)

| Item | Note |
| --- | --- |
| OpenCL fused attention | device kernel; host path is default today |
| Persistent device buffers | fewer host↔device copies on GPU path |
| SIMD (`std::simd` / target-cpu) | further matmul/attention speed |
