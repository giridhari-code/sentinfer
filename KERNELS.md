# Sentinel kernels: findings and plan (phase 3 start)

## What the kernels actually are

DeepInfer's kernels are **TileLang** (a Python DSL that JIT-compiles to GPU code), not CUDA C++.
They live in `deepinfer/model_executor/layers/tilelang_ops/`, with one module per hardware
backend (`cuda/`, `asc/` for Ascend NPUs) selected once in `ops.py` via `PLATFORM.tilelang_ops_backend`.
Many are large (several are 30-50 KB each), and many ship a torch `*_ref` reference function.

The attention and GEMM/MoE kernels (FlashMLA, DeepGEMM, DeepEP, DeepSelect and their Ascend
variants) are git submodules under `vendor/`. **They are empty in the uploaded zip**, so their
sources are not available here.

## What phase 3 delivers so far (`sentinel-kernels` crate)

* `Backend` trait with backend-owned buffer types (`Backend::Buffer<T>`): data stays on the device
  between ops; only `upload`/`download` cross the host boundary. It mirrors the `ops.py` dispatcher.
* `reference.rs`: slice-level oracle ports of the repo's `*_ref` functions.
* `CpuBackend`: reference backend (slow, correct).
* `conformance::check_against_reference`: randomized cross-check any new backend runs in its own tests.
  A test proves the harness fails on a wrong backend.

Only f32/i32 so far; bf16 and fp8 arrive with the first GPU backend.

## Op inventory

| op module | cuda | asc | Rust status |
| --- | --- | --- | --- |
| `__init__` | yes | yes | - |
| `cache_common` | - | - | - |
| `candidate_blocks` | yes | yes | - |
| `compress` | yes | yes | - |
| `compute_seqlens_k_zos` | yes | yes | yes |
| `dflash` | yes | yes | - |
| `dflash_copy` | yes | yes | - |
| `flash_mla_utils` | yes | yes | - |
| `fused_sample` | yes | yes | apply_top_p_top_k only |
| `fused_transform` | yes | yes | - |
| `gather` | yes | yes | - |
| `gather_engram_tokens_zos` | yes | yes | - |
| `gather_input_ids_zos` | yes | yes | - |
| `get_kv_index` | yes | yes | kv_index_prefill / kv_index_decode (not the SWA variant) |
| `image_embed_ops` | yes | yes | - |
| `kv_index_cat` | yes | yes | - |
| `load_paged_kv_cache` | yes | yes | - |
| `norm` | yes | yes | rms_norm (no fp8 cast / scale-factor outputs yet) |
| `outlier_detection` | yes | - | - |
| `prepare_cp_mirror` | yes | yes | - |
| `prepare_decode_metadata` | yes | - | - |
| `prepare_v4_decode_metadata` | - | yes | - |
| `quant` | yes | - | - |
| `random_sampling` | yes | yes | - |
| `sampler` | yes | yes | - |
| `schedule_mtp_varlen` | yes | yes | - |
| `state_cache` | yes | yes | - |
| `store_kv_cache` | yes | yes | - |
| `store_mtp_info` | yes | yes | - |
| `structure_output` | yes | yes | - |
| `sum_smaller_probs` | yes | yes | - |
| `swa_index` | yes | yes | - |

## Options for the GPU side

**A. Reuse the TileLang kernels (recommended for the first end-to-end run).**
1. Embed Python via PyO3 and call the existing `ops.*_tl` factories. Fastest route to a working Rust
   engine, but it keeps the Python + torch runtime for tensors and kernels.
2. AOT: have TileLang emit CUDA source per configuration at build time, then load it from Rust
   (e.g. with `cudarc`). No Python at runtime, but needs a build-time export step per shape
   configuration. I have not verified TileLang's export path; treat this as a hypothesis to test.

**B. Native rewrite per op (wgpu for cross-platform, or CUDA C/PTX).**
~35 ops, several very large (`fused_sample`, `fused_transform`). Each port is checked with
`check_against_reference`, so this can be done one op at a time and mixed with option A.

Suggested path: A(1) to get an end-to-end Rust engine, then replace the hottest ops with native
backends one by one, gated by the conformance check.


## GPU backend (OpenCL)

`GpuBackend` loads `libOpenCL.so.1` at runtime and builds these device kernels:

| Kernel | Work |
| --- | --- |
| `matmul_nt` | `X[m,k] @ W[n,k]^T` |
| `rms_norm` | row-wise RMSNorm |
| `silu_inplace` | SiLU / swish |
| `softmax_rows` | row softmax |
| `rope_inplace` | rotary embeddings |

If no OpenCL platform is present, `GpuBackend` falls back to `ParallelCpuBackend`
(`is_device_accelerated() == false`, name `gpu-fallback-cpu`).

```rust
use sentinel_kernels::{Backend, GpuBackend};
let g = GpuBackend::new();
eprintln!("{} accelerated={}", g.name(), g.is_device_accelerated());
let y = g.matmul_nt(&x, &w, m, k, n)?;
```

wgpu/CUDA crate backends are not linked by default (registry/toolchain limits); OpenCL
covers NVIDIA/AMD/Intel GPUs via the system ICD without extra Cargo deps.


## Kernel completeness update (Seek-V path)

DeepInfer TileLang sources / vendor FlashMLA are **not in the uploaded zip**. Rust ports
the reference semantics needed for Seek-V continuous batching:

| Op | Status |
| --- | --- |
| matmul_nt / parallel | yes |
| rms_norm | yes |
| silu / softmax / rope | yes |
| attention_causal | yes (CPU + OpenCL) |
| apply_top_p_top_k | yes |
| fused_sample / sample_from_logits | **yes** |
| compute_seqlens_k_zos | yes |
| kv_index_prefill / decode | yes |
| kv_index_cat | **yes** |
| gather_rows / gather_i32 | **yes** |
| load_paged_kv_cache | **yes** |
| quant (int8/int4 weights) | yes |
| FlashMLA / DeepGEMM / MoE | **reference yes** (`flash_mla`, `deep_gemm`, `moe` modules; vendor CUDA still absent) |
| dflash / image_embed / SWA kv | deferred (not Seek-V core) |


## FlashMLA / DeepGEMM / MoE (reference)

| Module | API |
| --- | --- |
| `flash_mla` | `flash_mla_ref`, `rope_partial_inplace`, `MlaDims` |
| `deep_gemm` | `deep_gemm_nt`, `deep_gemm_nt_tiled` |
| `moe` | `moe_forward`, `moe_router_topk`, `MoeWeights` |

These are **f32 reference** implementations. DeepInfer’s CUDA TileLang/vendor binaries
were not present in the uploaded tree; wire GPU kernels later when sources are available.
