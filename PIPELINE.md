# Full kernel pipeline (sentinel-kernels)

## What’s on *your* kernels today

```text
                    ┌─────────────────────────────────────────┐
                    │           sentinel-kernels               │
                    │  Backend: Cpu | ParallelCpu | Gpu/OpenCL │
                    └─────────────────────────────────────────┘
                                         │
     ┌───────────────────────────────────┼───────────────────────────────┐
     │                                   │                               │
     ▼                                   ▼                               ▼
┌─────────────┐                 ┌─────────────────┐              ┌──────────────┐
│ Core GEMM   │                 │ Attention       │              │ Sample / KV  │
│ matmul_nt   │                 │ attention_causal│              │ fused_sample │
│ deep_gemm   │                 │ flash_mla_ref   │              │ top_p_top_k  │
│ quant i8/i4 │                 │ rope / softmax  │              │ kv_index_*   │
└─────────────┘                 └─────────────────┘              │ paged_kv     │
                                                                 │ gather_*     │
                                                                 └──────────────┘
     │                                   │                               │
     └───────────────────────────────────┼───────────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  pipeline::run_layer / run_forward       │
                    │  rms → QKV → rope → attn → O → residual  │
                    │  → rms → SwiGLU → residual → norm → head │
                    │  → fused_sample                          │
                    └─────────────────────────────────────────┘
                                         │
                          optional: moe_forward · quant · OpenCL
```

## Stages (`PipelineStage`)

| Stage | Kernel |
| --- | --- |
| rms_attn | `rms_norm` |
| qkv_proj | `matmul_nt` / `deep_gemm_nt` |
| rope | `rope_inplace` |
| kv_append | (cache write) |
| attention_causal | `attention_causal` or `flash_mla_ref` |
| out_proj | `matmul_nt` |
| residual_1 | add |
| rms_mlp | `rms_norm` |
| swiglu | `matmul_nt` + `silu_inplace` |
| residual_2 | add |
| final_norm | `rms_norm` |
| lm_head | `matmul_nt` |
| fused_sample | `fused_sample` |

## API

```rust
use sentinel_kernels::{run_forward, run_layer, kernel_inventory, PipelineDims, LayerKvCache};

// inventory of every entry point
for (path, desc) in kernel_inventory() {
    println!("{path}: {desc}");
}

// one layer / full forward — see `pipeline` module tests
```

## Vendor CUDA (optional)

`vendor/deepseek-vendor-cuda.tar.gz` → FlashMLA, DeepGEMM, DeepEP (build with CUDA).
Not linked into the Rust pipeline by default.
