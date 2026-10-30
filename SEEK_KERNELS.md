# SeekKernels — senior research / engineering pipeline

One public kernel object. Everything else is implementation detail.

```text
┌──────────────────────────────────────────────────────────┐
│                     SeekKernels                          │
│   cpu()  ·  parallel()  ·  gpu()                         │
└────────────┬─────────────────────────────┬───────────────┘
             │                             │
     primitives                      full stack
     gemm / sample                   layer → forward
             │                             │
             ▼                             ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│ Backend             │     │ Decoder pipeline             │
│ CPU | Parallel | GPU│     │ RMS → QKV → RoPE → Attn →   │
│                     │     │ O → residual → SwiGLU → …   │
│                     │     │ → final norm → lm_head      │
│                     │     │ → sample                    │
└─────────────────────┘     └──────────────────────────────┘
```

## API

```rust
use sentinel_kernels::{SeekKernels, ModelDims, LayerWeights, LayerKvCache};

let k = SeekKernels::parallel(); // or ::cpu() / ::gpu()

// one block
k.layer(&mut x, dims, &weights, &mut cache)?;

// full model step
let (logits, tokens) = k.forward(
    x, dims, &layers, &mut caches, final_norm, lm_head, Some((0.0, 42)),
)?;
```

## Design rules

1. **One name** — `SeekKernels` is the product surface.
2. **Oracle path** — CPU/parallel `layer`/`forward` use the reference decoder math.
3. **Device path** — `gpu()` runs the same pipeline through the Backend trait.
4. **No scattered public kernels** — MLA/MoE/TP/quant live as internal modules; call via SeekKernels or keep rare escapes (`reference`, `quant`) for training/compat only.

## Not the public surface anymore

`flash_mla`, `moe`, `tensor_parallel`, `deep_gemm`, `cuda_vendor` — internal.
Wire them into `SeekKernels` methods only when a research experiment needs them.
