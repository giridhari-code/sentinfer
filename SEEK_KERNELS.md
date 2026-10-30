# SeekKernels

One public API for Seek-V kernels.

```rust
use sentinel_kernels::{SeekKernels, ModelDims, LayerWeights, LayerKvCache};

let k = SeekKernels::parallel(); // or cpu() / gpu()
k.layer(&mut x, dims, &w, &mut cache)?;
let (logits, tokens) = k.forward(x, dims, &layers, &mut caches, final_norm, lm_head, Some((0.0, 1)))?;
```

| Method | Role |
| --- | --- |
| `layer` | One decoder block + KV |
| `forward` | Full stack + optional sample |
| `gemm` | Matmul primitive |
| `sample` | Token from logits |

Internals (`reference`, OpenCL, quant, TP) are not the product surface.
