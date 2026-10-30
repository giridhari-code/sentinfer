# Vendor CUDA ↔ Rust bridge

## Layout

| Path | Role |
| --- | --- |
| `deepseek-vendor-cuda.tar.gz` | FlashMLA + DeepGEMM + DeepEP **sources** |
| `rust_abi/sentinel_vendor_abi.h` | C ABI expected by `sentinel_kernels::cuda_vendor` |
| Extracted trees | `./scripts/fetch-vendor.sh` |

## Rust API

```rust
use sentinel_kernels::{vendor_flash_mla, vendor_deep_gemm_nt, vendor_moe_forward, vendor_backend_name};

// Uses Rust reference unless SENTINEL_VENDOR_SO is set to a .so implementing rust_abi
let y = vendor_deep_gemm_nt(&x, &w, m, k, n)?;
eprintln!("{}", vendor_backend_name()); // "rust-reference" or "cuda-vendor-so"
```

## Why not link FlashMLA/DeepGEMM/DeepEP directly?

Those projects are **PyTorch C++ extensions** (`torch::Tensor`), not a stable `extern "C"` library.
Linking them pulls libtorch. Instead:

1. Build a small CUDA `.so` that implements `sentinel_vendor_abi.h` (wrapping their kernels or your own).
2. `export SENTINEL_VENDOR_SO=/path/to/libsentinel_vendor.so`
3. Rust `dlopen`s it at runtime.

## Extract sources

```bash
./scripts/fetch-vendor.sh
```
