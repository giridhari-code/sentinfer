/* ABI for optional CUDA vendor .so used by sentinel-kernels cuda_vendor bridge.
 * Build a shared library that implements these and point SENTINEL_VENDOR_SO at it.
 * Until then, Rust uses f32 reference kernels.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Y[m,n] = X[m,k] @ W[n,k]^T  (row-major f32) */
int sentinel_deep_gemm_nt_f32(
    const float* x, const float* w, float* out,
    int32_t m, int32_t k, int32_t n);

/* FlashMLA-style: see flash_mla_ref layout in Rust docs.
 * q: [tokens, heads, head_dim]
 * c_kv: [total, kv_lora_rank]
 * w_uk, w_uv: [heads*head_dim, kv_lora_rank]
 * out: [tokens, heads, head_dim]
 */
int sentinel_flash_mla_f32(
    const float* q, const float* c_kv, const float* w_uk, const float* w_uv,
    float* out,
    int32_t tokens, int32_t start_pos, int32_t num_heads, int32_t head_dim,
    int32_t kv_lora_rank);

/* Placeholder MoE dispatch (experts run on host reference if unimplemented). */
int sentinel_moe_available(void);

#ifdef __cplusplus
}
#endif
