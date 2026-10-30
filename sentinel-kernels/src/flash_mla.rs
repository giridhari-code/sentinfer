//! Multi-head Latent Attention (MLA) reference — FlashMLA-style decode/prefill path.
//!
//! Vendor CUDA FlashMLA is not in the DeepInfer zip; this is a correct f32 reference
//! matching the common MLA pattern:
//! * Q compressed via `w_q` / optional RoPE on split dims
//! * KV stored as latent `c_kv` then up-projected with `w_uk` / `w_uv` (absorbable)
//! * Causal attention over latent-up-projected K/V
//!
//! For simplicity this reference materializes K/V then runs [`crate::reference::attention_causal`].

use crate::error::{ensure, KernelError, Result};
use crate::reference::{attention_causal, matmul_nt, rope_inplace};

/// MLA config slice used by the reference kernel.
#[derive(Debug, Clone, Copy)]
pub struct MlaDims {
    pub tokens: usize,
    pub start_pos: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Latent KV rank (compressed dimension).
    pub kv_lora_rank: usize,
    /// RoPE dims applied on Q/K (often head_dim - nope_dim is non-rope).
    pub rope_dim: usize,
}

/// Prefill/decode MLA:
/// * `q` — `[tokens, num_heads, head_dim]` after Q projection (+ optional rope outside)
/// * `c_kv` — `[total_pos, kv_lora_rank]` latent cache including new positions
/// * `w_uk` — `[num_heads * head_dim, kv_lora_rank]` (K up-project, row-major n×k for matmul_nt)
/// * `w_uv` — `[num_heads * head_dim, kv_lora_rank]`
///
/// Returns attention output `[tokens, num_heads, head_dim]`.
pub fn flash_mla_ref(
    q: &[f32],
    c_kv: &[f32],
    w_uk: &[f32],
    w_uv: &[f32],
    dims: MlaDims,
) -> Result<Vec<f32>> {
    let MlaDims {
        tokens,
        start_pos,
        num_heads,
        head_dim,
        kv_lora_rank,
        rope_dim: _,
    } = dims;
    let total = start_pos + tokens;
    ensure!(tokens > 0 && num_heads > 0 && head_dim > 0 && kv_lora_rank > 0, KernelError::Shape, "mla dims");
    ensure!(q.len() == tokens * num_heads * head_dim, KernelError::Shape, "mla q");
    ensure!(c_kv.len() >= total * kv_lora_rank, KernelError::Shape, "mla c_kv");
    ensure!(w_uk.len() == num_heads * head_dim * kv_lora_rank, KernelError::Shape, "mla w_uk");
    ensure!(w_uv.len() == num_heads * head_dim * kv_lora_rank, KernelError::Shape, "mla w_uv");

    // Up-project all positions: K,V = c_kv @ W^T  → [total, num_heads*head_dim]
    let k_full = matmul_nt(c_kv, w_uk, total, kv_lora_rank, num_heads * head_dim)?;
    let v_full = matmul_nt(c_kv, w_uv, total, kv_lora_rank, num_heads * head_dim)?;
    // Treat as MHA with num_kv_heads == num_heads
    attention_causal(
        q,
        &k_full,
        &v_full,
        tokens,
        start_pos,
        num_heads,
        num_heads,
        head_dim,
    )
}

/// Apply RoPE only on the last `rope_dim` elements of each head (MLA partial rope).
pub fn rope_partial_inplace(
    x: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    start_pos: usize,
    theta: f32,
) -> Result<()> {
    ensure!(rope_dim <= head_dim && rope_dim % 2 == 0, KernelError::Shape, "rope_dim");
    if rope_dim == 0 {
        return Ok(());
    }
    // Extract rope slice per head, rope, write back — simple path
    let mut tmp = vec![0.0f32; tokens * heads * rope_dim];
    for t in 0..tokens {
        for h in 0..heads {
            let base = (t * heads + h) * head_dim + (head_dim - rope_dim);
            let dst = (t * heads + h) * rope_dim;
            tmp[dst..dst + rope_dim].copy_from_slice(&x[base..base + rope_dim]);
        }
    }
    rope_inplace(&mut tmp, tokens, heads, rope_dim, start_pos, theta)?;
    for t in 0..tokens {
        for h in 0..heads {
            let base = (t * heads + h) * head_dim + (head_dim - rope_dim);
            let dst = (t * heads + h) * rope_dim;
            x[base..base + rope_dim].copy_from_slice(&tmp[dst..dst + rope_dim]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mla_smoke_shapes() {
        let tokens = 2;
        let start = 0;
        let heads = 2;
        let hd = 4;
        let rank = 3;
        let q = vec![0.1f32; tokens * heads * hd];
        let c_kv = vec![0.2f32; (start + tokens) * rank];
        let w_uk = vec![0.05f32; heads * hd * rank];
        let w_uv = vec![0.04f32; heads * hd * rank];
        let out = flash_mla_ref(
            &q,
            &c_kv,
            &w_uk,
            &w_uv,
            MlaDims {
                tokens,
                start_pos: start,
                num_heads: heads,
                head_dim: hd,
                kv_lora_rank: rank,
                rope_dim: 2,
            },
        )
        .unwrap();
        assert_eq!(out.len(), tokens * heads * hd);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
