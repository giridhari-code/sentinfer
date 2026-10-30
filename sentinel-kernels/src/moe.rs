//! Mixture-of-Experts reference kernels.
//!
//! DeepInfer MoE routing often uses DeepEP; sources are not in the zip.
//! This implements a standard top-k softmax router + expert SwiGLU MLP combine.

use crate::error::{ensure, KernelError, Result};
use crate::reference::{matmul_nt, silu_inplace};

/// One expert's SwiGLU weights: gate/up `[inter, hidden]`, down `[hidden, inter]` (matmul_nt layout).
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

/// Router logits projection `[num_experts, hidden]` for matmul_nt.
#[derive(Debug, Clone)]
pub struct MoeWeights {
    pub router: Vec<f32>,
    pub experts: Vec<ExpertWeights>,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

/// Router: `x[tokens, hidden] @ router^T` → scores `[tokens, num_experts]`, then top-k softmax weights.
pub fn moe_router_topk(
    x: &[f32],
    router: &[f32],
    tokens: usize,
    hidden: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<(Vec<i32>, Vec<f32>)> {
    ensure!(top_k > 0 && top_k <= num_experts, KernelError::Shape, "moe top_k");
    ensure!(x.len() == tokens * hidden && router.len() == num_experts * hidden, KernelError::Shape, "moe router");
    let scores = matmul_nt(x, router, tokens, hidden, num_experts)?;
    let mut indices = vec![0i32; tokens * top_k];
    let mut weights = vec![0.0f32; tokens * top_k];
    for t in 0..tokens {
        let row = &scores[t * num_experts..(t + 1) * num_experts];
        let mut order: Vec<usize> = (0..num_experts).collect();
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut max = f32::NEG_INFINITY;
        for &e in order.iter().take(top_k) {
            max = max.max(row[e]);
        }
        let mut sum = 0.0f32;
        let mut local = vec![0.0f32; top_k];
        for (i, &e) in order.iter().take(top_k).enumerate() {
            local[i] = (row[e] - max).exp();
            sum += local[i];
            indices[t * top_k + i] = e as i32;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for i in 0..top_k {
            weights[t * top_k + i] = local[i] * inv;
        }
    }
    Ok((indices, weights))
}

fn expert_ffn(x: &[f32], tokens: usize, hidden: usize, inter: usize, e: &ExpertWeights) -> Result<Vec<f32>> {
    let mut gate = matmul_nt(x, &e.w_gate, tokens, hidden, inter)?;
    let up = matmul_nt(x, &e.w_up, tokens, hidden, inter)?;
    silu_inplace(&mut gate);
    for i in 0..gate.len() {
        gate[i] *= up[i];
    }
    matmul_nt(&gate, &e.w_down, tokens, inter, hidden)
}

/// Full MoE forward: route → expert FFN → weighted sum.
/// Returns `[tokens, hidden]`.
pub fn moe_forward(x: &[f32], tokens: usize, w: &MoeWeights) -> Result<Vec<f32>> {
    ensure!(x.len() == tokens * w.hidden, KernelError::Shape, "moe x");
    ensure!(w.experts.len() == w.num_experts, KernelError::Shape, "moe experts len");
    let (indices, weights) = moe_router_topk(x, &w.router, tokens, w.hidden, w.num_experts, w.top_k)?;
    let mut out = vec![0.0f32; tokens * w.hidden];
    // For reference: run each selected expert on the full batch then scatter-add (simple, not sparse).
    // Better sparse path: per-token expert — clearer correctness.
    for t in 0..tokens {
        let xt = &x[t * w.hidden..(t + 1) * w.hidden];
        for k in 0..w.top_k {
            let eidx = indices[t * w.top_k + k] as usize;
            let wgt = weights[t * w.top_k + k];
            let y = expert_ffn(xt, 1, w.hidden, w.intermediate, &w.experts[eidx])?;
            for c in 0..w.hidden {
                out[t * w.hidden + c] += wgt * y[c];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_expert(hidden: usize, inter: usize, seed: f32) -> ExpertWeights {
        let n = |len, s: f32| (0..len).map(|i| ((i as f32 + s) * 0.01).sin()).collect();
        ExpertWeights {
            w_gate: n(inter * hidden, seed),
            w_up: n(inter * hidden, seed + 1.0),
            w_down: n(hidden * inter, seed + 2.0),
        }
    }

    #[test]
    fn moe_forward_shapes_and_finite() {
        let hidden = 8;
        let inter = 16;
        let num_experts = 4;
        let top_k = 2;
        let tokens = 3;
        let router: Vec<f32> = (0..num_experts * hidden).map(|i| (i as f32 * 0.02).cos()).collect();
        let experts: Vec<_> = (0..num_experts).map(|e| dummy_expert(hidden, inter, e as f32)).collect();
        let w = MoeWeights {
            router,
            experts,
            hidden,
            intermediate: inter,
            num_experts,
            top_k,
        };
        let x: Vec<f32> = (0..tokens * hidden).map(|i| (i as f32 * 0.03).sin()).collect();
        let y = moe_forward(&x, tokens, &w).unwrap();
        assert_eq!(y.len(), tokens * hidden);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn router_topk_picks_k() {
        let tokens = 1;
        let hidden = 4;
        let ne = 3;
        let x = vec![1.0f32; hidden];
        // expert 1 strongly aligned
        let mut router = vec![0.0f32; ne * hidden];
        for c in 0..hidden {
            router[1 * hidden + c] = 10.0;
        }
        let (idx, wts) = moe_router_topk(&x, &router, tokens, hidden, ne, 2).unwrap();
        assert_eq!(idx[0], 1);
        assert!((wts[0] + wts[1] - 1.0).abs() < 1e-4);
    }
}
