//! Mixture-of-Experts FFN for Seek-V (DeepInfer / DeepSeek-style sparse MLP).
//!
//! Architecture:
//! * **Router** — linear `hidden → num_experts`, softmax, top-k selection.
//! * **Experts** — independent SwiGLU FFNs (`w_gate`, `w_up`, `w_down`).
//! * **Combine** — weighted sum of selected expert outputs.
//!
//! **DeepGEMM-inspired packing:** tokens routed to the same expert are packed into a
//! contiguous batch so one `matmul_nt` serves many tokens (better than per-token GEMM).
//! Full CUDA DeepGEMM is not vendored; this is the algorithmic equivalent on our kernels.
//!
//! **EP / EPLB:** expert-parallel sharding and load-balance are not implemented yet;
//! `ExpertParallelConfig` holds the future knobs.

use sentinel_core::{Result, SentinelError};
use sentinel_kernels::reference as kern;

use crate::ops::kerr;

/// Expert-parallel / load-balance scaffolding (DeepInfer EPLB).
#[derive(Debug, Clone, Default)]
pub struct ExpertParallelConfig {
    /// Number of expert-parallel ranks (1 = no EP).
    pub ep_size: usize,
    /// Enable expert load-balance metrics / future EPLB.
    pub enable_eplb: bool,
}

/// MoE hyperparameters.
#[derive(Debug, Clone)]
pub struct MoEConfig {
    pub num_experts: usize,
    /// How many experts each token activates (top-k).
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub ep: ExpertParallelConfig,
}

impl MoEConfig {
    pub fn validate(&self) -> Result<()> {
        if self.num_experts == 0 {
            return Err(SentinelError::Config("MoEConfig: num_experts must be > 0".into()));
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(SentinelError::Config(format!(
                "MoEConfig: top_k={} invalid for num_experts={}",
                self.top_k, self.num_experts
            )));
        }
        if self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err(SentinelError::Config("MoEConfig: sizes must be > 0".into()));
        }
        Ok(())
    }
}

/// One SwiGLU expert.
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

impl ExpertWeights {
    pub fn zeros(hidden: usize, intermediate: usize) -> Self {
        Self {
            w_gate: vec![0.0; intermediate * hidden],
            w_up: vec![0.0; intermediate * hidden],
            w_down: vec![0.0; hidden * intermediate],
        }
    }

    pub fn random(hidden: usize, intermediate: usize, seed: u64) -> Self {
        use crate::ops::SplitMix;
        let mut rng = SplitMix::new(seed);
        let mut mat = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.uniform(-0.1, 0.1)).collect()
        };
        Self {
            w_gate: mat(intermediate, hidden),
            w_up: mat(intermediate, hidden),
            w_down: mat(hidden, intermediate),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MoEWeights {
    pub router: Vec<f32>,
    pub experts: Vec<ExpertWeights>,
    pub cfg: MoEConfig,
}

impl MoEWeights {
    pub fn random(cfg: &MoEConfig, seed: u64) -> Result<Self> {
        cfg.validate()?;
        use crate::ops::SplitMix;
        let mut rng = SplitMix::new(seed);
        let router: Vec<f32> = (0..cfg.num_experts * cfg.hidden_size)
            .map(|_| rng.uniform(-0.1, 0.1))
            .collect();
        let experts = (0..cfg.num_experts)
            .map(|i| ExpertWeights::random(cfg.hidden_size, cfg.intermediate_size, seed + 1 + i as u64))
            .collect();
        Ok(Self { router, experts, cfg: cfg.clone() })
    }

    pub fn zeros(cfg: &MoEConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            router: vec![0.0; cfg.num_experts * cfg.hidden_size],
            experts: (0..cfg.num_experts)
                .map(|_| ExpertWeights::zeros(cfg.hidden_size, cfg.intermediate_size))
                .collect(),
            cfg: cfg.clone(),
        })
    }

    pub fn validate(&self) -> Result<()> {
        self.cfg.validate()?;
        let (e, d, inter) = (self.cfg.num_experts, self.cfg.hidden_size, self.cfg.intermediate_size);
        if self.router.len() != e * d {
            return Err(SentinelError::Config(format!(
                "MoE router len {} expected {}",
                self.router.len(),
                e * d
            )));
        }
        if self.experts.len() != e {
            return Err(SentinelError::Config("MoE experts count mismatch".into()));
        }
        for (i, ex) in self.experts.iter().enumerate() {
            if ex.w_gate.len() != inter * d || ex.w_up.len() != inter * d || ex.w_down.len() != d * inter {
                return Err(SentinelError::Config(format!("MoE expert {i} shape mismatch")));
            }
        }
        Ok(())
    }
}

fn softmax_rows(logits: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut logits[r * cols..(r + 1) * cols];
        let mut m = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

fn top_k_row(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    let sum: f32 = idx.iter().map(|&i| scores[i]).sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 / k as f32 };
    idx.into_iter().map(|i| (i, scores[i] * inv)).collect()
}

pub fn moe_ffn(x: &[f32], moe: &MoEWeights) -> Result<Vec<f32>> {
    moe.validate()?;
    let cfg = &moe.cfg;
    let (t, d, e, k, inter) = (
        x.len() / cfg.hidden_size,
        cfg.hidden_size,
        cfg.num_experts,
        cfg.top_k,
        cfg.intermediate_size,
    );
    if x.len() != t * d {
        return Err(SentinelError::InvalidState("moe_ffn: x length not divisible by hidden".into()));
    }

    let mut router_logits = kern::matmul_nt(x, &moe.router, t, d, e).map_err(kerr)?;
    softmax_rows(&mut router_logits, t, e);

    let mut choices: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t);
    for row in 0..t {
        let scores = &router_logits[row * e..(row + 1) * e];
        choices.push(top_k_row(scores, k));
    }

    let mut expert_tokens: Vec<Vec<usize>> = vec![Vec::new(); e];
    let mut expert_weights: Vec<Vec<f32>> = vec![Vec::new(); e];
    for (tok, ch) in choices.iter().enumerate() {
        for &(ei, w) in ch {
            expert_tokens[ei].push(tok);
            expert_weights[ei].push(w);
        }
    }

    let mut out = vec![0.0f32; t * d];

    for ei in 0..e {
        let idxs = &expert_tokens[ei];
        if idxs.is_empty() {
            continue;
        }
        let n = idxs.len();
        let mut packed = vec![0.0f32; n * d];
        for (j, &tok) in idxs.iter().enumerate() {
            packed[j * d..(j + 1) * d].copy_from_slice(&x[tok * d..(tok + 1) * d]);
        }
        let ex = &moe.experts[ei];
        let mut gate = kern::matmul_nt(&packed, &ex.w_gate, n, d, inter).map_err(kerr)?;
        let up = kern::matmul_nt(&packed, &ex.w_up, n, d, inter).map_err(kerr)?;
        kern::silu_inplace(&mut gate).map_err(kerr)?;
        for i in 0..gate.len() {
            gate[i] *= up[i];
        }
        let y = kern::matmul_nt(&gate, &ex.w_down, n, inter, d).map_err(kerr)?;

        for (j, &tok) in idxs.iter().enumerate() {
            let w = expert_weights[ei][j];
            for c in 0..d {
                out[tok * d + c] += w * y[j * d + c];
            }
        }
    }

    Ok(out)
}

pub fn dense_swiglu(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    tokens: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<Vec<f32>> {
    let mut gate = kern::matmul_nt(x, w_gate, tokens, hidden, intermediate).map_err(kerr)?;
    let up = kern::matmul_nt(x, w_up, tokens, hidden, intermediate).map_err(kerr)?;
    kern::silu_inplace(&mut gate).map_err(kerr)?;
    for i in 0..gate.len() {
        gate[i] *= up[i];
    }
    kern::matmul_nt(&gate, w_down, tokens, intermediate, hidden).map_err(kerr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_ffn_shape_and_finite() {
        let cfg = MoEConfig {
            num_experts: 4,
            top_k: 2,
            intermediate_size: 16,
            hidden_size: 8,
            ep: Default::default(),
        };
        let moe = MoEWeights::random(&cfg, 7).unwrap();
        let t = 3;
        let x = vec![0.1f32; t * cfg.hidden_size];
        let y = moe_ffn(&x, &moe).unwrap();
        assert_eq!(y.len(), t * cfg.hidden_size);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
