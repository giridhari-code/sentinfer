//! Mixture-of-Experts FFN for Seek-V.
//!
//! * [`moe_ffn`] — CPU reference
//! * [`moe_ffn_kernels`] — GEMMs via SeekKernels (GPU if available)
//! * [`moe_ffn_cached`] — OpenCL persistent expert weights + fused SiLU*up
//! * [`MoEGpuCache`] — upload once, reuse every step

use std::cmp::Ordering;

use sentinel_core::{Result, SentinelError};
use sentinel_kernels::reference as kern;

use crate::ops::kerr;

#[derive(Debug, Clone, Default)]
pub struct ExpertParallelConfig {
    pub ep_size: usize,
    pub enable_eplb: bool,
}

#[derive(Debug, Clone)]
pub struct MoEConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub ep: ExpertParallelConfig,
}

impl MoEConfig {
    pub fn validate(&self) -> Result<()> {
        if self.num_experts == 0 {
            return Err(SentinelError::Config("num_experts > 0".into()));
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(SentinelError::Config("bad top_k".into()));
        }
        if self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err(SentinelError::Config("sizes".into()));
        }
        Ok(())
    }
}

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
        let mut mat = |r: usize, c: usize| -> Vec<f32> {
            (0..r * c).map(|_| rng.uniform(-0.1, 0.1)).collect()
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
            return Err(SentinelError::Config("router shape".into()));
        }
        if self.experts.len() != e {
            return Err(SentinelError::Config("experts count".into()));
        }
        for (i, ex) in self.experts.iter().enumerate() {
            if ex.w_gate.len() != inter * d || ex.w_up.len() != inter * d || ex.w_down.len() != d * inter {
                return Err(SentinelError::Config(format!("expert {i} shape".into())));
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
            if v > m { m = v; }
        }
        let mut s = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for v in row.iter_mut() { *v *= inv; }
    }
}

fn top_k_row(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::equal));
    idx.truncate(k);
    let sum: f32 = idx.iter().map(|&i| scores[i]).sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 / k as f32 };
    idx.into_iter().map(|i| (i, scores[i] * inv)).collect()
}

pub fn moe_ffn(x: &[f32], moe: &MoEWeights) -> Result<Vec<f32>> {
    moe.validate()?;
    let cfg = &moe.cfg;
    let t = x.len() / cfg.hidden_size;
    let d = cfg.hidden_size;
    let e = cfg.num_experts;
    let k = cfg.top_k;
    let inter = cfg.intermediate_size;
    if x.len() != t * d {
        return Err(SentinelError::InvalidState("moe_ffn len".into()));
    }
    let mut router_logits = kern::matmul_nt(x, &moe.router, t, d, e).map_err(kerr)?;
    softmax_rows(&mut router_logits, t, e);
    let mut choices = Vec::with_capacity(t);
    for row in 0..t {
        choices.push(top_k_row(&router_logits[row * e..(row + 1) * e], k));
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
        if idxs.is_empty() { continue; }
        let n = idxs.len();
        let mut packed = vec![0.0f32; n * d];
        for (j, &tok) in idxs.iter().enumerate() {
            packed[j * d..(j + 1) * d].copy_from_slice(&x[tok * d..(tok + 1) * d]);
        }
        let ex = &moe.experts[ei];
        let mut gate = kern::matmul_nt(&packed, &ex.w_gate, n, d, inter).map_err(kerr)?;
        let up = kern::matmul_nt(&packed, &ex.w_up, n, d, inter).map_err(kerr)?;
        kern::silu_inplace(&mut gate).map_err(kerr)?;
        for i in 0..gate.len() { gate[i] *= up[i]; }
        let y = kern::matmul_nt(&gate, &ex.w_down, n, inter, d).map_err(kerr)?;
        for (j, &tok) in idxs.iter().enumerate() {
            let w = expert_weights[ei][j];
            for c in 0..d { out[tok * d + c] += w * y[j * d + c]; }
        }
    }
    Ok(out)
}

pub fn dense_swiglu(
    x: &[f32], w_gate: &[f32], w_up: &[f32], w_down: &[f32],
    tokens: usize, hidden: usize, intermediate: usize,
) -> Result<Vec<f32>> {
    let mut gate = kern::matmul_nt(x, w_gate, tokens, hidden, intermediate).map_err(kerr)?;
    let up = kern::matmul_nt(x, w_up, tokens, hidden, intermediate).map_err(kerr)?;
    kern::silu_inplace(&mut gate).map_err(kerr)?;
    for i in 0..gate.len() { gate[i] *= up[i]; }
    kern::matmul_nt(&gate, w_down, tokens, intermediate, hidden).map_err(kerr)
}

pub fn moe_ffn_kernels(
    x: &[f32], moe: &MoEWeights, kernels: &sentinel_kernels::SeekKernels,
) -> Result<Vec<f32>> {
    moe.validate()?;
    let cfg = &moe.cfg;
    let t = x.len() / cfg.hidden_size;
    let d = cfg.hidden_size;
    let e = cfg.num_experts;
    let k = cfg.top_k;
    let inter = cfg.intermediate_size;
    if x.len() != t * d {
        return Err(SentinelError::InvalidState("moe_ffn_kernels len".into()));
    }
    let mut router_logits = kernels.gemm(x, &moe.router, t, d, e).map_err(kerr)?;
    softmax_rows(&mut router_logits, t, e);
    let mut choices = Vec::with_capacity(t);
    for row in 0..t {
        choices.push(top_k_row(&router_logits[row * e..(row + 1) * e], k));
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
        if idxs.is_empty() { continue; }
        let n = idxs.len();
        let mut packed = vec![0.0f32; n * d];
        for (j, &tok) in idxs.iter().enumerate() {
            packed[j * d..(j + 1) * d].copy_from_slice(&x[tok * d..(tok + 1) * d]);
        }
        let ex = &moe.experts[ei];
        let gate = kernels.gemm(&packed, &ex.w_gate, n, d, inter).map_err(kerr)?;
        let up = kernels.gemm(&packed, &ex.w_up, n, d, inter).map_err(kerr)?;
        let y_mid = kernels.swiglu_fused(&gate, &up).map_err(kerr)?;
        let y = kernels.gemm(&y_mid, &ex.w_down, n, inter, d).map_err(kerr)?;
        for (j, &tok) in idxs.iter().enumerate() {
            let w = expert_weights[ei][j];
            for c in 0..d { out[tok * d + c] += w * y[j * d + c]; }
        }
    }
    Ok(out)
}

pub fn dense_swiglu_kernels(
    x: &[f32], w_gate: &[f32], w_up: &[f32], w_down: &[f32],
    tokens: usize, hidden: usize, intermediate: usize,
    kernels: &sentinel_kernels::SeekKernels,
) -> Result<Vec<f32>> {
    let gate = kernels.gemm(x, w_gate, tokens, hidden, intermediate).map_err(kerr)?;
    let up = kernels.gemm(x, w_up, tokens, hidden, intermediate).map_err(kerr)?;
    let y_mid = kernels.swiglu_fused(&gate, &up).map_err(kerr)?;
    kernels.gemm(&y_mid, w_down, tokens, intermediate, hidden).map_err(kerr)
}

/// Persistent OpenCL buffers for one MoE layer.
pub struct MoEGpuCache {
    pub router: sentinel_kernels::DeviceBuffer,
    pub gate: Vec<sentinel_kernels::DeviceBuffer>,
    pub up: Vec<sentinel_kernels::DeviceBuffer>,
    pub down: Vec<sentinel_kernels::DeviceBuffer>,
}

impl MoEGpuCache {
    /// Upload once. `None` if OpenCL not available.
    pub fn try_upload(moe: &MoEWeights) -> Result<Option<Self>> {
        moe.validate()?;
        let Some(dev) = sentinel_kernels::shared_opencl() else {
            return Ok(None);
        };
        let router = dev.upload_weights(&moe.router).map_err(kerr)?;
        let mut gate = Vec::with_capacity(moe.experts.len());
        let mut up = Vec::with_capacity(moe.experts.len());
        let mut down = Vec::with_capacity(moe.experts.len());
        for ex in &moe.experts {
            gate.push(dev.upload_weights(&ex.w_gate).map_err(kerr)?);
            up.push(dev.upload_weights(&ex.w_up).map_err(kerr)?);
            down.push(dev.upload_weights(&ex.w_down).map_err(kerr)?);
        }
        Ok(Some(Self { router, gate, up, down }))
    }
}

/// Cached expert weights + fused SwiGLU (OpenCL). Falls back to [`moe_ffn_kernels`].
pub fn moe_ffn_cached(
    x: &[f32],
    moe: &MoEWeights,
    cache: Option<&MoEGpuCache>,
    kernels: &sentinel_kernels::SeekKernels,
) -> Result<Vec<f32>> {
    let Some(cache) = cache else {
        return moe_ffn_kernels(x, moe, kernels);
    };
    let Some(dev) = sentinel_kernels::shared_opencl() else {
        return moe_ffn_kernels(x, moe, kernels);
    };
    moe.validate()?;
    let cfg = &moe.cfg;
    let t = x.len() / cfg.hidden_size;
    let d = cfg.hidden_size;
    let e = cfg.num_experts;
    let k = cfg.top_k;
    let inter = cfg.intermediate_size;
    if x.len() != t * d {
        return Err(SentinelError::InvalidState("moe_ffn_cached len".into()));
    }
    let mut router_logits = dev.matmul_nt_cached(x, &cache.router, t, d, e).map_err(kerr)?;
    softmax_rows(&mut router_logits, t, e);
    let mut choices = Vec::with_capacity(t);
    for row in 0..t {
        choices.push(top_k_row(&router_logits[row * e..(row + 1) * e], k));
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
        if idxs.is_empty() { continue; }
        let n = idxs.len();
        let mut packed = vec![0.0f32; n * d];
        for (j, &tok) in idxs.iter().enumerate() {
            packed[j * d..(j + 1) * d].copy_from_slice(&x[tok * d..(tok + 1) * d]);
        }
        let gate = dev.matmul_nt_cached(&packed, &cache.gate[ei], n, d, inter).map_err(kerr)?;
        let up = dev.matmul_nt_cached(&packed, &cache.up[ei], n, d, inter).map_err(kerr)?;
        let y_mid = kernels.swiglu_fused(&gate, &up).map_err(kerr)?;
        let y = dev.matmul_nt_cached(&y_mid, &cache.down[ei], n, inter, d).map_err(kerr)?;
        for (j, &tok) in idxs.iter().enumerate() {
            let w = expert_weights[ei][j];
            for c in 0..d { out[tok * d + c] += w * y[j * d + c]; }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moe_shape() {
        let cfg = MoEConfig {
            num_experts: 4, top_k: 2, intermediate_size: 16, hidden_size: 8, ep: Default::default(),
        };
        let moe = MoEWeights::random(&cfg, 7).unwrap();
        let x = vec![0.1f32; 3 * 8];
        let y = moe_ffn(&x, &moe).unwrap();
        assert_eq!(y.len(), 24);
    }
}
