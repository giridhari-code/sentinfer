//! # SeekKernels
//!
//! Unified kernel surface for Seek-V: device → layer → forward → sample.
//!
//! ```ignore
//! let k = SeekKernels::parallel();
//! let (logits, ids) = k.forward(...)?;
//! ```

pub mod backend;
pub(crate) mod conformance;
pub(crate) mod cpu;
pub mod error;
pub mod gpu;
pub(crate) mod gpu_opencl;
pub(crate) mod opencl_ffi;
pub(crate) mod parallel_cpu;
pub(crate) mod pipeline;
pub mod quant;
pub mod reference;
pub(crate) mod tensor_parallel;

pub use backend::Backend;
pub use error::{KernelError, Result};
pub use gpu::GpuBackend;
pub use pipeline::LayerKvCache;
pub use quant::{QuantMatrix, QuantScheme};

use backend::Backend as _;
use cpu::CpuBackend;
use gpu::GpuBackend as Gpu;
use parallel_cpu::ParallelCpuBackend;
use pipeline::{run_forward, run_layer, run_layer_backend, LayerWeightsView, PipelineDims};

/// Borrowed weights for one decoder block (`matmul_nt` layout `[out, in]`).
pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],
    pub mlp_norm: &'a [f32],
    pub w_gate: &'a [f32],
    pub w_up: &'a [f32],
    pub w_down: &'a [f32],
}

impl<'a> LayerWeights<'a> {
    fn view(&self) -> LayerWeightsView<'a> {
        LayerWeightsView {
            attn_norm: self.attn_norm,
            wq: self.wq,
            wk: self.wk,
            wv: self.wv,
            wo: self.wo,
            mlp_norm: self.mlp_norm,
            w_gate: self.w_gate,
            w_up: self.w_up,
            w_down: self.w_down,
        }
    }
}

/// Step geometry for layer / forward.
#[derive(Debug, Clone, Copy)]
pub struct ModelDims {
    pub tokens: usize,
    pub start_pos: usize,
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f32,
}

impl From<ModelDims> for PipelineDims {
    fn from(d: ModelDims) -> Self {
        PipelineDims {
            tokens: d.tokens,
            start_pos: d.start_pos,
            hidden: d.hidden,
            num_heads: d.num_heads,
            num_kv_heads: d.num_kv_heads,
            head_dim: d.head_dim,
            intermediate: d.intermediate,
            vocab: d.vocab,
            rms_eps: d.rms_eps,
        }
    }
}

enum Kind {
    Cpu(CpuBackend),
    Parallel(ParallelCpuBackend),
    Gpu(Gpu),
}

/// Single kernel engine for Seek-V research and production paths.
pub struct SeekKernels {
    kind: Kind,
    rope_theta: f32,
    rms_eps: f32,
}

impl SeekKernels {
    pub fn cpu() -> Self {
        Self { kind: Kind::Cpu(CpuBackend), rope_theta: 10_000.0, rms_eps: 1e-6 }
    }

    pub fn parallel() -> Self {
        Self { kind: Kind::Parallel(ParallelCpuBackend), rope_theta: 10_000.0, rms_eps: 1e-6 }
    }

    pub fn gpu() -> Self {
        Self { kind: Kind::Gpu(Gpu::new()), rope_theta: 10_000.0, rms_eps: 1e-6 }
    }

    pub fn with_rope_theta(mut self, theta: f32) -> Self {
        self.rope_theta = theta;
        self
    }

    pub fn with_rms_eps(mut self, eps: f32) -> Self {
        self.rms_eps = eps;
        self
    }

    pub fn name(&self) -> &'static str {
        match &self.kind {
            Kind::Cpu(b) => b.name(),
            Kind::Parallel(b) => b.name(),
            Kind::Gpu(b) => b.name(),
        }
    }

    pub fn is_accelerated(&self) -> bool {
        matches!(&self.kind, Kind::Gpu(b) if b.is_device_accelerated())
    }

    /// One transformer block; writes residual stream `x` and updates KV `cache`.
    pub fn layer(
        &self,
        x: &mut [f32],
        dims: ModelDims,
        w: &LayerWeights<'_>,
        cache: &mut LayerKvCache,
    ) -> Result<()> {
        let pd = dims.into();
        let view = w.view();
        match &self.kind {
            Kind::Gpu(b) => run_layer_backend(b, x, &pd, &view, cache, self.rope_theta),
            _ => run_layer(x, &pd, &view, cache, self.rope_theta),
        }
    }

    /// Full stack: layers → final norm → lm_head → optional sample.
    pub fn forward(
        &self,
        x: Vec<f32>,
        dims: ModelDims,
        layers: &[LayerWeights<'_>],
        caches: &mut [LayerKvCache],
        final_norm: &[f32],
        lm_head: &[f32],
        sample: Option<(f32, u64)>,
    ) -> Result<(Vec<f32>, Option<Vec<i32>>)> {
        let pd = dims.into();
        let views: Vec<_> = layers.iter().map(LayerWeights::view).collect();
        run_forward(x, &pd, &views, caches, final_norm, lm_head, self.rope_theta, sample)
    }

    /// `X[m,k] @ W[n,k]^T`.
    pub fn gemm(&self, x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        match &self.kind {
            Kind::Cpu(b) => download_gemm(b, x, w, m, k, n),
            Kind::Parallel(b) => download_gemm(b, x, w, m, k, n),
            Kind::Gpu(b) => download_gemm(b, x, w, m, k, n),
        }
    }

    /// Greedy/sample one token from a logits row (`vocab` long).
    pub fn sample(&self, logits: &mut [f32], vocab: usize, temperature: f32, seed: u64) -> Result<i32> {
        let ids = reference::fused_sample(logits, &[1.0], &[-1], 1, vocab, temperature, seed)?;
        Ok(ids[0])
    }
}

fn download_gemm<B: Backend>(b: &B, x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    let y = b.matmul_nt(&b.upload(x)?, &b.upload(w)?, m, k, n)?;
    b.download(&y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_smoke() {
        let k = SeekKernels::parallel();
        let dims = ModelDims {
            tokens: 2,
            start_pos: 0,
            hidden: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate: 16,
            vocab: 12,
            rms_eps: 1e-6,
        };
        let d = dims.hidden;
        let inter = dims.intermediate;
        let ones = |n| vec![1.0f32; n];
        let s = |n, t: f32| (0..n).map(|i| ((i as f32) * t).sin() * 0.05).collect::<Vec<_>>();
        let an = ones(d);
        let mn = ones(d);
        let wq = s(dims.num_heads * dims.head_dim * d, 0.01);
        let wk = s(dims.num_kv_heads * dims.head_dim * d, 0.02);
        let wv = s(dims.num_kv_heads * dims.head_dim * d, 0.03);
        let wo = s(d * dims.num_heads * dims.head_dim, 0.01);
        let wg = s(inter * d, 0.02);
        let wu = s(inter * d, 0.03);
        let wd = s(d * inter, 0.01);
        let layer = LayerWeights {
            attn_norm: &an,
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wo: &wo,
            mlp_norm: &mn,
            w_gate: &wg,
            w_up: &wu,
            w_down: &wd,
        };
        let final_n = ones(d);
        let lm = s(dims.vocab * d, 0.001);
        let mut caches = vec![LayerKvCache::new()];
        let (logits, ids) = k
            .forward(
                vec![0.1; dims.tokens * d],
                dims,
                &[layer],
                &mut caches,
                &final_n,
                &lm,
                Some((0.0, 1)),
            )
            .unwrap();
        assert_eq!(logits.len(), dims.vocab);
        assert!(ids.is_some());
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
