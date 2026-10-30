//! # SeekKernels
//!
//! One kernel surface for Seek-V research & engineering:
//! backends → decoder ops → full forward pipeline → sample.
//!
//! ```ignore
//! use sentinel_kernels::SeekKernels;
//! let k = SeekKernels::parallel();
//! let logits = k.forward(/* ... */)?;
//! ```
//!
//! Internal modules are implementation detail (reference math, OpenCL, TP, quant).

#![allow(dead_code)]

pub mod backend;
pub(crate) mod conformance;
pub(crate) mod cpu;
pub(crate) mod cuda_vendor;
pub(crate) mod deep_gemm;
pub mod error;
pub(crate) mod flash_mla;
pub mod gpu;
pub(crate) mod gpu_opencl;
pub(crate) mod moe;
pub(crate) mod opencl_ffi;
pub(crate) mod parallel_cpu;
pub(crate) mod pipeline;
pub mod quant;
pub mod reference;
pub(crate) mod tensor_parallel;

pub use error::{KernelError, Result};

// --- Compatibility exports (prefer SeekKernels for new code) ---
pub use quant::{QuantMatrix, QuantScheme};
pub use backend::Backend;
pub use gpu::GpuBackend;
pub use pipeline::LayerKvCache;


use cpu::CpuBackend;
use parallel_cpu::ParallelCpuBackend;
use pipeline::{run_forward, run_layer, LayerWeightsView, PipelineDims};

/// Layer weights borrowed for one decoder block (row-major `matmul_nt` layout).
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

/// Model-level dims for a forward step.
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
        }
    }
}

enum BackendKind {
    Cpu(CpuBackend),
    Parallel(ParallelCpuBackend),
    Gpu(GpuBackend),
}

/// **SeekKernels** — single entry for all kernel work.
///
/// Research pipeline:
/// 1. Choose device (`cpu` / `parallel` / `gpu`)
/// 2. `layer` — one transformer block + KV cache
/// 3. `forward` — full stack → logits (+ optional sample)
/// 4. `gemm` / `rms` / `sample` — primitive escapes when needed
pub struct SeekKernels {
    kind: BackendKind,
    rope_theta: f32,
}

impl SeekKernels {
    /// Scalar CPU (correctness / debug).
    pub fn cpu() -> Self {
        Self { kind: BackendKind::Cpu(CpuBackend), rope_theta: 10_000.0 }
    }

    /// Multi-core CPU (default engineering path).
    pub fn parallel() -> Self {
        Self { kind: BackendKind::Parallel(ParallelCpuBackend), rope_theta: 10_000.0 }
    }

    /// OpenCL GPU if present, else parallel CPU.
    pub fn gpu() -> Self {
        Self { kind: BackendKind::Gpu(GpuBackend::new()), rope_theta: 10_000.0 }
    }

    pub fn with_rope_theta(mut self, theta: f32) -> Self {
        self.rope_theta = theta;
        self
    }

    pub fn name(&self) -> &'static str {
        match &self.kind {
            BackendKind::Cpu(b) => b.name(),
            BackendKind::Parallel(b) => b.name(),
            BackendKind::Gpu(b) => b.name(),
        }
    }

    pub fn is_accelerated(&self) -> bool {
        match &self.kind {
            BackendKind::Gpu(b) => b.is_device_accelerated(),
            _ => false,
        }
    }

    /// One decoder layer; updates `x` in place and extends `cache`.
    pub fn layer(
        &self,
        x: &mut [f32],
        dims: ModelDims,
        w: &LayerWeights<'_>,
        cache: &mut LayerKvCache,
    ) -> Result<()> {
        let pd = dims.into();
        let view = LayerWeightsView {
            attn_norm: w.attn_norm,
            wq: w.wq,
            wk: w.wk,
            wv: w.wv,
            wo: w.wo,
            mlp_norm: w.mlp_norm,
            w_gate: w.w_gate,
            w_up: w.w_up,
            w_down: w.w_down,
        };
        // Reference path is the research oracle; backend path for device parity tests.
        match &self.kind {
            BackendKind::Cpu(_) | BackendKind::Parallel(_) => {
                run_layer(x, &pd, &view, cache, self.rope_theta)
            }
            BackendKind::Gpu(b) => {
                pipeline::run_layer_backend(b, x, &pd, &view, cache, self.rope_theta)
            }
        }
    }

    /// Full research forward: N layers → final RMSNorm → lm_head → optional greedy/sample.
    ///
    /// Returns `(logits[vocab], optional token ids)`.
    pub fn forward(
        &self,
        x: Vec<f32>,
        dims: ModelDims,
        layers: &[LayerWeights<'_>],
        caches: &mut [LayerKvCache],
        final_norm: &[f32],
        lm_head: &[f32],
        sample_temp_seed: Option<(f32, u64)>,
    ) -> Result<(Vec<f32>, Option<Vec<i32>>)> {
        let pd = dims.into();
        let views: Vec<LayerWeightsView<'_>> = layers
            .iter()
            .map(|w| LayerWeightsView {
                attn_norm: w.attn_norm,
                wq: w.wq,
                wk: w.wk,
                wv: w.wv,
                wo: w.wo,
                mlp_norm: w.mlp_norm,
                w_gate: w.w_gate,
                w_up: w.w_up,
                w_down: w.w_down,
            })
            .collect();
        run_forward(
            x,
            &pd,
            &views,
            caches,
            final_norm,
            lm_head,
            self.rope_theta,
            sample_temp_seed,
        )
    }

    /// Primitive GEMM `X[m,k] @ W[n,k]^T`.
    pub fn gemm(&self, x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        match &self.kind {
            BackendKind::Cpu(b) => {
                let y = b.matmul_nt(&b.upload(x)?, &b.upload(w)?, m, k, n)?;
                b.download(&y)
            }
            BackendKind::Parallel(b) => {
                let y = b.matmul_nt(&b.upload(x)?, &b.upload(w)?, m, k, n)?;
                b.download(&y)
            }
            BackendKind::Gpu(b) => {
                let y = b.matmul_nt(&b.upload(x)?, &b.upload(w)?, m, k, n)?;
                b.download(&y)
            }
        }
    }

    /// Fused top-p/top-k + sample one row of logits.
    pub fn sample(
        &self,
        logits: &mut [f32],
        vocab: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<i32> {
        let ids = reference::fused_sample(logits, &[1.0], &[-1], 1, vocab, temperature, seed)?;
        Ok(ids[0])
    }
}

// Public types needed to drive the pipeline without exposing internals.

#[cfg(test)]
mod seek_tests {
    use super::*;

    #[test]
    fn seek_kernels_forward_smoke() {
        let k = SeekKernels::parallel();
        assert!(!k.name().is_empty());
        let dims = ModelDims {
            tokens: 2,
            start_pos: 0,
            hidden: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate: 16,
            vocab: 12,
        };
        let d = dims.hidden;
        let inter = dims.intermediate;
        let an = vec![1.0f32; d];
        let mn = vec![1.0f32; d];
        let small = |n: usize, s: f32| (0..n).map(|i| ((i as f32) * s).sin() * 0.05).collect::<Vec<_>>();
        let wq = small(dims.num_heads * dims.head_dim * d, 0.01);
        let wk = small(dims.num_kv_heads * dims.head_dim * d, 0.02);
        let wv = small(dims.num_kv_heads * dims.head_dim * d, 0.03);
        let wo = small(d * dims.num_heads * dims.head_dim, 0.01);
        let wg = small(inter * d, 0.02);
        let wu = small(inter * d, 0.03);
        let wd = small(d * inter, 0.01);
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
        let final_n = vec![1.0f32; d];
        let lm = small(dims.vocab * d, 0.001);
        let mut caches = vec![LayerKvCache::new()];
        let x = vec![0.1f32; dims.tokens * d];
        let (logits, ids) = k
            .forward(x, dims, &[layer], &mut caches, &final_n, &lm, Some((0.0, 1)))
            .unwrap();
        assert_eq!(logits.len(), dims.vocab);
        assert!(ids.is_some());
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
