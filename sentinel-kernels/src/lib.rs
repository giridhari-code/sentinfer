//! # SeekKernels
//!
//! Unified kernel surface for Seek-V.
//!
//! * [`SeekKernels::auto`] — CUDA → OpenCL → parallel CPU
//! * [`SeekKernels::layer_with_rope`] — one decoder block on the selected backend
//! * [`SeekKernels::gemm`] — matmul for lm_head / projections

pub mod backend;
pub(crate) mod conformance;
pub(crate) mod cpu;
pub(crate) mod cuda_ffi;
pub mod error;
pub mod gpu;
pub(crate) mod gpu_cuda;
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
use pipeline::{run_layer_backend, LayerWeightsView, PipelineDims};

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

pub struct SeekKernels {
    kind: Kind,
}

impl Default for SeekKernels {
    fn default() -> Self {
        Self::auto()
    }
}

impl SeekKernels {
    pub fn cpu() -> Self {
        Self { kind: Kind::Cpu(CpuBackend) }
    }

    pub fn parallel() -> Self {
        Self { kind: Kind::Parallel(ParallelCpuBackend) }
    }

    pub fn gpu() -> Self {
        Self { kind: Kind::Gpu(GpuBackend::new()) }
    }

    /// CUDA → OpenCL → parallel CPU.
    pub fn auto() -> Self {
        let g = GpuBackend::new();
        if g.is_device_accelerated() {
            Self { kind: Kind::Gpu(g) }
        } else {
            Self::parallel()
        }
    }

    pub fn with_rope_theta(self, _theta: f32) -> Self {
        self
    }
    pub fn with_rms_eps(self, _eps: f32) -> Self {
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

    pub fn accel_kind(&self) -> Option<&'static str> {
        match &self.kind {
            Kind::Gpu(b) => b.accel_kind(),
            _ => None,
        }
    }

    /// One decoder layer on the active backend (GPU when available).
    pub fn layer(
        &self,
        x: &mut [f32],
        dims: ModelDims,
        weights: &LayerWeights<'_>,
        kv: &mut LayerKvCache,
    ) -> Result<()> {
        self.layer_with_rope(x, dims, weights, kv, 10000.0)
    }

    pub fn layer_with_rope(
        &self,
        x: &mut [f32],
        dims: ModelDims,
        weights: &LayerWeights<'_>,
        kv: &mut LayerKvCache,
        rope_theta: f32,
    ) -> Result<()> {
        let pd: PipelineDims = dims.into();
        let view = weights.view();
        match &self.kind {
            Kind::Cpu(b) => run_layer_backend(b, x, &pd, &view, kv, rope_theta),
            Kind::Parallel(b) => run_layer_backend(b, x, &pd, &view, kv, rope_theta),
            Kind::Gpu(b) => run_layer_backend(b, x, &pd, &view, kv, rope_theta),
        }
    }

    /// `out[m,n] = x[m,k] @ W[n,k]^T`
    pub fn gemm(&self, x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        match &self.kind {
            Kind::Cpu(b) => b.matmul_nt(&b.upload(x)?, &b.upload(w)?, m, k, n),
            Kind::Parallel(b) => b.matmul_nt(&b.upload(x)?, &b.upload(w)?, m, k, n),
            Kind::Gpu(b) => b.matmul_nt(&b.upload(x)?, &b.upload(w)?, m, k, n),
        }
    }
}
