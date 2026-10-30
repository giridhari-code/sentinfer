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
}

impl SeekKernels {
    pub fn cpu() -> Self {
        Self { kind: Kind::Cpu(CpuBackend) }
    }

    pub fn parallel() -> Self {
        Self { kind: Kind::Parallel(ParallelCpuBackend) }
    }

    /// Prefer CUDA, then OpenCL, else parallel CPU.
    pub fn gpu() -> Self {
        Self { kind: Kind::Gpu(GpuBackend::new()) }
    }

    pub fn name(&self) -> &'static str {
        match &self.kind {
            Kind::Cpu(b) => b.name(),
            Kind::Parallel(b) => b.name(),
            Kind::Gpu(b) => b.name(),
        }
    }
}
