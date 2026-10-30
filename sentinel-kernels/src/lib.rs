//! Sentinel kernel layer.
//!
//! | Backend | When to use |
//! | --- | --- |
//! | [`CpuBackend`] | correctness oracle |
//! | [`ParallelCpuBackend`] | multi-core CPU |
//! | [`GpuBackend`] | OpenCL GPU if present, else parallel CPU |

pub mod backend;
pub mod conformance;
pub mod cpu;
pub mod error;
pub mod gpu;
pub mod gpu_opencl;
pub mod opencl_ffi;
pub mod parallel_cpu;
pub mod quant;
pub mod moe;
pub mod deep_gemm;
pub mod flash_mla;
pub mod pipeline;
pub mod cuda_vendor;
pub mod reference;

pub use backend::{Backend, Element, RmsNormOutput};
pub use cpu::CpuBackend;
pub use error::{KernelError, Result};
pub use gpu::GpuBackend;
pub use parallel_cpu::{set_parallel_threads, ParallelCpuBackend};
pub use quant::{QuantMatrix, QuantScheme};

pub use flash_mla::{flash_mla_ref, rope_partial_inplace, MlaDims};
pub use deep_gemm::{deep_gemm_nt, deep_gemm_nt_tiled};
pub use moe::{moe_forward, moe_router_topk, ExpertWeights, MoeWeights};

pub use pipeline::{kernel_inventory, run_forward, run_layer, run_layer_backend, LayerKvCache, LayerWeightsView, PipelineDims, PipelineStage};

pub use cuda_vendor::{
    vendor_backend_name, vendor_deep_gemm_nt, vendor_flash_mla, vendor_moe_forward, vendor_so_loaded,
};
