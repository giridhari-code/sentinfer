//! GPU backend: CUDA (preferred) or OpenCL when present, else parallel CPU.
//!
//! * CUDA: runtime-loaded `libcuda.so.1` + embedded PTX ([`crate::gpu_cuda`]).
//! * OpenCL: runtime-loaded `libOpenCL.so.1` ([`crate::gpu_opencl`]).
//! No CUDA/OpenCL SDK required at compile time.

use std::sync::OnceLock;

use crate::backend::{Backend, Element, RmsNormOutput};
use crate::error::Result;
use crate::gpu_cuda::{try_default_cuda, CudaDevice};
use crate::gpu_opencl::{try_default_device, GpuDevice};
use crate::parallel_cpu::ParallelCpuBackend;

static CUDA: OnceLock<Option<CudaDevice>> = OnceLock::new();
static OPENCL: OnceLock<Option<GpuDevice>> = OnceLock::new();

fn cuda() -> Option<&'static CudaDevice> {
    CUDA.get_or_init(try_default_cuda).as_ref()
}

fn opencl() -> Option<&'static GpuDevice> {
    OPENCL.get_or_init(try_default_device).as_ref()
}

/// Device-backed backend. Prefer this for inference hot paths when a GPU is available.
#[derive(Debug, Default, Clone, Copy)]
pub struct GpuBackend;

impl GpuBackend {
    pub fn new() -> Self {
        Self
    }

    /// `true` when CUDA or OpenCL was opened successfully.
    pub fn is_device_accelerated(&self) -> bool {
        cuda().is_some() || opencl().is_some()
    }

    pub fn device_name(&self) -> Option<&'static str> {
        cuda()
            .map(|d| d.device_name())
            .or_else(|| opencl().map(|d| d.device_name()))
    }

    /// `"cuda"` | `"opencl"` | `None` if CPU fallback only.
    pub fn accel_kind(&self) -> Option<&'static str> {
        if cuda().is_some() {
            Some("cuda")
        } else if opencl().is_some() {
            Some("opencl")
        } else {
            None
        }
    }
}

#[allow(clippy::ptr_arg)]
impl Backend for GpuBackend {
    type Buffer<T: Element> = Vec<T>;

    fn name(&self) -> &'static str {
        match self.accel_kind() {
            Some("cuda") => "cuda-gpu",
            Some("opencl") => "opencl-gpu",
            _ => "gpu-fallback-cpu",
        }
    }

    fn upload<T: Element>(&self, data: &[T]) -> Result<Vec<T>> {
        Ok(data.to_vec())
    }
    fn download<T: Element>(&self, buf: &Vec<T>) -> Result<Vec<T>> {
        Ok(buf.clone())
    }

    fn rms_norm(
        &self,
        x: &Vec<f32>,
        weight: &Vec<f32>,
        residual: Option<&Vec<f32>>,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<RmsNormOutput<Vec<f32>>> {
        if residual.is_some() {
            return ParallelCpuBackend.rms_norm(x, weight, residual, rows, hidden, eps);
        }
        if let Some(dev) = opencl() {
            let out = dev.rms_norm(x, weight, rows, hidden, eps)?;
            return Ok(RmsNormOutput { out, residual: None });
        }
        ParallelCpuBackend.rms_norm(x, weight, residual, rows, hidden, eps)
    }

    fn matmul_nt(&self, x: &Vec<f32>, w: &Vec<f32>, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if let Some(dev) = cuda() {
            return dev.matmul_nt(x, w, m, k, n);
        }
        if let Some(dev) = opencl() {
            return dev.matmul_nt(x, w, m, k, n);
        }
        ParallelCpuBackend.matmul_nt(x, w, m, k, n)
    }

    fn silu_inplace(&self, x: &mut Vec<f32>) -> Result<()> {
        if let Some(dev) = cuda() {
            return dev.silu_inplace(x);
        }
        if let Some(dev) = opencl() {
            return dev.silu_inplace(x);
        }
        ParallelCpuBackend.silu_inplace(x)
    }

    fn softmax_rows_inplace(&self, v: &mut Vec<f32>, rows: usize, cols: usize) -> Result<()> {
        if let Some(dev) = opencl() {
            return dev.softmax_rows_inplace(v, rows, cols);
        }
        ParallelCpuBackend.softmax_rows_inplace(v, rows, cols)
    }

    fn rope_inplace(
        &self,
        x: &mut Vec<f32>,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        start_pos: usize,
        theta: f32,
    ) -> Result<()> {
        if let Some(dev) = opencl() {
            return dev.rope_inplace(x, tokens, heads, head_dim, start_pos, theta);
        }
        ParallelCpuBackend.rope_inplace(x, tokens, heads, head_dim, start_pos, theta)
    }

    fn attention_causal(
        &self,
        q: &Vec<f32>,
        k_cache: &Vec<f32>,
        v_cache: &Vec<f32>,
        tokens: usize,
        start_pos: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        if let Some(dev) = opencl() {
            return dev.attention_causal(
                q, k_cache, v_cache, tokens, start_pos, num_heads, num_kv_heads, head_dim,
            );
        }
        ParallelCpuBackend.attention_causal(
            q, k_cache, v_cache, tokens, start_pos, num_heads, num_kv_heads, head_dim,
        )
    }
}
