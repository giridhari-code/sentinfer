//! GPU backend: real OpenCL device kernels when a platform is present, else parallel CPU.
//!
//! OpenCL C sources live in [`crate::gpu_opencl`] (`matmul_nt`, `rms_norm`, `silu_inplace`,
//! `softmax_rows`, `rope_inplace`). They execute on GPU/accelerator ICDs discovered at runtime
//! via `libOpenCL.so.1` (no compile-time OpenCL SDK required).

use std::sync::OnceLock;

use crate::backend::{Backend, Element, RmsNormOutput};
use crate::error::Result;
use crate::gpu_opencl::{try_default_device, GpuDevice};
use crate::parallel_cpu::ParallelCpuBackend;

static DEVICE: OnceLock<Option<GpuDevice>> = OnceLock::new();

fn device() -> Option<&'static GpuDevice> {
    DEVICE.get_or_init(try_default_device).as_ref()
}

/// Device-backed backend. Prefer this for inference hot paths when a GPU is available.
#[derive(Debug, Default, Clone, Copy)]
pub struct GpuBackend;

impl GpuBackend {
    pub fn new() -> Self {
        Self
    }

    /// `true` when an OpenCL device was opened and kernels were built successfully.
    pub fn is_device_accelerated(&self) -> bool {
        device().is_some()
    }

    pub fn device_name(&self) -> Option<&'static str> {
        device().map(|d| d.device_name())
    }
}

#[allow(clippy::ptr_arg)]
impl Backend for GpuBackend {
    type Buffer<T: Element> = Vec<T>;

    fn name(&self) -> &'static str {
        if device().is_some() {
            "opencl-gpu"
        } else {
            "gpu-fallback-cpu"
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
            // residual path stays on CPU for now (fused add not in GPU kernel yet)
            return ParallelCpuBackend.rms_norm(x, weight, residual, rows, hidden, eps);
        }
        if let Some(dev) = device() {
            let out = dev.rms_norm(x, weight, rows, hidden, eps)?;
            return Ok(RmsNormOutput { out, residual: None });
        }
        ParallelCpuBackend.rms_norm(x, weight, residual, rows, hidden, eps)
    }

    fn matmul_nt(&self, x: &Vec<f32>, w: &Vec<f32>, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if let Some(dev) = device() {
            return dev.matmul_nt(x, w, m, k, n);
        }
        ParallelCpuBackend.matmul_nt(x, w, m, k, n)
    }

    fn silu_inplace(&self, x: &mut Vec<f32>) -> Result<()> {
        if let Some(dev) = device() {
            return dev.silu_inplace(x);
        }
        ParallelCpuBackend.silu_inplace(x)
    }

    fn softmax_rows_inplace(&self, v: &mut Vec<f32>, rows: usize, cols: usize) -> Result<()> {
        if let Some(dev) = device() {
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
        if let Some(dev) = device() {
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
        if let Some(dev) = device() {
            return dev.attention_causal(q, k_cache, v_cache, tokens, start_pos, num_heads, num_kv_heads, head_dim);
        }
        ParallelCpuBackend.attention_causal(q, k_cache, v_cache, tokens, start_pos, num_heads, num_kv_heads, head_dim)
    }

    fn apply_top_p_top_k(
        &self,
        logits: &mut Vec<f32>,
        top_p: &Vec<f32>,
        top_k: &Vec<i32>,
        rows: usize,
        vocab: usize,
    ) -> Result<()> {
        ParallelCpuBackend.apply_top_p_top_k(logits, top_p, top_k, rows, vocab)
    }
    fn compute_seqlens_k_zos(
        &self,
        seqlens_k: &mut Vec<i32>,
        seqlens_q: &Vec<i32>,
        remap: &Vec<i32>,
        prev_num_accepted_tokens: &Vec<i32>,
        nextn: i32,
    ) -> Result<()> {
        ParallelCpuBackend.compute_seqlens_k_zos(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens, nextn)
    }
    fn kv_index_prefill(&self, offset_len: &Vec<i32>, n: usize, length: usize) -> Result<Vec<i32>> {
        ParallelCpuBackend.kv_index_prefill(offset_len, n, length)
    }
    fn kv_index_decode(
        &self,
        offset_len: &Vec<i32>,
        n: usize,
        length: usize,
        token_per_block: usize,
        block_tables: &Vec<i32>,
    ) -> Result<Vec<i32>> {
        ParallelCpuBackend.kv_index_decode(offset_len, n, length, token_per_block, block_tables)
    }
    fn gather_rows(&self, src: &Vec<f32>, indices: &Vec<i32>, cols: usize) -> Result<Vec<f32>> {
        ParallelCpuBackend.gather_rows(src, indices, cols)
    }
    fn gather_i32(&self, src: &Vec<i32>, indices: &Vec<i32>) -> Result<Vec<i32>> {
        ParallelCpuBackend.gather_i32(src, indices)
    }
    fn kv_index_cat(&self, a: &Vec<i32>, b: &Vec<i32>, n: usize, l0: usize, l1: usize) -> Result<Vec<i32>> {
        ParallelCpuBackend.kv_index_cat(a, b, n, l0, l1)
    }
    fn load_paged_kv_cache(
        &self,
        paged: &Vec<f32>,
        block_tables: &Vec<i32>,
        num_seqs: usize,
        max_blocks: usize,
        block_size: usize,
        stride: usize,
        context_lens: &Vec<i32>,
    ) -> Result<Vec<f32>> {
        ParallelCpuBackend.load_paged_kv_cache(paged, block_tables, num_seqs, max_blocks, block_size, stride, context_lens)
    }
    fn sample_from_logits(
        &self,
        logits: &Vec<f32>,
        rows: usize,
        vocab: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<i32>> {
        ParallelCpuBackend.sample_from_logits(logits, rows, vocab, temperature, seed)
    }
    fn fused_sample(
        &self,
        logits: &mut Vec<f32>,
        top_p: &Vec<f32>,
        top_k: &Vec<i32>,
        rows: usize,
        vocab: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<i32>> {
        ParallelCpuBackend.fused_sample(logits, top_p, top_k, rows, vocab, temperature, seed)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformance::check_against_reference;
    use crate::reference;

    #[test]
    fn gpu_backend_reports_status() {
        let g = GpuBackend::new();
        eprintln!(
            "GpuBackend name={} accelerated={:?} device={:?}",
            g.name(),
            g.is_device_accelerated(),
            g.device_name()
        );
        // Always usable (CPU fallback).
        assert!(g.name() == "opencl-gpu" || g.name() == "gpu-fallback-cpu");
    }

    #[test]
    fn gpu_or_fallback_matches_reference_matmul() {
        let g = GpuBackend::new();
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.05).collect();
        let w: Vec<f32> = (0..64).map(|i| (i as f32 * 0.03).sin()).collect();
        let a = reference::matmul_nt(&x, &w, 4, 8, 8).unwrap();
        let b = g.matmul_nt(&x, &w, 4, 8, 8).unwrap();
        for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
            assert!((u - v).abs() < 1e-3, "idx {i}: {u} vs {v}");
        }
    }

    #[test]
    fn gpu_backend_conformance_loose() {
        // Parallel CPU path always; OpenCL if present — allow 1e-3 for GPU fp32.
        check_against_reference(&GpuBackend, 11, 1e-3).unwrap();
    }
}
