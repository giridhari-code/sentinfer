//! Multi-threaded CPU backend: same numerics as [`CpuBackend`], matmul parallelized over rows.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::backend::{Backend, Element, RmsNormOutput};
use crate::cpu::CpuBackend;
use crate::error::Result;
use crate::reference;

static DEFAULT_THREADS: AtomicUsize = AtomicUsize::new(0);

fn worker_threads() -> usize {
    let n = DEFAULT_THREADS.load(Ordering::Relaxed);
    if n > 0 {
        n
    } else {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
    }
}

/// Set the thread count used by [`ParallelCpuBackend`] matmul (0 = auto).
pub fn set_parallel_threads(n: usize) {
    DEFAULT_THREADS.store(n, Ordering::Relaxed);
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ParallelCpuBackend;

#[allow(clippy::ptr_arg)]
impl Backend for ParallelCpuBackend {
    type Buffer<T: Element> = Vec<T>;

    fn name(&self) -> &'static str {
        "cpu-parallel"
    }

    fn upload<T: Element>(&self, data: &[T]) -> Result<Vec<T>> {
        CpuBackend.upload(data)
    }
    fn download<T: Element>(&self, buf: &Vec<T>) -> Result<Vec<T>> {
        CpuBackend.download(buf)
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
        CpuBackend.rms_norm(x, weight, residual, rows, hidden, eps)
    }
    fn matmul_nt(&self, x: &Vec<f32>, w: &Vec<f32>, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        reference::matmul_nt_parallel(x, w, m, k, n, worker_threads())
    }
    fn silu_inplace(&self, x: &mut Vec<f32>) -> Result<()> {
        CpuBackend.silu_inplace(x)
    }
    fn softmax_rows_inplace(&self, v: &mut Vec<f32>, rows: usize, cols: usize) -> Result<()> {
        CpuBackend.softmax_rows_inplace(v, rows, cols)
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
        CpuBackend.rope_inplace(x, tokens, heads, head_dim, start_pos, theta)
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
        reference::attention_causal(q, k_cache, v_cache, tokens, start_pos, num_heads, num_kv_heads, head_dim)
    }

    fn apply_top_p_top_k(
        &self,
        logits: &mut Vec<f32>,
        top_p: &Vec<f32>,
        top_k: &Vec<i32>,
        rows: usize,
        vocab: usize,
    ) -> Result<()> {
        CpuBackend.apply_top_p_top_k(logits, top_p, top_k, rows, vocab)
    }
    fn compute_seqlens_k_zos(
        &self,
        seqlens_k: &mut Vec<i32>,
        seqlens_q: &Vec<i32>,
        remap: &Vec<i32>,
        prev_num_accepted_tokens: &Vec<i32>,
        nextn: i32,
    ) -> Result<()> {
        CpuBackend.compute_seqlens_k_zos(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens, nextn)
    }
    fn kv_index_prefill(&self, offset_len: &Vec<i32>, n: usize, length: usize) -> Result<Vec<i32>> {
        CpuBackend.kv_index_prefill(offset_len, n, length)
    }
    fn kv_index_decode(
        &self,
        offset_len: &Vec<i32>,
        n: usize,
        length: usize,
        token_per_block: usize,
        block_tables: &Vec<i32>,
    ) -> Result<Vec<i32>> {
        CpuBackend.kv_index_decode(offset_len, n, length, token_per_block, block_tables)
    }

    fn gather_rows(&self, src: &Vec<f32>, indices: &Vec<i32>, cols: usize) -> Result<Vec<f32>> {
        reference::gather_rows(src, indices, cols)
    }
    fn gather_i32(&self, src: &Vec<i32>, indices: &Vec<i32>) -> Result<Vec<i32>> {
        reference::gather_i32(src, indices)
    }
    fn kv_index_cat(&self, a: &Vec<i32>, b: &Vec<i32>, n: usize, l0: usize, l1: usize) -> Result<Vec<i32>> {
        reference::kv_index_cat(a, b, n, l0, l1)
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
        reference::load_paged_kv_cache(paged, block_tables, num_seqs, max_blocks, block_size, stride, context_lens)
    }
    fn sample_from_logits(
        &self,
        logits: &Vec<f32>,
        rows: usize,
        vocab: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<i32>> {
        reference::sample_from_logits(logits, rows, vocab, temperature, seed)
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
        reference::fused_sample(logits, top_p, top_k, rows, vocab, temperature, seed)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformance::check_against_reference;

    #[test]
    fn matches_reference_conformance() {
        set_parallel_threads(2);
        check_against_reference(&ParallelCpuBackend, 9, 1e-4).unwrap();
    }

    #[test]
    fn matmul_matches_serial() {
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let w: Vec<f32> = (0..128).map(|i| (i as f32 * 0.02).sin()).collect();
        // m=8 k=8 n=16
        let a = reference::matmul_nt(&x, &w, 8, 8, 16).unwrap();
        let b = reference::matmul_nt_parallel(&x, &w, 8, 8, 16, 4).unwrap();
        for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
            assert!((u - v).abs() < 1e-4, "mismatch at {i}: {u} vs {v}");
        }
    }
}
