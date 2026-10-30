//! CPU reference backend: host `Vec`s as buffers, ops delegate to `reference`.

use crate::backend::{Backend, Element, RmsNormOutput};
use crate::error::Result;
use crate::reference;

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

#[allow(clippy::ptr_arg)]
impl Backend for CpuBackend {
    type Buffer<T: Element> = Vec<T>;

    fn name(&self) -> &'static str {
        "cpu-reference"
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
        let (out, residual) = reference::rms_norm(x, weight, residual.map(|r| r.as_slice()), rows, hidden, eps)?;
        Ok(RmsNormOutput { out, residual })
    }

    fn matmul_nt(&self, x: &Vec<f32>, w: &Vec<f32>, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        reference::matmul_nt(x, w, m, k, n)
    }

    fn silu_inplace(&self, x: &mut Vec<f32>) -> Result<()> {
        reference::silu_inplace(x);
        Ok(())
    }

    fn softmax_rows_inplace(&self, v: &mut Vec<f32>, rows: usize, cols: usize) -> Result<()> {
        reference::softmax_rows_inplace(v, rows, cols)
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
        reference::rope_inplace(x, tokens, heads, head_dim, start_pos, theta)
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
        reference::apply_top_p_top_k(logits, top_p, top_k, rows, vocab)
    }

    fn compute_seqlens_k_zos(
        &self,
        seqlens_k: &mut Vec<i32>,
        seqlens_q: &Vec<i32>,
        remap: &Vec<i32>,
        prev_num_accepted_tokens: &Vec<i32>,
        nextn: i32,
    ) -> Result<()> {
        reference::compute_seqlens_k_zos(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens, nextn)
    }

    fn kv_index_prefill(&self, offset_len: &Vec<i32>, n: usize, length: usize) -> Result<Vec<i32>> {
        reference::kv_index_prefill(offset_len, n, length)
    }

    fn kv_index_decode(
        &self,
        offset_len: &Vec<i32>,
        n: usize,
        length: usize,
        token_per_block: usize,
        block_tables: &Vec<i32>,
    ) -> Result<Vec<i32>> {
        reference::kv_index_decode(offset_len, n, length, token_per_block, block_tables)
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
    fn upload_download_roundtrip() {
        let b = CpuBackend;
        let buf = b.upload(&[1i32, 2, 3]).unwrap();
        assert_eq!(b.download(&buf).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn rms_norm_through_the_trait() {
        let b = CpuBackend;
        let x = b.upload(&[3.0f32, 4.0]).unwrap();
        let w = b.upload(&[1.0f32, 1.0]).unwrap();
        let out = b.rms_norm(&x, &w, None, 1, 2, 0.0).unwrap();
        let v = b.download(&out.out).unwrap();
        assert!((v[0] - 0.848_528_1).abs() < 1e-5);
        assert!(out.residual.is_none());
    }

    #[test]
    fn matmul_through_the_trait() {
        let b = CpuBackend;
        let x = b.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let w = b.upload(&[1.0f32, 0.0, 0.0, 1.0]).unwrap();
        let out = b.matmul_nt(&x, &w, 2, 2, 2).unwrap();
        assert_eq!(b.download(&out).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn passes_its_own_conformance_suite() {
        for seed in [1, 2, 3, 42, 1234] {
            check_against_reference(&CpuBackend, seed, 1e-5).unwrap();
        }
    }
}
