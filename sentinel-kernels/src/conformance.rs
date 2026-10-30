//! Randomized cross-check of any [`Backend`] against the slice-level reference.
//!
//! A new backend (wgpu, CUDA, ...) calls `check_against_reference(&backend, seed, rtol)` from
//! its own tests. Inputs come from a small built-in PRNG so the check is reproducible and needs
//! no dependencies. Masking patterns of top-p/top-k and all integer outputs must match exactly;
//! float outputs must match within `rtol`.

use crate::backend::Backend;
use crate::error::{KernelError, Result};
use crate::reference;

/// xorshift64* generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }

    /// Uniform integer in `lo..=hi`.
    pub fn i32_in(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi as i64 - lo as i64 + 1) as u64;
        lo + (self.next_u64() % span) as i32
    }
}

fn close(got: f32, want: f32, rtol: f32) -> bool {
    if got == want || (got.is_nan() && want.is_nan()) {
        return true; // also covers equal infinities
    }
    (got - want).abs() <= rtol * (1.0 + want.abs())
}

fn compare_f32(op: &str, got: &[f32], want: &[f32], rtol: f32) -> Result<()> {
    if got.len() != want.len() {
        return Err(KernelError::Backend(format!("{op}: got {} elements, want {}", got.len(), want.len())));
    }
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if !close(g, w, rtol) {
            return Err(KernelError::Backend(format!("{op}: mismatch at {i}: got {g}, want {w}")));
        }
    }
    Ok(())
}

fn compare_i32(op: &str, got: &[i32], want: &[i32]) -> Result<()> {
    if got.len() != want.len() {
        return Err(KernelError::Backend(format!("{op}: got {} elements, want {}", got.len(), want.len())));
    }
    match got.iter().zip(want).position(|(g, w)| g != w) {
        Some(i) => Err(KernelError::Backend(format!("{op}: mismatch at {i}: got {}, want {}", got[i], want[i]))),
        None => Ok(()),
    }
}

fn vec_f32(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.f32_in(lo, hi)).collect()
}

fn vec_i32(rng: &mut Rng, n: usize, lo: i32, hi: i32) -> Vec<i32> {
    (0..n).map(|_| rng.i32_in(lo, hi)).collect()
}

fn check_rms_norm<B: Backend>(b: &B, rng: &mut Rng, rtol: f32, with_residual: bool) -> Result<()> {
    let (rows, hidden, eps) = (7, 33, 1e-6f32);
    let x = vec_f32(rng, rows * hidden, -3.0, 3.0);
    let w = vec_f32(rng, hidden, 0.5, 1.5);
    let res = with_residual.then(|| vec_f32(rng, rows * hidden, -3.0, 3.0));

    let (want_out, want_res) = reference::rms_norm(&x, &w, res.as_deref(), rows, hidden, eps)?;

    let bx = b.upload(&x)?;
    let bw = b.upload(&w)?;
    let bres = res.as_ref().map(|r| b.upload(r)).transpose()?;
    let got = b.rms_norm(&bx, &bw, bres.as_ref(), rows, hidden, eps)?;

    compare_f32("rms_norm.out", &b.download::<f32>(&got.out)?, &want_out, rtol)?;
    match (&got.residual, &want_res) {
        (Some(g), Some(w)) => compare_f32("rms_norm.residual", &b.download::<f32>(g)?, w, rtol),
        (None, None) => Ok(()),
        _ => Err(KernelError::Backend("rms_norm: residual output presence differs from reference".into())),
    }
}

fn check_top_p_top_k<B: Backend>(b: &B, rng: &mut Rng, rtol: f32) -> Result<()> {
    let (rows, vocab) = (5, 50);
    let logits = vec_f32(rng, rows * vocab, -5.0, 5.0);
    let top_p: Vec<f32> = (0..rows).map(|i| if i % 2 == 0 { rng.f32_in(0.05, 1.0) } else { 1.0 }).collect();
    let top_k = vec_i32(rng, rows, -1, vocab as i32 + 2);

    let mut want = logits.clone();
    reference::apply_top_p_top_k(&mut want, &top_p, &top_k, rows, vocab)?;

    let mut bl = b.upload(&logits)?;
    let bp = b.upload(&top_p)?;
    let bk = b.upload(&top_k)?;
    b.apply_top_p_top_k(&mut bl, &bp, &bk, rows, vocab)?;
    compare_f32("apply_top_p_top_k", &b.download::<f32>(&bl)?, &want, rtol)
}

fn check_seqlens_k_zos<B: Backend>(b: &B, rng: &mut Rng) -> Result<()> {
    let (n, prev_n, nextn) = (64usize, 70usize, 3i32);
    let k0 = vec_i32(rng, n, 100, 200);
    let q = vec_i32(rng, n, 1, nextn);
    let remap = vec_i32(rng, n, -1, prev_n as i32 - 1);
    let prev = vec_i32(rng, prev_n, 1, nextn);

    let mut want = k0.clone();
    reference::compute_seqlens_k_zos(&mut want, &q, &remap, &prev, nextn)?;

    let mut bk = b.upload(&k0)?;
    let bq = b.upload(&q)?;
    let br = b.upload(&remap)?;
    let bp = b.upload(&prev)?;
    b.compute_seqlens_k_zos(&mut bk, &bq, &br, &bp, nextn)?;
    compare_i32("compute_seqlens_k_zos", &b.download::<i32>(&bk)?, &want)
}

fn check_kv_index<B: Backend>(b: &B, rng: &mut Rng) -> Result<()> {
    let (n, length, tpb) = (8usize, 16usize, 4usize);
    let mut offset_len = Vec::with_capacity(n * 2);
    for _ in 0..n {
        offset_len.push(rng.i32_in(0, 40));
        offset_len.push(rng.i32_in(0, length as i32));
    }
    let block_tables = vec_i32(rng, 32, 0, 100); // positions reach at most 55 -> block 13

    let want_prefill = reference::kv_index_prefill(&offset_len, n, length)?;
    let want_decode = reference::kv_index_decode(&offset_len, n, length, tpb, &block_tables)?;

    let bo = b.upload(&offset_len)?;
    let bt = b.upload(&block_tables)?;
    compare_i32("kv_index_prefill", &b.download::<i32>(&b.kv_index_prefill(&bo, n, length)?)?, &want_prefill)?;
    compare_i32("kv_index_decode", &b.download::<i32>(&b.kv_index_decode(&bo, n, length, tpb, &bt)?)?, &want_decode)
}

/// Runs every op on `candidate` with random inputs and compares against the reference.
pub fn check_against_reference<B: Backend>(candidate: &B, seed: u64, rtol: f32) -> Result<()> {
    let mut rng = Rng::new(seed);
    check_rms_norm(candidate, &mut rng, rtol, false)?;
    check_rms_norm(candidate, &mut rng, rtol, true)?;
    check_top_p_top_k(candidate, &mut rng, rtol)?;
    check_seqlens_k_zos(candidate, &mut rng)?;
    check_kv_index(candidate, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Element, RmsNormOutput};
    use crate::cpu::CpuBackend;

    /// CPU backend whose rms_norm is deliberately wrong, to prove the harness can fail.
    struct Broken(CpuBackend);

    impl Backend for Broken {
        type Buffer<T: Element> = Vec<T>;

        fn name(&self) -> &'static str {
            "broken"
        }
        fn upload<T: Element>(&self, data: &[T]) -> Result<Vec<T>> {
            self.0.upload(data)
        }
        fn download<T: Element>(&self, buf: &Vec<T>) -> Result<Vec<T>> {
            self.0.download(buf)
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
            let mut o = self.0.rms_norm(x, weight, residual, rows, hidden, eps)?;
            o.out[0] += 1.0;
            Ok(o)
        }
        fn matmul_nt(&self, x: &Vec<f32>, w: &Vec<f32>, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
            self.0.matmul_nt(x, w, m, k, n)
        }
        fn silu_inplace(&self, x: &mut Vec<f32>) -> Result<()> {
            self.0.silu_inplace(x)
        }
        fn softmax_rows_inplace(&self, v: &mut Vec<f32>, rows: usize, cols: usize) -> Result<()> {
            self.0.softmax_rows_inplace(v, rows, cols)
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
            self.0.rope_inplace(x, tokens, heads, head_dim, start_pos, theta)
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
            self.0.attention_causal(q, k_cache, v_cache, tokens, start_pos, num_heads, num_kv_heads, head_dim)
        }
        fn apply_top_p_top_k(&self, l: &mut Vec<f32>, p: &Vec<f32>, k: &Vec<i32>, r: usize, v: usize) -> Result<()> {
            self.0.apply_top_p_top_k(l, p, k, r, v)
        }
        fn compute_seqlens_k_zos(&self, k: &mut Vec<i32>, q: &Vec<i32>, m: &Vec<i32>, p: &Vec<i32>, n: i32) -> Result<()> {
            self.0.compute_seqlens_k_zos(k, q, m, p, n)
        }
        fn kv_index_prefill(&self, o: &Vec<i32>, n: usize, l: usize) -> Result<Vec<i32>> {
            self.0.kv_index_prefill(o, n, l)
        }
        fn kv_index_decode(&self, o: &Vec<i32>, n: usize, l: usize, t: usize, b: &Vec<i32>) -> Result<Vec<i32>> {
            self.0.kv_index_decode(o, n, l, t, b)
        }
        fn gather_rows(&self, src: &Vec<f32>, indices: &Vec<i32>, cols: usize) -> Result<Vec<f32>> {
            self.0.gather_rows(src, indices, cols)
        }
        fn gather_i32(&self, src: &Vec<i32>, indices: &Vec<i32>) -> Result<Vec<i32>> {
            self.0.gather_i32(src, indices)
        }
        fn kv_index_cat(&self, a: &Vec<i32>, b: &Vec<i32>, n: usize, l0: usize, l1: usize) -> Result<Vec<i32>> {
            self.0.kv_index_cat(a, b, n, l0, l1)
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
            self.0.load_paged_kv_cache(paged, block_tables, num_seqs, max_blocks, block_size, stride, context_lens)
        }
        fn sample_from_logits(
            &self,
            logits: &Vec<f32>,
            rows: usize,
            vocab: usize,
            temperature: f32,
            seed: u64,
        ) -> Result<Vec<i32>> {
            self.0.sample_from_logits(logits, rows, vocab, temperature, seed)
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
            self.0.fused_sample(logits, top_p, top_k, rows, vocab, temperature, seed)
        }

    }

    #[test]
    fn harness_detects_a_wrong_backend() {
        let err = check_against_reference(&Broken(CpuBackend), 7, 1e-5).unwrap_err();
        assert!(err.to_string().contains("rms_norm.out"), "{err}");
    }

    #[test]
    fn rng_is_reproducible_and_in_range() {
        let mut a = Rng::new(5);
        let mut b = Rng::new(5);
        for _ in 0..100 {
            let (x, y) = (a.i32_in(-3, 3), b.i32_in(-3, 3));
            assert_eq!(x, y);
            assert!((-3..=3).contains(&x));
            let f = a.f32_in(0.5, 1.5);
            b.f32_in(0.5, 1.5);
            assert!((0.5..1.5).contains(&f));
        }
    }

    #[test]
    fn close_handles_infinities() {
        assert!(close(f32::NEG_INFINITY, f32::NEG_INFINITY, 1e-5));
        assert!(!close(f32::NEG_INFINITY, 3.0, 1e-5));
        assert!(close(1.0, 1.000_001, 1e-5));
    }
}
