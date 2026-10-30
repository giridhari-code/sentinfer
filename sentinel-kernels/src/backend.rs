//! The kernel backend interface for Seek-V hot paths.
//!
//! A backend owns its buffer type so a GPU backend can keep data on-device between ops;
//! only `upload` / `download` cross the host boundary.
//!
//! Phase 3a adds the math ops Seek-V needs: matmul, silu, softmax, rope — alongside the
//! earlier scheduler helpers (rms_norm, top-p/k, kv index).

use crate::error::Result;

/// Element types a backend can hold.
pub trait Element: Copy + Send + Sync + PartialEq + std::fmt::Debug + 'static {}
impl Element for f32 {}
impl Element for i32 {}

/// Output of [`Backend::rms_norm`].
#[derive(Debug, Clone, PartialEq)]
pub struct RmsNormOutput<B> {
    pub out: B,
    /// `x + residual`, returned only when a residual was passed in.
    pub residual: Option<B>,
}

pub trait Backend {
    type Buffer<T: Element>;

    fn name(&self) -> &'static str;

    fn upload<T: Element>(&self, data: &[T]) -> Result<Self::Buffer<T>>;
    fn download<T: Element>(&self, buf: &Self::Buffer<T>) -> Result<Vec<T>>;

    /// RMSNorm over the last dimension: `w * x / sqrt(mean(x^2) + eps)`, optional fused residual.
    fn rms_norm(
        &self,
        x: &Self::Buffer<f32>,
        weight: &Self::Buffer<f32>,
        residual: Option<&Self::Buffer<f32>>,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<RmsNormOutput<Self::Buffer<f32>>>;

    /// `out[m, n] = x[m, k] @ w[n, k]^T` (weights `[out, in]`).
    fn matmul_nt(
        &self,
        x: &Self::Buffer<f32>,
        w: &Self::Buffer<f32>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self::Buffer<f32>>;

    /// SiLU / swish in place: `x / (1 + exp(-x))`.
    fn silu_inplace(&self, x: &mut Self::Buffer<f32>) -> Result<()>;

    /// Softmax over the last dim of `[rows, cols]`, in place.
    fn softmax_rows_inplace(&self, v: &mut Self::Buffer<f32>, rows: usize, cols: usize) -> Result<()>;

    /// Rotary embedding on `[tokens, heads, head_dim]`.
    fn rope_inplace(
        &self,
        x: &mut Self::Buffer<f32>,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        start_pos: usize,
        theta: f32,
    ) -> Result<()>;


    /// Causal attention; see [`crate::reference::attention_causal`].
    fn attention_causal(
        &self,
        q: &Self::Buffer<f32>,
        k_cache: &Self::Buffer<f32>,
        v_cache: &Self::Buffer<f32>,
        tokens: usize,
        start_pos: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self::Buffer<f32>>;

    /// Stable top-p then top-k; masks logits `[rows, vocab]` with `-inf`.
    fn apply_top_p_top_k(
        &self,
        logits: &mut Self::Buffer<f32>,
        top_p: &Self::Buffer<f32>,
        top_k: &Self::Buffer<i32>,
        rows: usize,
        vocab: usize,
    ) -> Result<()>;

    fn compute_seqlens_k_zos(
        &self,
        seqlens_k: &mut Self::Buffer<i32>,
        seqlens_q: &Self::Buffer<i32>,
        remap: &Self::Buffer<i32>,
        prev_num_accepted_tokens: &Self::Buffer<i32>,
        nextn: i32,
    ) -> Result<()>;

    fn kv_index_prefill(&self, offset_len: &Self::Buffer<i32>, n: usize, length: usize) -> Result<Self::Buffer<i32>>;

    fn kv_index_decode(
        &self,
        offset_len: &Self::Buffer<i32>,
        n: usize,
        length: usize,
        token_per_block: usize,
        block_tables: &Self::Buffer<i32>,
    ) -> Result<Self::Buffer<i32>>;


    fn gather_rows(&self, src: &Self::Buffer<f32>, indices: &Self::Buffer<i32>, cols: usize) -> Result<Self::Buffer<f32>>;
    fn gather_i32(&self, src: &Self::Buffer<i32>, indices: &Self::Buffer<i32>) -> Result<Self::Buffer<i32>>;
    fn kv_index_cat(
        &self,
        a: &Self::Buffer<i32>,
        b: &Self::Buffer<i32>,
        n: usize,
        l0: usize,
        l1: usize,
    ) -> Result<Self::Buffer<i32>>;
    fn load_paged_kv_cache(
        &self,
        paged: &Self::Buffer<f32>,
        block_tables: &Self::Buffer<i32>,
        num_seqs: usize,
        max_blocks: usize,
        block_size: usize,
        stride: usize,
        context_lens: &Self::Buffer<i32>,
    ) -> Result<Self::Buffer<f32>>;
    fn sample_from_logits(
        &self,
        logits: &Self::Buffer<f32>,
        rows: usize,
        vocab: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Self::Buffer<i32>>;
    fn fused_sample(
        &self,
        logits: &mut Self::Buffer<f32>,
        top_p: &Self::Buffer<f32>,
        top_k: &Self::Buffer<i32>,
        rows: usize,
        vocab: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Self::Buffer<i32>>;
}
