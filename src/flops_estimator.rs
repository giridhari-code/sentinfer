//! Scheduling-time FLOPs estimate for Seek-V (standard decoder-only transformer).
//!
//! Returns `(attention_flops, total_flops)` for a scheduled `(q, k)` step so the
//! scheduler can fill `attn_tflops` / `total_tflops` on its outputs.
//!
//! Coefficients use the same FP8-equivalent weights as common MFU accounting:
//! multiply-add counts as 2 FLOPs in BF16 terms (weight 4 relative to FP8).

/// Multiply-add weight relative to FP8-equivalent FLOPs.
pub mod coef {
    pub const FP8: i64 = 2;
    pub const BF16: i64 = 4;
}

/// Architecture numbers for Seek-V / LLaMA-style decoder (GQA + SwiGLU MLP).
#[derive(Debug, Clone)]
pub struct DecoderFlopsConfig {
    pub n_layer: usize,
    pub n_embed: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    /// Sliding-window size; 0 = full causal attention.
    pub window_size: usize,
    /// Use FP8 coefficients for linear projections when true.
    pub fp8_proj: bool,
}

impl Default for DecoderFlopsConfig {
    fn default() -> Self {
        Self {
            n_layer: 12,
            n_embed: 768,
            n_head: 12,
            n_kv_head: 12,
            head_dim: 64,
            intermediate: 2048,
            vocab_size: 32000,
            window_size: 0,
            fp8_proj: false,
        }
    }
}

impl DecoderFlopsConfig {
    /// Match a `SeekVConfig` from `sentinel-model` (call sites pass the fields explicitly
    /// to avoid a crate dependency cycle).
    pub fn from_seekv(
        n_layer: usize,
        n_embed: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        intermediate: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            n_layer,
            n_embed,
            n_head,
            n_kv_head,
            head_dim,
            intermediate,
            vocab_size,
            window_size: 0,
            fp8_proj: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlopsEstimator {
    cfg: DecoderFlopsConfig,
}

impl FlopsEstimator {
    pub fn new(cfg: DecoderFlopsConfig) -> Self {
        Self { cfg }
    }

    /// Convenience alias used by older call sites.
    pub fn decoder(cfg: DecoderFlopsConfig) -> Self {
        Self::new(cfg)
    }

    /// Returns `(attention_flops, total_flops)` for one sequence's scheduled step.
    ///
    /// * `q` — query tokens this step (`seqlen_q`)
    /// * `k` — key length after this step (`seqlen_k`)
    /// * `is_prompt` — prefill vs decode
    pub fn calc_flops(&self, q: usize, k: usize, is_prompt: bool) -> (i64, i64) {
        calc_decoder(&self.cfg, q, k, is_prompt)
    }

    /// Same as [`calc_flops`], ignoring extra DeepSeek-style arguments for API stability
    /// with the scheduler helper.
    pub fn calc_flops_ext(
        &self,
        q: usize,
        k: usize,
        is_prompt: bool,
        _full_len: i64,
        _swa_valid_start: usize,
    ) -> (i64, i64) {
        self.calc_flops(q, k, is_prompt)
    }
}

fn calc_decoder(c: &DecoderFlopsConfig, q: usize, k: usize, is_prompt: bool) -> (i64, i64) {
    let proj = if c.fp8_proj { coef::FP8 } else { coef::BF16 };
    let h = c.n_embed as i64;
    let heads = c.n_head as i64;
    let kv = c.n_kv_head as i64;
    let d = c.head_dim as i64;
    let inter = c.intermediate as i64;
    let layers = c.n_layer as i64;
    let vocab = c.vocab_size as i64;

    // Per-token projection cost (QKV + O + SwiGLU MLP) across all layers.
    let qkv_o = proj * (h * h + 2 * h * kv * d + h * h);
    let mlp = proj * 3 * h * inter;
    let non_attn_per_tok = layers * (qkv_o + mlp);

    // Attention QK^T + AV: 2 * heads * d * pairs, in BF16.
    let attn_coef = coef::BF16 * heads * 2 * d;
    let pairs = if c.window_size > 0 {
        sliding_window_pairs(q, k, c.window_size)
    } else if is_prompt {
        let q = q as i64;
        let k = k as i64;
        q * (2 * k - q + 1) / 2
    } else {
        (q as i64) * (k as i64)
    };
    let attn = attn_coef * pairs * layers;

    // LM head: one row on prefill (last token), q rows on decode.
    let logit_rows = if is_prompt { 1i64 } else { q as i64 };
    let logits = coef::BF16 * h * vocab * logit_rows;

    let non_attn = non_attn_per_tok * (q as i64);
    (attn, attn + non_attn + logits)
}

fn sliding_window_pairs(q: usize, k: usize, window: usize) -> i64 {
    let start = k.saturating_sub(q);
    let mut total = 0i64;
    for pos in start..k {
        let visible = (pos + 1).min(window);
        total += visible as i64;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_is_positive_and_grows_with_length() {
        let est = FlopsEstimator::new(DecoderFlopsConfig::default());
        let (a1, t1) = est.calc_flops(16, 16, true);
        let (a2, t2) = est.calc_flops(32, 32, true);
        assert!(a1 > 0 && t1 >= a1);
        assert!(a2 > a1 && t2 > t1);
    }

    #[test]
    fn decode_scales_with_kv_length() {
        let est = FlopsEstimator::new(DecoderFlopsConfig::default());
        let (a_short, _) = est.calc_flops(1, 16, false);
        let (a_long, _) = est.calc_flops(1, 128, false);
        assert!(a_long > a_short);
    }

    #[test]
    fn from_seekv_matches_fields() {
        let c = DecoderFlopsConfig::from_seekv(6, 256, 4, 2, 64, 512, 258);
        assert_eq!(c.n_layer, 6);
        assert_eq!(c.n_kv_head, 2);
        assert_eq!(c.vocab_size, 258);
    }
}
