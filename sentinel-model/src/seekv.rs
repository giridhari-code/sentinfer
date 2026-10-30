//! Seek-V: decoder-only transformer (RMSNorm, RoPE, GQA, dense or MoE SwiGLU).
//!
//! Dense MLP by default. Set `num_experts > 0` for DeepInfer-style sparse FFN
//! (router + top-k experts, DeepGEMM-style packed expert GEMMs).

use std::collections::HashMap;

use sentinel_core::DecoderFlopsConfig;
use sentinel_core::{Result, SeqId, SentinelError};
use sentinel_kernels::{LayerKvCache, LayerWeights as KernLayerWeights, ModelDims, SeekKernels};
use sentinel_kernels::reference as kern;

use crate::model::{Model, StepInput};
use crate::ops::{kerr, SplitMix};

#[derive(Debug, Clone)]
pub struct SeekVConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    /// 0 = dense MLP. >0 enables MoE FFN.
    pub num_experts: usize,
    /// Top-k experts per token (when MoE on).
    pub num_experts_per_tok: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub max_positions: usize,
    pub eos_token_id: i32,
}

impl SeekVConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 258,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            num_experts: 0,
            num_experts_per_tok: 1,
            rope_theta: 10000.0,
            rms_eps: 1e-6,
            max_positions: 512,
            eos_token_id: 257,
        }
    }

    pub fn small() -> Self {
        Self {
            vocab_size: 258,
            hidden_size: 256,
            num_layers: 6,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 32,
            intermediate_size: 704,
            num_experts: 0,
            num_experts_per_tok: 1,
            rope_theta: 10000.0,
            rms_eps: 1e-6,
            max_positions: 512,
            eos_token_id: 257,
        }
    }

    pub fn micro(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 8,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 12,
            num_experts: 0,
            num_experts_per_tok: 1,
            rope_theta: 10000.0,
            rms_eps: 1e-6,
            max_positions: 64,
            eos_token_id: vocab_size as i32 - 1,
        }
    }

    /// Tiny MoE demo: 4 experts, top-2.
    pub fn tiny_moe() -> Self {
        let mut c = Self::tiny();
        c.num_experts = 4;
        c.num_experts_per_tok = 2;
        c
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    pub fn flops_config(&self) -> DecoderFlopsConfig {
        DecoderFlopsConfig::from_seekv(
            self.num_layers,
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.intermediate_size,
            self.vocab_size,
        )
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(SentinelError::Config(format!("SeekVConfig: {m}")));
        if [self.vocab_size, self.hidden_size, self.num_layers, self.num_heads, self.num_kv_heads, self.head_dim, self.intermediate_size, self.max_positions].contains(&0) {
            return bad("sizes must be > 0");
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return bad("num_heads must be a multiple of num_kv_heads");
        }
        if self.head_dim % 2 != 0 {
            return bad("head_dim must be even (rotary embeddings rotate pairs)");
        }
        if self.eos_token_id < 0 || self.eos_token_id as usize >= self.vocab_size {
            return bad("eos_token_id must be inside the vocabulary");
        }
        if self.num_experts > 0
            && (self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts)
        {
            return bad("num_experts_per_tok must be in 1..=num_experts when MoE is on");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SeekVWeights {
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    /// Per-layer MoE when `cfg.num_experts > 0`.
    pub moe: Option<Vec<crate::moe::MoEWeights>>,
}

impl SeekVWeights {
    pub fn random(cfg: &SeekVConfig, seed: u64) -> Self {
        let mut rng = SplitMix::new(seed);
        let mut mat = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.uniform(-0.1, 0.1)).collect()
        };
        let (d, h, hk, hd, inter) =
            (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.intermediate_size);
        let embed = mat(cfg.vocab_size, d);
        let layers = (0..cfg.num_layers)
            .map(|_| LayerWeights {
                attn_norm: vec![1.0; d],
                wq: mat(h * hd, d),
                wk: mat(hk * hd, d),
                wv: mat(hk * hd, d),
                wo: mat(d, h * hd),
                mlp_norm: vec![1.0; d],
                w_gate: mat(inter, d),
                w_up: mat(inter, d),
                w_down: mat(d, inter),
            })
            .collect();
        let lm_head = mat(cfg.vocab_size, d);
        let moe = if cfg.is_moe() {
            let mcfg = crate::moe::MoEConfig {
                num_experts: cfg.num_experts,
                top_k: cfg.num_experts_per_tok,
                intermediate_size: cfg.intermediate_size,
                hidden_size: cfg.hidden_size,
                ep: Default::default(),
            };
            Some(
                (0..cfg.num_layers)
                    .map(|i| {
                        crate::moe::MoEWeights::random(&mcfg, seed + 100 + i as u64).expect("moe")
                    })
                    .collect(),
            )
        } else {
            None
        };
        Self {
            embed,
            layers,
            final_norm: vec![1.0; d],
            lm_head,
            moe,
        }
    }

    pub fn zeros(cfg: &SeekVConfig) -> Self {
        let mut w = Self::random(cfg, 0);
        for (_, t) in w.named_tensors_mut() {
            t.fill(0.0);
        }
        w
    }

    pub fn named_tensors(&self) -> Vec<(String, &[f32])> {
        let mut v: Vec<(String, &[f32])> = vec![("embed".to_string(), self.embed.as_slice())];
        for (i, l) in self.layers.iter().enumerate() {
            v.push((format!("layers.{i}.attn_norm"), l.attn_norm.as_slice()));
            v.push((format!("layers.{i}.wq"), l.wq.as_slice()));
            v.push((format!("layers.{i}.wk"), l.wk.as_slice()));
            v.push((format!("layers.{i}.wv"), l.wv.as_slice()));
            v.push((format!("layers.{i}.wo"), l.wo.as_slice()));
            v.push((format!("layers.{i}.mlp_norm"), l.mlp_norm.as_slice()));
            v.push((format!("layers.{i}.w_gate"), l.w_gate.as_slice()));
            v.push((format!("layers.{i}.w_up"), l.w_up.as_slice()));
            v.push((format!("layers.{i}.w_down"), l.w_down.as_slice()));
        }
        v.push(("final_norm".to_string(), self.final_norm.as_slice()));
        v.push(("lm_head".to_string(), self.lm_head.as_slice()));
        v
    }

    pub fn named_tensors_mut(&mut self) -> Vec<(String, &mut [f32])> {
        let mut v: Vec<(String, &mut [f32])> = vec![("embed".to_string(), self.embed.as_mut_slice())];
        for (i, l) in self.layers.iter_mut().enumerate() {
            v.push((format!("layers.{i}.attn_norm"), l.attn_norm.as_mut_slice()));
            v.push((format!("layers.{i}.wq"), l.wq.as_mut_slice()));
            v.push((format!("layers.{i}.wk"), l.wk.as_mut_slice()));
            v.push((format!("layers.{i}.wv"), l.wv.as_mut_slice()));
            v.push((format!("layers.{i}.wo"), l.wo.as_mut_slice()));
            v.push((format!("layers.{i}.mlp_norm"), l.mlp_norm.as_mut_slice()));
            v.push((format!("layers.{i}.w_gate"), l.w_gate.as_mut_slice()));
            v.push((format!("layers.{i}.w_up"), l.w_up.as_mut_slice()));
            v.push((format!("layers.{i}.w_down"), l.w_down.as_mut_slice()));
        }
        v.push(("final_norm".to_string(), self.final_norm.as_mut_slice()));
        v.push(("lm_head".to_string(), self.lm_head.as_mut_slice()));
        v
    }

    pub fn num_params(&self) -> usize {
        self.named_tensors().iter().map(|(_, t)| t.len()).sum()
    }

    pub fn add_assign(&mut self, other: &Self) {
        for ((_, a), (_, b)) in self.named_tensors_mut().into_iter().zip(other.named_tensors()) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += *y;
            }
        }
    }

    pub fn scale(&mut self, factor: f32) {
        for (_, t) in self.named_tensors_mut() {
            for x in t.iter_mut() {
                *x *= factor;
            }
        }
    }

    pub fn validate(&self, cfg: &SeekVConfig) -> Result<()> {
        let (d, h, hk, hd, inter, v) =
            (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.intermediate_size, cfg.vocab_size);
        let check = |name: &str, got: usize, want: usize| {
            if got == want {
                Ok(())
            } else {
                Err(SentinelError::Config(format!("weight `{name}` has {got} elements, expected {want}")))
            }
        };
        check("embed", self.embed.len(), v * d)?;
        check("lm_head", self.lm_head.len(), v * d)?;
        check("final_norm", self.final_norm.len(), d)?;
        check("layers", self.layers.len(), cfg.num_layers)?;
        for (i, l) in self.layers.iter().enumerate() {
            check(&format!("layers.{i}.attn_norm"), l.attn_norm.len(), d)?;
            check(&format!("layers.{i}.wq"), l.wq.len(), h * hd * d)?;
            check(&format!("layers.{i}.wk"), l.wk.len(), hk * hd * d)?;
            check(&format!("layers.{i}.wv"), l.wv.len(), hk * hd * d)?;
            check(&format!("layers.{i}.wo"), l.wo.len(), d * h * hd)?;
            check(&format!("layers.{i}.mlp_norm"), l.mlp_norm.len(), d)?;
            check(&format!("layers.{i}.w_gate"), l.w_gate.len(), inter * d)?;
            check(&format!("layers.{i}.w_up"), l.w_up.len(), inter * d)?;
            check(&format!("layers.{i}.w_down"), l.w_down.len(), d * inter)?;
        }
        if cfg.is_moe() {
            let moe = self.moe.as_ref().ok_or_else(|| {
                SentinelError::Config("MoE config set but weights.moe is None".into())
            })?;
            if moe.len() != cfg.num_layers {
                return Err(SentinelError::Config("moe layers count mismatch".into()));
            }
            for m in moe {
                m.validate()?;
            }
        }
        Ok(())
    }
}

struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

pub struct SeekV {
    cfg: SeekVConfig,
    w: SeekVWeights,
    caches: HashMap<SeqId, KvCache>,
    kernels: SeekKernels,
}

impl SeekV {
    pub fn new(cfg: SeekVConfig, weights: SeekVWeights) -> Result<Self> {
        cfg.validate()?;
        weights.validate(&cfg)?;
        let kernels = SeekKernels::parallel()
            .with_rope_theta(cfg.rope_theta)
            .with_rms_eps(cfg.rms_eps);
        Ok(Self {
            cfg,
            w: weights,
            caches: HashMap::new(),
            kernels,
        })
    }

    pub fn random(cfg: SeekVConfig, seed: u64) -> Result<Self> {
        Self::new(cfg.clone(), SeekVWeights::random(&cfg, seed))
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let (cfg, weights) = crate::checkpoint::load_weights(path)?;
        Self::new(cfg, weights)
    }

    pub fn config(&self) -> &SeekVConfig {
        &self.cfg
    }

    pub fn weights(&self) -> &SeekVWeights {
        &self.w
    }

    pub fn from_weights(cfg: SeekVConfig, weights: SeekVWeights) -> Result<Self> {
        Self::new(cfg, weights)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        crate::checkpoint::save_weights(path, &self.cfg, &self.w)
    }

    pub fn kv_len(&self, seq_id: SeqId) -> usize {
        self.caches.get(&seq_id).map_or(0, |c| c.len)
    }

    fn forward_one(&mut self, step: &StepInput<'_>) -> Result<Vec<f32>> {
        let t = step.tokens.len();
        if t == 0 {
            return Err(SentinelError::InvalidState(format!("sequence {}: empty step", step.seq_id)));
        }
        if step.start_pos + t > self.cfg.max_positions {
            return Err(SentinelError::InvalidState(format!(
                "sequence {}: positions {}..{} exceed max_positions {}",
                step.seq_id,
                step.start_pos,
                step.start_pos + t,
                self.cfg.max_positions
            )));
        }
        if let Some(bad) = step.tokens.iter().find(|&&tok| tok < 0 || tok as usize >= self.cfg.vocab_size) {
            return Err(SentinelError::InvalidState(format!(
                "sequence {}: token {bad} outside vocabulary",
                step.seq_id
            )));
        }

        let mut cache = if step.start_pos == 0 {
            KvCache {
                k: vec![Vec::new(); self.cfg.num_layers],
                v: vec![Vec::new(); self.cfg.num_layers],
                len: 0,
            }
        } else {
            match self.caches.get(&step.seq_id) {
                None => {
                    return Err(SentinelError::InvalidState(format!(
                        "sequence {}: no KV cache but start_pos = {}",
                        step.seq_id, step.start_pos
                    )))
                }
                Some(c) if c.len != step.start_pos => {
                    return Err(SentinelError::InvalidState(format!(
                        "sequence {}: KV cache holds {} positions but the step starts at {}",
                        step.seq_id, c.len, step.start_pos
                    )))
                }
                Some(_) => {}
            }
            self.caches.remove(&step.seq_id).ok_or_else(|| {
                SentinelError::InvalidState(format!("sequence {}: KV cache vanished", step.seq_id))
            })?
        };

        let (d, h, hk, hd, inter, vocab) = (
            self.cfg.hidden_size,
            self.cfg.num_heads,
            self.cfg.num_kv_heads,
            self.cfg.head_dim,
            self.cfg.intermediate_size,
            self.cfg.vocab_size,
        );
        let (eps, start) = (self.cfg.rms_eps, step.start_pos);
        let kv_stride = hk * hd;
        let w = &self.w;

        let mut x = vec![0.0f32; t * d];
        for (i, &tok) in step.tokens.iter().enumerate() {
            let tok = tok as usize;
            x[i * d..(i + 1) * d].copy_from_slice(&w.embed[tok * d..(tok + 1) * d]);
        }

        let dims = ModelDims {
            tokens: t,
            start_pos: start,
            hidden: d,
            num_heads: h,
            num_kv_heads: hk,
            head_dim: hd,
            intermediate: inter,
            vocab,
            rms_eps: eps,
        };
        let use_moe = w.moe.is_some();
        for (l, lw) in w.layers.iter().enumerate() {
            let mut layer_cache = LayerKvCache {
                k: std::mem::take(&mut cache.k[l]),
                v: std::mem::take(&mut cache.v[l]),
                len: start,
            };
            if use_moe {
                Self::layer_attn_then_moe(
                    &mut x,
                    lw,
                    w.moe.as_ref().unwrap().get(l).expect("moe layer"),
                    &mut layer_cache,
                    t,
                    start,
                    d,
                    h,
                    hk,
                    hd,
                    eps,
                    self.cfg.rope_theta,
                )?;
            } else {
                let kw = KernLayerWeights {
                    attn_norm: &lw.attn_norm,
                    wq: &lw.wq,
                    wk: &lw.wk,
                    wv: &lw.wv,
                    wo: &lw.wo,
                    mlp_norm: &lw.mlp_norm,
                    w_gate: &lw.w_gate,
                    w_up: &lw.w_up,
                    w_down: &lw.w_down,
                };
                self.kernels.layer(&mut x, dims, &kw, &mut layer_cache).map_err(kerr)?;
            }
            let end = layer_cache.len * kv_stride;
            cache.k[l] = layer_cache.k[..end.min(layer_cache.k.len())].to_vec();
            cache.v[l] = layer_cache.v[..end.min(layer_cache.v.len())].to_vec();
        }

        let (last, _) =
            kern::rms_norm(&x[(t - 1) * d..t * d], &w.final_norm, None, 1, d, eps).map_err(kerr)?;
        let logits = self.kernels.gemm(&last, &w.lm_head, 1, d, vocab).map_err(kerr)?;

        cache.len = start + t;
        self.caches.insert(step.seq_id, cache);
        Ok(logits)
    }

    /// Attention then MoE FFN + residuals.
    fn layer_attn_then_moe(
        x: &mut [f32],
        lw: &LayerWeights,
        moe: &crate::moe::MoEWeights,
        cache: &mut LayerKvCache,
        t: usize,
        start: usize,
        d: usize,
        h: usize,
        hk: usize,
        hd: usize,
        eps: f32,
        rope_theta: f32,
    ) -> Result<()> {
        let residual = x.to_vec();
        let (h_in, _) = kern::rms_norm(x, &lw.attn_norm, None, t, d, eps).map_err(kerr)?;
        let mut q = kern::matmul_nt(&h_in, &lw.wq, t, d, h * hd).map_err(kerr)?;
        let mut k = kern::matmul_nt(&h_in, &lw.wk, t, d, hk * hd).map_err(kerr)?;
        let mut v = kern::matmul_nt(&h_in, &lw.wv, t, d, hk * hd).map_err(kerr)?;
        kern::rope_inplace(&mut q, t, h, hd, start, rope_theta).map_err(kerr)?;
        kern::rope_inplace(&mut k, t, hk, hd, start, rope_theta).map_err(kerr)?;

        let total = start + t;
        let stride = hk * hd;
        let need = total * stride;
        if cache.k.len() < need {
            cache.k.resize(need, 0.0);
            cache.v.resize(need, 0.0);
        }
        for i in 0..t {
            let pos = start + i;
            let dst = pos * stride;
            cache.k[dst..dst + stride].copy_from_slice(&k[i * stride..(i + 1) * stride]);
            cache.v[dst..dst + stride].copy_from_slice(&v[i * stride..(i + 1) * stride]);
        }
        cache.len = total;

        let attn = kern::attention_causal(
            &q,
            &cache.k[..total * stride],
            &cache.v[..total * stride],
            t,
            start,
            h,
            hk,
            hd,
        )
        .map_err(kerr)?;
        let mut attn_out = kern::matmul_nt(&attn, &lw.wo, t, h * hd, d).map_err(kerr)?;
        for (a, b) in attn_out.iter_mut().zip(residual.iter()) {
            *a += *b;
        }

        let residual2 = attn_out.clone();
        let (h2, _) = kern::rms_norm(&attn_out, &lw.mlp_norm, None, t, d, eps).map_err(kerr)?;
        let mut mlp = crate::moe::moe_ffn(&h2, moe)?;
        for (a, b) in mlp.iter_mut().zip(residual2.iter()) {
            *a += *b;
        }
        x.copy_from_slice(&mlp);
        Ok(())
    }
}

impl Model for SeekV {
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn eos_token_id(&self) -> i32 {
        self.cfg.eos_token_id
    }

    fn max_positions(&self) -> usize {
        self.cfg.max_positions
    }

    fn forward(&mut self, batch: &[StepInput<'_>]) -> Result<Vec<f32>> {
        let mut logits = Vec::with_capacity(batch.len() * self.cfg.vocab_size);
        for step in batch {
            logits.extend(self.forward_one(step)?);
        }
        Ok(logits)
    }

    fn release(&mut self, seq_id: SeqId) {
        self.caches.remove(&seq_id);
    }

    fn truncate(&mut self, seq_id: SeqId, len: usize) {
        if let Some(cache) = self.caches.get_mut(&seq_id) {
            if len == 0 {
                self.caches.remove(&seq_id);
                return;
            }
            if len >= cache.len {
                return;
            }
            let stride = self.cfg.num_kv_heads * self.cfg.head_dim;
            let keep = len * stride;
            for layer in 0..cache.k.len() {
                cache.k[layer].truncate(keep);
                cache.v[layer].truncate(keep);
            }
            cache.len = len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> SeekV {
        SeekV::random(SeekVConfig::tiny(), 1234).unwrap()
    }

    fn step(seq_id: SeqId, tokens: &[i32], start_pos: usize) -> StepInput<'_> {
        StepInput {
            seq_id,
            tokens,
            start_pos,
        }
    }

    #[test]
    fn logits_have_vocab_shape_and_are_finite() {
        let mut m = model();
        let logits = m.forward(&[step(1, &[1, 2, 3], 0)]).unwrap();
        assert_eq!(logits.len(), 258);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn truncate_shortens_kv() {
        let mut m = model();
        m.forward(&[step(1, &[1, 2, 3, 4], 0)]).unwrap();
        assert_eq!(m.kv_len(1), 4);
        m.truncate(1, 2);
        assert_eq!(m.kv_len(1), 2);
    }

    #[test]
    fn moe_forward_runs() {
        let mut m = SeekV::random(SeekVConfig::tiny_moe(), 99).unwrap();
        assert!(m.config().is_moe());
        let logits = m.forward(&[step(1, &[1, 2, 3], 0)]).unwrap();
        assert_eq!(logits.len(), 258);
        assert!(logits.iter().all(|x| x.is_finite()));
    }
}
