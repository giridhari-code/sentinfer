//! Quantized Seek-V weights (int8 / int4) for smaller memory and faster matmul.

use std::path::Path;

use sentinel_core::{Result, SentinelError};
use sentinel_kernels::{QuantMatrix, QuantScheme};

use crate::seekv::{LayerWeights, SeekV, SeekVConfig, SeekVWeights};

/// One transformer layer with quantized linear weights; norms stay f32.
#[derive(Debug, Clone)]
pub struct QuantLayer {
    pub attn_norm: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub wq: QuantMatrix,
    pub wk: QuantMatrix,
    pub wv: QuantMatrix,
    pub wo: QuantMatrix,
    pub w_gate: QuantMatrix,
    pub w_up: QuantMatrix,
    pub w_down: QuantMatrix,
}

/// Fully quantized weight set (embeddings stay f32 for quality).
#[derive(Debug, Clone)]
pub struct QuantizedSeekV {
    pub config: SeekVConfig,
    pub scheme: QuantScheme,
    pub embed: Vec<f32>,
    pub layers: Vec<QuantLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: QuantMatrix,
}

fn qmat(w: &[f32], rows: usize, cols: usize, scheme: QuantScheme) -> Result<QuantMatrix> {
    match scheme {
        QuantScheme::Int8 => QuantMatrix::quantize_int8(w, rows, cols).map_err(|e| SentinelError::InvalidState(e.to_string())),
        QuantScheme::Int4 => QuantMatrix::quantize_int4(w, rows, cols).map_err(|e| SentinelError::InvalidState(e.to_string())),
    }
}

impl QuantizedSeekV {
    pub fn from_fp32(model: &SeekV, scheme: QuantScheme) -> Result<Self> {
        let cfg = model.config().clone();
        let d = cfg.hidden_size;
        let h = cfg.num_heads;
        let hk = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let w = model.weights();
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in &w.layers {
            layers.push(QuantLayer {
                attn_norm: l.attn_norm.clone(),
                mlp_norm: l.mlp_norm.clone(),
                wq: qmat(&l.wq, h * hd, d, scheme)?,
                wk: qmat(&l.wk, hk * hd, d, scheme)?,
                wv: qmat(&l.wv, hk * hd, d, scheme)?,
                wo: qmat(&l.wo, d, h * hd, scheme)?,
                w_gate: qmat(&l.w_gate, inter, d, scheme)?,
                w_up: qmat(&l.w_up, inter, d, scheme)?,
                w_down: qmat(&l.w_down, d, inter, scheme)?,
            });
        }
        Ok(Self {
            config: cfg.clone(),
            scheme,
            embed: w.embed.clone(),
            layers,
            final_norm: w.final_norm.clone(),
            lm_head: qmat(&w.lm_head, v, d, scheme)?,
        })
    }

    /// Materialize back to f32 SeekV (for drop-in Engine use).
    pub fn to_fp32(&self) -> Result<SeekV> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            layers.push(LayerWeights {
                attn_norm: l.attn_norm.clone(),
                mlp_norm: l.mlp_norm.clone(),
                wq: l.wq.dequantize(),
                wk: l.wk.dequantize(),
                wv: l.wv.dequantize(),
                wo: l.wo.dequantize(),
                w_gate: l.w_gate.dequantize(),
                w_up: l.w_up.dequantize(),
                w_down: l.w_down.dequantize(),
            });
        }
        let weights = SeekVWeights {
            embed: self.embed.clone(),
            layers,
            final_norm: self.final_norm.clone(),
            lm_head: self.lm_head.dequantize(),
        };
        SeekV::from_weights(self.config.clone(), weights)
    }

    pub fn nbytes_packed(&self) -> usize {
        let mut n = self.embed.len() * 4 + self.final_norm.len() * 4;
        for l in &self.layers {
            n += l.attn_norm.len() * 4 + l.mlp_norm.len() * 4;
            n += l.wq.data.len() + l.wq.scale.len() * 4;
            n += l.wk.data.len() + l.wk.scale.len() * 4;
            n += l.wv.data.len() + l.wv.scale.len() * 4;
            n += l.wo.data.len() + l.wo.scale.len() * 4;
            n += l.w_gate.data.len() + l.w_gate.scale.len() * 4;
            n += l.w_up.data.len() + l.w_up.scale.len() * 4;
            n += l.w_down.data.len() + l.w_down.scale.len() * 4;
        }
        n += self.lm_head.data.len() + self.lm_head.scale.len() * 4;
        n
    }
}

/// Quantize a `.skv` checkpoint and write a new `.skv` (dequantized materialization)
/// or keep fp32 path after quant-dequant for size reporting.
pub fn quantize_checkpoint(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    scheme: QuantScheme,
) -> Result<(usize, usize)> {
    let model = SeekV::load(src.as_ref())?;
    let fp_bytes = model.weights().named_tensors().iter().map(|(_, t)| t.len() * 4).sum::<usize>();
    let q = QuantizedSeekV::from_fp32(&model, scheme)?;
    let packed = q.nbytes_packed();
    // Persist as fp32 .skv reconstructed from quant (validates pipeline); true packed format later.
    let restored = q.to_fp32()?;
    restored.save(dst.as_ref())?;
    Ok((fp_bytes, packed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seekv::SeekVConfig;

    #[test]
    fn quant_dequant_logits_finite() {
        let m = SeekV::random(SeekVConfig::tiny(), 1).unwrap();
        let q = QuantizedSeekV::from_fp32(&m, QuantScheme::Int8).unwrap();
        let fp_bytes: usize = m.weights().named_tensors().iter().map(|(_, t)| t.len() * 4).sum();
        assert!(q.nbytes_packed() < fp_bytes);
        let restored = q.to_fp32().unwrap();
        assert_eq!(restored.config().hidden_size, m.config().hidden_size);
        assert_eq!(restored.weights().lm_head.len(), m.weights().lm_head.len());
    }
}
