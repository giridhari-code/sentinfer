//! Quantized Seek-V **runtime**: linear weights stay int8/int4; matmul uses
//! [`QuantMatrix::matmul_nt`] (no full dequant each step).

use std::collections::HashMap;
use std::path::Path;

use sentinel_core::{Result, SeqId, SentinelError};
use sentinel_kernels::reference as kern;
use sentinel_kernels::{QuantMatrix, QuantScheme};

use crate::model::{Model, StepInput};
use crate::ops::kerr;
use crate::seekv::{SeekV, SeekVConfig};

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

#[derive(Debug, Clone)]
struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

/// Seek-V with quantized linears (embed + norms remain f32).
#[derive(Debug, Clone)]
pub struct QuantizedSeekV {
    cfg: SeekVConfig,
    scheme: QuantScheme,
    embed: Vec<f32>,
    layers: Vec<QuantLayer>,
    final_norm: Vec<f32>,
    lm_head: QuantMatrix,
    caches: HashMap<SeqId, KvCache>,
}

fn qmat(w: &[f32], rows: usize, cols: usize, scheme: QuantScheme) -> Result<QuantMatrix> {
    match scheme {
        QuantScheme::Int8 => QuantMatrix::quantize_int8(w, rows, cols).map_err(|e| SentinelError::InvalidState(e.to_string())),
        QuantScheme::Int4 => QuantMatrix::quantize_int4(w, rows, cols).map_err(|e| SentinelError::InvalidState(e.to_string())),
    }
}

impl QuantizedSeekV {
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    pub fn config(&self) -> &SeekVConfig {
        &self.cfg
    }

    pub fn nbytes_packed(&self) -> usize {
        let mut n = self.embed.len() * 4 + self.final_norm.len() * 4;
        for l in &self.layers {
            n += l.attn_norm.len() * 4 + l.mlp_norm.len() * 4;
            for m in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
                n += m.data.len() + m.scale.len() * 4;
            }
        }
        n += self.lm_head.data.len() + self.lm_head.scale.len() * 4;
        n
    }

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
            cfg,
            scheme,
            embed: w.embed.clone(),
            layers,
            final_norm: w.final_norm.clone(),
            lm_head: qmat(&w.lm_head, v, d, scheme)?,
            caches: HashMap::new(),
        })
    }

    /// Materialize fp32 SeekV (debug / export).
    pub fn to_fp32(&self) -> Result<SeekV> {
        use crate::seekv::{LayerWeights, SeekVWeights};
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
        SeekV::from_weights(self.cfg.clone(), weights)
    }

    fn forward_one(&mut self, step: &StepInput<'_>) -> Result<Vec<f32>> {
        let t = step.tokens.len();
        if t == 0 {
            return Err(SentinelError::InvalidState(format!("sequence {}: empty step", step.seq_id)));
        }
        if step.start_pos + t > self.cfg.max_positions {
            return Err(SentinelError::InvalidState(format!(
                "sequence {}: positions exceed max",
                step.seq_id
            )));
        }
        if let Some(bad) = step.tokens.iter().find(|&&tok| tok < 0 || tok as usize >= self.cfg.vocab_size) {
            return Err(SentinelError::InvalidState(format!("sequence {}: bad token {bad}", step.seq_id)));
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
                        "sequence {}: no KV cache",
                        step.seq_id
                    )))
                }
                Some(c) if c.len != step.start_pos => {
                    return Err(SentinelError::InvalidState(format!(
                        "sequence {}: cache len {} != start {}",
                        step.seq_id, c.len, step.start_pos
                    )))
                }
                Some(_) => {}
            }
            self.caches.remove(&step.seq_id).expect("cache")
        };

        let d = self.cfg.hidden_size;
        let h = self.cfg.num_heads;
        let hk = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size;
        let eps = self.cfg.rms_eps;
        let theta = self.cfg.rope_theta;
        let start = step.start_pos;
        let kv_stride = hk * hd;

        let mut x = vec![0.0f32; t * d];
        for (i, &tok) in step.tokens.iter().enumerate() {
            let tok = tok as usize;
            x[i * d..(i + 1) * d].copy_from_slice(&self.embed[tok * d..(tok + 1) * d]);
        }

        for (l, lw) in self.layers.iter().enumerate() {
            let (hn, _) = kern::rms_norm(&x, &lw.attn_norm, None, t, d, eps).map_err(kerr)?;
            let mut q = lw.wq.matmul_nt(&hn, t).map_err(kerr)?;
            let mut k = lw.wk.matmul_nt(&hn, t).map_err(kerr)?;
            let v = lw.wv.matmul_nt(&hn, t).map_err(kerr)?;
            kern::rope_inplace(&mut q, t, h, hd, start, theta).map_err(kerr)?;
            kern::rope_inplace(&mut k, t, hk, hd, start, theta).map_err(kerr)?;
            cache.k[l].extend_from_slice(&k);
            cache.v[l].extend_from_slice(&v);
            let attn = kern::attention_causal(&q, &cache.k[l], &cache.v[l], t, start, h, hk, hd).map_err(kerr)?;
            let o = lw.wo.matmul_nt(&attn, t).map_err(kerr)?;
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += *oi;
            }

            let (mn, _) = kern::rms_norm(&x, &lw.mlp_norm, None, t, d, eps).map_err(kerr)?;
            let mut gate = lw.w_gate.matmul_nt(&mn, t).map_err(kerr)?;
            let up = lw.w_up.matmul_nt(&mn, t).map_err(kerr)?;
            kern::silu_inplace(&mut gate);
            for i in 0..gate.len() {
                gate[i] *= up[i];
            }
            let down = lw.w_down.matmul_nt(&gate, t).map_err(kerr)?;
            for (xi, di) in x.iter_mut().zip(&down) {
                *xi += *di;
            }
        }

        let (last, _) = kern::rms_norm(&x[(t - 1) * d..t * d], &self.final_norm, None, 1, d, eps).map_err(kerr)?;
        let logits = self.lm_head.matmul_nt(&last, 1).map_err(kerr)?;
        let _ = kv_stride;
        cache.len = start + t;
        self.caches.insert(step.seq_id, cache);
        Ok(logits)
    }

    /// Write packed `.skq` (magic SKQ1 + FNV checksum).
    pub fn save_packed(&self, path: impl AsRef<Path>) -> Result<()> {
        use crate::checkpoint::{write_atomic, ByteWriter};
        let mut w = ByteWriter::new();
        w.bytes(b"SKQ1");
        w.u32(1);
        w.u32(match self.scheme {
            QuantScheme::Int8 => 0,
            QuantScheme::Int4 => 1,
        });
        skq_write_config(&mut w, &self.cfg);
        w.f32_slice(&self.embed);
        w.u32(self.layers.len() as u32);
        for l in &self.layers {
            w.f32_slice(&l.attn_norm);
            w.f32_slice(&l.mlp_norm);
            skq_write_qmat(&mut w, &l.wq);
            skq_write_qmat(&mut w, &l.wk);
            skq_write_qmat(&mut w, &l.wv);
            skq_write_qmat(&mut w, &l.wo);
            skq_write_qmat(&mut w, &l.w_gate);
            skq_write_qmat(&mut w, &l.w_up);
            skq_write_qmat(&mut w, &l.w_down);
        }
        w.f32_slice(&self.final_norm);
        skq_write_qmat(&mut w, &self.lm_head);
        write_atomic(path.as_ref(), &w.finish())
    }

    pub fn load_packed(path: impl AsRef<Path>) -> Result<Self> {
        use crate::checkpoint::ByteReader;
        let bytes = std::fs::read(path.as_ref()).map_err(|e| SentinelError::Io(e.to_string()))?;
        let mut r = ByteReader::checked(&bytes).map_err(|e| e)?;
        let magic = r.bytes(4).map_err(|e| e)?;
        if magic != b"SKQ1" {
            return Err(SentinelError::InvalidState("not an SKQ1 file".into()));
        }
        let ver = r.u32()?;
        if ver != 1 {
            return Err(SentinelError::InvalidState(format!("skq version {ver}")));
        }
        let scheme = match r.u32()? {
            0 => QuantScheme::Int8,
            1 => QuantScheme::Int4,
            s => return Err(SentinelError::InvalidState(format!("scheme {s}"))),
        };
        let cfg = skq_read_config(&mut r)?;
        let embed = skq_read_f32s(&mut r)?;
        let n_layers = r.u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(QuantLayer {
                attn_norm: skq_read_f32s(&mut r)?,
                mlp_norm: skq_read_f32s(&mut r)?,
                wq: skq_read_qmat(&mut r, scheme)?,
                wk: skq_read_qmat(&mut r, scheme)?,
                wv: skq_read_qmat(&mut r, scheme)?,
                wo: skq_read_qmat(&mut r, scheme)?,
                w_gate: skq_read_qmat(&mut r, scheme)?,
                w_up: skq_read_qmat(&mut r, scheme)?,
                w_down: skq_read_qmat(&mut r, scheme)?,
            });
        }
        let final_norm = skq_read_f32s(&mut r)?;
        let lm_head = skq_read_qmat(&mut r, scheme)?;
        Ok(Self {
            cfg,
            scheme,
            embed,
            layers,
            final_norm,
            lm_head,
            caches: HashMap::new(),
        })
    }
}

impl Model for QuantizedSeekV {
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
}


fn skq_write_config(w: &mut crate::checkpoint::ByteWriter, cfg: &SeekVConfig) {
    w.u64(cfg.vocab_size as u64);
    w.u64(cfg.hidden_size as u64);
    w.u64(cfg.num_layers as u64);
    w.u64(cfg.num_heads as u64);
    w.u64(cfg.num_kv_heads as u64);
    w.u64(cfg.head_dim as u64);
    w.u64(cfg.intermediate_size as u64);
    w.u64(cfg.max_positions as u64);
    w.f32(cfg.rope_theta);
    w.f32(cfg.rms_eps);
    w.i32(cfg.eos_token_id);
}

fn skq_read_config(r: &mut crate::checkpoint::ByteReader<'_>) -> Result<SeekVConfig> {
    Ok(SeekVConfig {
        vocab_size: r.u64()? as usize,
        hidden_size: r.u64()? as usize,
        num_layers: r.u64()? as usize,
        num_heads: r.u64()? as usize,
        num_kv_heads: r.u64()? as usize,
        head_dim: r.u64()? as usize,
        intermediate_size: r.u64()? as usize,
        max_positions: r.u64()? as usize,
        rope_theta: r.f32()?,
        rms_eps: r.f32()?,
        eos_token_id: r.i32()?,
    })
}

fn skq_read_f32s(r: &mut crate::checkpoint::ByteReader<'_>) -> Result<Vec<f32>> {
    let n = r.u64()? as usize;
    let mut v = vec![0.0f32; n];
    for slot in &mut v {
        *slot = r.f32()?;
    }
    Ok(v)
}

fn skq_write_qmat(w: &mut crate::checkpoint::ByteWriter, m: &QuantMatrix) {
    w.u32(m.rows as u32);
    w.u32(m.cols as u32);
    w.f32_slice(&m.scale);
    w.u64(m.data.len() as u64);
    for &b in &m.data {
        w.bytes(&(b as u8).to_le_bytes());
    }
}

fn skq_read_qmat(r: &mut crate::checkpoint::ByteReader<'_>, scheme: QuantScheme) -> Result<QuantMatrix> {
    let rows = r.u32()? as usize;
    let cols = r.u32()? as usize;
    let scale = skq_read_f32s(r)?;
    let n = r.u64()? as usize;
    let raw = r.bytes(n)?;
    let data: Vec<i8> = raw.iter().map(|&b| b as i8).collect();
    Ok(QuantMatrix { scheme, rows, cols, data, scale })
}

/// Quantize checkpoint and write **fp32** `.skv` (compat). Prefer [`QuantizedSeekV::from_fp32`] for runtime.
pub fn quantize_checkpoint(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    scheme: QuantScheme,
) -> Result<(usize, usize)> {
    let model = SeekV::load(src.as_ref())?;
    let fp_bytes = model.weights().named_tensors().iter().map(|(_, t)| t.len() * 4).sum::<usize>();
    let q = QuantizedSeekV::from_fp32(&model, scheme)?;
    let packed = q.nbytes_packed();
    q.to_fp32()?.save(dst.as_ref())?;
    Ok((fp_bytes, packed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seekv::SeekVConfig;

    #[test]
    fn quant_runtime_forward_finite() {
        let m = SeekV::random(SeekVConfig::tiny(), 1).unwrap();
        let mut q = QuantizedSeekV::from_fp32(&m, QuantScheme::Int8).unwrap();
        assert!(q.nbytes_packed() < m.weights().named_tensors().iter().map(|(_, t)| t.len() * 4).sum::<usize>());
        let step = StepInput {
            seq_id: 1,
            tokens: &[1, 2, 3],
            start_pos: 0,
        };
        let logits = q.forward(&[step]).unwrap();
        assert_eq!(logits.len(), q.vocab_size());
        assert!(logits.iter().all(|x| x.is_finite()));
        // decode step
        let step2 = StepInput {
            seq_id: 1,
            tokens: &[4],
            start_pos: 3,
        };
        let logits2 = q.forward(&[step2]).unwrap();
        assert!(logits2.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn packed_roundtrip_int8() {
        let m = SeekV::random(SeekVConfig::tiny(), 3).unwrap();
        let q = QuantizedSeekV::from_fp32(&m, QuantScheme::Int8).unwrap();
        let dir = std::env::temp_dir().join("skq_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t.skq");
        q.save_packed(&path).unwrap();
        let mut q2 = QuantizedSeekV::load_packed(&path).unwrap();
        assert_eq!(q2.config().hidden_size, q.config().hidden_size);
        let step = StepInput { seq_id: 1, tokens: &[1, 2], start_pos: 0 };
        assert!(q2.forward(&[step]).unwrap().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn int4_runtime_forward_finite() {
        let m = SeekV::random(SeekVConfig::tiny(), 2).unwrap();
        let mut q = QuantizedSeekV::from_fp32(&m, QuantScheme::Int4).unwrap();
        let step = StepInput {
            seq_id: 7,
            tokens: &[0, 1],
            start_pos: 0,
        };
        let logits = q.forward(&[step]).unwrap();
        assert_eq!(logits.len(), q.vocab_size());
        assert!(logits.iter().all(|x| x.is_finite()));
    }
}
