//! AdamW, global-norm gradient clipping and the learning-rate schedule.

use sentinel_core::Result;
use sentinel_model::checkpoint::{ByteReader, ByteWriter};
use sentinel_model::{SeekVConfig, SeekVWeights};

#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Decoupled weight decay. Applied to matrices and embeddings, not to norm scales.
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay: 0.1 }
    }
}

pub struct AdamW {
    cfg: AdamWConfig,
    m: SeekVWeights,
    v: SeekVWeights,
    t: u64,
}

impl AdamW {
    pub fn new(model_cfg: &SeekVConfig, cfg: AdamWConfig) -> Self {
        Self { cfg, m: SeekVWeights::zeros(model_cfg), v: SeekVWeights::zeros(model_cfg), t: 0 }
    }

    /// Number of optimizer steps taken so far.
    pub fn steps(&self) -> u64 {
        self.t
    }

    pub fn step(&mut self, weights: &mut SeekVWeights, grads: &SeekVWeights, lr: f32) {
        self.t += 1;
        let t = self.t.min(i32::MAX as u64) as i32;
        let (b1, b2, eps, wd) = (self.cfg.beta1, self.cfg.beta2, self.cfg.eps, self.cfg.weight_decay);
        let bias1 = 1.0 - b1.powi(t);
        let bias2 = 1.0 - b2.powi(t);

        let params = weights.named_tensors_mut();
        let grads = grads.named_tensors();
        let ms = self.m.named_tensors_mut();
        let vs = self.v.named_tensors_mut();
        for ((((name, p), (_, g)), (_, m)), (_, v)) in params.into_iter().zip(grads).zip(ms).zip(vs) {
            let decay = !name.ends_with("norm");
            for i in 0..p.len() {
                let gi = g[i];
                m[i] = b1 * m[i] + (1.0 - b1) * gi;
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                let m_hat = m[i] / bias1;
                let v_hat = v[i] / bias2;
                if decay {
                    p[i] -= lr * wd * p[i];
                }
                p[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }

    /// Appends the optimizer state (step count, first and second moments) for resuming.
    pub fn write_state(&self, w: &mut ByteWriter) {
        w.u64(self.t);
        for (_, t) in self.m.named_tensors() {
            w.f32_slice(t);
        }
        for (_, t) in self.v.named_tensors() {
            w.f32_slice(t);
        }
    }

    pub fn read_state(&mut self, r: &mut ByteReader<'_>) -> Result<()> {
        self.t = r.u64()?;
        for (_, t) in self.m.named_tensors_mut() {
            r.f32_into(t)?;
        }
        for (_, t) in self.v.named_tensors_mut() {
            r.f32_into(t)?;
        }
        Ok(())
    }
}

/// L2 norm over every gradient element.
pub fn global_norm(g: &SeekVWeights) -> f32 {
    let mut sum = 0.0f64;
    for (_, t) in g.named_tensors() {
        for &x in t {
            sum += (x as f64) * (x as f64);
        }
    }
    sum.sqrt() as f32
}

/// Scales `g` so its global norm is at most `max_norm`. Returns the norm before clipping.
pub fn clip_grad_norm(g: &mut SeekVWeights, max_norm: f32) -> f32 {
    let norm = global_norm(g);
    if norm.is_finite() && norm > max_norm && max_norm > 0.0 {
        g.scale(max_norm / norm);
    }
    norm
}

/// Linear warmup, then cosine decay from `peak` to `min`.
#[derive(Debug, Clone)]
pub struct LrSchedule {
    pub peak: f32,
    pub min: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LrSchedule {
    /// Learning rate for the update at 0-based `step`.
    pub fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.peak * (step + 1) as f32 / self.warmup_steps as f32;
        }
        if step >= self.total_steps || self.total_steps <= self.warmup_steps {
            return self.min;
        }
        let progress = (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
        self.min + 0.5 * (self.peak - self.min) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SeekVConfig {
        SeekVConfig::micro(10)
    }

    fn ones(cfg: &SeekVConfig) -> SeekVWeights {
        let mut g = SeekVWeights::zeros(cfg);
        for (_, t) in g.named_tensors_mut() {
            t.fill(1.0);
        }
        g
    }

    #[test]
    fn first_step_moves_every_parameter_by_about_lr() {
        let cfg = cfg();
        let mut w = SeekVWeights::random(&cfg, 1);
        let before = w.clone();
        let mut opt = AdamW::new(&cfg, AdamWConfig { weight_decay: 0.0, ..Default::default() });
        opt.step(&mut w, &ones(&cfg), 0.01);
        // m_hat = g = 1, v_hat = 1 -> update = lr * 1 / (1 + eps)
        assert!((before.embed[0] - w.embed[0] - 0.01).abs() < 1e-6);
        assert!((before.layers[1].wo[3] - w.layers[1].wo[3] - 0.01).abs() < 1e-6);
        assert_eq!(opt.steps(), 1);
    }

    #[test]
    fn weight_decay_skips_norm_scales() {
        let cfg = cfg();
        let mut w = SeekVWeights::random(&cfg, 2);
        let before = w.clone();
        let mut opt = AdamW::new(&cfg, AdamWConfig { weight_decay: 0.5, ..Default::default() });
        opt.step(&mut w, &SeekVWeights::zeros(&cfg), 0.1);
        // zero gradient: matrices shrink by (1 - lr * wd) = 0.95, norm scales stay at 1.0
        assert!((w.embed[5] - before.embed[5] * 0.95).abs() < 1e-7);
        assert!((w.layers[0].w_gate[2] - before.layers[0].w_gate[2] * 0.95).abs() < 1e-7);
        assert_eq!(w.final_norm, before.final_norm);
        assert_eq!(w.layers[0].attn_norm, before.layers[0].attn_norm);
    }

    #[test]
    fn descends_on_a_quadratic() {
        // minimize 0.5 * ||p||^2: the gradient is p itself
        let cfg = cfg();
        let mut w = SeekVWeights::random(&cfg, 3);
        let start = global_norm(&w);
        let mut opt = AdamW::new(&cfg, AdamWConfig { weight_decay: 0.0, ..Default::default() });
        for _ in 0..50 {
            let g = w.clone();
            opt.step(&mut w, &g, 0.01);
        }
        assert!(global_norm(&w) < start);
    }

    #[test]
    fn clipping_limits_the_global_norm() {
        let cfg = cfg();
        let mut g = ones(&cfg);
        let before = global_norm(&g);
        assert!(before > 1.0);
        let reported = clip_grad_norm(&mut g, 1.0);
        assert!((reported - before).abs() < 1e-3);
        assert!((global_norm(&g) - 1.0).abs() < 1e-4);

        let mut small = ones(&cfg);
        small.scale(1e-4);
        let n = global_norm(&small);
        clip_grad_norm(&mut small, 1.0);
        assert!((global_norm(&small) - n).abs() < 1e-9, "small gradients must stay untouched");
    }

    #[test]
    fn schedule_warms_up_then_decays_to_the_minimum() {
        let s = LrSchedule { peak: 1.0, min: 0.1, warmup_steps: 10, total_steps: 110 };
        assert!((s.lr_at(0) - 0.1).abs() < 1e-6); // 1/10 of peak
        assert!((s.lr_at(9) - 1.0).abs() < 1e-6);
        assert!((s.lr_at(10) - 1.0).abs() < 1e-6);
        assert!((s.lr_at(60) - 0.55).abs() < 1e-5); // halfway through the cosine
        assert!((s.lr_at(110) - 0.1).abs() < 1e-6);
        assert!((s.lr_at(5000) - 0.1).abs() < 1e-6);
        let mut prev = f32::MAX;
        for step in 10..110 {
            let lr = s.lr_at(step);
            assert!(lr <= prev + 1e-7);
            prev = lr;
        }
    }

    #[test]
    fn schedule_without_decay_range_is_constant_minimum_after_warmup() {
        let s = LrSchedule { peak: 1.0, min: 0.2, warmup_steps: 5, total_steps: 5 };
        assert_eq!(s.lr_at(5), 0.2);
        let no_warmup = LrSchedule { peak: 1.0, min: 0.0, warmup_steps: 0, total_steps: 10 };
        assert!((no_warmup.lr_at(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn optimizer_state_roundtrips_exactly() {
        let cfg = cfg();
        let mut w1 = SeekVWeights::random(&cfg, 4);
        let mut opt1 = AdamW::new(&cfg, AdamWConfig::default());
        for _ in 0..2 {
            let g = w1.clone();
            opt1.step(&mut w1, &g, 0.01);
        }
        let mut writer = ByteWriter::new();
        opt1.write_state(&mut writer);
        let bytes = writer.finish();

        let mut opt2 = AdamW::new(&cfg, AdamWConfig::default());
        let mut reader = ByteReader::checked(&bytes).unwrap();
        opt2.read_state(&mut reader).unwrap();
        assert!(reader.is_empty());
        assert_eq!(opt2.steps(), 2);

        let mut w2 = w1.clone();
        let g = w1.clone();
        opt1.step(&mut w1, &g, 0.01);
        opt2.step(&mut w2, &g, 0.01);
        assert_eq!(w1.embed, w2.embed);
        assert_eq!(w1.lm_head, w2.lm_head);
    }
}
