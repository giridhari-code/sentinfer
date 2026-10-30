//! Training-side optimizations shared by the trainer loop.
//!
//! * **Gradient accumulation** — average micro-batch grads over `N` steps before AdamW.
//! * **In-place grad scale / add** — avoid extra weight buffers when possible.
//! * Scaffolding for future: activation checkpointing, mixed precision (bf16).

use sentinel_model::SeekVWeights;

/// Add `src` into `dst` elementwise (`dst += src`).
pub fn add_grads(dst: &mut SeekVWeights, src: &SeekVWeights) {
    for ((_, d), (_, s)) in dst.named_tensors_mut().into_iter().zip(src.named_tensors()) {
        for (di, &si) in d.iter_mut().zip(s.iter()) {
            *di += si;
        }
    }
}

/// Scale every gradient by `factor` in place.
pub fn scale_grads(g: &mut SeekVWeights, factor: f32) {
    if (factor - 1.0).abs() < f32::EPSILON {
        return;
    }
    g.scale(factor);
}

/// Average accumulated gradients over `n` micro-batches (`g *= 1/n`).
pub fn average_grads(g: &mut SeekVWeights, n: usize) {
    if n <= 1 {
        return;
    }
    scale_grads(g, 1.0 / n as f32);
}

/// Options that control memory / throughput tradeoffs during training.
#[derive(Debug, Clone)]
pub struct TrainOptConfig {
    /// Number of micro-batches whose gradients are averaged before one AdamW step.
    /// Effective batch size ≈ `batch_size * grad_accum_steps`.
    pub grad_accum_steps: usize,
    /// When true, autograd may drop intermediate activations and recompute on backward
    /// (lower peak RAM, more compute). Not fully wired in all layers yet.
    pub activation_checkpointing: bool,
    /// Reserved: store master weights in f32 and run matmuls in bf16/fp16 when backends support it.
    pub mixed_precision: bool,
}

impl Default for TrainOptConfig {
    fn default() -> Self {
        Self {
            grad_accum_steps: 1,
            activation_checkpointing: false,
            mixed_precision: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sentinel_model::SeekVConfig;

    #[test]
    fn add_and_average() {
        let cfg = SeekVConfig::micro(8);
        let mut a = SeekVWeights::zeros(&cfg);
        let mut b = SeekVWeights::zeros(&cfg);
        for (_, t) in a.named_tensors_mut() {
            t.fill(2.0);
        }
        for (_, t) in b.named_tensors_mut() {
            t.fill(4.0);
        }
        add_grads(&mut a, &b);
        average_grads(&mut a, 2);
        // (2+4)/2 = 3
        assert!((a.embed[0] - 3.0).abs() < 1e-5);
    }
}
