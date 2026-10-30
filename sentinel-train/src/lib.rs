//! Training pipeline for Seek-V (CPU, f32, no external dependencies).
//!
//! * `math`: linear / RMSNorm forward+backward building blocks.
//! * `autograd`: full-sequence forward pass with saved activations, hand-written backward pass,
//!   cross-entropy loss, and parallel gradient computation over a batch.
//! * `optim`: AdamW, gradient clipping, warmup + cosine learning-rate schedule.
//! * `data`: token datasets, window sampling, train/validation split.
//! * `trainer`: the training loop with evaluation, checkpointing and resume.
//! * `train_opt`: gradient accumulation helpers, future mixed-precision / checkpointing flags.

pub mod autograd;
pub mod data;
pub mod math;
pub mod optim;
pub mod train_opt;
pub mod trainer;

pub use train_opt::{add_grads, average_grads, scale_grads, TrainOptConfig};
