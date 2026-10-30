//! Sentinel core: scheduler, sequences, and block manager for the Seek-V engine.
//!
//! Phase 1–2: KV block manager, sequence state, scheduler (prefill/decode, MTP,
//! prompted/fork, multimodal chunk adjust), FLOPs estimate, SequenceMetadata,
//! host-side prepare_sample. See `PORTING.md`.
//!
//! Phase 3: BufferPool (activation scratch reuse) + BlockManager usage stats.

pub mod block_manager;
pub mod error;
pub mod flops_estimator;
pub mod memory_pool;
pub mod mtp_varlen;
pub mod scheduler;
pub mod sequence;
pub mod sequence_metadata;

pub use block_manager::{AllocStatus, BlockId, BlockManager, BlockPoolStats};
pub use error::{Result, SentinelError};
pub use flops_estimator::{DecoderFlopsConfig, FlopsEstimator};
pub use memory_pool::{BufferPool, PooledBuf};
pub use mtp_varlen::MtpVarlen;
pub use scheduler::{
    EngineType, MmTokenOffset, ProcessedOutputs, Scheduler, SchedulerConfig, SchedulerOutputs,
};
pub use sequence::{SamplingScalars, SeqId, Sequence, SequenceInit, SequenceStatus};
pub use sequence_metadata::{prepare_sampler_inputs, SamplerInputs, SequenceMetadata};
