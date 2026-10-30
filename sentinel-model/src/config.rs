//! Engine configuration (DeepInfer `EngineArgs` / scheduler config subset for Seek-V).

use sentinel_core::scheduler::{EngineType, SchedulerConfig};

/// How to size the KV block pool and scheduler for a model.
#[derive(Debug, Clone)]
pub struct EngineArgs {
    pub block_size: usize,
    pub num_gpu_blocks: Option<usize>,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_model_len: Option<usize>,
    pub chunked_prefill: bool,
    pub stop_preempted_request: bool,
    pub log_stats: bool,
}

impl Default for EngineArgs {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_gpu_blocks: None,
            max_num_seqs: 64,
            max_num_batched_tokens: 2048,
            max_model_len: None,
            chunked_prefill: true,
            stop_preempted_request: false,
            log_stats: false,
        }
    }
}

impl EngineArgs {
    pub fn scheduler_config(&self, model_max_positions: usize, eos: i32, vocab: i32) -> SchedulerConfig {
        let max_model_len = self.max_model_len.unwrap_or(model_max_positions).min(model_max_positions);
        SchedulerConfig {
            block_size: self.block_size,
            eos_token_id: eos,
            max_model_len,
            runtime_max_model_len: max_model_len,
            max_num_batched_tokens: self.max_num_batched_tokens,
            max_num_seqs: self.max_num_seqs,
            stop_preempted_request: self.stop_preempted_request,
            chunked_prefill: self.chunked_prefill,
            engine_type: EngineType::Normal,
            prefill_attempt_interval: 1,
            prefill_any_attempt_interval: 1,
            lookahead_tokens: 0,
            prefill_max_sum_seqlen_k: max_model_len.max(self.max_num_batched_tokens),
            cp_size: 1,
            nextn: 0,
            vocab_size: vocab,
        }
    }

    pub fn num_blocks(&self, model_max_positions: usize) -> usize {
        self.num_gpu_blocks.unwrap_or_else(|| {
            // Enough for max_num_seqs full sequences + headroom.
            let per = (model_max_positions + self.block_size - 1) / self.block_size + 1;
            (per * self.max_num_seqs + 8).max(64)
        })
    }
}
