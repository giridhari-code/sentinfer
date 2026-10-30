//! Sequence state (port of `csrc/engine/sequence.hpp`).
//!
//! Python-object fields (`py::object` sampling params, multimodal data and the
//! structured-output matcher) are intentionally not carried over here; they come
//! back with the PyO3 bindings in phase 4. Sampling parameters are plain structs.

use std::collections::VecDeque;

use crate::block_manager::BlockId;
use crate::error::{ensure, Result, SentinelError};

pub type SeqId = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum SequenceStatus {
    #[default]
    Waiting,
    Running,
    Swapped,
    // Everything from here on counts as finished (see `is_finished`).
    FinishedStopped,
    FinishedLengthCapped,
    FinishedAborted,
    FinishedIgnored,
    FinishedPreempted,
    FinishedSoFailed,
    FinishedVeFailed,
}

impl SequenceStatus {
    pub fn is_finished(self) -> bool {
        self >= SequenceStatus::FinishedStopped
    }

    pub fn finished_reason(self) -> &'static str {
        match self {
            Self::FinishedStopped => "stop",
            Self::FinishedLengthCapped => "length",
            Self::FinishedAborted => "abort",
            Self::FinishedIgnored => "prompt_length_exceeded",
            Self::FinishedPreempted => "insufficient_system_resource",
            Self::FinishedSoFailed => "xgrammar runtime errors or timeout",
            Self::FinishedVeFailed => "ve_error",
            _ => "",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SequenceData {
    pub token_ids: Vec<i32>,
    pub prompt_len: usize,
}

impl SequenceData {
    pub fn new(prompt: Vec<i32>) -> Self {
        let prompt_len = prompt.len();
        Self { token_ids: prompt, prompt_len }
    }
    pub fn append_token_id(&mut self, token_id: i32) {
        self.token_ids.push(token_id);
    }
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }
    pub fn output_len(&self) -> usize {
        self.len().saturating_sub(self.prompt_len)
    }
    pub fn last_token_id(&self) -> Option<i32> {
        self.token_ids.last().copied()
    }
}

/// Per-step sampler inputs.
#[derive(Debug, Clone, Default)]
pub struct SamplingScalars {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i32,
    pub seq_seed: u64,
    pub top_logprobs: u32,
    pub need_prompt_logprobs: bool,
    pub first_n_token_no_eos: u32,
    pub eos_id: i64,
    pub ignore_eos: bool,
}

#[derive(Debug, Clone, Default)]
pub struct StoppingCriteria {
    pub max_tokens: usize,
    pub ignore_eos: bool,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationState {
    pub sampling: SamplingScalars,
    pub stopping: StoppingCriteria,
    pub prompt_logprobs: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Default)]
pub struct MtpState {
    pub golden_tokens: Option<Vec<i64>>,
    /// Last output token, or -1 when there is none yet.
    pub last_output_token: i32,
    pub accepted_tokens: Vec<i32>,
    pub draft_tokens: Vec<i32>,
    pub accept_rates: Vec<f32>,
    pub prev_accept_rates: Vec<f32>,
    pub confidence_raw: Vec<f32>,
    pub prev_confidence_raw: Vec<f32>,
    /// Slot in the global MTP info pool; 0 means padding / unassigned.
    pub info_slot: usize,
    pub has_draft_info: bool,
    pub process_cnt: u64,
    pub schedule_cnt: u64,
}

impl MtpState {
    /// Keeps `accepted_tokens` (needed to update the SO matcher after recompute)
    /// and `prev_confidence_raw` (metrics only).
    pub fn reset_for_recompute(&mut self) {
        self.draft_tokens.clear();
        self.accept_rates.clear();
        self.prev_accept_rates.clear();
        self.confidence_raw.clear();
        self.has_draft_info = false;
        self.process_cnt = 0;
        self.schedule_cnt = 0;
    }

    pub fn clear_info_slot(&mut self) {
        self.info_slot = 0;
        self.has_draft_info = false;
    }

    pub fn take_draft_info_slot(&mut self) -> usize {
        let slot = if self.has_draft_info { self.info_slot } else { 0 };
        self.has_draft_info = false;
        slot
    }

    pub fn next_golden_tokens(&self, output_len: usize, nextn: usize) -> Option<Vec<i64>> {
        let golden = self.golden_tokens.as_ref()?;
        if output_len >= golden.len() {
            return None;
        }
        Some((output_len..output_len + nextn * 2 + 1).map(|i| golden.get(i).copied().unwrap_or(-1)).collect())
    }
}

#[derive(Debug, Clone, Default)]
pub struct SequenceRequest {
    pub seq_id: SeqId,
    pub request_id: String,
    pub user_id: String,
    pub arrival_time: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SequenceInput {
    pub tokens: SequenceData,
    pub compressed_tokens: SequenceData,
}

#[derive(Debug, Clone, Default)]
pub struct KvCacheState {
    pub block_size: usize,
    pub previous_num_tokens: usize,
    pub num_tokens: usize,
    pub hit_tokens: usize,
    pub swa_valid_start: usize,
    pub block_table: Vec<BlockId>,
    /// State slot id; `None` where the C++ code uses -1.
    pub state_id: Option<BlockId>,
    pub last_copy_state_token_idx: usize,
}

impl KvCacheState {
    pub fn set_num_tokens(&mut self, value: usize) {
        self.num_tokens = value;
        self.previous_num_tokens = value;
    }

    pub fn advance_optimistically(&mut self, token_count: usize) {
        self.previous_num_tokens = self.num_tokens;
        self.num_tokens += token_count;
    }

    pub fn commit_processed_tokens(&mut self, token_count: usize) {
        self.num_tokens = self.previous_num_tokens + token_count;
    }

    pub fn reset_for_recompute(&mut self) {
        self.previous_num_tokens = 0;
        self.num_tokens = 0;
        self.swa_valid_start = 0;
    }
}

/// 0 = not started, 1 = just started (boundary), 2 = started, -1 = aborted.
#[derive(Debug, Clone, Copy, Default)]
pub struct StructuredOutputState {
    pub status: i32,
}

#[derive(Debug, Clone)]
pub struct SequenceScheduleState {
    pub status: SequenceStatus,
    pub schedulable: bool,
    pub branch_id: String,
    pub chunked_rest_len: usize,
    pub chunk_checkpoints: VecDeque<usize>,
    pub total_inflight_tokens: usize,
    pub last_inflight_tokens: usize,
}

impl Default for SequenceScheduleState {
    fn default() -> Self {
        Self {
            status: SequenceStatus::Waiting,
            schedulable: false,
            branch_id: "default".to_string(),
            chunked_rest_len: 0,
            chunk_checkpoints: VecDeque::new(),
            total_inflight_tokens: 0,
            last_inflight_tokens: 0,
        }
    }
}

impl SequenceScheduleState {
    pub fn start_inflight_step(&mut self, token_count: usize) {
        self.total_inflight_tokens += token_count;
        self.last_inflight_tokens = token_count;
    }

    pub fn finish_inflight_step(&mut self) {
        if self.total_inflight_tokens == self.last_inflight_tokens {
            self.total_inflight_tokens = 0;
            self.last_inflight_tokens = 0;
        } else {
            self.total_inflight_tokens = self.last_inflight_tokens;
        }
    }

    pub fn reset_for_recompute(&mut self) {
        self.total_inflight_tokens = 0;
        self.last_inflight_tokens = 0;
    }
}

#[derive(Debug, Clone, Default)]
pub struct SequenceInit {
    pub request: SequenceRequest,
    pub input: SequenceInput,
    pub generation: GenerationState,
    pub cache: KvCacheState,
    pub mtp: MtpState,
    pub structured_output: StructuredOutputState,
    pub schedule: SequenceScheduleState,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub request: SequenceRequest,
    pub input: SequenceInput,
    pub generation: GenerationState,
    pub cache: KvCacheState,
    pub mtp: MtpState,
    pub structured_output: StructuredOutputState,
    pub schedule: SequenceScheduleState,
}

impl Sequence {
    pub fn new(init: SequenceInit) -> Result<Self> {
        ensure!(init.cache.block_size > 0, SentinelError::Config, "cache.block_size must be > 0");
        let mut seq = Self {
            request: init.request,
            input: init.input,
            generation: init.generation,
            cache: init.cache,
            mtp: init.mtp,
            structured_output: init.structured_output,
            schedule: init.schedule,
        };
        seq.mtp.last_output_token = seq.input.tokens.last_token_id().unwrap_or(-1);
        Ok(seq)
    }

    /// Convenience constructor: a plain waiting text sequence.
    pub fn simple(seq_id: SeqId, request_id: &str, prompt: Vec<i32>, max_tokens: usize, block_size: usize) -> Result<Self> {
        let mut init = SequenceInit::default();
        init.request.seq_id = seq_id;
        init.request.request_id = request_id.to_string();
        init.input.tokens = SequenceData::new(prompt);
        init.generation.stopping.max_tokens = max_tokens;
        init.cache.block_size = block_size;
        init.schedule.schedulable = true;
        Self::new(init)
    }

    pub fn reserve_output_tokens(&mut self) {
        let extra = self.generation.stopping.max_tokens;
        self.input.tokens.token_ids.reserve(extra);
        self.input.compressed_tokens.token_ids.reserve(extra);
    }

    pub fn get_num_blocks(&self) -> usize {
        let bs = self.cache.block_size;
        (self.len() + self.schedule.total_inflight_tokens + self.nextn_len() + bs - 1) / bs
    }

    pub fn append_token_id(&mut self, token_id: i32) {
        self.input.tokens.append_token_id(token_id);
    }
    pub fn append_compressed_token_id(&mut self, token_id: i32) {
        self.input.compressed_tokens.append_token_id(token_id);
    }
    pub fn nextn_len(&self) -> usize {
        self.mtp.draft_tokens.len()
    }
    pub fn len(&self) -> usize {
        self.input.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.input.tokens.is_empty()
    }
    pub fn get_prompt_len(&self) -> usize {
        self.input.tokens.prompt_len()
    }
    pub fn get_output_len(&self) -> usize {
        self.input.tokens.output_len()
    }
    pub fn get_last_token_id(&self) -> Option<i32> {
        self.input.tokens.last_token_id()
    }
    pub fn token_ids(&self) -> &[i32] {
        &self.input.tokens.token_ids
    }
    pub fn compressed_token_ids(&self) -> &[i32] {
        &self.input.compressed_tokens.token_ids
    }
    pub fn is_finished(&self) -> bool {
        self.schedule.status.is_finished()
    }

    pub fn reset_for_recompute(&mut self) {
        self.cache.reset_for_recompute();
        self.schedule.reset_for_recompute();
        self.mtp.reset_for_recompute();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_ordering_matches_finished_semantics() {
        assert!(!SequenceStatus::Waiting.is_finished());
        assert!(!SequenceStatus::Running.is_finished());
        assert!(!SequenceStatus::Swapped.is_finished());
        assert!(SequenceStatus::FinishedStopped.is_finished());
        assert!(SequenceStatus::FinishedVeFailed.is_finished());
        assert_eq!(SequenceStatus::FinishedLengthCapped.finished_reason(), "length");
        assert_eq!(SequenceStatus::Running.finished_reason(), "");
    }

    #[test]
    fn num_blocks_counts_inflight_and_drafts() {
        let mut s = Sequence::simple(1, "r", vec![0; 10], 8, 16).unwrap();
        assert_eq!(s.get_num_blocks(), 1);
        s.schedule.start_inflight_step(6);
        assert_eq!(s.get_num_blocks(), 1); // 16 tokens
        s.mtp.draft_tokens = vec![1];
        assert_eq!(s.get_num_blocks(), 2); // 17 tokens
    }

    #[test]
    fn inflight_step_bookkeeping() {
        let mut st = SequenceScheduleState::default();
        st.start_inflight_step(6);
        st.finish_inflight_step();
        assert_eq!((st.total_inflight_tokens, st.last_inflight_tokens), (0, 0));
        st.start_inflight_step(6);
        st.start_inflight_step(6);
        st.finish_inflight_step();
        assert_eq!(st.total_inflight_tokens, 6);
    }

    #[test]
    fn recompute_reset_keeps_accepted_tokens() {
        let mut s = Sequence::simple(1, "r", vec![1, 2, 3], 8, 16).unwrap();
        s.mtp.accepted_tokens = vec![7];
        s.mtp.draft_tokens = vec![9];
        s.cache.num_tokens = 3;
        s.reset_for_recompute();
        assert_eq!(s.cache.num_tokens, 0);
        assert!(s.mtp.draft_tokens.is_empty());
        assert_eq!(s.mtp.accepted_tokens, vec![7]);
    }

    #[test]
    fn golden_tokens_window_pads_with_minus_one() {
        let mut m = MtpState::default();
        m.golden_tokens = Some(vec![10, 11, 12]);
        assert_eq!(m.next_golden_tokens(1, 1), Some(vec![11, 12, -1]));
        assert_eq!(m.next_golden_tokens(3, 1), None);
    }

    #[test]
    fn rejects_zero_block_size() {
        assert!(Sequence::simple(1, "r", vec![1], 1, 0).is_err());
    }
}
