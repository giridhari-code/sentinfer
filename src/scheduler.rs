//! Scheduler core (port of `csrc/engine/scheduler.hpp`).
//!
//! Ownership model differs from C++: the scheduler owns every `Sequence` in a map
//! and queues hold `SeqId`s, instead of `shared_ptr<Sequence>` in every queue.
//! Callers read/modify sequences through `sequence()` / `sequence_mut()` and take
//! finished ones back with `remove_sequence()`.
//!
//! Phase 2d additions: prompted queue + fork state/block cloning,
//! multimodal image-pad chunk adjustment, clone_states / zero_states.
//! Still deferred: data-parallel voting (assumes dp_size == 1), PyO3 bindings.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::block_manager::{allocate_seq, AllocStatus, BlockId, BlockManager};
use crate::error::{ensure, Result, SentinelError};
use crate::flops_estimator::FlopsEstimator;
use crate::mtp_varlen::MtpVarlen;
use crate::sequence::{MtpState, SeqId, Sequence, SequenceStatus};
use crate::sequence_metadata::{prepare_sampler_inputs, SequenceMetadata};

const GPU_CACHE_PREFILL_WATERMARK: f32 = 0.9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineType {
    /// Prefill and decode in the same engine.
    Normal,
    /// Prefill only.
    Prompt,
    /// Decode only.
    Completion,
}

/// Multimodal special-token offsets (vocab_size + offset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MmTokenOffset {
    ImageStart = 0,
    ImagePad = 1,
    Image = 2,
    ImageNewLine = 3,
    ImageEnd = 4,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub block_size: usize,
    pub eos_token_id: i32,
    pub max_model_len: usize,
    pub runtime_max_model_len: usize,
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub stop_preempted_request: bool,
    pub chunked_prefill: bool,
    pub engine_type: EngineType,
    pub prefill_attempt_interval: usize,
    pub prefill_any_attempt_interval: usize,
    pub lookahead_tokens: usize,
    pub prefill_max_sum_seqlen_k: usize,
    pub cp_size: usize,
    /// Max draft length for MTP decoding; 0 disables MTP.
    pub nextn: usize,
    /// Vocabulary size; used to derive multimodal sentinel token ids
    /// (`vocab_size + MmTokenOffset`). Set 0 to disable image-pad chunk adjustment.
    pub vocab_size: i32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            eos_token_id: 2,
            max_model_len: 4096,
            runtime_max_model_len: 8192,
            max_num_batched_tokens: 2048,
            max_num_seqs: 64,
            stop_preempted_request: false,
            chunked_prefill: false,
            engine_type: EngineType::Normal,
            prefill_attempt_interval: 1,
            prefill_any_attempt_interval: 1,
            lookahead_tokens: 0,
            prefill_max_sum_seqlen_k: 8192,
            cp_size: 1,
            nextn: 0,
            vocab_size: 0,
        }
    }
}

impl SchedulerConfig {
    /// Same variable and default (5) as the C++ `get_nextn_from_env`.
    pub fn nextn_from_env() -> usize {
        std::env::var("DEEPINFER_USE_NEXTN_DECODE")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .map(|v| v.max(0) as usize)
            .unwrap_or(5)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SchedulerOutputs {
    pub scheduled_seqs: Vec<SeqId>,
    pub prompt_run: bool,
    pub num_batched_tokens: usize,
    pub finished_seqs: Vec<SeqId>,
    pub ignored_seqs: Vec<SeqId>,
    pub seqlens_q: Vec<usize>,
    pub sum_seqlen_qxk: u64,
    pub max_seqlen_k: usize,
    pub min_seqlen_k: usize,
    pub num_cached_tokens_gpu: usize,
    /// Flat (src, dst) pairs of KV blocks to copy before the step (fork).
    pub clone_blocks: Vec<BlockId>,
    /// Flat (src, dst) pairs of state slots to copy before the step (fork).
    pub clone_states: Vec<BlockId>,
    /// State slots that must be zeroed before the step.
    pub zero_states: Vec<BlockId>,
    /// Scheduling-time attention TFLOPs estimate (0 until FLOPs estimator is wired).
    pub attn_tflops: f64,
    /// Scheduling-time total TFLOPs estimate (0 until FLOPs estimator is wired).
    pub total_tflops: f64,
}

impl SchedulerOutputs {
    pub fn is_empty(&self) -> bool {
        self.scheduled_seqs.is_empty()
    }
}

/// Per-step results of `process_outputs`.
#[derive(Debug, Clone, Default)]
pub struct ProcessedOutputs {
    /// Time from arrival to first token (ms) for sequences that produced their first token.
    pub first_token_latencies: Vec<f64>,
    /// Tokens pushed per scheduled sequence before it finished (0 if skipped or chunked).
    pub pushed_tokens_before_finish: Vec<usize>,
}

pub struct Scheduler {
    cfg: SchedulerConfig,
    prompt_limit: usize,
    cp_align: usize,
    block_manager: BlockManager,
    state_manager: Option<BlockManager>,
    seqs: HashMap<SeqId, Sequence>,
    waiting: VecDeque<SeqId>,
    running: Vec<SeqId>,
    /// Sequences that finished prefill on a prompt-only engine and are waiting to
    /// be admitted into the completion engine's running queue. The bool is `true`
    /// when the sequence is a fork branch that still needs state/block cloning.
    prompted: VecDeque<(SeqId, bool)>,
    free_mtp_info_slots: Vec<usize>,
    decode_step: usize,
    schedule_counter: u64,
    num_preempted: u64,
    num_aborted: u64,
    tokens_chunked_prefill_remaining: usize,
    last_step_num_batched_tokens: usize,
    clone_blocks: Vec<BlockId>,
    clone_states: Vec<BlockId>,
    zero_states: Vec<BlockId>,
    mm_token_image: i32,
    mm_token_image_pad: i32,
    mtp_varlen: MtpVarlen,
    engram_lookup_table: Vec<i32>,
    flops_estimator: Option<FlopsEstimator>,
}

fn now_ms() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0)
}

fn unknown_seq(id: SeqId) -> SentinelError {
    SentinelError::InvalidState(format!("unknown sequence {id}"))
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig, block_manager: BlockManager, state_manager: Option<BlockManager>) -> Result<Self> {
        ensure!(cfg.block_size > 0, SentinelError::Config, "block_size must be > 0");
        ensure!(
            cfg.engine_type != EngineType::Completion || cfg.stop_preempted_request,
            SentinelError::Config,
            "COMPLETION engine cannot recompute preempted requests (no prefill channel); \
             it must be deployed with stop_preempted_request=true"
        );
        ensure!(cfg.prefill_attempt_interval >= 1, SentinelError::Config, "prefill_attempt_interval must be >= 1");
        ensure!(
            cfg.prefill_any_attempt_interval >= 1,
            SentinelError::Config,
            "prefill_any_attempt_interval must be >= 1"
        );
        ensure!(
            cfg.max_num_batched_tokens >= cfg.block_size,
            SentinelError::Config,
            "max_num_batched_tokens ({}) must be >= block_size ({})",
            cfg.max_num_batched_tokens,
            cfg.block_size
        );
        let prompt_limit =
            if cfg.chunked_prefill { cfg.max_model_len } else { cfg.max_model_len.min(cfg.max_num_batched_tokens) };
        ensure!(
            cfg.prefill_max_sum_seqlen_k >= prompt_limit,
            SentinelError::Config,
            "prefill_max_sum_seqlen_k={} too small, prompt_limit={}",
            cfg.prefill_max_sum_seqlen_k,
            prompt_limit
        );
        let cp_align = if cfg.cp_size > 1 { cfg.cp_size * 2 } else { 1 };
        // Slot 0 is reserved for padding; real requests use [1, max_num_seqs].
        let free_mtp_info_slots: Vec<usize> = (1..=cfg.max_num_seqs).rev().collect();
        let (mm_token_image, mm_token_image_pad) = if cfg.vocab_size > 0 {
            (
                cfg.vocab_size + MmTokenOffset::Image as i32,
                cfg.vocab_size + MmTokenOffset::ImagePad as i32,
            )
        } else {
            // Disabled: no token will ever match these sentinel ids.
            (i32::MIN, i32::MIN)
        };
        Ok(Self {
            prompt_limit,
            cp_align,
            block_manager,
            state_manager,
            seqs: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::with_capacity(cfg.max_num_seqs),
            prompted: VecDeque::new(),
            free_mtp_info_slots,
            decode_step: 0,
            schedule_counter: 0,
            num_preempted: 0,
            num_aborted: 0,
            tokens_chunked_prefill_remaining: 0,
            last_step_num_batched_tokens: 0,
            clone_blocks: Vec::new(),
            clone_states: Vec::new(),
            zero_states: Vec::new(),
            mm_token_image,
            mm_token_image_pad,
            mtp_varlen: MtpVarlen::default(),
            engram_lookup_table: Vec::new(),
            flops_estimator: None,
            cfg,
        })
    }

    // ---- sequence access -------------------------------------------------

    pub fn add_sequence(&mut self, mut seq: Sequence) -> Result<SeqId> {
        let id = seq.request.seq_id;
        ensure!(!self.seqs.contains_key(&id), SentinelError::InvalidState, "duplicate seq_id {id}");
        seq.request.arrival_time = now_ms();
        if self.cfg.engine_type == EngineType::Normal {
            seq.reserve_output_tokens();
            let cap = self.block_table_capacity(&seq);
            seq.cache.block_table.reserve(cap);
        }
        self.seqs.insert(id, seq);
        self.waiting.push_back(id);
        Ok(id)
    }

    pub fn sequence(&self, id: SeqId) -> Option<&Sequence> {
        self.seqs.get(&id)
    }

    pub fn sequence_mut(&mut self, id: SeqId) -> Option<&mut Sequence> {
        self.seqs.get_mut(&id)
    }

    /// Takes a sequence out of the scheduler. Only call this for sequences that are
    /// no longer in the waiting/running queues (e.g. ones reported in `finished_seqs`).
    pub fn remove_sequence(&mut self, id: SeqId) -> Option<Sequence> {
        self.seqs.remove(&id)
    }

    /// Marks matching unfinished sequences as aborted; they are reaped by the next schedule.
    pub fn abort_sequence(&mut self, request_ids: &[String]) -> Vec<SeqId> {
        let mut aborted = Vec::new();
        for request_id in request_ids {
            // Failed multimodal requests can sit in `waiting` with schedulable=false,
            // so both queues are scanned.
            for id in self.running.iter().chain(self.waiting.iter()) {
                if let Some(seq) = self.seqs.get_mut(id) {
                    if &seq.request.request_id == request_id && !seq.is_finished() {
                        seq.schedule.status = SequenceStatus::FinishedAborted;
                        self.num_aborted += 1;
                        aborted.push(*id);
                    }
                }
            }
        }
        aborted
    }

    pub fn set_schedulable(&mut self, request_id: &str) {
        for id in &self.waiting {
            if let Some(seq) = self.seqs.get_mut(id) {
                if seq.request.request_id == request_id {
                    seq.schedule.schedulable = true;
                }
            }
        }
    }

    /// Moves a sequence whose prefill finished into the running queue.
    /// In C++ this happens in the output-processing path (ported in phase 2).
    pub fn admit_running(&mut self, id: SeqId) -> Result<()> {
        let seq = self.seqs.get(&id).ok_or_else(|| unknown_seq(id))?;
        ensure!(
            seq.schedule.status == SequenceStatus::Running,
            SentinelError::InvalidState,
            "cannot admit sequence {id} with status {:?}",
            seq.schedule.status
        );
        self.running.push(id);
        Ok(())
    }

    pub fn has_unfinished_seqs(&self) -> bool {
        self.num_unfinished_seqs() > 0
    }

    pub fn num_unfinished_seqs(&self) -> usize {
        self.waiting.len() + self.running.len() + self.prompted.len()
    }

    /// Number of sequences waiting to be admitted from the prompted queue.
    pub fn num_prompted(&self) -> usize {
        self.prompted.len()
    }

    /// Enqueue a sequence that has finished prefill on a prompt engine into the
    /// completion engine's prompted queue. `forked` requests state/block cloning
    /// on admit (branch ids starting with `_`).
    pub fn enqueue_prompted(&mut self, id: SeqId, forked: bool) -> Result<()> {
        ensure!(self.seqs.contains_key(&id), SentinelError::InvalidState, "unknown sequence {id}");
        self.prompted.push_back((id, forked));
        Ok(())
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn num_preempted(&self) -> u64 {
        self.num_preempted
    }

    pub fn num_aborted(&self) -> u64 {
        self.num_aborted
    }

    pub fn schedule_counter(&self) -> u64 {
        self.schedule_counter
    }

    pub fn last_step_num_batched_tokens(&self) -> usize {
        self.last_step_num_batched_tokens
    }

    pub fn tokens_chunked_prefill_remaining(&self) -> usize {
        self.tokens_chunked_prefill_remaining
    }

    pub fn num_free_gpu_blocks(&self) -> usize {
        self.block_manager.num_free_blocks()
    }

    pub fn gpu_cache_usage(&self, num_gpu_blocks_to_use: usize) -> f32 {
        let total = self.block_manager.num_blocks();
        let used = total - self.block_manager.num_free_blocks() + num_gpu_blocks_to_use;
        used as f32 / total as f32
    }

    // ---- internal helpers --------------------------------------------------

    fn block_table_capacity(&self, seq: &Sequence) -> usize {
        (seq.len() + seq.generation.stopping.max_tokens + self.cfg.block_size - 1) / self.cfg.block_size + 1
    }

    /// Block + state joint check: if no state slot can be opened, treat blocks as LATER too.
    fn can_allocate_with_state(&self, num_blocks: usize) -> AllocStatus {
        let status = self.block_manager.can_allocate(num_blocks);
        if status == AllocStatus::Ok {
            if let Some(sm) = &self.state_manager {
                if sm.can_allocate(1) == AllocStatus::Later {
                    return AllocStatus::Later;
                }
            }
        }
        status
    }

    /// Clone the state slot for a forked sequence: allocate a new state, record
    /// (old, new) in `clone_states`, and swap the sequence's state_id.
    fn clone_state(&mut self, seq: &mut Sequence) -> Result<()> {
        let Some(sm) = self.state_manager.as_mut() else {
            return Ok(());
        };
        let old = seq.cache.state_id.ok_or_else(|| {
            SentinelError::InvalidState("forked sequence has no state_id to clone".into())
        })?;
        let new_state = sm.allocate_single()?;
        self.clone_states.push(old);
        self.clone_states.push(new_state);
        seq.cache.state_id = Some(new_state);
        Ok(())
    }

    /// True when position `k_len` falls inside an `IMAGE + IMAGE_PAD*N` image unit.
    /// Boundary at `IMAGE` itself is allowed to split (returns false).
    fn is_inside_image_pad_region(&self, k_len: usize, token_ids: &[i32]) -> bool {
        if k_len >= token_ids.len() || token_ids[k_len] != self.mm_token_image_pad {
            return false;
        }
        for i in (0..k_len).rev() {
            if token_ids[i] == self.mm_token_image {
                return true;
            }
            if token_ids[i] != self.mm_token_image_pad {
                return false;
            }
        }
        false
    }

    /// Adjust chunk size so multimodal image units are not split mid-image.
    /// Layout in token_ids: `MM_TOKEN_IMAGE` + `MM_TOKEN_IMAGE_PAD * N`.
    fn adjust_chunk_size(
        &self,
        q_len: usize,
        old_k_len: usize,
        token_ids: &[i32],
        make_larger: bool,
    ) -> usize {
        let mut k_len = q_len + old_k_len;
        if k_len >= token_ids.len() {
            return q_len;
        }
        if !self.is_inside_image_pad_region(k_len, token_ids) {
            return q_len;
        }
        if make_larger {
            while k_len < token_ids.len() && token_ids[k_len] == self.mm_token_image_pad {
                k_len += 1;
            }
        } else {
            while k_len > old_k_len && token_ids[k_len - 1] == self.mm_token_image_pad {
                k_len -= 1;
            }
            if k_len > old_k_len && token_ids[k_len - 1] == self.mm_token_image {
                k_len -= 1;
            }
        }
        k_len.saturating_sub(old_k_len)
    }

    fn has_prefill_waiting(&self) -> bool {
        self.waiting.front().and_then(|id| self.seqs.get(id)).map(|s| s.schedule.schedulable).unwrap_or(false)
    }

    /// Temporarily takes `id` out of the map so the closure can use `&mut Self` and
    /// `&mut Sequence` together. The sequence is always put back, even on error.
    fn with_seq<R>(&mut self, id: SeqId, f: impl FnOnce(&mut Self, &mut Sequence) -> Result<R>) -> Result<R> {
        let mut seq = self.seqs.remove(&id).ok_or_else(|| unknown_seq(id))?;
        let result = f(self, &mut seq);
        self.seqs.insert(id, seq);
        result
    }

    fn allocate_mtp_info_slot(&mut self, seq: &mut Sequence) -> Result<()> {
        if self.cfg.nextn == 0 || seq.mtp.info_slot != 0 {
            return Ok(());
        }
        let slot = self.free_mtp_info_slots.pop();
        let Some(slot) = slot else {
            return Err(SentinelError::InvalidState(format!(
                "no free MTP info slot for request {}",
                seq.request.request_id
            )));
        };
        seq.mtp.info_slot = slot;
        seq.mtp.has_draft_info = false;
        Ok(())
    }

    fn release_mtp_info_slot(&mut self, seq: &mut Sequence) -> Result<()> {
        if seq.mtp.info_slot == 0 {
            return Ok(());
        }
        ensure!(
            seq.mtp.info_slot <= self.cfg.max_num_seqs,
            SentinelError::InvalidState,
            "MTP info slot {} out of range",
            seq.mtp.info_slot
        );
        self.free_mtp_info_slots.push(seq.mtp.info_slot);
        seq.mtp.clear_info_slot();
        Ok(())
    }

    fn num_used_mtp_info_slots(&self) -> usize {
        self.cfg.max_num_seqs - self.free_mtp_info_slots.len()
    }

    /// Releases MTP slot, state slot and KV blocks of `seq`.
    /// Unlike C++, `state_id` is reset to `None` so a recompute cannot double-free it.
    fn free_seq_inner(&mut self, seq: &mut Sequence) -> Result<()> {
        self.release_mtp_info_slot(seq)?;
        if let Some(state_id) = seq.cache.state_id.take() {
            let sm = self
                .state_manager
                .as_mut()
                .ok_or_else(|| SentinelError::InvalidState("sequence has a state slot but no state manager".into()))?;
            sm.free_blocks(&[state_id])?;
        }
        self.block_manager.free_blocks(&seq.cache.block_table)?;
        seq.cache.block_table.clear();
        Ok(())
    }

    pub fn free_seq(&mut self, id: SeqId) -> Result<()> {
        self.with_seq(id, |this, seq| this.free_seq_inner(seq))
    }

    fn free_finished_seqs(&mut self) -> Result<Vec<SeqId>> {
        let running = std::mem::take(&mut self.running);
        let mut keep = Vec::with_capacity(running.len());
        let mut finished = Vec::new();
        for id in running {
            if self.seqs.get(&id).ok_or_else(|| unknown_seq(id))?.is_finished() {
                finished.push(id);
            } else {
                keep.push(id);
            }
        }
        self.running = keep;
        for &id in &finished {
            self.with_seq(id, |this, seq| this.release_mtp_info_slot(seq))?;
        }
        Ok(finished)
    }

    fn preempt(&mut self, id: SeqId, ignored: &mut Vec<SeqId>) -> Result<()> {
        self.with_seq(id, |this, seq| {
            ensure!(
                seq.schedule.status == SequenceStatus::Running,
                SentinelError::InvalidState,
                "preempt: status {:?}",
                seq.schedule.status
            );
            // Structured-output requests cannot be recomputed, so they stop instead.
            let so_blocks_recompute = seq.structured_output.status != 0;
            if this.cfg.stop_preempted_request || so_blocks_recompute {
                eprintln!("[sentinel] request {} is preempted and stops", seq.request.request_id);
                seq.schedule.status = SequenceStatus::FinishedPreempted;
                this.free_seq_inner(seq)?;
                ignored.push(id);
            } else {
                eprintln!("[sentinel] request {} is preempted and will be recomputed", seq.request.request_id);
                seq.reset_for_recompute();
                this.waiting.push_front(id);
                seq.schedule.status = SequenceStatus::Waiting;
                this.free_seq_inner(seq)?;
            }
            this.num_preempted += 1;
            Ok(())
        })
    }

    fn append_slot(&mut self, id: SeqId) -> Result<()> {
        self.with_seq(id, |this, seq| {
            let num_blocks = seq.get_num_blocks();
            if seq.cache.block_table.len() < num_blocks {
                let b = this.block_manager.allocate_single()?;
                seq.cache.block_table.push(b);
            } else if seq.cache.block_table.len() > num_blocks {
                let redundant = seq.cache.block_table.split_off(num_blocks);
                this.block_manager.free_blocks(&redundant)?;
            }
            ensure!(
                seq.cache.block_table.len() == num_blocks,
                SentinelError::InvalidState,
                "len(block_table) = {}, but should be {}. seq_len = {}",
                seq.cache.block_table.len(),
                num_blocks,
                seq.len()
            );
            Ok(())
        })
    }

    fn check_stop_seq(cfg: &SchedulerConfig, seq: &mut Sequence) {
        let Some(token_id) = seq.get_last_token_id() else {
            return;
        };
        if !seq.generation.stopping.ignore_eos && token_id == cfg.eos_token_id {
            seq.schedule.status = SequenceStatus::FinishedStopped;
        } else if seq.len() > cfg.max_model_len || seq.get_output_len() == seq.generation.stopping.max_tokens {
            seq.schedule.status = SequenceStatus::FinishedLengthCapped;
        }
    }

    /// Finish conditions after a token was appended.
    pub fn check_stop(&mut self, id: SeqId) -> Result<()> {
        let seq = self.seqs.get_mut(&id).ok_or_else(|| unknown_seq(id))?;
        Self::check_stop_seq(&self.cfg, seq);
        Ok(())
    }

    // ---- MTP / engram configuration ------------------------------------------------

    pub fn set_decode_perf_curve(&mut self, curve: Vec<(usize, f32)>) -> Result<()> {
        self.mtp_varlen.set_decode_perf_curve(curve)
    }

    /// Feeds measured decode step timings back into the MTP planner.
    pub fn update_mtp_runtime(&mut self, num_seqs: usize, graph_size: usize, forward_ms: f64, sampler_ms: f64) -> Result<()> {
        self.mtp_varlen.update_runtime(num_seqs, graph_size, forward_ms, sampler_ms)
    }

    pub fn set_engram_lookup_table(&mut self, table: Vec<i32>) {
        self.engram_lookup_table = table;
    }

    /// Attach a FLOPs estimator used to fill `attn_tflops` / `total_tflops` on outputs.
    pub fn set_flops_estimator(&mut self, estimator: FlopsEstimator) {
        self.flops_estimator = Some(estimator);
    }

    /// Build per-sequence metadata for the last schedule outputs and prepare sampler inputs.
    pub fn build_sampler_inputs(
        &mut self,
        outputs: &SchedulerOutputs,
    ) -> std::result::Result<(Vec<SequenceMetadata>, crate::sequence_metadata::SamplerInputs), SentinelError> {
        let nextn = self.cfg.nextn;
        let mut metas = Vec::with_capacity(outputs.scheduled_seqs.len());
        for (i, &id) in outputs.scheduled_seqs.iter().enumerate() {
            let seqlen_q = outputs.seqlens_q.get(i).copied().unwrap_or(1);
            let seq = self.seqs.get_mut(&id).ok_or_else(|| unknown_seq(id))?;
            let mut meta = SequenceMetadata::from_sequence(seq, seqlen_q, outputs.prompt_run, nextn);
            meta.refresh_for_sampling(seq, nextn);
            metas.push(meta);
        }
        let inputs = prepare_sampler_inputs(&metas, nextn)
            .map_err(|e| SentinelError::InvalidState(e))?;
        Ok((metas, inputs))
    }

    fn accumulate_flops(
        &self,
        scheduled: &[SeqId],
        seqlens_q: &[usize],
        prompt_run: bool,
    ) -> (f64, f64) {
        let Some(est) = &self.flops_estimator else {
            return (0.0, 0.0);
        };
        let mut attn = 0i64;
        let mut total = 0i64;
        for (i, &id) in scheduled.iter().enumerate() {
            let q = seqlens_q.get(i).copied().unwrap_or(1);
            let Some(seq) = self.seqs.get(&id) else { continue };
            let k = seq.cache.num_tokens + seq.schedule.total_inflight_tokens + q;
            let (a, t) = est.calc_flops(q, k, prompt_run);
            attn += a;
            total += t;
        }
        // Convert raw FLOPs to TFLOPs.
        (attn as f64 / 1e12, total as f64 / 1e12)
    }

    fn map_engram(&self, token_id: i32) -> Result<i32> {
        if self.engram_lookup_table.is_empty() {
            return Ok(token_id);
        }
        usize::try_from(token_id)
            .ok()
            .and_then(|i| self.engram_lookup_table.get(i).copied())
            .ok_or_else(|| SentinelError::InvalidState(format!("token {token_id} outside the engram lookup table")))
    }

    // ---- output processing ---------------------------------------------------------

    /// Applies the results of a model step to the scheduled sequences (port of `process_outputs`).
    ///
    /// * `accepted_token_ids`: flat `[sum(seqlens_q)]`; each request owns `seqlens_q[i]` slots, the
    ///   first `seqlens_q[i] - 1` are draft tokens (-1 = rejected), the last is the main-token slot
    ///   (unused). Pass an empty slice for MTP = 0 or prefill.
    /// * `output_token_ids`: one already-decided token per scheduled sequence.
    /// * `so_status`: optional structured-output status per sequence.
    pub fn process_outputs(
        &mut self,
        outputs: &SchedulerOutputs,
        accepted_token_ids: &[i32],
        output_token_ids: &[i32],
        so_status: Option<&[i32]>,
    ) -> Result<ProcessedOutputs> {
        let ts = now_ms();
        let n = outputs.scheduled_seqs.len();
        ensure!(
            output_token_ids.len() == n && outputs.seqlens_q.len() == n,
            SentinelError::InvalidState,
            "expected one output token and one seqlen_q per scheduled sequence ({n}), got {} and {}",
            output_token_ids.len(),
            outputs.seqlens_q.len()
        );
        if let Some(so) = so_status {
            ensure!(so.len() == n, SentinelError::InvalidState, "so_status has {} entries, expected {n}", so.len());
        }
        let has_accepted = !accepted_token_ids.is_empty();
        if has_accepted {
            let total: usize = outputs.seqlens_q.iter().sum();
            ensure!(
                accepted_token_ids.len() == total,
                SentinelError::InvalidState,
                "accepted_token_ids has {} entries, expected sum(seqlens_q) = {total}",
                accepted_token_ids.len()
            );
        }

        let mut result = ProcessedOutputs {
            first_token_latencies: Vec::new(),
            pushed_tokens_before_finish: Vec::with_capacity(n),
        };
        let mut flat_offset = 0usize;
        for i in 0..n {
            let id = outputs.scheduled_seqs[i];
            let seqlen_q = outputs.seqlens_q[i];
            let accepted = if has_accepted { Some(&accepted_token_ids[flat_offset..flat_offset + seqlen_q]) } else { None };
            if has_accepted {
                flat_offset += seqlen_q;
            }
            let main_token = output_token_ids[i];
            let so = so_status.map(|s| s[i]);
            let prompt_run = outputs.prompt_run;

            let (pushed, latency) = self.with_seq(id, |this, seq| {
                this.process_one(id, seq, prompt_run, seqlen_q, accepted, main_token, so, ts)
            })?;
            result.pushed_tokens_before_finish.push(pushed);
            if let Some(l) = latency {
                result.first_token_latencies.push(l);
            }
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn process_one(
        &mut self,
        id: SeqId,
        seq: &mut Sequence,
        prompt_run: bool,
        seqlen_q: usize,
        accepted: Option<&[i32]>,
        main_token: i32,
        so: Option<i32>,
        ts: f64,
    ) -> Result<(usize, Option<f64>)> {
        seq.schedule.finish_inflight_step();

        // Already aborted while the step was in flight.
        if seq.is_finished() {
            self.release_mtp_info_slot(seq)?;
            if prompt_run {
                self.running.push(id);
            }
            return Ok((0, None));
        }

        // Async scheduling: preempted for recompute while this step was in flight (decode steps
        // only; a prompt step's sequences are not in `running`, so they cannot be victims).
        // `preempt` already reset its counters and put it back in `waiting`: this output is void.
        if !prompt_run && seq.schedule.status == SequenceStatus::Waiting {
            return Ok((0, None));
        }

        seq.cache.advance_optimistically(seqlen_q);
        if let Some(status) = so {
            seq.structured_output.status = status;
        }

        // Invariants after the optimistic advance (`input.tokens` holds committed tokens only):
        //   prompt step: seqlen_q covers the prompt only, so num_tokens <= len; == len means done.
        //   decode step: seqlen_q = 1 + verified drafts, so num_tokens lands in [len, len + nextn];
        //                the surplus is corrected below by the number of tokens actually accepted.
        ensure!(
            seq.cache.num_tokens <= seq.len() + seq.nextn_len(),
            SentinelError::InvalidState,
            "num_tokens ({}) exceeds len + drafts ({} + {})",
            seq.cache.num_tokens,
            seq.len(),
            seq.nextn_len()
        );
        let prefill_finished_or_decode = seq.cache.num_tokens >= seq.len();
        ensure!(
            prompt_run || prefill_finished_or_decode,
            SentinelError::InvalidState,
            "after a decode step num_tokens must be >= len, got {} < {}",
            seq.cache.num_tokens,
            seq.len()
        );

        if !prefill_finished_or_decode {
            // Chunked prefill that has not finished yet: queue it for the next chunk.
            ensure!(
                seq.schedule.status == SequenceStatus::Waiting,
                SentinelError::InvalidState,
                "a partially prefilled sequence must be waiting, got {:?}",
                seq.schedule.status
            );
            self.waiting.push_front(id);
            return Ok((0, None));
        }

        // A prefill engine does not change status: the sequence is handed to a completion engine.
        if self.cfg.engine_type == EngineType::Normal {
            ensure!(
                seq.schedule.status == SequenceStatus::Running,
                SentinelError::InvalidState,
                "status should be Running (set when the prefill was scheduled), got {:?}",
                seq.schedule.status
            );
        }
        if prompt_run {
            self.running.push(id);
        }

        // Verified draft tokens of this step (layout: see `process_outputs`).
        let mut token_ids: Vec<i32> = Vec::new();
        if let Some(acc) = accepted {
            for &t in acc.iter().take(seqlen_q.saturating_sub(1)) {
                if t == -1 {
                    break;
                }
                token_ids.push(t);
            }
        }
        seq.mtp.accepted_tokens = token_ids.clone();
        seq.mtp.last_output_token = main_token;
        token_ids.push(main_token);

        let mut token_cnt = 0usize;
        // Judge SO failure by the ledger, not the per-step report: the sampler returns None for
        // batches without an active SO request, and a sequence already at -1 (e.g. migrated after
        // failing in a prefill engine) must not generate unconstrained.
        if seq.structured_output.status == -1 {
            seq.schedule.status = SequenceStatus::FinishedSoFailed;
            seq.append_token_id(0);
            seq.append_compressed_token_id(0);
            token_cnt += 1;
        } else {
            for t in token_ids {
                seq.append_token_id(t);
                let mapped = self.map_engram(t)?;
                seq.append_compressed_token_id(mapped);
                token_cnt += 1;
                Self::check_stop_seq(&self.cfg, seq);
                if seq.is_finished() {
                    break;
                }
            }
        }
        if seq.is_finished() {
            self.release_mtp_info_slot(seq)?;
        }

        if !prompt_run {
            // Replace the optimistic advance with the accepted count (rejected drafts hold no KV).
            seq.cache.commit_processed_tokens(token_cnt);
        }
        ensure!(
            seq.cache.num_tokens <= seq.len(),
            SentinelError::InvalidState,
            "num_tokens ({}) exceeds len ({})",
            seq.cache.num_tokens,
            seq.len()
        );

        // Prompt steps produce the first token (the prompt never uses drafts).
        let latency = (seq.get_output_len() == 1).then(|| ts - seq.request.arrival_time);
        Ok((token_cnt, latency))
    }

    // ---- scheduling ----------------------------------------------------------

    /// One scheduling step: picks the step type, schedules, and updates per-sequence
    /// bookkeeping (inflight tokens, schedule counters, seqlen statistics).
    pub fn schedule(&mut self) -> Result<SchedulerOutputs> {
        let mut outputs = self.schedule_inner()?;
        ensure!(
            outputs.num_batched_tokens <= self.cfg.max_num_batched_tokens,
            SentinelError::InvalidState,
            "scheduled {} tokens, over max_num_batched_tokens {}",
            outputs.num_batched_tokens,
            self.cfg.max_num_batched_tokens
        );

        self.schedule_counter += 1;
        if outputs.prompt_run {
            self.decode_step = 0;
        } else {
            self.running = outputs.scheduled_seqs.clone();
            self.decode_step += 1;
            if self.cfg.nextn > 0 {
                ensure!(
                    self.num_used_mtp_info_slots() == self.running.len(),
                    SentinelError::InvalidState,
                    "used MTP info slots ({}) must match running queue size ({})",
                    self.num_used_mtp_info_slots(),
                    self.running.len()
                );
            }
        }

        let nextn = self.cfg.nextn;
        let max_seqlen_q = nextn + 1;
        let mut sum_qxk: u64 = 0;
        let mut max_k = 0usize;
        let mut min_k = 0usize;
        for (i, &id) in outputs.scheduled_seqs.iter().enumerate() {
            let seqlen_q = outputs.seqlens_q[i];
            let seq = self.seqs.get_mut(&id).ok_or_else(|| unknown_seq(id))?;
            seq.mtp.schedule_cnt += 1;
            let seqlen_k = seq.cache.num_tokens + seq.schedule.total_inflight_tokens + seqlen_q;
            // Exclusive upper bound of positions touched this step; RoPE-style buffers only
            // have `runtime_max_model_len` rows.
            let position_end =
                if outputs.prompt_run { seqlen_k } else { seqlen_k - seqlen_q + max_seqlen_q + nextn };
            ensure!(
                position_end <= self.cfg.runtime_max_model_len,
                SentinelError::InvalidState,
                "position_end={} exceeds runtime_max_model_len={}: request_id={} len={} num_tokens={} inflight={} \
                 seqlen_q={} prompt_run={}",
                position_end,
                self.cfg.runtime_max_model_len,
                seq.request.request_id,
                seq.len(),
                seq.cache.num_tokens,
                seq.schedule.total_inflight_tokens,
                seqlen_q,
                outputs.prompt_run
            );
            sum_qxk += seqlen_q as u64 * seqlen_k as u64;
            max_k = max_k.max(seqlen_k);
            if min_k == 0 || min_k > seqlen_k {
                min_k = seqlen_k;
            }
            seq.schedule.start_inflight_step(max_seqlen_q);
        }
        outputs.sum_seqlen_qxk = sum_qxk;
        outputs.max_seqlen_k = max_k;
        outputs.min_seqlen_k = min_k;
        self.last_step_num_batched_tokens = outputs.num_batched_tokens;
        Ok(outputs)
    }

    fn schedule_inner(&mut self) -> Result<SchedulerOutputs> {
        self.clone_blocks.clear();
        self.clone_states.clear();
        self.zero_states.clear();
        self.tokens_chunked_prefill_remaining = 0;

        match self.cfg.engine_type {
            // Prefill-only engines always return a result, even an empty one.
            EngineType::Prompt => {
                Ok(self.schedule_prefill()?.unwrap_or_else(|| SchedulerOutputs { prompt_run: true, ..Default::default() }))
            }
            EngineType::Completion => self.schedule_decode(),
            EngineType::Normal => {
                // Decide whether this step is prefill or decode. With dp_size == 1 no
                // cross-rank voting is needed (DP voting is ported in the distributed phase).
                let is_any_point = self.decode_step != 0 && self.decode_step % self.cfg.prefill_any_attempt_interval == 0;
                let is_all_point = self.decode_step % self.cfg.prefill_attempt_interval == 0;
                if !is_any_point && !is_all_point {
                    return self.schedule_decode();
                }
                let want = self.gpu_cache_usage(0) <= GPU_CACHE_PREFILL_WATERMARK && self.has_prefill_waiting();
                if !want {
                    return self.schedule_decode();
                }
                match self.schedule_prefill()? {
                    Some(outputs) => Ok(outputs),
                    // Prefill came back empty (slots full, not enough KV blocks): decode instead,
                    // otherwise decode_step would reset and we could loop forever.
                    None => self.schedule_decode(),
                }
            }
        }
    }

    /// Picks runnable prompts from `waiting`.
    /// For the NORMAL engine, returns `None` when nothing was scheduled or ignored.
    fn schedule_prefill(&mut self) -> Result<Option<SchedulerOutputs>> {
        let mut ignored: Vec<SeqId> = Vec::new();
        let mut scheduled: Vec<SeqId> = Vec::new();
        let mut seqlens_q: Vec<usize> = Vec::new();
        let mut num_curr_seqs = self.running.len();
        let mut num_batched_tokens = 0usize;
        let mut prefill_sum_seqlen_k_remain = self.cfg.prefill_max_sum_seqlen_k;
        let align = self.cp_align;

        while let Some(&id) = self.waiting.front() {
            let (status, schedulable, len, cached, table_empty, is_fork_branch, num_blocks) = {
                let s = self.seqs.get(&id).ok_or_else(|| unknown_seq(id))?;
                (
                    s.schedule.status,
                    s.schedule.schedulable,
                    s.len(),
                    s.cache.num_tokens,
                    s.cache.block_table.is_empty(),
                    s.schedule.branch_id.starts_with('_'),
                    s.get_num_blocks(),
                )
            };

            // Sequences already finished before being scheduled go back with their finish reason.
            if matches!(
                status,
                SequenceStatus::FinishedAborted | SequenceStatus::FinishedVeFailed | SequenceStatus::FinishedSoFailed
            ) {
                ignored.push(id);
                self.waiting.pop_front();
                continue;
            }
            if !schedulable {
                break;
            }
            ensure!(
                cached <= len,
                SentinelError::InvalidState,
                "sequence {id}: cached tokens ({cached}) exceed length ({len})"
            );
            let num_prompt_tokens = len - cached;
            let mut prefix_caching_hit = false;

            if table_empty {
                prefix_caching_hit = true;
                if len > self.prompt_limit {
                    eprintln!(
                        "[sentinel] input prompt ({len} tokens, request {id}) exceeds limit of {}",
                        self.prompt_limit
                    );
                    self.mark_ignored(id)?;
                    ignored.push(id);
                    self.waiting.pop_front();
                    continue;
                }
                match self.can_allocate_with_state(num_blocks) {
                    AllocStatus::Later => break,
                    AllocStatus::Never => {
                        eprintln!(
                            "[sentinel] input prompt ({num_prompt_tokens} tokens) exceeds the capacity of the block manager"
                        );
                        self.mark_ignored(id)?;
                        ignored.push(id);
                        self.waiting.pop_front();
                        continue;
                    }
                    AllocStatus::Ok => {}
                }
            } else if is_fork_branch {
                return Err(SentinelError::Unsupported("forked sequences (branch ids starting with '_')".into()));
            }

            if num_curr_seqs + 1 > self.cfg.max_num_seqs {
                break;
            }
            if prefill_sum_seqlen_k_remain < len {
                eprintln!("[sentinel] prefill_sum_seqlen_k will exceed limit ({})", self.cfg.prefill_max_sum_seqlen_k);
                break;
            }
            prefill_sum_seqlen_k_remain -= len;

            if !prefix_caching_hit {
                ensure!(
                    cached == 0 || cached >= self.cfg.lookahead_tokens,
                    SentinelError::InvalidState,
                    "expected cached == 0 || cached >= lookahead_tokens (cached={cached}, lookahead={})",
                    self.cfg.lookahead_tokens
                );
            }

            let seq_workload = num_prompt_tokens;
            ensure!(
                seq_workload >= self.cfg.lookahead_tokens,
                SentinelError::InvalidState,
                "expected seq_workload >= lookahead_tokens (workload={seq_workload}, lookahead={})",
                self.cfg.lookahead_tokens
            );

            // Truncate the prompt to what still fits in this step (chunked prefill).
            // The remaining budget is aligned down to the context-parallel pad.
            let budget = (self.cfg.max_num_batched_tokens - num_batched_tokens) / align * align;
            let mut seqlen_q = seq_workload;
            if seq_workload > budget {
                seqlen_q = budget;
            }

            // Honor chunk checkpoints (e.g. segment boundaries that must end a chunk).
            {
                let seq = self.seqs.get_mut(&id).ok_or_else(|| unknown_seq(id))?;
                while let Some(&front) = seq.schedule.chunk_checkpoints.front() {
                    if front <= cached {
                        seq.schedule.chunk_checkpoints.pop_front();
                    } else {
                        break;
                    }
                }
                let checkpoint = seq.schedule.chunk_checkpoints.front().copied().unwrap_or(0);
                if cached + seqlen_q >= checkpoint && checkpoint > cached {
                    seqlen_q = checkpoint - cached;
                    seq.schedule.chunk_checkpoints.pop_front();
                }
            }

            // Multimodal: do not split an IMAGE + IMAGE_PAD*N unit mid-image.
            if self.cfg.vocab_size > 0 {
                let token_ids: Vec<i32> = self
                    .seqs
                    .get(&id)
                    .map(|s| s.token_ids().to_vec())
                    .unwrap_or_default();
                let old_k = cached;
                let adjusted = self.adjust_chunk_size(seqlen_q, old_k, &token_ids, scheduled.is_empty());
                if adjusted != seqlen_q {
                    // If shrinking to zero and this is the first sequence, try expanding instead.
                    if adjusted == 0 && scheduled.is_empty() {
                        seqlen_q = self.adjust_chunk_size(seqlen_q, old_k, &token_ids, true);
                    } else {
                        seqlen_q = adjusted;
                    }
                }
            }

            if seqlen_q == 0 {
                ensure!(
                    !scheduled.is_empty(),
                    SentinelError::InvalidState,
                    "a request is waiting but cannot be scheduled: len={len}, cached={cached}, \
                     num_batched_tokens={num_batched_tokens}, max_num_batched_tokens={}, cp_align={align}",
                    self.cfg.max_num_batched_tokens
                );
                break;
            }

            let is_chunked = seqlen_q < seq_workload;
            if is_chunked && !self.cfg.chunked_prefill {
                break;
            }
            if seqlen_q < self.cfg.lookahead_tokens.max(1) {
                break;
            }

            // ---- commit: from here on the sequence is scheduled ----
            num_batched_tokens += (seqlen_q + align - 1) / align * align;

            self.with_seq(id, |this, seq| {
                ensure!(
                    seq.schedule.status == SequenceStatus::Waiting,
                    SentinelError::InvalidState,
                    "prefill: status {:?}",
                    seq.schedule.status
                );
                if seq.cache.block_table.is_empty() {
                    allocate_seq(&mut this.block_manager, this.state_manager.as_mut(), seq)?;
                }
                if !is_chunked {
                    seq.schedule.status = SequenceStatus::Running;
                    this.allocate_mtp_info_slot(seq)?;
                }
                ensure!(
                    seqlen_q + seq.cache.num_tokens <= seq.len(),
                    SentinelError::InvalidState,
                    "expected seqlen_q + cached <= len (seqlen_q={seqlen_q}, cached={}, len={})",
                    seq.cache.num_tokens,
                    seq.len()
                );
                if is_chunked {
                    seq.schedule.chunked_rest_len = seq.len() - (seqlen_q + seq.cache.num_tokens);
                    this.tokens_chunked_prefill_remaining += seq.len();
                } else {
                    seq.schedule.chunked_rest_len = 0;
                }
                Ok(())
            })?;
            if !is_chunked {
                num_curr_seqs += 1;
            }
            scheduled.push(id);
            seqlens_q.push(seqlen_q);
            self.waiting.pop_front();

            if is_chunked && self.cfg.engine_type == EngineType::Normal {
                break;
            }
        }

        if self.cfg.engine_type != EngineType::Normal || !scheduled.is_empty() || !ignored.is_empty() {
            let (attn_tflops, total_tflops) =
                self.accumulate_flops(&scheduled, &seqlens_q, true);
            return Ok(Some(SchedulerOutputs {
                scheduled_seqs: scheduled,
                prompt_run: true,
                num_batched_tokens,
                ignored_seqs: ignored,
                seqlens_q,
                clone_blocks: self.clone_blocks.clone(),
                clone_states: self.clone_states.clone(),
                zero_states: self.zero_states.clone(),
                attn_tflops,
                total_tflops,
                ..Default::default()
            }));
        }
        Ok(None)
    }

    fn mark_ignored(&mut self, id: SeqId) -> Result<()> {
        self.seqs.get_mut(&id).ok_or_else(|| unknown_seq(id))?.schedule.status = SequenceStatus::FinishedIgnored;
        Ok(())
    }

    /// Decode scheduling: admit prompted (with optional fork cloning), then pick
    /// sequences LCFS; when KV blocks run out it preempts the shortest output first.
    fn schedule_decode(&mut self) -> Result<SchedulerOutputs> {
        let mut finished = self.free_finished_seqs()?;

        // Finished sequences stuck in `waiting` (abort / VE failure) are only reaped by the
        // scheduler. Clean them here too, otherwise a rank with only such sequences would
        // report unfinished work forever.
        let mut ignored: Vec<SeqId> = Vec::new();
        let mut keep = VecDeque::with_capacity(self.waiting.len());
        for id in std::mem::take(&mut self.waiting) {
            if self.seqs.get(&id).ok_or_else(|| unknown_seq(id))?.is_finished() {
                ignored.push(id);
            } else {
                keep.push_back(id);
            }
        }
        self.waiting = keep;

        let max_seqs = self.cfg.max_num_seqs;

        // Admit prompted sequences into running (fork branches clone state + last block).
        while !self.prompted.is_empty() && self.running.len() < max_seqs {
            let (id, forked) = *self.prompted.front().unwrap();
            let status = self.seqs.get(&id).ok_or_else(|| unknown_seq(id))?.schedule.status;
            if status == SequenceStatus::FinishedAborted {
                finished.push(id);
                self.prompted.pop_front();
                continue;
            }
            if forked {
                let can_state = self
                    .state_manager
                    .as_ref()
                    .map(|sm| sm.can_allocate_single())
                    .unwrap_or(true);
                if !can_state || !self.block_manager.can_allocate_single() {
                    break;
                }
                self.with_seq(id, |this, seq| {
                    this.clone_state(seq)?;
                    let new_block = this.block_manager.allocate_single()?;
                    let block_size = seq.cache.block_size;
                    let num_blocks = (seq.cache.num_tokens + block_size - 1) / block_size;
                    ensure!(
                        num_blocks <= seq.cache.block_table.len(),
                        SentinelError::InvalidState,
                        "fork: num_blocks={num_blocks} > block_table.len()={}",
                        seq.cache.block_table.len()
                    );
                    let old_last = seq.cache.block_table[num_blocks - 1];
                    this.clone_blocks.push(old_last);
                    this.clone_blocks.push(new_block);
                    for i in num_blocks..seq.cache.block_table.len() {
                        this.clone_blocks.push(seq.cache.block_table[i]);
                        this.clone_blocks.push(0);
                    }
                    seq.cache.block_table[num_blocks - 1] = new_block;
                    seq.cache.block_table.truncate(num_blocks);
                    Ok(())
                })?;
            }
            let status = self.seqs.get(&id).ok_or_else(|| unknown_seq(id))?.schedule.status;
            if status == SequenceStatus::Waiting {
                self.with_seq(id, |this, seq| {
                    seq.schedule.status = SequenceStatus::Running;
                    this.allocate_mtp_info_slot(seq)
                })?;
                self.running.push(id);
            } else {
                ensure!(
                    self.seqs.get(&id).map(|s| s.is_finished()).unwrap_or(false),
                    SentinelError::InvalidState,
                    "prompted sequence {id} has unexpected status {status:?}"
                );
                finished.push(id);
            }
            self.prompted.pop_front();
        }

        // LCFS: longest output first in the vector, so the back is the youngest request.
        let mut running = std::mem::take(&mut self.running);
        running.sort_by_key(|id| std::cmp::Reverse(self.seqs.get(id).map(|s| s.get_output_len()).unwrap_or(0)));

        let mut scheduled: Vec<SeqId> = Vec::with_capacity(max_seqs.min(running.len()));
        let mut l = 0usize;
        let mut r = running.len();
        while l < r && scheduled.len() < max_seqs {
            let id = running[r - 1];
            while !self.block_manager.can_allocate_single() && r - l > 1 {
                let victim = running[l];
                self.preempt(victim, &mut ignored)?;
                l += 1;
            }
            if self.block_manager.can_allocate_single() {
                self.append_slot(id)?;
                scheduled.push(id);
            } else {
                self.preempt(id, &mut ignored)?;
            }
            r -= 1;
        }
        ensure!(
            l == r,
            SentinelError::InvalidState,
            "decode: {} scheduled, {} left unscheduled in running; should not happen",
            scheduled.len(),
            r - l
        );

        scheduled.sort_by_key(|id| std::cmp::Reverse(self.seqs.get(id).map(|s| s.len()).unwrap_or(0)));

        let seqlens_q = {
            let mtp_states: Vec<&MtpState> = scheduled
                .iter()
                .map(|id| self.seqs.get(id).map(|s| &s.mtp).ok_or_else(|| unknown_seq(*id)))
                .collect::<Result<_>>()?;
            self.mtp_varlen.schedule(&mtp_states)?
        };
        let num_batched_tokens: usize = seqlens_q.iter().sum();

        let (attn_tflops, total_tflops) =
            self.accumulate_flops(&scheduled, &seqlens_q, false);
        Ok(SchedulerOutputs {
            scheduled_seqs: scheduled,
            prompt_run: false,
            num_batched_tokens,
            finished_seqs: finished,
            ignored_seqs: ignored,
            seqlens_q,
            clone_blocks: self.clone_blocks.clone(),
            clone_states: self.clone_states.clone(),
            zero_states: self.zero_states.clone(),
            attn_tflops,
            total_tflops,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(cfg: SchedulerConfig, blocks: usize) -> Scheduler {
        Scheduler::new(cfg, BlockManager::new(blocks).unwrap(), None).unwrap()
    }

    fn seq(id: i64, prompt_len: usize, max_tokens: usize) -> Sequence {
        Sequence::simple(id, &format!("req-{id}"), vec![5; prompt_len], max_tokens, 16).unwrap()
    }

    #[test]
    fn prefill_then_decode() {
        let mut s = sched(SchedulerConfig::default(), 64);
        s.add_sequence(seq(1, 10, 8)).unwrap();

        let out = s.schedule().unwrap();
        assert!(out.prompt_run);
        assert_eq!(out.scheduled_seqs, vec![1]);
        assert_eq!(out.seqlens_q, vec![10]);
        assert_eq!(out.num_batched_tokens, 10);
        assert_eq!(s.sequence(1).unwrap().schedule.status, SequenceStatus::Running);
        assert_eq!(s.sequence(1).unwrap().cache.block_table.len(), 1);

        s.admit_running(1).unwrap();
        let out = s.schedule().unwrap();
        assert!(!out.prompt_run);
        assert_eq!(out.scheduled_seqs, vec![1]);
        assert_eq!(s.num_running(), 1);
    }

    #[test]
    fn prompt_larger_than_pool_is_ignored() {
        let mut s = sched(SchedulerConfig::default(), 4);
        s.add_sequence(seq(1, 100, 8)).unwrap();
        let out = s.schedule().unwrap();
        assert!(out.prompt_run);
        assert_eq!(out.ignored_seqs, vec![1]);
        assert_eq!(s.sequence(1).unwrap().schedule.status, SequenceStatus::FinishedIgnored);
        assert!(!s.has_unfinished_seqs());
    }

    #[test]
    fn prompt_over_limit_is_ignored() {
        let mut cfg = SchedulerConfig::default();
        cfg.max_num_batched_tokens = 64;
        cfg.prefill_max_sum_seqlen_k = 4096;
        let mut s = sched(cfg, 64);
        s.add_sequence(seq(1, 100, 8)).unwrap();
        let out = s.schedule().unwrap();
        assert_eq!(out.ignored_seqs, vec![1]);
    }

    #[test]
    fn chunked_prefill_splits_long_prompt() {
        let mut cfg = SchedulerConfig::default();
        cfg.chunked_prefill = true;
        cfg.max_num_batched_tokens = 64;
        cfg.prefill_max_sum_seqlen_k = 4096;
        let mut s = sched(cfg, 64);
        s.add_sequence(seq(1, 100, 8)).unwrap();
        let out = s.schedule().unwrap();
        assert!(out.prompt_run);
        assert_eq!(out.seqlens_q, vec![64]);
        // A chunked sequence is not Running yet and keeps the remainder.
        assert_eq!(s.sequence(1).unwrap().schedule.status, SequenceStatus::Waiting);
        assert_eq!(s.sequence(1).unwrap().schedule.chunked_rest_len, 36);
    }

    #[test]
    fn chunked_prompt_rejected_without_chunked_prefill() {
        // prompt fits prompt_limit (2048) but not the remaining step budget after another prompt
        let mut cfg = SchedulerConfig::default();
        cfg.max_num_batched_tokens = 128;
        cfg.prefill_max_sum_seqlen_k = 4096;
        let mut s = sched(cfg, 64);
        s.add_sequence(seq(1, 100, 8)).unwrap();
        s.add_sequence(seq(2, 100, 8)).unwrap();
        let out = s.schedule().unwrap();
        assert_eq!(out.scheduled_seqs, vec![1]); // 2 would need chunking, so it waits
        assert_eq!(s.num_waiting(), 1);
    }

    #[test]
    fn abort_is_reaped_by_next_schedule() {
        let mut s = sched(SchedulerConfig::default(), 64);
        s.add_sequence(seq(1, 10, 8)).unwrap();
        let aborted = s.abort_sequence(&["req-1".to_string()]);
        assert_eq!(aborted, vec![1]);
        let out = s.schedule().unwrap();
        assert_eq!(out.ignored_seqs, vec![1]);
        assert_eq!(s.num_aborted(), 1);
    }

    #[test]
    fn decode_preempts_and_recomputes_when_blocks_run_out() {
        // 3 usable blocks of 16 tokens; two sequences of 16 tokens each need a 2nd block.
        let mut s = sched(SchedulerConfig::default(), 4);
        s.add_sequence(seq(1, 16, 8)).unwrap();
        s.add_sequence(seq(2, 16, 8)).unwrap();
        s.schedule().unwrap(); // prefill both (1 block each)
        s.admit_running(1).unwrap();
        s.admit_running(2).unwrap();
        // inflight tokens from the prefill step make each need 2 blocks; only 1 is free.
        let out = s.schedule().unwrap();
        assert!(!out.prompt_run);
        assert_eq!(out.scheduled_seqs.len(), 1);
        assert_eq!(s.num_preempted(), 1);
        assert_eq!(s.num_waiting(), 1); // the victim is queued for recompute
    }

    #[test]
    fn preempted_request_stops_when_configured() {
        let mut cfg = SchedulerConfig::default();
        cfg.stop_preempted_request = true;
        let mut s = sched(cfg, 4);
        s.add_sequence(seq(1, 16, 8)).unwrap();
        s.add_sequence(seq(2, 16, 8)).unwrap();
        s.schedule().unwrap();
        s.admit_running(1).unwrap();
        s.admit_running(2).unwrap();
        let out = s.schedule().unwrap();
        assert_eq!(out.ignored_seqs.len(), 1);
        let victim = out.ignored_seqs[0];
        assert_eq!(s.sequence(victim).unwrap().schedule.status, SequenceStatus::FinishedPreempted);
    }

    #[test]
    fn mtp_slots_follow_running_queue() {
        let mut cfg = SchedulerConfig::default();
        cfg.nextn = 2;
        let mut s = sched(cfg, 64);
        s.add_sequence(seq(1, 10, 8)).unwrap();
        s.schedule().unwrap();
        assert_ne!(s.sequence(1).unwrap().mtp.info_slot, 0);
        s.admit_running(1).unwrap();
        let out = s.schedule().unwrap(); // slot count check inside schedule()
        assert_eq!(out.seqlens_q, vec![1]); // no draft tokens yet
    }

    #[test]
    fn finished_sequences_free_their_slots() {
        let mut s = sched(SchedulerConfig::default(), 64);
        s.add_sequence(seq(1, 10, 8)).unwrap();
        s.schedule().unwrap();
        s.admit_running(1).unwrap();
        let free_before = s.num_free_gpu_blocks();
        s.sequence_mut(1).unwrap().schedule.status = SequenceStatus::FinishedStopped;
        let out = s.schedule().unwrap();
        assert_eq!(out.finished_seqs, vec![1]);
        // blocks are released by the caller via free_seq (as in C++, after outputs are processed)
        s.free_seq(1).unwrap();
        assert!(s.num_free_gpu_blocks() > free_before);
    }

    #[test]
    fn check_stop_conditions() {
        let mut s = sched(SchedulerConfig::default(), 64);
        s.add_sequence(seq(1, 3, 2)).unwrap();
        s.sequence_mut(1).unwrap().append_token_id(2); // eos
        s.check_stop(1).unwrap();
        assert_eq!(s.sequence(1).unwrap().schedule.status, SequenceStatus::FinishedStopped);

        s.add_sequence(seq(2, 3, 2)).unwrap();
        s.sequence_mut(2).unwrap().append_token_id(7);
        s.sequence_mut(2).unwrap().append_token_id(8);
        s.check_stop(2).unwrap();
        assert_eq!(s.sequence(2).unwrap().schedule.status, SequenceStatus::FinishedLengthCapped);
    }

    #[test]
    fn config_validation() {
        let mut cfg = SchedulerConfig::default();
        cfg.engine_type = EngineType::Completion;
        assert!(Scheduler::new(cfg, BlockManager::new(8).unwrap(), None).is_err());

        let mut cfg = SchedulerConfig::default();
        cfg.prefill_max_sum_seqlen_k = 10;
        assert!(Scheduler::new(cfg, BlockManager::new(8).unwrap(), None).is_err());
    }

    // ---- process_outputs -------------------------------------------------------

    #[test]
    fn process_prefill_output_moves_sequence_to_running() {
        let mut s = sched(SchedulerConfig::default(), 64);
        s.add_sequence(seq(1, 10, 8)).unwrap();
        let out = s.schedule().unwrap();
        let r = s.process_outputs(&out, &[], &[9], None).unwrap();
        assert_eq!(r.pushed_tokens_before_finish, vec![1]);
        assert_eq!(r.first_token_latencies.len(), 1);
        assert_eq!(s.num_running(), 1);
        let q = s.sequence(1).unwrap();
        assert_eq!(q.len(), 11);
        assert_eq!(q.cache.num_tokens, 10);
        assert_eq!(q.mtp.last_output_token, 9);
    }

    #[test]
    fn process_output_eos_finishes_and_frees_mtp_slot() {
        let mut cfg = SchedulerConfig::default();
        cfg.nextn = 2;
        let mut s = sched(cfg, 64);
        s.add_sequence(seq(1, 10, 8)).unwrap();
        let out = s.schedule().unwrap();
        assert_ne!(s.sequence(1).unwrap().mtp.info_slot, 0);
        s.process_outputs(&out, &[], &[2], None).unwrap();
        let q = s.sequence(1).unwrap();
        assert_eq!(q.schedule.status, SequenceStatus::FinishedStopped);
        assert_eq!(q.mtp.info_slot, 0);
    }

    #[test]
    fn process_output_for_aborted_sequence_is_dropped() {
        let mut s = sched(SchedulerConfig::default(), 64);
        s.add_sequence(seq(1, 10, 8)).unwrap();
        let out = s.schedule().unwrap();
        s.sequence_mut(1).unwrap().schedule.status = SequenceStatus::FinishedAborted;
        let r = s.process_outputs(&out, &[], &[9], None).unwrap();
        assert_eq!(r.pushed_tokens_before_finish, vec![0]);
        assert_eq!(s.sequence(1).unwrap().len(), 10);
        assert_eq!(s.num_running(), 1); // reaped by the next decode schedule
    }

    #[test]
    fn chunked_prefill_requeues_until_prompt_is_done() {
        let mut cfg = SchedulerConfig::default();
        cfg.chunked_prefill = true;
        cfg.max_num_batched_tokens = 64;
        cfg.prefill_max_sum_seqlen_k = 4096;
        let mut s = sched(cfg, 64);
        s.add_sequence(seq(1, 100, 8)).unwrap();

        let out = s.schedule().unwrap();
        assert_eq!(out.seqlens_q, vec![64]);
        let r = s.process_outputs(&out, &[], &[0], None).unwrap();
        assert_eq!(r.pushed_tokens_before_finish, vec![0]);
        assert_eq!(s.num_waiting(), 1);
        assert_eq!(s.sequence(1).unwrap().cache.num_tokens, 64);

        let out = s.schedule().unwrap();
        assert!(out.prompt_run);
        assert_eq!(out.seqlens_q, vec![36]);
        let r = s.process_outputs(&out, &[], &[9], None).unwrap();
        assert_eq!(r.pushed_tokens_before_finish, vec![1]);
        assert_eq!(s.num_running(), 1);
        assert_eq!(s.sequence(1).unwrap().len(), 101);
    }

    #[test]
    fn decode_commits_only_accepted_draft_tokens() {
        let mut cfg = SchedulerConfig::default();
        cfg.nextn = 2;
        let mut s = sched(cfg, 64);
        s.add_sequence(seq(1, 10, 16)).unwrap();
        let out = s.schedule().unwrap();
        s.process_outputs(&out, &[], &[9], None).unwrap();
        s.sequence_mut(1).unwrap().mtp.draft_tokens = vec![4, 5];

        let out = s.schedule().unwrap();
        assert!(!out.prompt_run);
        assert_eq!(out.seqlens_q, vec![3]);
        // first draft accepted, second rejected (-1), last slot is the unused main-token slot
        let r = s.process_outputs(&out, &[4, -1, 0], &[7], None).unwrap();
        assert_eq!(r.pushed_tokens_before_finish, vec![2]);
        let q = s.sequence(1).unwrap();
        assert_eq!(q.mtp.accepted_tokens, vec![4]);
        assert_eq!(q.len(), 13);
        assert_eq!(q.cache.num_tokens, 12);
    }

    #[test]
    fn engram_table_maps_compressed_tokens() {
        let mut s = sched(SchedulerConfig::default(), 64);
        s.set_engram_lookup_table((0..32).map(|i| i + 100).collect());
        s.add_sequence(seq(1, 10, 8)).unwrap();
        let out = s.schedule().unwrap();
        s.process_outputs(&out, &[], &[9], None).unwrap();
        assert_eq!(s.sequence(1).unwrap().compressed_token_ids(), &[109]);
    }

    #[test]
    fn structured_output_failure_finishes_sequence() {
        let mut s = sched(SchedulerConfig::default(), 64);
        s.add_sequence(seq(1, 10, 8)).unwrap();
        let out = s.schedule().unwrap();
        s.process_outputs(&out, &[], &[9], Some(&[-1])).unwrap();
        let q = s.sequence(1).unwrap();
        assert_eq!(q.schedule.status, SequenceStatus::FinishedSoFailed);
        assert_eq!(q.get_last_token_id(), Some(0));
    }

    #[test]
    fn process_outputs_validates_lengths() {
        let mut s = sched(SchedulerConfig::default(), 64);
        s.add_sequence(seq(1, 10, 8)).unwrap();
        let out = s.schedule().unwrap();
        assert!(s.process_outputs(&out, &[], &[9, 9], None).is_err());
        assert!(s.process_outputs(&out, &[1, 2], &[9], None).is_err());
    }
}
