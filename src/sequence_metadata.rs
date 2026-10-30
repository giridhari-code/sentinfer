//! Per-step sequence view handed to the model / sampler (port of C++ `SequenceMetadata`).
//!
//! In C++ this is an object-pool type that borrows into the live `Sequence`. Here we
//! snapshot the fields the engine needs so the scheduler can release locks / finish
//! the step without keeping a mutable borrow of every sequence.

use crate::sequence::{SamplingScalars, SeqId, Sequence, SequenceStatus};

/// Lightweight metadata built for one scheduled step of one sequence.
#[derive(Debug, Clone)]
pub struct SequenceMetadata {
    pub request_id: String,
    pub seq_id: SeqId,
    pub is_placeholder: bool,
    pub is_prompt: bool,
    pub seqlen_q: usize,
    pub seqlen_k: usize,
    pub block_table: Vec<i32>,
    pub token_ids: Vec<i32>,
    pub compressed_token_ids: Vec<i32>,
    pub sampling: SamplingScalars,
    pub golden_token: Option<Vec<i64>>,
    pub swa_valid_start: usize,
    pub state_id: Option<i32>,
    pub last_output_token: i32,
    pub mtp_info_read_slot: usize,
    pub mtp_info_write_slot: usize,
    pub accepted_tokens: Vec<i32>,
    pub draft_tokens: Vec<i32>,
    pub so_status: i32,
    /// True when this prefill chunk reaches the end of the prompt.
    pub is_last_prompt_chunk: bool,
}

impl SequenceMetadata {
    /// Build metadata for a scheduled sequence.
    ///
    /// * `seqlen_q` — tokens this step will process
    /// * `is_prompt` — prefill vs decode
    /// * `nextn` — MTP draft budget (for golden tokens)
    pub fn from_sequence(seq: &Sequence, seqlen_q: usize, is_prompt: bool, nextn: usize) -> Self {
        let seqlen_k = seq.cache.num_tokens + seq.schedule.total_inflight_tokens + seqlen_q;
        let is_last_prompt_chunk = is_prompt && seqlen_k >= seq.len();
        let golden = seq.mtp.next_golden_tokens(seq.get_output_len(), nextn);
        let mut last_output = seq.mtp.last_output_token;
        if is_prompt && !is_last_prompt_chunk {
            last_output = -1;
        }
        Self {
            request_id: seq.request.request_id.clone(),
            seq_id: seq.request.seq_id,
            is_placeholder: false,
            is_prompt,
            seqlen_q,
            seqlen_k,
            block_table: seq.cache.block_table.iter().map(|&b| b as i32).collect(),
            token_ids: seq.token_ids().to_vec(),
            compressed_token_ids: seq.compressed_token_ids().to_vec(),
            sampling: seq.generation.sampling.clone(),
            golden_token: golden,
            swa_valid_start: if is_prompt { seq.cache.swa_valid_start } else { 0 },
            state_id: seq.cache.state_id.map(|b| b as i32),
            last_output_token: last_output,
            mtp_info_read_slot: 0, // filled by refresh_for_sampling
            mtp_info_write_slot: seq.mtp.info_slot,
            accepted_tokens: seq.mtp.accepted_tokens.clone(),
            draft_tokens: seq.mtp.draft_tokens.clone(),
            so_status: seq.structured_output.status,
            is_last_prompt_chunk,
        }
    }

    /// Update sampling-related slots after the sequence has been scheduled
    /// (matches C++ `refresh_for_sampling`).
    pub fn refresh_for_sampling(&mut self, seq: &mut Sequence, nextn: usize) {
        self.last_output_token = seq.mtp.last_output_token;
        self.mtp_info_read_slot = seq.mtp.take_draft_info_slot();
        if seq.mtp.info_slot != 0 {
            self.mtp_info_write_slot = seq.mtp.info_slot;
        }
        self.so_status = seq.structured_output.status;
        self.golden_token = seq.mtp.next_golden_tokens(seq.get_output_len(), nextn);
        self.accepted_tokens = seq.mtp.accepted_tokens.clone();
        self.draft_tokens = seq.mtp.draft_tokens.clone();
    }

    pub fn status_is_active(status: SequenceStatus) -> bool {
        matches!(status, SequenceStatus::Waiting | SequenceStatus::Running | SequenceStatus::Swapped)
    }
}

/// Host-side sampling batch prepared for the CPU/GPU sampler kernels.
///
/// Replaces the torch-tensor `SamplerInputs` from C++ `prepare_sample.cpp` with
/// plain vectors the Rust engine can consume without a tensor runtime.
#[derive(Debug, Clone, Default)]
pub struct SamplerInputs {
    pub selected_token_indices: Vec<i32>,
    pub selected_prompt_indices: Vec<i32>,
    pub temperatures: Vec<f32>,
    pub top_ps: Vec<f32>,
    pub top_ks: Vec<i32>,
    pub seq_seeds: Vec<u64>,
    pub sample_positions: Vec<i32>,
    pub is_prompt: bool,
    pub max_top_logprobs: u32,
    /// Prefill-only: prompt tokens for logprob rows.
    pub prompt_logprob_tokens: Vec<i32>,
    pub prompt_logprob_lens: Vec<usize>,
    pub prompt_logprob_enabled: Vec<bool>,
    /// Prefill-only: rows where EOS must be suppressed.
    pub eos_suppression_rows: Vec<i32>,
    pub eos_token_ids: Vec<i32>,
    /// Decode / MTP.
    pub mtp_first_token_positions: Vec<i32>,
    pub mtp_seqlens_q: Vec<i32>,
    pub mtp_info_read_rows: Vec<i32>,
    pub mtp_info_write_slots: Vec<i32>,
    pub nextn_tokens: usize,
}

/// Build host-side sampler inputs from a batch of sequence metadata.
///
/// Builds host-side sampler inputs without a tensor runtime.
pub fn prepare_sampler_inputs(
    seqs: &[SequenceMetadata],
    max_mtp_tokens: usize,
) -> Result<SamplerInputs, String> {
    if seqs.is_empty() {
        return Err("Sampling requires at least one sequence.".into());
    }
    let is_prompt = seqs[0].is_prompt;
    let nextn = if is_prompt { 0 } else { max_mtp_tokens };

    let mut total_batch_size = 0usize;
    let mut prompt_tokens_size = 0usize;
    let mut num_eos_suppressions = 0usize;
    for m in seqs {
        if m.is_prompt != is_prompt {
            return Err("Mixed prefill and decode batches are unsupported.".into());
        }
        if m.seqlen_q == 0 {
            return Err(format!(
                "Invalid sequence lengths: seqlen_q={}, seqlen_k={}",
                m.seqlen_q, m.seqlen_k
            ));
        }
        if m.seqlen_k < m.seqlen_q {
            return Err(format!(
                "Invalid sequence lengths: seqlen_q={}, seqlen_k={}",
                m.seqlen_q, m.seqlen_k
            ));
        }
        if is_prompt {
            total_batch_size += 1;
            if m.sampling.need_prompt_logprobs {
                let n = m.seqlen_q.saturating_sub(usize::from(m.is_last_prompt_chunk));
                prompt_tokens_size += n;
            }
            if m.sampling.first_n_token_no_eos == 1 && m.is_last_prompt_chunk {
                num_eos_suppressions += 1;
            }
        } else {
            total_batch_size += m.seqlen_q;
        }
    }

    let num_sampling_positions = if is_prompt { total_batch_size } else { seqs.len() };
    let mut out = SamplerInputs {
        selected_token_indices: vec![0; total_batch_size],
        selected_prompt_indices: Vec::with_capacity(prompt_tokens_size),
        temperatures: vec![0.0; num_sampling_positions],
        top_ps: vec![0.0; num_sampling_positions],
        top_ks: vec![0; num_sampling_positions],
        seq_seeds: vec![0; num_sampling_positions],
        sample_positions: vec![0; num_sampling_positions],
        is_prompt,
        max_top_logprobs: 0,
        prompt_logprob_tokens: Vec::with_capacity(prompt_tokens_size),
        prompt_logprob_lens: Vec::with_capacity(seqs.len()),
        prompt_logprob_enabled: Vec::with_capacity(seqs.len()),
        eos_suppression_rows: Vec::with_capacity(num_eos_suppressions),
        eos_token_ids: Vec::with_capacity(num_eos_suppressions),
        mtp_first_token_positions: Vec::new(),
        mtp_seqlens_q: Vec::new(),
        mtp_info_read_rows: Vec::new(),
        mtp_info_write_slots: Vec::new(),
        nextn_tokens: nextn,
    };

    let mut token_offset = 0i32;
    let mut sample_i = 0usize;
    for (batch_i, m) in seqs.iter().enumerate() {
        out.max_top_logprobs = out.max_top_logprobs.max(m.sampling.top_logprobs);

        if is_prompt {
            // One sample position per sequence (last prompt token of the chunk).
            out.temperatures[sample_i] = m.sampling.temperature as f32;
            out.top_ps[sample_i] = m.sampling.top_p as f32;
            out.top_ks[sample_i] = m.sampling.top_k;
            out.seq_seeds[sample_i] = m.sampling.seq_seed;
            out.sample_positions[sample_i] = sample_i as i32;
            out.selected_token_indices[sample_i] = token_offset + m.seqlen_q as i32 - 1;

            let n_logprobs = if m.sampling.need_prompt_logprobs {
                m.seqlen_q.saturating_sub(usize::from(m.is_last_prompt_chunk))
            } else {
                0
            };
            out.prompt_logprob_lens.push(n_logprobs);
            out.prompt_logprob_enabled.push(m.sampling.need_prompt_logprobs);
            if n_logprobs > 0 {
                let start = m.seqlen_k - m.seqlen_q;
                for j in 0..n_logprobs {
                    let abs = start + j;
                    out.selected_prompt_indices.push(token_offset + j as i32);
                    let tok = m.token_ids.get(abs + 1).copied().unwrap_or(0);
                    out.prompt_logprob_tokens.push(tok);
                }
            }
            if m.sampling.first_n_token_no_eos == 1 && m.is_last_prompt_chunk {
                out.eos_suppression_rows.push(sample_i as i32);
                out.eos_token_ids.push(m.sampling.eos_id as i32);
            }
            token_offset += m.seqlen_q as i32;
            sample_i += 1;
        } else {
            // Decode: seqlen_q sample rows per sequence, but one temperature/seed per seq.
            out.temperatures[batch_i] = m.sampling.temperature as f32;
            out.top_ps[batch_i] = m.sampling.top_p as f32;
            out.top_ks[batch_i] = m.sampling.top_k;
            out.seq_seeds[batch_i] = m.sampling.seq_seed;
            out.sample_positions[batch_i] = batch_i as i32;
            for j in 0..m.seqlen_q {
                out.selected_token_indices[token_offset as usize + j] = token_offset + j as i32;
            }
            if nextn > 0 {
                out.mtp_seqlens_q.push(m.seqlen_q as i32);
                out.mtp_first_token_positions.push(token_offset);
                out.mtp_info_read_rows.push(m.mtp_info_read_slot as i32);
                out.mtp_info_write_slots.push(m.mtp_info_write_slot as i32);
            }
            token_offset += m.seqlen_q as i32;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::Sequence;

    fn make_seq(id: i64, prompt_len: usize) -> Sequence {
        Sequence::simple(id, &format!("req-{id}"), vec![7; prompt_len], 8, 16).unwrap()
    }

    #[test]
    fn prefill_metadata_marks_last_chunk() {
        let seq = make_seq(1, 10);
        let m = SequenceMetadata::from_sequence(&seq, 10, true, 0);
        assert!(m.is_last_prompt_chunk);
        assert_eq!(m.seqlen_q, 10);
        assert_eq!(m.seqlen_k, 10);
    }

    #[test]
    fn prepare_prefill_sampler_inputs() {
        let seq = make_seq(1, 8);
        let m = SequenceMetadata::from_sequence(&seq, 8, true, 0);
        let inputs = prepare_sampler_inputs(&[m], 0).unwrap();
        assert!(inputs.is_prompt);
        assert_eq!(inputs.selected_token_indices.len(), 1);
        assert_eq!(inputs.temperatures.len(), 1);
        assert_eq!(inputs.selected_token_indices[0], 7);
    }

    #[test]
    fn prepare_decode_sampler_inputs_with_mtp() {
        let mut seq = make_seq(2, 4);
        // Simulate that prefill finished and we are decoding with 2 draft tokens.
        seq.cache.num_tokens = 4;
        seq.schedule.status = SequenceStatus::Running;
        let m = SequenceMetadata::from_sequence(&seq, 3, false, 2);
        let inputs = prepare_sampler_inputs(&[m], 2).unwrap();
        assert!(!inputs.is_prompt);
        assert_eq!(inputs.nextn_tokens, 2);
        assert_eq!(inputs.selected_token_indices.len(), 3);
        assert_eq!(inputs.mtp_seqlens_q, vec![3]);
    }

    #[test]
    fn rejects_mixed_batch() {
        let s1 = make_seq(1, 4);
        let s2 = make_seq(2, 4);
        let m1 = SequenceMetadata::from_sequence(&s1, 4, true, 0);
        let m2 = SequenceMetadata::from_sequence(&s2, 1, false, 0);
        assert!(prepare_sampler_inputs(&[m1, m2], 0).is_err());
    }
}
