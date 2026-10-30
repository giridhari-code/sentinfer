//! Request / response types for the inference framework (DeepInfer-style surface).

use crate::engine::{FinishedRequest, GenerationParams};
use sentinel_core::sequence::SamplingScalars;
use sentinel_core::SeqId;

/// High-level sampling parameters (maps onto [`GenerationParams`] + scalars).
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i32,
    pub seed: u64,
    pub ignore_eos: bool,
    /// Stop when any of these token ids is generated (in addition to EOS).
    pub stop_token_ids: Vec<i32>,
    /// Optional stop strings checked on decoded text (caller must provide decode).
    pub stop_strings: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
            seed: 1,
            ignore_eos: false,
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
        }
    }
}

impl SamplingParams {
    pub fn greedy(max_tokens: usize) -> Self {
        Self { max_tokens, temperature: 0.0, ..Default::default() }
    }

    pub fn to_generation_params(&self) -> GenerationParams {
        GenerationParams {
            max_tokens: self.max_tokens,
            ignore_eos: self.ignore_eos,
            sampling: SamplingScalars {
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                seq_seed: self.seed,
                ..Default::default()
            },
        }
    }
}

/// One user-facing generation request.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub request_id: String,
    pub prompt_token_ids: Vec<i32>,
    pub params: SamplingParams,
    /// Optional original prompt text (for stop-string checks / logging).
    pub prompt_text: Option<String>,
}

/// Output of a finished request (DeepInfer `RequestOutput`-style).
#[derive(Debug, Clone, PartialEq)]
pub struct RequestOutput {
    pub request_id: String,
    pub seq_id: SeqId,
    pub prompt_token_ids: Vec<i32>,
    pub output_token_ids: Vec<i32>,
    /// Full text if a tokenizer was used; otherwise empty.
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl RequestOutput {
    pub fn from_finished(
        finished: FinishedRequest,
        prompt_token_ids: Vec<i32>,
        text: String,
    ) -> Self {
        let completion_tokens = finished.output_token_ids.len();
        Self {
            request_id: finished.request_id,
            seq_id: finished.seq_id,
            prompt_token_ids,
            output_token_ids: finished.output_token_ids,
            text,
            finish_reason: finished.finish_reason.to_string(),
            prompt_tokens: finished.prompt_len,
            completion_tokens,
        }
    }
}
