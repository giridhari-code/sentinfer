//! Multi-Token Prediction / self-speculative decoding helpers.
//!
//! **Propose:** after a committed token, run greedy one-step forwards to build up to
//! `nextn` draft tokens (KV grows; caller must [`Model::truncate`] back to the
//! committed length after proposing so the next real step starts clean).
//!
//! **Verify:** given draft tokens, re-run autoregressive steps from the committed
//! state and accept the longest matching prefix. The first mismatch (or the token
//! after all drafts match) becomes the new `main_token`.

use sentinel_core::{Result, SeqId, SentinelError};

use crate::model::{Model, StepInput};
use crate::sampler::sample;
use sentinel_core::sequence::SamplingScalars;

/// Greedy argmax over a logits row.
pub fn greedy_token(logits: &[f32]) -> i32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as i32
}

/// Propose up to `nextn` draft tokens by greedy self-speculation.
///
/// `last_token` is the most recently **committed** token (already in the sequence).
/// `kv_len` is the model KV length after that token was processed.
/// After return, the model KV may be longer — call `truncate(seq_id, kv_len)` to roll back.
pub fn propose_drafts<M: Model>(
    model: &mut M,
    seq_id: SeqId,
    last_token: i32,
    kv_len: usize,
    nextn: usize,
    vocab: usize,
) -> Result<Vec<i32>> {
    if nextn == 0 {
        return Ok(Vec::new());
    }
    let mut drafts = Vec::with_capacity(nextn);
    let mut tok = last_token;
    let mut pos = kv_len;
    for _ in 0..nextn {
        let batch = [StepInput {
            seq_id,
            tokens: std::slice::from_ref(&tok),
            start_pos: pos,
        }];
        // start_pos = current kv length: append one position
        let logits = model.forward(&batch)?;
        if logits.len() < vocab {
            return Err(SentinelError::InvalidState(format!(
                "mtp propose: expected {vocab} logits, got {}",
                logits.len()
            )));
        }
        let next = greedy_token(&logits[..vocab]);
        drafts.push(next);
        tok = next;
        pos += 1;
    }
    Ok(drafts)
}

/// Result of verifying drafts against the target model.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Accepted draft tokens (prefix of the proposal).
    pub accepted: Vec<i32>,
    /// First rejected prediction, or the next token after all drafts matched.
    pub main_token: i32,
    /// How many drafts matched.
    pub num_accepted: usize,
}

/// Verify `drafts` starting after the committed token at KV length `kv_len`.
///
/// Uses greedy (or `sampling` when temperature is 0) at each step. KV is left at
/// `kv_len + num_accepted` (the accepted drafts are real). Caller should commit those
/// tokens to the sequence; the `main_token` still needs one more forward after commit
/// if you only accept drafts here — typically `main_token` is produced by the last
/// verify step and is the token to append after accepted drafts.
pub fn verify_drafts<M: Model>(
    model: &mut M,
    seq_id: SeqId,
    last_token: i32,
    kv_len: usize,
    drafts: &[i32],
    vocab: usize,
    sampling: &SamplingScalars,
    seq_len_for_seed: usize,
) -> Result<VerifyResult> {
    let mut accepted = Vec::new();
    let mut tok = last_token;
    let mut pos = kv_len;

    for (i, &draft) in drafts.iter().enumerate() {
        let batch = [StepInput {
            seq_id,
            tokens: std::slice::from_ref(&tok),
            start_pos: pos,
        }];
        let logits = model.forward(&batch)?;
        if logits.len() < vocab {
            return Err(SentinelError::InvalidState(format!(
                "mtp verify: expected {vocab} logits, got {}",
                logits.len()
            )));
        }
        let row = &logits[..vocab];
        let pred = if sampling.temperature <= 0.0 {
            greedy_token(row)
        } else {
            sample(row, sampling, seq_len_for_seed + accepted.len())?
        };
        if pred == draft {
            accepted.push(draft);
            tok = draft;
            pos += 1;
        } else {
            // Reject from here; KV already includes this forward's position for `tok`.
            // Model has advanced one step past committed+accepted; the predicted token
            // is the bonus main token. Truncate is NOT done here — engine decides.
            return Ok(VerifyResult {
                num_accepted: accepted.len(),
                accepted,
                main_token: pred,
            });
        }
        let _ = i;
    }

    // All drafts accepted — sample one more token as main.
    let batch = [StepInput {
        seq_id,
        tokens: std::slice::from_ref(&tok),
        start_pos: pos,
    }];
    let logits = model.forward(&batch)?;
    let row = &logits[..vocab];
    let main = if sampling.temperature <= 0.0 {
        greedy_token(row)
    } else {
        sample(row, sampling, seq_len_for_seed + accepted.len())?
    };
    Ok(VerifyResult {
        num_accepted: accepted.len(),
        accepted,
        main_token: main,
    })
}
