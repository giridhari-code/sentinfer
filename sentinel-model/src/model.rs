use sentinel_core::{Result, SeqId};

/// One sequence's share of a model step.
#[derive(Debug, Clone, Copy)]
pub struct StepInput<'a> {
    pub seq_id: SeqId,
    /// Tokens to run through the model: the whole prompt (or a chunk of it) for a prefill step,
    /// the last generated token for a decode step.
    pub tokens: &'a [i32],
    /// Position of `tokens[0]` in the sequence. 0 means "start from scratch" (new request, or
    /// recompute after preemption), so the model must drop any cached state for the sequence.
    pub start_pos: usize,
}

/// What the engine needs from a model.
pub trait Model {
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> i32;
    /// Longest sequence the model can handle.
    fn max_positions(&self) -> usize;
    /// Runs one step for every entry and returns the logits of each entry's last token,
    /// row-major `[batch.len(), vocab_size]`.
    fn forward(&mut self, batch: &[StepInput<'_>]) -> Result<Vec<f32>>;
    /// Drops all cached state of a finished or aborted sequence.
    fn release(&mut self, seq_id: SeqId);
}
