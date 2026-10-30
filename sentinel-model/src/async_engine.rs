//! Asynchronous scheduling scaffolding (DeepInfer-style).
//!
//! Pipeline: schedule → forward → sample/commit, with optional lookahead so the
//! next schedule can be prepared after the previous commit. True GPU overlap
//! needs non-blocking device streams; this is the host-side structure.

use sentinel_core::{Result, Scheduler, SchedulerConfig, SchedulerOutputs, SeqId, SentinelError};

use crate::engine::{Engine, FinishedRequest, GenerationParams};
use crate::model::{Model, StepInput};
use crate::sampler::sample;

#[derive(Debug)]
struct InflightBatch {
    outputs: SchedulerOutputs,
    work: Vec<(SeqId, Vec<i32>, usize)>,
}

/// Engine wrapper that pipelines schedule with compute.
pub struct PipelinedEngine<M: Model> {
    inner: Engine<M>,
    inflight: Option<InflightBatch>,
    pub enable_lookahead: bool,
}

impl<M: Model> PipelinedEngine<M> {
    pub fn new(cfg: SchedulerConfig, num_blocks: usize, model: M) -> Result<Self> {
        Ok(Self {
            inner: Engine::new(cfg, num_blocks, model)?,
            inflight: None,
            enable_lookahead: true,
        })
    }

    pub fn from_engine(engine: Engine<M>) -> Self {
        Self { inner: engine, inflight: None, enable_lookahead: true }
    }

    pub fn engine(&self) -> &Engine<M> {
        &self.inner
    }

    pub fn engine_mut(&mut self) -> &mut Engine<M> {
        &mut self.inner
    }

    pub fn scheduler(&self) -> &Scheduler {
        self.inner.scheduler()
    }

    pub fn add_request(
        &mut self,
        request_id: &str,
        prompt: Vec<i32>,
        params: GenerationParams,
    ) -> Result<SeqId> {
        self.inner.add_request(request_id, prompt, params)
    }

    pub fn abort(&mut self, request_id: &str) {
        self.inner.abort(request_id);
    }

    /// Synchronous step (drains inflight first).
    pub fn step(&mut self) -> Result<Vec<FinishedRequest>> {
        if self.inflight.is_some() {
            let _ = self.commit_inflight()?;
        }
        self.inner.step()
    }

    /// Pipelined step: commit inflight (if any), then schedule next batch.
    pub fn step_pipelined(&mut self) -> Result<Vec<FinishedRequest>> {
        let mut finished = Vec::new();
        if self.inflight.is_some() {
            finished.extend(self.commit_inflight()?);
        }
        if self.enable_lookahead && self.inner.scheduler().has_unfinished_seqs() {
            let _ = self.schedule_into_inflight()?;
        }
        Ok(finished)
    }

    /// Schedule a batch into inflight without running the model.
    pub fn schedule_into_inflight(&mut self) -> Result<bool> {
        if self.inflight.is_some() {
            return Err(SentinelError::InvalidState(
                "inflight batch already pending; commit it first".into(),
            ));
        }
        let outputs = self.inner.scheduler_mut().schedule()?;

        let mut finished_ids: Vec<SeqId> = outputs.finished_seqs.clone();
        finished_ids.extend_from_slice(&outputs.ignored_seqs);

        // Reap finished/ignored now (no forward needed).
        // Caller of commit will not see them again.
        // We store them on the inflight only if we need to return them from commit;
        // for simplicity, reap here and return empty from schedule.
        for id in finished_ids {
            let _ = self.inner.reap(id)?;
        }

        if outputs.scheduled_seqs.is_empty() {
            return Ok(false);
        }

        let mut work = Vec::with_capacity(outputs.scheduled_seqs.len());
        for (i, &id) in outputs.scheduled_seqs.iter().enumerate() {
            let seq = self
                .inner
                .scheduler()
                .sequence(id)
                .ok_or_else(|| SentinelError::InvalidState(format!("unknown sequence {id}")))?;
            let start = seq.cache.num_tokens;
            let end = start + outputs.seqlens_q[i];
            if end > seq.len() {
                return Err(SentinelError::InvalidState(format!(
                    "sequence {id}: step covers {start}..{end} but only {} tokens",
                    seq.len()
                )));
            }
            work.push((id, seq.token_ids()[start..end].to_vec(), start));
        }

        self.inflight = Some(InflightBatch { outputs, work });
        Ok(true)
    }

    /// Forward + sample + process_outputs for the inflight batch.
    pub fn commit_inflight(&mut self) -> Result<Vec<FinishedRequest>> {
        let InflightBatch { outputs, work } = self
            .inflight
            .take()
            .ok_or_else(|| SentinelError::InvalidState("no inflight batch to commit".into()))?;

        if work.is_empty() {
            return Ok(Vec::new());
        }

        let batch: Vec<StepInput<'_>> = work
            .iter()
            .map(|(id, tokens, start)| StepInput { seq_id: *id, tokens, start_pos: *start })
            .collect();

        let logits = self.inner.model().forward(&batch)?;
        let n = batch.len();
        let vocab = logits.len() / n;
        if logits.len() != n * vocab {
            return Err(SentinelError::InvalidState(format!(
                "logits len {} not divisible by batch {n}",
                logits.len()
            )));
        }

        let mut tokens = Vec::with_capacity(n);
        for (i, (id, _, _)) in work.iter().enumerate() {
            let seq = self
                .inner
                .scheduler()
                .sequence(*id)
                .ok_or_else(|| SentinelError::InvalidState(format!("unknown sequence {id}")))?;
            tokens.push(sample(
                &logits[i * vocab..(i + 1) * vocab],
                &seq.generation.sampling,
                seq.len(),
            )?);
        }

        self.inner.scheduler_mut().process_outputs(&outputs, &[], &tokens, None)?;
        Ok(Vec::new())
    }

    pub fn run_to_completion(&mut self, max_steps: usize) -> Result<Vec<FinishedRequest>> {
        let mut results = Vec::new();
        let mut steps = 0usize;
        while self.inner.scheduler().has_unfinished_seqs() || self.inflight.is_some() {
            if steps >= max_steps {
                return Err(SentinelError::InvalidState(format!(
                    "not finished after {max_steps} pipelined steps"
                )));
            }
            // Prefer pipeline; fall back to sync step if nothing progressed.
            let before = results.len();
            results.extend(self.step_pipelined()?);
            if results.len() == before && self.inflight.is_none() && self.inner.scheduler().has_unfinished_seqs()
            {
                results.extend(self.inner.step()?);
            }
            steps += 1;
        }
        Ok(results)
    }

    pub fn has_unfinished(&self) -> bool {
        self.inner.scheduler().has_unfinished_seqs() || self.inflight.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seekv::{SeekV, SeekVConfig};

    #[test]
    fn pipelined_generates() {
        let model = SeekV::random(SeekVConfig::tiny(), 7).unwrap();
        let mut pe = PipelinedEngine::new(SchedulerConfig::default(), 64, model).unwrap();
        pe.add_request(
            "r",
            vec![1, 2, 3],
            GenerationParams { max_tokens: 4, ignore_eos: true, ..Default::default() },
        )
        .unwrap();
        let done = pe.run_to_completion(200).unwrap();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].output_token_ids.len(), 4);
    }
}
