//! Per-step and cumulative inference stats (subset of DeepInfer `StatsLogger`).

use std::time::Instant;

#[derive(Debug, Clone, Default)]
pub struct StepStats {
    pub step: u64,
    pub prompt_run: bool,
    pub num_scheduled: usize,
    pub num_batched_tokens: usize,
    pub num_finished: usize,
    pub attn_tflops: f64,
    pub total_tflops: f64,
    pub step_ms: f64,
    pub tokens_per_sec: f64,
}

#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub steps: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_finished: u64,
    pub total_aborted: u64,
}

pub struct StatsLogger {
    pub engine: EngineStats,
    last_step: Instant,
    enabled: bool,
}

impl StatsLogger {
    pub fn new(enabled: bool) -> Self {
        Self { engine: EngineStats::default(), last_step: Instant::now(), enabled }
    }

    pub fn begin_step(&mut self) {
        self.last_step = Instant::now();
    }

    pub fn end_step(
        &mut self,
        prompt_run: bool,
        num_scheduled: usize,
        num_batched_tokens: usize,
        num_finished: usize,
        attn_tflops: f64,
        total_tflops: f64,
    ) -> StepStats {
        let step_ms = self.last_step.elapsed().as_secs_f64() * 1000.0;
        self.engine.steps += 1;
        self.engine.total_finished += num_finished as u64;
        if prompt_run {
            self.engine.total_prompt_tokens += num_batched_tokens as u64;
        } else {
            self.engine.total_completion_tokens += num_batched_tokens as u64;
        }
        let tokens_per_sec = if step_ms > 0.0 {
            (num_batched_tokens as f64) / (step_ms / 1000.0)
        } else {
            0.0
        };
        let stats = StepStats {
            step: self.engine.steps,
            prompt_run,
            num_scheduled,
            num_batched_tokens,
            num_finished,
            attn_tflops,
            total_tflops,
            step_ms,
            tokens_per_sec,
        };
        if self.enabled {
            let kind = if prompt_run { "prefill" } else { "decode" };
            eprintln!(
                "[stats] step {} {kind} sched={} tokens={} finished={} {:.1} tok/s {:.2} ms",
                stats.step, stats.num_scheduled, stats.num_batched_tokens, stats.num_finished, stats.tokens_per_sec, stats.step_ms
            );
        }
        stats
    }
}
