//! High-level inference API (DeepInfer `LLM` / `LLMEngine` surface for Seek-V).
//!
//! ```ignore
//! let mut llm = Llm::from_checkpoint("best.skv", EngineArgs::default())?;
//! let outs = llm.generate(vec!["Hello".into()], SamplingParams::greedy(32))?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use sentinel_core::{Result, SeqId, SentinelError};

use crate::config::EngineArgs;
use crate::engine::{Engine, FinishedRequest};
use crate::request::{CompletionRequest, RequestOutput, SamplingParams};
use crate::quantized::QuantizedSeekV;
use crate::seekv::SeekV;
use sentinel_kernels::QuantScheme;
use crate::stats::{EngineStats, StatsLogger, StepStats};
use crate::tokenizer::{ByteTokenizer, PieceTokenizer, Tokenizer};

/// One streaming event from [`Llm::generate_stream`].
#[derive(Debug, Clone)]
pub struct StreamDelta {
    pub index: usize,
    pub request_id: String,
    pub seq_id: SeqId,
    pub token_ids: Vec<i32>,
    pub text: String,
    pub text_so_far: String,
    pub finished: bool,
    pub finish_reason: Option<String>,
}

/// Full inference service: tokenize → continuous batch → detokenize.
enum EngineKind {
    Fp32(Engine<SeekV>),
    Quant(Engine<QuantizedSeekV>),
}


impl EngineKind {
    fn add_request(
        &mut self,
        request_id: &str,
        prompt: Vec<i32>,
        params: crate::engine::GenerationParams,
    ) -> Result<SeqId> {
        match self {
            Self::Fp32(e) => e.add_request(request_id, prompt, params),
            Self::Quant(e) => e.add_request(request_id, prompt, params),
        }
    }

    fn abort(&mut self, request_id: &str) {
        match self {
            Self::Fp32(e) => e.abort(request_id),
            Self::Quant(e) => e.abort(request_id),
        }
    }

    fn step(&mut self) -> Result<Vec<crate::engine::FinishedRequest>> {
        match self {
            Self::Fp32(e) => e.step(),
            Self::Quant(e) => e.step(),
        }
    }

    fn has_unfinished_seqs(&self) -> bool {
        match self {
            Self::Fp32(e) => e.scheduler().has_unfinished_seqs(),
            Self::Quant(e) => e.scheduler().has_unfinished_seqs(),
        }
    }

    fn num_unfinished_seqs(&self) -> usize {
        match self {
            Self::Fp32(e) => e.scheduler().num_unfinished_seqs(),
            Self::Quant(e) => e.scheduler().num_unfinished_seqs(),
        }
    }

    fn sequence(&self, seq_id: SeqId) -> Option<&sentinel_core::Sequence> {
        match self {
            Self::Fp32(e) => e.scheduler().sequence(seq_id),
            Self::Quant(e) => e.scheduler().sequence(seq_id),
        }
    }
}

/// Full inference service: tokenize → continuous batch → detokenize.
pub struct Llm {
    engine: EngineKind,
    tokenizer: Box<dyn Tokenizer>,
    /// prompt tokens retained until the request finishes (for RequestOutput).
    prompts: HashMap<SeqId, Vec<i32>>,
    prompt_texts: HashMap<SeqId, Option<String>>,
    stats: StatsLogger,
    next_req: u64,
}

impl Llm {
    pub fn new(model: SeekV, args: EngineArgs) -> Result<Self> {
        Self::new_with_tokenizer(model, args, Box::new(ByteTokenizer))
    }

    pub fn new_with_tokenizer(model: SeekV, args: EngineArgs, tokenizer: Box<dyn Tokenizer>) -> Result<Self> {
        let cfg_model = model.config().clone();
        if tokenizer.vocab_size() != cfg_model.vocab_size {
            return Err(SentinelError::InvalidState(format!(
                "tokenizer vocab {} != model vocab {}",
                tokenizer.vocab_size(),
                cfg_model.vocab_size
            )));
        }
        let sched = args.scheduler_config(cfg_model.max_positions, cfg_model.eos_token_id, cfg_model.vocab_size as i32);
        let blocks = args.num_blocks(cfg_model.max_positions);
        let engine = EngineKind::Fp32(Engine::for_seekv(sched, blocks, model)?);
        Ok(Self {
            engine,
            tokenizer,
            prompts: HashMap::new(),
            prompt_texts: HashMap::new(),
            stats: StatsLogger::new(args.log_stats),
            next_req: 1,
        })
    }

    /// Runtime int8/int4 model ([`QuantizedSeekV`]).
    pub fn from_quantized(model: QuantizedSeekV, args: EngineArgs) -> Result<Self> {
        Self::from_quantized_with_tokenizer(model, args, Box::new(ByteTokenizer))
    }

    pub fn from_quantized_with_tokenizer(
        model: QuantizedSeekV,
        args: EngineArgs,
        tokenizer: Box<dyn Tokenizer>,
    ) -> Result<Self> {
        let cfg = model.config().clone();
        if tokenizer.vocab_size() != cfg.vocab_size {
            return Err(SentinelError::InvalidState(format!(
                "tokenizer vocab {} != model vocab {}",
                tokenizer.vocab_size(),
                cfg.vocab_size
            )));
        }
        let sched = args.scheduler_config(cfg.max_positions, cfg.eos_token_id, cfg.vocab_size as i32);
        let blocks = args.num_blocks(cfg.max_positions);
        let engine = EngineKind::Quant(Engine::for_quantized_seekv(sched, blocks, model)?);
        Ok(Self {
            engine,
            tokenizer,
            prompts: HashMap::new(),
            prompt_texts: HashMap::new(),
            stats: StatsLogger::new(args.log_stats),
            next_req: 1,
        })
    }

    /// Load `.skv` then quantize in memory, or load `.skq` packed.
    pub fn from_checkpoint_quant(
        path: impl AsRef<Path>,
        scheme: QuantScheme,
        vocab: Option<&Path>,
        args: EngineArgs,
    ) -> Result<Self> {
        let path = path.as_ref();
        let q = if path.extension().and_then(|e| e.to_str()) == Some("skq") {
            QuantizedSeekV::load_packed(path)?
        } else {
            let m = SeekV::load(path)?;
            QuantizedSeekV::from_fp32(&m, scheme)?
        };
        let tok: Box<dyn Tokenizer> = match vocab {
            Some(v) => Box::new(PieceTokenizer::from_vocab_file(v)?),
            None => Box::new(ByteTokenizer),
        };
        Self::from_quantized_with_tokenizer(q, args, tok)
    }

    pub fn from_checkpoint(path: impl AsRef<Path>, args: EngineArgs) -> Result<Self> {
        let model = SeekV::load(path.as_ref())?;
        Self::new(model, args)
    }

    /// Checkpoint + optional `PieceTokenizer` vocab file (must match model `vocab_size`).
    pub fn from_checkpoint_with_vocab(
        path: impl AsRef<Path>,
        vocab: Option<&Path>,
        args: EngineArgs,
    ) -> Result<Self> {
        let model = SeekV::load(path.as_ref())?;
        let tok: Box<dyn Tokenizer> = match vocab {
            Some(v) => Box::new(PieceTokenizer::from_vocab_file(v)?),
            None => Box::new(ByteTokenizer),
        };
        Self::new_with_tokenizer(model, args, tok)
    }

    pub fn from_random(cfg: crate::seekv::SeekVConfig, seed: u64, args: EngineArgs) -> Result<Self> {
        Self::new(SeekV::random(cfg, seed)?, args)
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.engine, EngineKind::Quant(_))
    }

    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    pub fn stats(&self) -> &EngineStats {
        &self.stats.engine
    }

    /// Add a text prompt (byte-tokenized).
    pub fn add_prompt(&mut self, request_id: &str, prompt: &str, params: SamplingParams) -> Result<SeqId> {
        let ids = self.tokenizer.encode(prompt, false);
        self.add_tokens(request_id, ids, params, Some(prompt.to_string()))
    }

    /// Add a pre-tokenized prompt.
    pub fn add_tokens(
        &mut self,
        request_id: &str,
        prompt_token_ids: Vec<i32>,
        params: SamplingParams,
        prompt_text: Option<String>,
    ) -> Result<SeqId> {
        let gen = params.to_generation_params();
        let seq_id = self.engine.add_request(request_id, prompt_token_ids.clone(), gen)?;
        self.prompts.insert(seq_id, prompt_token_ids);
        self.prompt_texts.insert(seq_id, prompt_text);
        Ok(seq_id)
    }

    pub fn abort(&mut self, request_id: &str) {
        self.engine.abort(request_id);
    }

    pub fn has_unfinished(&self) -> bool {
        self.engine.has_unfinished_seqs()
    }

    pub fn num_unfinished(&self) -> usize {
        self.engine.num_unfinished_seqs()
    }

    /// One scheduling + model step; returns newly finished outputs.
    pub fn step(&mut self) -> Result<(Vec<RequestOutput>, StepStats)> {
        self.stats.begin_step();
        // Peek schedule stats via a step that returns finished only — we need
        // scheduler outputs. Re-implement thin wrapper using engine.step and
        // approximate stats from finished count.
        let before = self.num_unfinished();
        let finished = self.engine.step()?;
        let after = self.num_unfinished();
        let num_finished = finished.len();
        // Rough: if unfinished grew or stayed with finishes, prefer decode accounting in logger.
        let prompt_run = before > 0 && num_finished == 0 && after >= before;
        let outs = self.finish_to_outputs(finished)?;
        let stats = self.stats.end_step(prompt_run, 0, 0, num_finished, 0.0, 0.0);
        Ok((outs, stats))
    }

    fn finish_to_outputs(&mut self, finished: Vec<FinishedRequest>) -> Result<Vec<RequestOutput>> {
        let mut outs = Vec::with_capacity(finished.len());
        for f in finished {
            let prompt = self.prompts.remove(&f.seq_id).unwrap_or_default();
            let _text_prompt = self.prompt_texts.remove(&f.seq_id);
            let text = self.tokenizer.decode(&f.output_token_ids);
            outs.push(RequestOutput::from_finished(f, prompt, text));
        }
        Ok(outs)
    }

    /// Run until the pool is empty (DeepInfer `_run_engine`).
    pub fn run_to_completion(&mut self, max_steps: usize) -> Result<Vec<RequestOutput>> {
        let mut all = Vec::new();
        let mut steps = 0usize;
        while self.has_unfinished() {
            if steps >= max_steps {
                return Err(SentinelError::InvalidState(format!(
                    "not finished after {max_steps} steps ({} still in flight)",
                    self.num_unfinished()
                )));
            }
            let (outs, _) = self.step()?;
            all.extend(outs);
            steps += 1;
        }
        Ok(all)
    }

    /// Convenience: encode prompts, generate, return outputs in request order.
    pub fn generate(&mut self, prompts: Vec<String>, params: SamplingParams) -> Result<Vec<RequestOutput>> {
        let mut ids = Vec::with_capacity(prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            let rid = format!("req-{}", self.next_req);
            self.next_req += 1;
            let seq = self.add_prompt(&rid, p, params.clone())?;
            ids.push((i, seq, rid));
        }
        let mut finished = self.run_to_completion(1_000_000)?;
        // Stable order by original index via request_id map
        let mut by_id: HashMap<String, RequestOutput> = finished.drain(..).map(|o| (o.request_id.clone(), o)).collect();
        let mut ordered = Vec::with_capacity(ids.len());
        for (_, _, rid) in ids {
            ordered.push(by_id.remove(&rid).ok_or_else(|| {
                SentinelError::InvalidState(format!("missing output for {rid}"))
            })?);
        }
        Ok(ordered)
    }

    /// Batch of heterogeneous requests.
    pub fn generate_requests(&mut self, requests: Vec<CompletionRequest>) -> Result<Vec<RequestOutput>> {
        let mut order = Vec::with_capacity(requests.len());
        for r in requests {
            let seq = self.add_tokens(&r.request_id, r.prompt_token_ids, r.params, r.prompt_text)?;
            order.push((seq, r.request_id));
        }
        let mut finished = self.run_to_completion(1_000_000)?;
        let mut by_id: HashMap<String, RequestOutput> = finished.drain(..).map(|o| (o.request_id.clone(), o)).collect();
        let mut ordered = Vec::with_capacity(order.len());
        for (_, rid) in order {
            ordered.push(by_id.remove(&rid).ok_or_else(|| {
                SentinelError::InvalidState(format!("missing output for {rid}"))
            })?);
        }
        Ok(ordered)
    }

    /// Completion tokens generated so far for a live sequence.
    pub fn completion_token_ids(&self, seq_id: SeqId) -> Vec<i32> {
        let Some(seq) = self.engine.sequence(seq_id) else {
            return Vec::new();
        };
        let prompt_len = self
            .prompts
            .get(&seq_id)
            .map(|p| p.len())
            .unwrap_or_else(|| seq.get_prompt_len());
        let ids = seq.token_ids();
        if ids.len() <= prompt_len {
            Vec::new()
        } else {
            ids[prompt_len..].to_vec()
        }
    }

    /// Batch generate with streaming callbacks (kLLM Phase B).
    ///
    /// Adds every prompt, then steps the engine. `on_delta` receives new tokens and a final
    /// event per request with `finished: true`.
    pub fn generate_stream<F>(
        &mut self,
        prompts: Vec<String>,
        params: SamplingParams,
        max_steps: usize,
        mut on_delta: F,
    ) -> Result<Vec<RequestOutput>>
    where
        F: FnMut(StreamDelta),
    {
        if prompts.is_empty() {
            return Err(SentinelError::InvalidState("generate_stream: empty prompt list".into()));
        }
        let mut seq_ids = Vec::with_capacity(prompts.len());
        let mut req_ids = Vec::with_capacity(prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            let rid = format!("stream-{}-{}", self.next_req, i);
            self.next_req += 1;
            let sid = self.add_prompt(&rid, p, params.clone())?;
            seq_ids.push(sid);
            req_ids.push(rid);
        }
        let mut last_len = vec![0usize; seq_ids.len()];
        let mut finished_map: HashMap<SeqId, RequestOutput> = HashMap::new();
        let mut steps = 0usize;
        while finished_map.len() < seq_ids.len() {
            if steps >= max_steps {
                return Err(SentinelError::InvalidState(format!(
                    "stream not finished after {max_steps} steps ({}/{} done)",
                    finished_map.len(),
                    seq_ids.len()
                )));
            }
            let (finished, _) = self.step()?;
            steps += 1;
            for (index, &seq_id) in seq_ids.iter().enumerate() {
                if finished_map.contains_key(&seq_id) {
                    continue;
                }
                if let Some(o) = finished.iter().find(|o| o.seq_id == seq_id) {
                    let tokens = o.output_token_ids.clone();
                    let prev = last_len[index];
                    let new_ids = if tokens.len() > prev { tokens[prev..].to_vec() } else { Vec::new() };
                    let text = self.tokenizer.decode(&new_ids);
                    let text_so_far = self.tokenizer.decode(&tokens);
                    on_delta(StreamDelta {
                        index,
                        request_id: o.request_id.clone(),
                        seq_id,
                        token_ids: new_ids,
                        text,
                        text_so_far,
                        finished: true,
                        finish_reason: Some(o.finish_reason.clone()),
                    });
                    last_len[index] = tokens.len();
                    finished_map.insert(seq_id, o.clone());
                } else {
                    let tokens = self.completion_token_ids(seq_id);
                    let prev = last_len[index];
                    if tokens.len() > prev {
                        let new_ids = tokens[prev..].to_vec();
                        let text = self.tokenizer.decode(&new_ids);
                        let text_so_far = self.tokenizer.decode(&tokens);
                        last_len[index] = tokens.len();
                        on_delta(StreamDelta {
                            index,
                            request_id: req_ids[index].clone(),
                            seq_id,
                            token_ids: new_ids,
                            text,
                            text_so_far,
                            finished: false,
                            finish_reason: None,
                        });
                    }
                }
            }
        }
        let mut ordered = Vec::with_capacity(seq_ids.len());
        for sid in seq_ids {
            ordered.push(finished_map.remove(&sid).expect("finished"));
        }
        Ok(ordered)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seekv::SeekVConfig;

    #[test]
    fn generate_one_prompt() {
        let mut llm = Llm::from_random(SeekVConfig::tiny(), 7, EngineArgs::default()).unwrap();
        let outs = llm.generate(vec!["hi".into()], SamplingParams::greedy(4)).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].completion_tokens, 4);
        assert_eq!(outs[0].finish_reason, "length");
    }

    #[test]
    fn generate_stream_batch_multi_prompt() {
        let mut llm = Llm::from_random(SeekVConfig::tiny(), 9, EngineArgs::default()).unwrap();
        let mut finished_idx = vec![];
        let outs = llm
            .generate_stream(
                vec!["a".into(), "bb".into(), "ccc".into()],
                SamplingParams::greedy(3),
                10_000,
                |d| {
                    if d.finished {
                        finished_idx.push(d.index);
                    }
                },
            )
            .unwrap();
        assert_eq!(outs.len(), 3);
        finished_idx.sort();
        assert_eq!(finished_idx, vec![0, 1, 2]);
        for o in &outs {
            assert_eq!(o.completion_tokens, 3);
        }
    }

    #[test]
    fn generate_stream_emits_tokens() {
        let mut llm = Llm::from_random(SeekVConfig::tiny(), 5, EngineArgs::default()).unwrap();
        let mut pieces = Vec::new();
        let outs = llm
            .generate_stream(
                vec!["hi".into()],
                SamplingParams::greedy(4),
                10_000,
                |d| {
                    if !d.token_ids.is_empty() {
                        pieces.push(d.text.clone());
                    }
                    if d.finished {
                        assert!(d.finish_reason.is_some());
                    }
                },
            )
            .unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].completion_tokens, 4);
        assert!(!pieces.is_empty());
    }

    #[test]
    fn generate_batch_preserves_order() {
        let mut llm = Llm::from_random(SeekVConfig::tiny(), 3, EngineArgs::default()).unwrap();
        let outs = llm
            .generate(vec!["a".into(), "bb".into(), "ccc".into()], SamplingParams::greedy(2))
            .unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].prompt_tokens, 1);
        assert_eq!(outs[1].prompt_tokens, 2);
        assert_eq!(outs[2].prompt_tokens, 3);
    }
}
