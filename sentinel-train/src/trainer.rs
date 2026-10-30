//! The training loop: sample a batch, compute gradients, clip, AdamW update, evaluate, checkpoint.

use std::path::{Path, PathBuf};
use std::time::Instant;

use sentinel_core::{Result, SentinelError};
use sentinel_model::checkpoint::{load_weights, save_weights, write_atomic, ByteReader, ByteWriter};
use sentinel_model::ops::SplitMix;
use sentinel_model::{SeekVConfig, SeekVWeights};

use crate::autograd::{compute_gradients, mean_loss};
use crate::data::TokenDataset;
use crate::optim::{clip_grad_norm, AdamW, AdamWConfig, LrSchedule};
use crate::train_opt::{add_grads, average_grads};

const STATE_MAGIC: &[u8; 4] = b"SKVT";
const STATE_VERSION: u32 = 1;
pub const WEIGHTS_FILE: &str = "weights.skv";
pub const BEST_FILE: &str = "best.skv";
pub const STATE_FILE: &str = "train_state.skt";
pub const METRICS_FILE: &str = "metrics.csv";

#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Total optimizer steps (the learning-rate schedule is laid out over this many).
    pub steps: usize,
    pub batch_size: usize,
    /// Input tokens per window (each window holds `seq_len + 1` tokens).
    pub seq_len: usize,
    pub peak_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    /// Maximum global gradient norm; 0 disables clipping.
    pub grad_clip: f32,
    pub adam: AdamWConfig,
    pub threads: usize,
    /// Evaluate every this many steps (0 = never).
    pub eval_interval: usize,
    /// Validation windows per evaluation.
    pub eval_windows: usize,
    /// Save weights and resume state every this many steps (0 = only at the end).
    pub save_interval: usize,
    pub seed: u64,
    /// Where checkpoints go; `None` disables saving.
    pub out_dir: Option<PathBuf>,
    /// Decode a short sample every N steps (0 = disabled).
    pub sample_interval: usize,
    /// Prompt for periodic samples (byte-level models).
    pub sample_prompt: String,
    /// Max new tokens for periodic samples.
    pub sample_tokens: usize,
    /// Micro-batches averaged before one optimizer step (effective batch = batch_size * this).
    pub grad_accum_steps: usize,
    /// Recompute activations on backward to save RAM (partial support).
    pub activation_checkpointing: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 1000,
            batch_size: 8,
            seq_len: 128,
            peak_lr: 3e-4,
            min_lr: 3e-5,
            warmup_steps: 100,
            grad_clip: 1.0,
            adam: AdamWConfig::default(),
            threads: 1,
            eval_interval: 100,
            eval_windows: 32,
            save_interval: 200,
            seed: 1,
            out_dir: None,
            sample_interval: 0,
            sample_prompt: "Once upon a time".into(),
            sample_tokens: 32,
            grad_accum_steps: 1,
            activation_checkpointing: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepLog {
    /// Steps completed, including this one.
    pub step: usize,
    pub loss: f32,
    pub lr: f32,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
    pub tokens_per_sec: f32,
    pub val_loss: Option<f32>,
    /// Periodic greedy sample (prompt + completion), if requested this step.
    pub sample: Option<String>,
}

pub struct Trainer {
    cfg: SeekVConfig,
    pub weights: SeekVWeights,
    opt: AdamW,
    train: TokenDataset,
    val: Option<TokenDataset>,
    tc: TrainConfig,
    schedule: LrSchedule,
    step: usize,
    tokens_seen: u64,
    rng: SplitMix,
    best_val: f32,
}

fn invalid(msg: String) -> SentinelError {
    SentinelError::InvalidState(msg)
}

impl Trainer {
    pub fn new(
        cfg: SeekVConfig,
        weights: SeekVWeights,
        train: TokenDataset,
        val: Option<TokenDataset>,
        tc: TrainConfig,
    ) -> Result<Self> {
        cfg.validate()?;
        weights.validate(&cfg)?;
        if tc.batch_size == 0 || tc.seq_len == 0 || tc.steps == 0 || tc.grad_accum_steps == 0 {
            return Err(invalid("batch_size, seq_len, steps and grad_accum_steps must be > 0".into()));
        }
        if tc.seq_len > cfg.max_positions {
            return Err(invalid(format!("seq_len {} exceeds the model's max_positions {}", tc.seq_len, cfg.max_positions)));
        }
        let window = tc.seq_len + 1;
        if train.len() < window {
            return Err(invalid(format!("training set has {} tokens, need at least {window}", train.len())));
        }
        let val = val.filter(|v| v.len() >= window);
        for (name, ds) in std::iter::once(("training", &train)).chain(val.as_ref().map(|v| ("validation", v))) {
            if let Some(max) = ds.max_token() {
                if max as usize >= cfg.vocab_size {
                    return Err(invalid(format!(
                        "{name} data contains token {max} but the vocabulary has {} entries (raise --vocab-size)",
                        cfg.vocab_size
                    )));
                }
            }
            if ds.tokens().iter().any(|&t| t < 0) {
                return Err(invalid(format!("{name} data contains negative token ids")));
            }
        }
        let schedule = LrSchedule { peak: tc.peak_lr, min: tc.min_lr, warmup_steps: tc.warmup_steps, total_steps: tc.steps };
        let opt = AdamW::new(&cfg, tc.adam.clone());
        let rng = SplitMix::new(tc.seed);
        Ok(Self { cfg, weights, opt, train, val, tc, schedule, step: 0, tokens_seen: 0, rng, best_val: f32::INFINITY })
    }

    pub fn step_count(&self) -> usize {
        self.step
    }

    pub fn tokens_seen(&self) -> u64 {
        self.tokens_seen
    }

    pub fn config(&self) -> &SeekVConfig {
        &self.cfg
    }

    /// Mean loss on the validation set, or `None` if there is no usable one.
    pub fn evaluate(&self) -> Result<Option<f32>> {
        let Some(val) = &self.val else {
            return Ok(None);
        };
        let windows = val.eval_windows(self.tc.seq_len + 1, self.tc.eval_windows.max(1));
        if windows.is_empty() {
            return Ok(None);
        }
        Ok(Some(mean_loss(&self.cfg, &self.weights, &windows, self.tc.threads)?))
    }

    /// Greedy decode from the current weights (byte-level vocab only).
    pub fn sample_decode(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        use sentinel_core::sequence::SamplingScalars;
        use sentinel_model::{ByteTokenizer, Engine, GenerationParams, SeekV, Tokenizer};

        if self.cfg.vocab_size != ByteTokenizer::VOCAB_SIZE {
            return Err(invalid(format!(
                "sample_decode only supports byte-level vocab {}, model has {}",
                ByteTokenizer::VOCAB_SIZE,
                self.cfg.vocab_size
            )));
        }
        let model = SeekV::new(self.cfg.clone(), self.weights.clone())?;
        let tok = ByteTokenizer;
        let ids = Tokenizer::encode(&tok, prompt, false);
        if ids.is_empty() {
            return Err(invalid("sample prompt is empty".into()));
        }
        let blocks = self.cfg.max_positions / 16 + 8;
        let mut engine = Engine::for_seekv(Default::default(), blocks, model)?;
        engine.add_request(
            "sample",
            ids,
            GenerationParams {
                max_tokens,
                ignore_eos: false,
                sampling: SamplingScalars {
                    temperature: 0.0,
                    ..Default::default()
                },
            },
        )?;
        let done = engine.run_to_completion(100_000)?;
        let out = done.into_iter().next().ok_or_else(|| invalid("sample produced no output".into()))?;
        Ok(format!("{}{}", prompt, Tokenizer::decode(&tok, &out.output_token_ids)))
    }

    /// One optimizer step (optionally over `grad_accum_steps` micro-batches).
    pub fn train_one_step(&mut self) -> Result<StepLog> {
        let started = Instant::now();
        let lr = self.schedule.lr_at(self.step);
        let accum = self.tc.grad_accum_steps.max(1);
        let mut total_loss = 0.0f32;
        let mut grads_acc: Option<SeekVWeights> = None;

        for micro in 0..accum {
            let windows = self.train.sample_windows(&mut self.rng, self.tc.batch_size, self.tc.seq_len + 1)?;
            let (loss, grads) = compute_gradients(&self.cfg, &self.weights, &windows, self.tc.threads)?;
            if !loss.is_finite() {
                return Err(invalid(format!(
                    "loss is {loss} at step {} micro {micro}: training diverged (try a lower learning rate)",
                    self.step + 1
                )));
            }
            total_loss += loss;
            match grads_acc.as_mut() {
                None => grads_acc = Some(grads),
                Some(acc) => add_grads(acc, &grads),
            }
        }

        let mut grads = grads_acc.expect("at least one micro-batch");
        average_grads(&mut grads, accum);
        let loss = total_loss / accum as f32;

        let grad_norm = if self.tc.grad_clip > 0.0 {
            clip_grad_norm(&mut grads, self.tc.grad_clip)
        } else {
            crate::optim::global_norm(&grads)
        };
        if !grad_norm.is_finite() {
            return Err(invalid(format!("gradient norm is {grad_norm} at step {}: training diverged", self.step + 1)));
        }
        self.opt.step(&mut self.weights, &grads, lr);
        self.step += 1;
        let tokens = (self.tc.batch_size * self.tc.seq_len * accum) as u64;
        self.tokens_seen += tokens;
        let secs = started.elapsed().as_secs_f32().max(1e-6);
        Ok(StepLog { step: self.step, loss, lr, grad_norm, tokens_per_sec: tokens as f32 / secs, val_loss: None, sample: None })
    }

    pub fn run(&mut self, on_step: &mut dyn FnMut(&StepLog)) -> Result<()> {
        let mut metrics = self.tc.out_dir.as_ref().and_then(|dir| {
            std::fs::create_dir_all(dir).ok()?;
            let path = dir.join(METRICS_FILE);
            let (mut f, write_header) = if self.step == 0 {
                (std::fs::File::create(&path).ok()?, true)
            } else {
                let exists = path.exists();
                (
                    std::fs::OpenOptions::new().create(true).append(true).open(&path).ok()?,
                    !exists,
                )
            };
            if write_header {
                use std::io::Write;
                let _ = writeln!(f, "step,loss,lr,grad_norm,tokens_per_sec,val_loss,tokens_seen");
            }
            Some(f)
        });
        while self.step < self.tc.steps {
            let mut log = self.train_one_step()?;
            let done = self.step;
            if self.tc.eval_interval > 0 && done % self.tc.eval_interval == 0 {
                log.val_loss = self.evaluate()?;
                if let (Some(v), Some(dir)) = (log.val_loss, self.tc.out_dir.clone()) {
                    if v < self.best_val {
                        self.best_val = v;
                        save_weights(&dir.join(BEST_FILE), &self.cfg, &self.weights)?;
                    }
                }
            }
            if self.tc.sample_interval > 0 && done % self.tc.sample_interval == 0 {
                match self.sample_decode(&self.tc.sample_prompt, self.tc.sample_tokens) {
                    Ok(s) => log.sample = Some(s),
                    Err(e) => eprintln!("[sample] skipped: {e}"),
                }
            }
            if let Some(f) = metrics.as_mut() {
                use std::io::Write;
                let val = log.val_loss.map(|v| format!("{v:.6}")).unwrap_or_default();
                let _ = writeln!(
                    f,
                    "{},{:.6},{:.6e},{:.6},{:.1},{},{}",
                    log.step, log.loss, log.lr, log.grad_norm, log.tokens_per_sec, val, self.tokens_seen
                );
            }
            on_step(&log);
            if let Some(dir) = self.tc.out_dir.clone() {
                if self.tc.save_interval > 0 && done % self.tc.save_interval == 0 {
                    self.save(&dir)?;
                }
            }
        }
        if let Some(dir) = self.tc.out_dir.clone() {
            self.save(&dir)?;
        }
        Ok(())
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        save_weights(&dir.join(WEIGHTS_FILE), &self.cfg, &self.weights)?;
        let mut w = ByteWriter::new();
        w.bytes(STATE_MAGIC);
        w.u32(STATE_VERSION);
        w.u64(self.step as u64);
        w.u64(self.tokens_seen);
        w.u64(self.rng.state());
        w.f32(self.best_val);
        self.opt.write_state(&mut w);
        write_atomic(&dir.join(STATE_FILE), &w.finish())
    }

    pub fn resume(&mut self, dir: &Path) -> Result<()> {
        let (cfg, weights) = load_weights(&dir.join(WEIGHTS_FILE))?;
        let same = cfg.vocab_size == self.cfg.vocab_size
            && cfg.hidden_size == self.cfg.hidden_size
            && cfg.num_layers == self.cfg.num_layers
            && cfg.num_heads == self.cfg.num_heads
            && cfg.num_kv_heads == self.cfg.num_kv_heads
            && cfg.head_dim == self.cfg.head_dim
            && cfg.intermediate_size == self.cfg.intermediate_size;
        if !same {
            return Err(invalid("checkpoint architecture differs from the configured model".into()));
        }
        let path = dir.join(STATE_FILE);
        let data = std::fs::read(&path).map_err(|e| sentinel_model::checkpoint::io_err(&format!("reading {}", path.display()), e))?;
        let mut r = ByteReader::checked(&data)?;
        if r.bytes(4)? != STATE_MAGIC {
            return Err(SentinelError::Io("not a training state file (bad magic)".into()));
        }
        let version = r.u32()?;
        if version != STATE_VERSION {
            return Err(SentinelError::Io(format!("unsupported training state version {version}")));
        }
        let step = r.u64()? as usize;
        let tokens_seen = r.u64()?;
        let rng_state = r.u64()?;
        let best_val = r.f32()?;
        self.opt.read_state(&mut r)?;
        if !r.is_empty() {
            return Err(SentinelError::Io("trailing bytes in training state".into()));
        }
        self.weights = weights;
        self.step = step;
        self.tokens_seen = tokens_seen;
        self.rng = SplitMix::new(rng_state);
        self.best_val = best_val;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern_dataset(len: usize) -> TokenDataset {
        TokenDataset::from_tokens((0..len).map(|i| (i % 4) as i32 + 1).collect())
    }

    fn config(steps: usize) -> TrainConfig {
        TrainConfig {
            steps,
            batch_size: 8,
            seq_len: 8,
            peak_lr: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 10,
            grad_clip: 1.0,
            threads: 1,
            eval_interval: 0,
            save_interval: 0,
            ..Default::default()
        }
    }

    fn trainer(steps: usize) -> Trainer {
        let cfg = SeekVConfig::micro(11);
        let weights = SeekVWeights::random(&cfg, 1);
        Trainer::new(cfg, weights, pattern_dataset(400), Some(pattern_dataset(100)), config(steps)).unwrap()
    }

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("sentinel-trainer-test-{}-{name}", std::process::id()))
    }

    #[test]
    fn learns_a_repeating_pattern() {
        let mut t = trainer(200);
        let mut first = None;
        let mut last = 0.0;
        t.run(&mut |log| {
            first.get_or_insert(log.loss);
            last = log.loss;
        })
        .unwrap();
        let first = first.unwrap();
        assert!(last < 0.5 * first && last < 1.0, "loss went from {first} to {last}");
        assert_eq!(t.step_count(), 200);
        assert_eq!(t.tokens_seen(), 200 * 8 * 8);
        let val = t.evaluate().unwrap().unwrap();
        assert!(val < 1.0, "validation loss {val}");
    }

    #[test]
    fn grad_accum_runs() {
        let mut tc = config(5);
        tc.grad_accum_steps = 2;
        let cfg = SeekVConfig::micro(11);
        let mut t = Trainer::new(cfg.clone(), SeekVWeights::random(&cfg, 1), pattern_dataset(400), None, tc).unwrap();
        t.train_one_step().unwrap();
        assert_eq!(t.tokens_seen(), 8 * 8 * 2);
    }
}
