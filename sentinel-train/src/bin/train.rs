//! `train`: train a Seek-V model from a text file or a pre-tokenized file.

use std::collections::HashMap;
use std::path::PathBuf;

use sentinel_model::{SeekVConfig, SeekVWeights};
use sentinel_train::data::TokenDataset;
use sentinel_train::trainer::{StepLog, TrainConfig, Trainer};

const USAGE: &str = "\
usage: train --data FILE | --tokens-u16 FILE | --tokens-u32 FILE  --out DIR [options]

data:
  --data FILE          UTF-8 text, tokenized as raw bytes (vocabulary 258)
  --split-docs         with --data: blank lines separate documents (EOS token between them)
  --tokens-u16 FILE    pre-tokenized ids, little-endian u16
  --tokens-u32 FILE    pre-tokenized ids, little-endian u32
  --val-fraction F     share of the data held out for validation (default 0.05)

model (preset, then overrides):
  --preset tiny|small  default small (about 4.6M parameters)
  --vocab-size N  --hidden N  --layers N  --heads N  --kv-heads N  --head-dim N
  --inter N  --max-pos N  --eos-id N

training:
  --steps N (1000)  --batch N (8)  --seq-len N (128)  --lr F (3e-4)  --min-lr F (3e-5)
  --warmup N (100)  --wd F (0.1)  --clip F (1.0)  --threads N (all cores)  --seed N (1)
  --eval-every N (100)  --eval-windows N (32)  --save-every N (200)
  --sample-every N     greedy decode sample every N steps (0=off)
  --sample-prompt TEXT (default: Once upon a time)
  --sample-tokens N (32)
  --resume             continue from the checkpoint in --out
";

struct Args {
    values: HashMap<String, String>,
    flags: Vec<String>,
}

const FLAG_NAMES: [&str; 2] = ["--resume", "--split-docs"];
const VALUE_NAMES: [&str; 31] = [
    "--data", "--tokens-u16", "--tokens-u32", "--out", "--val-fraction", "--preset", "--vocab-size", "--hidden",
    "--layers", "--heads", "--kv-heads", "--head-dim", "--inter", "--max-pos", "--eos-id", "--steps", "--batch",
    "--seq-len", "--lr", "--min-lr", "--warmup", "--wd", "--clip", "--threads", "--seed", "--eval-every",
    "--eval-windows", "--save-every", "--sample-every", "--sample-prompt", "--sample-tokens",
];

fn parse_args() -> Result<Args, String> {
    let mut values = HashMap::new();
    let mut flags = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if FLAG_NAMES.contains(&arg.as_str()) {
            flags.push(arg);
        } else if VALUE_NAMES.contains(&arg.as_str()) {
            let v = it.next().ok_or_else(|| format!("{arg} needs a value"))?;
            values.insert(arg, v);
        } else {
            return Err(format!("unknown argument `{arg}`"));
        }
    }
    Ok(Args { values, flags })
}

impl Args {
    fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    fn get<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, String> {
        match self.values.get(name) {
            None => Ok(None),
            Some(v) => v.parse::<T>().map(Some).map_err(|_| format!("{name}: cannot parse `{v}`")),
        }
    }

    fn or<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        Ok(self.get(name)?.unwrap_or(default))
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let e = |err: sentinel_core::SentinelError| err.to_string();

    let out: PathBuf = args.get::<String>("--out")?.ok_or("--out is required")?.into();

    let mut cfg = match args.or("--preset", "small".to_string())?.as_str() {
        "tiny" => SeekVConfig::tiny(),
        "small" => SeekVConfig::small(),
        other => return Err(format!("unknown preset `{other}` (use tiny or small)")),
    };
    cfg.vocab_size = args.or("--vocab-size", cfg.vocab_size)?;
    cfg.hidden_size = args.or("--hidden", cfg.hidden_size)?;
    cfg.num_layers = args.or("--layers", cfg.num_layers)?;
    cfg.num_heads = args.or("--heads", cfg.num_heads)?;
    cfg.num_kv_heads = args.or("--kv-heads", cfg.num_kv_heads)?;
    cfg.head_dim = args.or("--head-dim", cfg.head_dim)?;
    cfg.intermediate_size = args.or("--inter", cfg.intermediate_size)?;
    cfg.max_positions = args.or("--max-pos", cfg.max_positions)?;
    cfg.eos_token_id = args.or("--eos-id", cfg.eos_token_id)?;
    cfg.validate().map_err(e)?;

    let dataset = match (args.get::<String>("--data")?, args.get::<String>("--tokens-u16")?, args.get::<String>("--tokens-u32")?) {
        (Some(p), None, None) => TokenDataset::from_text_file(p.as_ref(), args.flag("--split-docs")).map_err(e)?,
        (None, Some(p), None) => TokenDataset::from_u16_file(p.as_ref()).map_err(e)?,
        (None, None, Some(p)) => TokenDataset::from_u32_file(p.as_ref()).map_err(e)?,
        _ => return Err("give exactly one of --data, --tokens-u16, --tokens-u32".into()),
    };
    let val_fraction: f32 = args.or("--val-fraction", 0.05)?;
    let (train, val) = dataset.split(val_fraction);

    let threads_default = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let tc = TrainConfig {
        steps: args.or("--steps", 1000)?,
        batch_size: args.or("--batch", 8)?,
        seq_len: args.or("--seq-len", 128)?,
        peak_lr: args.or("--lr", 3e-4)?,
        min_lr: args.or("--min-lr", 3e-5)?,
        warmup_steps: args.or("--warmup", 100)?,
        grad_clip: args.or("--clip", 1.0)?,
        threads: args.or("--threads", threads_default)?,
        eval_interval: args.or("--eval-every", 100)?,
        eval_windows: args.or("--eval-windows", 32)?,
        save_interval: args.or("--save-every", 200)?,
        seed: args.or("--seed", 1)?,
        out_dir: Some(out.clone()),
        sample_interval: args.or("--sample-every", 0usize)?,
        sample_prompt: args.or("--sample-prompt", "Once upon a time".to_string())?,
        sample_tokens: args.or("--sample-tokens", 32usize)?,
        adam: sentinel_train::optim::AdamWConfig { weight_decay: args.or("--wd", 0.1)?, ..Default::default() },
    };

    let weights = SeekVWeights::random(&cfg, tc.seed);
    println!(
        "model: {} parameters ({} layers, hidden {}, {} heads / {} kv heads, vocab {})",
        weights.num_params(),
        cfg.num_layers,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.vocab_size
    );
    println!("data: {} training tokens, {} validation tokens", train.len(), val.len());
    println!(
        "plan: {} steps x {} windows x {} tokens, peak lr {}, {} threads, output {}",
        tc.steps,
        tc.batch_size,
        tc.seq_len,
        tc.peak_lr,
        tc.threads,
        out.display()
    );

    let steps = tc.steps;
    let mut trainer = Trainer::new(cfg, weights, train, Some(val), tc).map_err(e)?;
    if args.flag("--resume") {
        trainer.resume(&out).map_err(e)?;
        println!("resumed at step {}", trainer.step_count());
    }

    trainer
        .run(&mut |log: &StepLog| {
            let mut line = format!(
                "step {:>6}/{steps}  loss {:.4}  lr {:.2e}  grad {:.2}  {:.0} tok/s",
                log.step, log.loss, log.lr, log.grad_norm, log.tokens_per_sec
            );
            if let Some(v) = log.val_loss {
                line.push_str(&format!("  | val loss {:.4}  ppl {:.2}", v, v.exp()));
            }
            println!("{line}");
            if let Some(s) = &log.sample {
                println!("  sample: {s}");
            }
        })
        .map_err(e)?;
    println!("done. weights: {}", out.join(sentinel_train::trainer::WEIGHTS_FILE).display());
    println!("metrics: {}", out.join(sentinel_train::trainer::METRICS_FILE).display());
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}\n\n{USAGE}");
        std::process::exit(2);
    }
}
