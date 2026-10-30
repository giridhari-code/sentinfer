//! `generate`: load a Seek-V checkpoint and complete one or more prompts.

use std::collections::HashMap;
use std::path::Path;

use sentinel_model::{EngineArgs, Llm, SamplingParams};

const USAGE: &str = "\
usage: generate --checkpoint FILE.skv --prompt TEXT [--prompt TEXT ...]
                [--max-tokens N] [--temperature F] [--top-k N] [--top-p F] [--seed N]
                [--log-stats]

Byte-level models only (vocabulary 258). Defaults: 64 tokens, temperature 0.8,
top-k 40, top-p 0.95, seed 1. Temperature 0 is greedy.
";

fn run() -> Result<(), String> {
    let mut values: HashMap<String, String> = HashMap::new();
    let mut prompts: Vec<String> = Vec::new();
    let mut flags: Vec<String> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--prompt" => {
                let v = it.next().ok_or_else(|| format!("{arg} needs a value"))?;
                prompts.push(v);
            }
            "--checkpoint" | "--max-tokens" | "--temperature" | "--top-k" | "--top-p" | "--seed" => {
                let v = it.next().ok_or_else(|| format!("{arg} needs a value"))?;
                values.insert(arg, v);
            }
            "--log-stats" => flags.push(arg),
            _ => return Err(format!("unknown argument `{arg}`")),
        }
    }
    fn num<T: std::str::FromStr>(values: &HashMap<String, String>, name: &str, default: T) -> Result<T, String> {
        match values.get(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("{name}: cannot parse `{v}`")),
        }
    }

    let checkpoint = values.get("--checkpoint").ok_or("--checkpoint is required")?;
    if prompts.is_empty() {
        return Err("at least one --prompt is required".into());
    }

    let mut args = EngineArgs::default();
    args.log_stats = flags.iter().any(|f| f == "--log-stats");
    let mut llm = Llm::from_checkpoint(Path::new(checkpoint), args).map_err(|e| e.to_string())?;

    let params = SamplingParams {
        max_tokens: num(&values, "--max-tokens", 64usize)?,
        temperature: num(&values, "--temperature", 0.8f64)?,
        top_k: num(&values, "--top-k", 40i32)?,
        top_p: num(&values, "--top-p", 0.95f64)?,
        seed: num(&values, "--seed", 1u64)?,
        ..Default::default()
    };

    let outs = llm.generate(prompts.clone(), params).map_err(|e| e.to_string())?;
    for (i, o) in outs.iter().enumerate() {
        if prompts.len() > 1 {
            println!("=== [{}] {} tokens, {} ===", o.request_id, o.completion_tokens, o.finish_reason);
        }
        println!("{}{}", prompts[i], o.text);
        if prompts.len() > 1 {
            println!();
        }
        eprintln!(
            "[{} new tokens, finished: {}, total stats: {} steps, {} completion tokens]",
            o.completion_tokens,
            o.finish_reason,
            llm.stats().steps,
            llm.stats().total_completion_tokens
        );
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}\n\n{USAGE}");
        std::process::exit(2);
    }
}
