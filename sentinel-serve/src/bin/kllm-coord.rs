//! Coordinator: fan-out prompts to kLLM workers.
//!
//!   kllm-coord --workers 127.0.0.1:9001,127.0.0.1:9002 --prompt "Hello" --max-tokens 32

use sentinel_model::Coordinator;

fn main() {
    let mut workers = Vec::new();
    let mut prompts = Vec::new();
    let mut max_tokens = 32usize;
    let mut temperature = 0.8f64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--workers" => {
                let list = it.next().unwrap_or_default();
                workers = list.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            }
            "--prompt" => prompts.push(it.next().unwrap_or_default()),
            "--max-tokens" => max_tokens = it.next().and_then(|s| s.parse().ok()).unwrap_or(32),
            "--temperature" => temperature = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.8),
            "-h" | "--help" => {
                eprintln!("usage: kllm-coord --workers host:port,... --prompt TEXT [--max-tokens N]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown {other}");
                std::process::exit(2);
            }
        }
    }
    if workers.is_empty() || prompts.is_empty() {
        eprintln!("--workers and --prompt required");
        std::process::exit(2);
    }
    let coord = Coordinator::new(workers);
    match coord.generate_batch(prompts, max_tokens, temperature) {
        Ok(outs) => {
            for (i, o) in outs.iter().enumerate() {
                println!("[{i}] {o}");
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
