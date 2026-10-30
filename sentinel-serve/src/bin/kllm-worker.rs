//! Data-parallel worker for multi-node kLLM.
//!
//!   kllm-worker --checkpoint model.skv --bind 0.0.0.0:9001 --id worker-0

use sentinel_model::Worker;

fn main() {
    let mut checkpoint = None;
    let mut bind = "0.0.0.0:9001".to_string();
    let mut id = "worker-0".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--checkpoint" => checkpoint = it.next(),
            "--bind" => bind = it.next().unwrap_or(bind),
            "--id" => id = it.next().unwrap_or(id),
            "-h" | "--help" => {
                eprintln!("usage: kllm-worker --checkpoint FILE.skv [--bind ADDR] [--id NAME]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown {other}");
                std::process::exit(2);
            }
        }
    }
    let checkpoint = checkpoint.expect("--checkpoint required");
    let w = Worker::from_checkpoint(&checkpoint, id).unwrap_or_else(|e| {
        eprintln!("load: {e}");
        std::process::exit(1);
    });
    if let Err(e) = w.serve(&bind) {
        eprintln!("serve: {e}");
        std::process::exit(1);
    }
}
