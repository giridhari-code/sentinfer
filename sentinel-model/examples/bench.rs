//! Micro-benchmarks: matmul, attention, end-to-end decode tok/s.
//!
//!   cargo run -p sentinel-model --example bench --release

use std::time::Instant;

use sentinel_kernels::reference;
use sentinel_model::{Engine, GenerationParams, SeekV, SeekVConfig};
use sentinel_core::sequence::SamplingScalars;

fn bench(name: &str, iters: usize, mut f: impl FnMut()) {
    // warmup
    for _ in 0..3 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("{name:<40}  {iters:>6} iters  {:>8.2} ms  {:>10.1} /s", ms, iters as f64 / (ms / 1000.0));
}

fn main() {
    println!("kLLM / Sentinel Phase C benches (release)\n");

    // matmul
    let m = 64usize;
    let k = 256usize;
    let n = 256usize;
    let x = vec![0.01f32; m * k];
    let w = vec![0.02f32; n * k];
    bench(&format!("matmul_nt {m}x{k}x{n}"), 50, || {
        let _ = reference::matmul_nt(&x, &w, m, k, n).unwrap();
    });
    bench(&format!("matmul_nt_parallel {m}x{k}x{n}"), 50, || {
        let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
        let _ = reference::matmul_nt_parallel(&x, &w, m, k, n, threads).unwrap();
    });

    // attention decode-like: 1 token, long context
    for ctx in [128usize, 512, 1024] {
        let heads = 8usize;
        let kv = 2usize;
        let hd = 32usize;
        let q = vec![0.01f32; heads * hd];
        let k_cache = vec![0.01f32; ctx * kv * hd];
        let v_cache = vec![0.02f32; ctx * kv * hd];
        bench(&format!("attention_causal tok=1 ctx={ctx}"), 30, || {
            let _ = reference::attention_causal(&q, &k_cache, &v_cache, 1, ctx - 1, heads, kv, hd).unwrap();
        });
    }

    // end-to-end decode tok/s (tiny model)
    let model = SeekV::random(SeekVConfig::tiny(), 1).unwrap();
    let blocks = 128;
    let mut engine = Engine::for_seekv(Default::default(), blocks, model).unwrap();
    let prompt: Vec<i32> = (1..17).collect();
    let max_tokens = 32usize;
    engine
        .add_request(
            "bench",
            prompt,
            GenerationParams {
                max_tokens,
                ignore_eos: true,
                sampling: SamplingScalars { temperature: 0.0, ..Default::default() },
            },
        )
        .unwrap();
    let t0 = Instant::now();
    let done = engine.run_to_completion(100_000).unwrap();
    let secs = t0.elapsed().as_secs_f64().max(1e-9);
    let ntok = done[0].output_token_ids.len();
    println!(
        "{:<40}  {:>6} toks  {:>8.2} ms  {:>10.1} tok/s",
        "e2e decode (SeekV::tiny)",
        ntok,
        secs * 1000.0,
        ntok as f64 / secs
    );

    // small preset if memory allows
    let model = SeekV::random(SeekVConfig::small(), 2).unwrap();
    let blocks = 256;
    let mut engine = Engine::for_seekv(Default::default(), blocks, model).unwrap();
    let prompt: Vec<i32> = (1..33).collect();
    engine
        .add_request(
            "bench-small",
            prompt,
            GenerationParams {
                max_tokens: 16,
                ignore_eos: true,
                sampling: SamplingScalars { temperature: 0.0, ..Default::default() },
            },
        )
        .unwrap();
    let t0 = Instant::now();
    let done = engine.run_to_completion(100_000).unwrap();
    let secs = t0.elapsed().as_secs_f64().max(1e-9);
    let ntok = done[0].output_token_ids.len();
    println!(
        "{:<40}  {:>6} toks  {:>8.2} ms  {:>10.1} tok/s",
        "e2e decode (SeekV::small)",
        ntok,
        secs * 1000.0,
        ntok as f64 / secs
    );
}
