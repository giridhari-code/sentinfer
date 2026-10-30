# Seek-V model and engine loop

`sentinel-model` is what Sentinel runs. Architecture: decoder-only transformer with RMSNorm,
rotary embeddings, grouped-query attention and a SwiGLU MLP. CPU, f32.

## What exists

* `Model` trait (`model.rs`): `forward(batch) -> logits`, `release(seq)`, vocab/EOS/length.
* `SeekV` (`seekv.rs`): reference implementation. Presets: `tiny()`, `small()`, `micro(vocab)`.
  `SeekVConfig::flops_config()` feeds the scheduler's FLOPs estimator.
* `Engine` (`engine.rs`): `add_request`, `step`, `run_to_completion`, `abort`.
  Prefer `Engine::for_seekv(...)` so TFLOPs accounting is wired automatically.
* `sample` (`sampler.rs`): greedy, temperature, top-p, top-k, reproducible per request.
* `ByteTokenizer`: byte-level tokenizer (vocab 258) for the default training path.
* Checkpoints: `SeekV::save` / `SeekV::load` (`.skv`, shared with `sentinel-train`).

## Limits

* Weights are random until you train them (`sentinel-train`) or load a checkpoint.
* KV cache is contiguous per sequence; the scheduler block table is capacity accounting only.
  Paged KV arrives with GPU kernels.
* Speculative (MTP) decoding is not wired into the engine loop (`nextn` must be 0).
* Single-threaded CPU reference — slow by design. Fast path is the `Backend` trait in
  `sentinel-kernels`.
