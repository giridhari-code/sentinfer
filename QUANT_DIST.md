# Quantization & distributed inference

## Quantization (int8 / int4)

```bash
cargo run -p sentinel-train --bin quantize -- \
  -i runs/best.skv -o runs/best-q8.skv --scheme int8

# int4
cargo run -p sentinel-train --bin quantize -- \
  -i runs/best.skv -o runs/best-q4.skv --scheme int4
```

- Per-row scales on linear weights (`wq/wk/wv/wo`, MLP, `lm_head`)
- Embeddings stay f32
- Packed size estimate printed; checkpoint is rewritten via quant→dequant validation path
- Runtime path: `QuantMatrix::matmul_nt` / `matmul_nt_i8` in `sentinel-kernels::quant`

## Distributed (data-parallel multi-node)

```bash
# node A
kllm-worker --checkpoint model.skv --bind 0.0.0.0:9001 --id a

# node B
kllm-worker --checkpoint model.skv --bind 0.0.0.0:9002 --id b

# coordinator
kllm-coord --workers 10.0.0.1:9001,10.0.0.2:9002 \
  --prompt "Hello" --prompt "Once upon a time" --max-tokens 32
```

Request-level sharding (round-robin / batch fan-out). Not tensor-parallel.
