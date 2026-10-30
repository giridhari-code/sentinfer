# Sentinel Rust port (Seek-V)

## Scope

This workspace is the **Seek-V** inference + training stack in Rust. It is not a
DeepSeek model loader. Architecture: standard decoder (RMSNorm, RoPE, GQA, SwiGLU).

## Engine mapping (historical C++ names → Rust)

| Historical C++ name | Rust |
| ------------------- | ---- |
| `block_manager` | `block_manager.rs` |
| `sequence` | `sequence.rs` |
| `scheduler` | `scheduler.rs` |
| `mtp_varlen` | `mtp_varlen.rs` |
| FLOPs estimate | `flops_estimator.rs` (Seek-V decoder only) |
| `SequenceMetadata` / prepare_sample | `sequence_metadata.rs` |

## Design

- Scheduler owns sequences in a `HashMap<SeqId, Sequence>`; queues hold ids.
- No locks in the block manager; `&mut` ownership at the scheduler.
- Errors are `Result`, not exceptions.
- Sampler inputs are host `Vec`s (no torch dependency in core).

## Ported

Block manager, sequences, prefill/decode scheduling (LCFS, preemption, chunked
prefill), MTP varlen planner, process_outputs, prompted queue + fork admit,
multimodal image-pad chunk adjust, Seek-V FLOPs estimator, SequenceMetadata,
prepare_sampler_inputs.

## Deferred

| Item | Notes |
| ---- | ----- |
| Data-parallel voting | Needs a process group |
| Extra GPU kernels | Phase 3 (`KERNELS.md`) |
| PyO3 bindings | Phase 4 |

## Training

Use `sentinel-train` on a text corpus (byte-level or pre-tokenized ids). See
`sentinel-train/TRAINING.md`. Checkpoints load in the inference engine via
`SeekV::load`.

## Status

Rust 1.75: workspace unit tests pass for core / kernels / model / train.
