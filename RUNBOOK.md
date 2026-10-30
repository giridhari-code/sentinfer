# Build and test

Needs Rust 1.70+ ([rustup](https://rustup.rs)).

```bash
./check.sh                 # writes build.log and test.log
# or per crate:
cargo test -p sentinel-core
cargo test -p sentinel-kernels
cargo test -p sentinel-model
cargo test -p sentinel-train --release
```

## Reading failures

| Failing test | Likely cause |
| --- | --- |
| `gradients_match_finite_differences_*` | backward-pass bug (tensor + index in the message) |
| `training_forward_matches_the_inference_model` | train vs inference forward disagree |
| `learns_a_repeating_pattern` / `trained_weights_work_in_the_inference_engine` | training bug, or tight thresholds |
| `resume_continues_exactly_*` | non-determinism or incomplete resume state |
| `preempted_requests_recompute_*` / `concurrent_requests_*` | scheduler/engine bookkeeping |
| scheduler / `process_outputs` | core scheduling logic |

## Notes

* Gradient checks use f32 finite differences (step `1e-2`); a single borderline element can
  flake from rounding without a real bug. Consistent multi-element failures are real.
* Release builds of `sentinel-train` with fat LTO can be memory-heavy; use debug tests if the
  process is killed (`SIGKILL`).
* Workspace unit tests have been run successfully on Rust 1.75.
