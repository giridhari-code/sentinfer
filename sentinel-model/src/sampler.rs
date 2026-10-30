use sentinel_core::sequence::SamplingScalars;
use sentinel_core::{Result, SentinelError};
use sentinel_kernels::reference;

use crate::ops::{argmax, kerr, softmax_inplace, SplitMix};

/// Picks the next token from one row of logits.
///
/// `temperature <= 0` is greedy. Otherwise logits are scaled, filtered with top-p / top-k
/// (`top_p` outside (0, 1) and `top_k <= 0` disable the filter) and sampled with a generator
/// seeded from `seq_seed` and `position`, so the same request always samples the same way,
/// even if it is preempted and recomputed.
pub fn sample(logits: &[f32], params: &SamplingScalars, position: usize) -> Result<i32> {
    if logits.is_empty() {
        return Err(SentinelError::InvalidState("cannot sample from empty logits".into()));
    }
    if params.temperature <= 0.0 {
        return Ok(argmax(logits) as i32);
    }
    let mut row: Vec<f32> = logits.iter().map(|l| l / params.temperature as f32).collect();
    let top_p = if params.top_p > 0.0 && params.top_p < 1.0 { params.top_p as f32 } else { 1.0 };
    let top_k = if params.top_k > 0 { params.top_k } else { -1 };
    reference::apply_top_p_top_k(&mut row, &[top_p], &[top_k], 1, logits.len()).map_err(kerr)?;
    softmax_inplace(&mut row);

    let seed = params.seq_seed ^ (position as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let u = SplitMix::new(seed).unit();
    let mut acc = 0.0f32;
    for (i, &p) in row.iter().enumerate() {
        acc += p;
        if u < acc {
            return Ok(i as i32);
        }
    }
    Ok(argmax(&row) as i32) // rounding left a sliver of probability mass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(temperature: f64) -> SamplingScalars {
        SamplingScalars { temperature, ..Default::default() }
    }

    #[test]
    fn zero_temperature_is_greedy() {
        assert_eq!(sample(&[0.1, 2.0, 0.5], &params(0.0), 3).unwrap(), 1);
    }

    #[test]
    fn top_k_one_always_picks_the_best_token() {
        let mut p = params(1.5);
        p.top_k = 1;
        for pos in 0..20 {
            p.seq_seed = pos as u64 * 31;
            assert_eq!(sample(&[0.1, 2.0, 0.5, 1.9], &p, pos).unwrap(), 1);
        }
    }

    #[test]
    fn sampling_is_reproducible_per_seed_and_position() {
        let mut p = params(1.0);
        p.seq_seed = 77;
        let logits = [0.3, 0.1, 0.9, 0.4, 0.2, 0.8];
        let first: Vec<i32> = (0..10).map(|pos| sample(&logits, &p, pos).unwrap()).collect();
        let second: Vec<i32> = (0..10).map(|pos| sample(&logits, &p, pos).unwrap()).collect();
        assert_eq!(first, second);
        assert!(first.iter().any(|&t| t != first[0]), "sampling should not be constant here");
    }

    #[test]
    fn top_p_restricts_to_the_nucleus() {
        // softmax([0,1,2,3]): top token has p = 0.64, so top_p = 0.5 leaves only that token
        let mut p = params(1.0);
        p.top_p = 0.5;
        for pos in 0..20 {
            p.seq_seed = pos as u64;
            assert_eq!(sample(&[0.0, 1.0, 2.0, 3.0], &p, pos).unwrap(), 3);
        }
    }

    #[test]
    fn empty_logits_are_an_error() {
        assert!(sample(&[], &params(0.0), 0).is_err());
    }
}
