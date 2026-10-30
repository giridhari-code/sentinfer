//! Small f32 CPU math used by the reference model.

use sentinel_core::SentinelError;
use sentinel_kernels::KernelError;

pub(crate) fn kerr(e: KernelError) -> SentinelError {
    SentinelError::InvalidState(e.to_string())
}

/// SplitMix64: small, fast, good enough for weight init and sampling.
#[derive(Debug, Clone)]
pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Internal state, for saving and resuming a generator.
    pub fn state(&self) -> u64 {
        self.0
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// `out[i][j] = sum_c x[i][c] * w[j][c]`; `x` is `[m, k]`, `w` is `[n, k]` (weights stored as
/// `[out, in]`), result `[m, n]`. Each output element is an independent sequential dot product, so
/// a row's result does not depend on how many rows are in the batch.
pub fn matmul_nt(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(w.len(), n * k);
    // Prefer the kernel-layer blocked implementation (same numerics, better cache).
    sentinel_kernels::reference::matmul_nt(x, w, m, k, n).unwrap_or_else(|_| {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let xi = &x[i * k..(i + 1) * k];
            for j in 0..n {
                let wj = &w[j * k..(j + 1) * k];
                let mut acc = 0.0f32;
                for c in 0..k {
                    acc += xi[c] * wj[c];
                }
                out[i * n + j] = acc;
            }
        }
        out
    })
}

fn rope_impl(x: &mut [f32], tokens: usize, heads: usize, head_dim: usize, start_pos: usize, theta: f32, sign: f32) {
    let half = head_dim / 2;
    for t in 0..tokens {
        let pos = (start_pos + t) as f32;
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            for i in 0..half {
                let freq = theta.powf(-((2 * i) as f32) / head_dim as f32);
                let (sin, cos) = (pos * freq).sin_cos();
                let sin = sin * sign;
                let (x0, x1) = (x[base + i], x[base + i + half]);
                x[base + i] = x0 * cos - x1 * sin;
                x[base + i + half] = x1 * cos + x0 * sin;
            }
        }
    }
}

/// Rotary embedding on `[tokens, heads, head_dim]`, rotating pairs `(i, i + head_dim / 2)`.
/// Token `t` sits at absolute position `start_pos + t`.
pub fn rope_inplace(x: &mut [f32], tokens: usize, heads: usize, head_dim: usize, start_pos: usize, theta: f32) {
    rope_impl(x, tokens, heads, head_dim, start_pos, theta, 1.0);
}

/// The inverse rotation (and the gradient of `rope_inplace`, since rotations are orthogonal).
pub fn rope_inverse_inplace(x: &mut [f32], tokens: usize, heads: usize, head_dim: usize, start_pos: usize, theta: f32) {
    rope_impl(x, tokens, heads, head_dim, start_pos, theta, -1.0);
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

/// Index of the first maximum.
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_nt_known_values() {
        // x = [[1, 2]], w = [[3, 4], [5, 6]] -> [11, 17]
        assert_eq!(matmul_nt(&[1.0, 2.0], &[3.0, 4.0, 5.0, 6.0], 1, 2, 2), vec![11.0, 17.0]);
    }

    #[test]
    fn matmul_rows_are_independent_of_batch_size() {
        let mut rng = SplitMix::new(1);
        let x: Vec<f32> = (0..3 * 8).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let w: Vec<f32> = (0..5 * 8).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let all = matmul_nt(&x, &w, 3, 8, 5);
        for i in 0..3 {
            let one = matmul_nt(&x[i * 8..(i + 1) * 8], &w, 1, 8, 5);
            assert_eq!(&all[i * 5..(i + 1) * 5], &one[..]);
        }
    }

    #[test]
    fn rope_is_identity_at_position_zero_and_preserves_norm() {
        let orig: Vec<f32> = (0..2 * 4).map(|i| i as f32 * 0.3 - 1.0).collect(); // 1 token, 2 heads, dim 4
        let mut x = orig.clone();
        rope_inplace(&mut x, 1, 2, 4, 0, 10000.0);
        assert_eq!(x, orig);

        let mut y = orig.clone();
        rope_inplace(&mut y, 1, 2, 4, 7, 10000.0);
        assert_ne!(y, orig);
        let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
        assert!((norm(&y) - norm(&orig)).abs() < 1e-4);
    }

    #[test]
    fn rope_depends_only_on_absolute_position() {
        let base: Vec<f32> = (0..4).map(|i| i as f32 + 1.0).collect();
        // two tokens at positions 3, 4 in one call vs two calls
        let mut both = [base.clone(), base.clone()].concat();
        rope_inplace(&mut both, 2, 1, 4, 3, 10000.0);
        let mut a = base.clone();
        rope_inplace(&mut a, 1, 1, 4, 3, 10000.0);
        let mut b = base.clone();
        rope_inplace(&mut b, 1, 1, 4, 4, 10000.0);
        assert_eq!(&both[..4], &a[..]);
        assert_eq!(&both[4..], &b[..]);
    }

    #[test]
    fn rope_inverse_undoes_rope() {
        let orig: Vec<f32> = (0..3 * 2 * 8).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut x = orig.clone();
        rope_inplace(&mut x, 3, 2, 8, 5, 10000.0);
        rope_inverse_inplace(&mut x, 3, 2, 8, 5, 10000.0);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_sums_to_one_and_handles_neg_inf() {
        let mut v = vec![1.0, 2.0, f32::NEG_INFINITY];
        softmax_inplace(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn argmax_picks_first_maximum() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn splitmix_is_reproducible_and_in_range() {
        let (mut a, mut b) = (SplitMix::new(9), SplitMix::new(9));
        for _ in 0..50 {
            let u = a.unit();
            assert_eq!(u, b.unit());
            assert!((0.0..1.0).contains(&u));
        }
    }
}
