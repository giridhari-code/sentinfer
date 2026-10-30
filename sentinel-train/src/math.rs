//! Forward and backward building blocks. Every matrix is row-major; weights are `[out, in]`.

/// Backward of `y = x W^T` with `x: [m, k]`, `W: [n, k]`, `y, dy: [m, n]`.
/// Accumulates `dW += dy^T x` (`[n, k]`) and returns `dx = dy W` (`[m, k]`).
pub fn linear_bwd(dy: &[f32], x: &[f32], w: &[f32], m: usize, k: usize, n: usize, dw: &mut [f32]) -> Vec<f32> {
    debug_assert_eq!(dy.len(), m * n);
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(dw.len(), n * k);
    let mut dx = vec![0.0f32; m * k];
    for t in 0..m {
        let xt = &x[t * k..(t + 1) * k];
        let dxt = &mut dx[t * k..(t + 1) * k];
        for j in 0..n {
            let g = dy[t * n + j];
            let wj = &w[j * k..(j + 1) * k];
            let dwj = &mut dw[j * k..(j + 1) * k];
            for c in 0..k {
                dwj[c] += g * xt[c];
                dxt[c] += g * wj[c];
            }
        }
    }
    dx
}

/// RMSNorm forward: `y = x * r * w` with `r = 1 / sqrt(mean(x^2) + eps)` per row.
/// Same arithmetic (and operation order) as `sentinel_kernels::reference::rms_norm`, so training
/// and inference agree. Returns `(y, r)`.
pub fn rmsnorm_fwd(x: &[f32], w: &[f32], rows: usize, n: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(x.len(), rows * n);
    debug_assert_eq!(w.len(), n);
    let mut y = vec![0.0f32; rows * n];
    let mut inv_rms = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &x[r * n..(r + 1) * n];
        let mut sum_sq = 0.0f32;
        for &v in row {
            sum_sq += v * v;
        }
        let rsqrt = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        inv_rms[r] = rsqrt;
        for c in 0..n {
            y[r * n + c] = row[c] * rsqrt * w[c];
        }
    }
    (y, inv_rms)
}

/// RMSNorm backward. With `y_c = w_c x_c r` and `dr/dx_c = -r^3 x_c / n`:
/// `dx_c = r w_c dy_c - (r^3 x_c / n) sum_k dy_k w_k x_k`, `dw_c += dy_c x_c r`.
pub fn rmsnorm_bwd(dy: &[f32], x: &[f32], w: &[f32], inv_rms: &[f32], rows: usize, n: usize, dw: &mut [f32]) -> Vec<f32> {
    let mut dx = vec![0.0f32; rows * n];
    for r in 0..rows {
        let xr = &x[r * n..(r + 1) * n];
        let dyr = &dy[r * n..(r + 1) * n];
        let rs = inv_rms[r];
        let mut dot = 0.0f32;
        for c in 0..n {
            dot += dyr[c] * w[c] * xr[c];
            dw[c] += dyr[c] * xr[c] * rs;
        }
        let coef = rs * rs * rs / n as f32 * dot;
        for c in 0..n {
            dx[r * n + c] = rs * w[c] * dyr[c] - xr[c] * coef;
        }
    }
    dx
}

/// Mean cross-entropy over `t` positions of `[t, v]` logits, and its gradient w.r.t. the logits
/// (`(softmax - onehot) / t`).
pub fn cross_entropy(logits: &[f32], targets: &[i32], t: usize, v: usize) -> (f32, Vec<f32>) {
    debug_assert_eq!(logits.len(), t * v);
    debug_assert_eq!(targets.len(), t);
    let mut loss = 0.0f64;
    let mut dlogits = vec![0.0f32; t * v];
    let inv_t = 1.0 / t as f32;
    for ti in 0..t {
        let row = &logits[ti * v..(ti + 1) * v];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &l in row {
            sum += (l - max).exp();
        }
        let lse = max + sum.ln();
        let target = targets[ti] as usize;
        loss += (lse - row[target]) as f64;
        for j in 0..v {
            dlogits[ti * v + j] = (row[j] - max).exp() / sum * inv_t;
        }
        dlogits[ti * v + target] -= inv_t;
    }
    ((loss / t as f64) as f32, dlogits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn linear_bwd_known_values() {
        // y = x W^T with x = [[1, 2]], W = [[3, 4], [5, 6]]; dy = [[1, 10]]
        let mut dw = vec![0.0; 4];
        let dx = linear_bwd(&[1.0, 10.0], &[1.0, 2.0], &[3.0, 4.0, 5.0, 6.0], 1, 2, 2, &mut dw);
        assert_eq!(dw, vec![1.0, 2.0, 10.0, 20.0]); // dy^T x
        assert_eq!(dx, vec![53.0, 64.0]); // dy W = [1*3 + 10*5, 1*4 + 10*6]
    }

    #[test]
    fn linear_bwd_accumulates_into_dw() {
        let mut dw = vec![1.0; 2];
        linear_bwd(&[2.0], &[3.0, 4.0], &[0.0, 0.0], 1, 2, 1, &mut dw);
        assert_eq!(dw, vec![7.0, 9.0]);
    }

    #[test]
    fn rmsnorm_fwd_matches_the_inference_kernel() {
        let x: Vec<f32> = (0..2 * 6).map(|i| (i as f32 * 0.7).sin() * 2.0).collect();
        let w: Vec<f32> = (0..6).map(|i| 0.5 + i as f32 * 0.1).collect();
        let (y, _) = rmsnorm_fwd(&x, &w, 2, 6, 1e-6);
        let (want, _) = sentinel_kernels_ref(&x, &w);
        assert_eq!(y, want);
    }

    // The inference model uses sentinel-kernels; re-derive it here without the dependency.
    fn sentinel_kernels_ref(x: &[f32], w: &[f32]) -> (Vec<f32>, ()) {
        let n = w.len();
        let mut out = vec![0.0f32; x.len()];
        for r in 0..x.len() / n {
            let mut ss = 0.0f32;
            for c in 0..n {
                ss += x[r * n + c] * x[r * n + c];
            }
            let rs = 1.0 / (ss / n as f32 + 1e-6).sqrt();
            for c in 0..n {
                out[r * n + c] = x[r * n + c] * rs * w[c];
            }
        }
        (out, ())
    }

    #[test]
    fn rmsnorm_bwd_matches_finite_differences() {
        let (rows, n, eps) = (2usize, 5usize, 1e-6f32);
        let x: Vec<f32> = (0..rows * n).map(|i| (i as f32 * 0.9).cos() + 0.3).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.8 + i as f32 * 0.15).collect();
        let dy: Vec<f32> = (0..rows * n).map(|i| (i as f32 * 1.3).sin()).collect();
        // scalar objective: L = sum(dy * y)
        let objective = |x: &[f32], w: &[f32]| -> f32 {
            let (y, _) = rmsnorm_fwd(x, w, rows, n, eps);
            y.iter().zip(&dy).map(|(a, b)| a * b).sum()
        };
        let (_, r) = rmsnorm_fwd(&x, &w, rows, n, eps);
        let mut dw = vec![0.0; n];
        let dx = rmsnorm_bwd(&dy, &x, &w, &r, rows, n, &mut dw);

        let h = 1e-2f32;
        for i in 0..x.len() {
            let (mut p, mut m) = (x.clone(), x.clone());
            p[i] += h;
            m[i] -= h;
            let num = (objective(&p, &w) - objective(&m, &w)) / (2.0 * h);
            assert!((num - dx[i]).abs() < 2e-3, "dx[{i}]: numeric {num}, analytic {}", dx[i]);
        }
        for i in 0..n {
            let (mut p, mut m) = (w.clone(), w.clone());
            p[i] += h;
            m[i] -= h;
            let num = (objective(&x, &p) - objective(&x, &m)) / (2.0 * h);
            assert!((num - dw[i]).abs() < 2e-3, "dw[{i}]: numeric {num}, analytic {}", dw[i]);
        }
    }

    #[test]
    fn cross_entropy_uniform_logits_give_ln_v() {
        let (loss, d) = cross_entropy(&[0.0; 8], &[1, 3], 2, 4);
        assert!(close(loss, 4.0f32.ln()));
        // gradient rows sum to zero and the target entry is negative
        for t in 0..2 {
            assert!(d[t * 4..(t + 1) * 4].iter().sum::<f32>().abs() < 1e-6);
        }
        assert!(d[1] < 0.0 && d[4 + 3] < 0.0);
    }

    #[test]
    fn cross_entropy_is_small_for_confident_correct_predictions() {
        let (loss, _) = cross_entropy(&[10.0, 0.0, 0.0, 0.0], &[0], 1, 4);
        assert!(loss < 1e-3);
    }

    #[test]
    fn cross_entropy_gradient_matches_finite_differences() {
        let logits = [0.3f32, -1.2, 0.8, 0.1, 1.5, -0.4];
        let targets = [2, 1];
        let (_, d) = cross_entropy(&logits, &targets, 2, 3);
        let h = 1e-2f32;
        for i in 0..6 {
            let (mut p, mut m) = (logits, logits);
            p[i] += h;
            m[i] -= h;
            let num = (cross_entropy(&p, &targets, 2, 3).0 - cross_entropy(&m, &targets, 2, 3).0) / (2.0 * h);
            assert!((num - d[i]).abs() < 1e-3, "d[{i}]: numeric {num}, analytic {}", d[i]);
        }
    }
}
