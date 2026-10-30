//! DeepGEMM-style blocked GEMM reference.
//!
//! Real DeepGEMM is a CUDA FP8/BF16 library (vendor submodule empty here).
//! This module provides a **cache-tiled** `matmul_nt` that matches the reference
//! `matmul_nt` numerics while being friendlier for large shapes on CPU.

use crate::error::{ensure, KernelError, Result};
use crate::reference;

const DEFAULT_TM: usize = 64;
const DEFAULT_TN: usize = 64;
const DEFAULT_TK: usize = 64;

/// Tiled `X[m,k] @ W[n,k]^T` → `[m,n]`. Same math as [`reference::matmul_nt`].
pub fn deep_gemm_nt(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    deep_gemm_nt_tiled(x, w, m, k, n, DEFAULT_TM, DEFAULT_TN, DEFAULT_TK)
}

pub fn deep_gemm_nt_tiled(
    x: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
    tm: usize,
    tn: usize,
    tk: usize,
) -> Result<Vec<f32>> {
    ensure!(x.len() == m * k && w.len() == n * k, KernelError::Shape, "deep_gemm shape");
    if m * n == 0 {
        return Ok(vec![]);
    }
    let tm = tm.max(1);
    let tn = tn.max(1);
    let tk = tk.max(1);
    let mut out = vec![0.0f32; m * n];
    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + tm).min(m);
        let mut j0 = 0;
        while j0 < n {
            let j1 = (j0 + tn).min(n);
            // clear tile
            for i in i0..i1 {
                for j in j0..j1 {
                    out[i * n + j] = 0.0;
                }
            }
            let mut p0 = 0;
            while p0 < k {
                let p1 = (p0 + tk).min(k);
                for i in i0..i1 {
                    for j in j0..j1 {
                        let mut acc = out[i * n + j];
                        for p in p0..p1 {
                            acc += x[i * k + p] * w[j * k + p];
                        }
                        out[i * n + j] = acc;
                    }
                }
                p0 = p1;
            }
            j0 = j1;
        }
        i0 = i1;
    }
    Ok(out)
}

/// Verify tiled GEMM matches oracle on a small random-ish pattern.
pub fn deep_gemm_matches_reference(m: usize, k: usize, n: usize) -> Result<()> {
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin()).collect();
    let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.013).cos()).collect();
    let a = reference::matmul_nt(&x, &w, m, k, n)?;
    let b = deep_gemm_nt(&x, &w, m, k, n)?;
    for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
        ensure!((u - v).abs() < 1e-4, KernelError::Backend, "deep_gemm mismatch at {i}: {u} vs {v}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_matches_naive() {
        deep_gemm_matches_reference(17, 33, 19).unwrap();
        deep_gemm_matches_reference(64, 64, 64).unwrap();
    }
}
