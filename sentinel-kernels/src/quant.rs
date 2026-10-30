//! Weight quantization: int8 / int4 with per-row scales.
//!
//! Layout for int8 matrix `W[n, k]` (same as matmul_nt weight):
//! * `q: [n * k]` i8
//! * `scale: [n]` f32 — dequant: `w[j,c] = q[j*k+c] as f32 * scale[j]`
//!
//! Int4 packs two values per byte (low nibble = even index).

use crate::error::{ensure, KernelError, Result};
use crate::reference;

/// Quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    Int8,
    Int4,
}

/// Row-scaled quantized matrix (weight side of `matmul_nt`).
#[derive(Debug, Clone)]
pub struct QuantMatrix {
    pub scheme: QuantScheme,
    pub rows: usize, // n
    pub cols: usize, // k
    /// Packed storage: int8 = one i8 per value; int4 = ceil(n*k/2) bytes as i8 storage.
    pub data: Vec<i8>,
    pub scale: Vec<f32>,
}

impl QuantMatrix {
    pub fn quantize_int8(w: &[f32], rows: usize, cols: usize) -> Result<Self> {
        ensure!(
            w.len() == rows * cols,
            KernelError::Shape,
            "quantize_int8: len {} != {}x{}",
            w.len(),
            rows,
            cols
        );
        let mut data = vec![0i8; rows * cols];
        let mut scale = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let mut amax = 0.0f32;
            for &v in row {
                amax = amax.max(v.abs());
            }
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            scale[r] = s;
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for c in 0..cols {
                let q = (row[c] * inv).round().clamp(-127.0, 127.0) as i8;
                data[r * cols + c] = q;
            }
        }
        Ok(Self { scheme: QuantScheme::Int8, rows, cols, data, scale })
    }

    pub fn quantize_int4(w: &[f32], rows: usize, cols: usize) -> Result<Self> {
        ensure!(
            w.len() == rows * cols,
            KernelError::Shape,
            "quantize_int4: len {} != {}x{}",
            w.len(),
            rows,
            cols
        );
        let packed_len = (rows * cols + 1) / 2;
        let mut data = vec![0i8; packed_len];
        let mut scale = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let mut amax = 0.0f32;
            for &v in row {
                amax = amax.max(v.abs());
            }
            let s = if amax > 0.0 { amax / 7.0 } else { 1.0 };
            scale[r] = s;
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for c in 0..cols {
                let q = (row[c] * inv).round().clamp(-7.0, 7.0) as i8;
                let idx = r * cols + c;
                let byte = idx / 2;
                if idx % 2 == 0 {
                    data[byte] = (data[byte] as u8 & 0xF0) as i8 | (q as u8 & 0x0F) as i8;
                } else {
                    data[byte] = ((q as u8 & 0x0F) << 4) as i8 | (data[byte] as u8 & 0x0F) as i8;
                }
            }
        }
        Ok(Self { scheme: QuantScheme::Int4, rows, cols, data, scale })
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        match self.scheme {
            QuantScheme::Int8 => {
                for r in 0..self.rows {
                    let s = self.scale[r];
                    for c in 0..self.cols {
                        out[r * self.cols + c] = self.data[r * self.cols + c] as f32 * s;
                    }
                }
            }
            QuantScheme::Int4 => {
                for r in 0..self.rows {
                    let s = self.scale[r];
                    for c in 0..self.cols {
                        let idx = r * self.cols + c;
                        let byte = self.data[idx / 2] as u8;
                        let nibble = if idx % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                        // sign-extend 4-bit two's complement in -8..7; we used -7..7
                        let q = if nibble >= 8 { (nibble as i8) - 16 } else { nibble as i8 };
                        out[r * self.cols + c] = q as f32 * s;
                    }
                }
            }
        }
        out
    }

    /// `X[m,k] @ W_q[n,k]^T` with on-the-fly dequant of W.
    pub fn matmul_nt(&self, x: &[f32], m: usize) -> Result<Vec<f32>> {
        ensure!(x.len() == m * self.cols, KernelError::Shape, "quant matmul: x shape");
        match self.scheme {
            QuantScheme::Int8 => matmul_nt_i8(x, &self.data, &self.scale, m, self.cols, self.rows),
            QuantScheme::Int4 => {
                let w = self.dequantize();
                reference::matmul_nt(x, &w, m, self.cols, self.rows)
            }
        }
    }
}

/// Int8 GEMM: accumulate in f32 using row scales.
pub fn matmul_nt_i8(x: &[f32], q: &[i8], scale: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    ensure!(x.len() == m * k && q.len() == n * k && scale.len() == n, KernelError::Shape, "matmul_nt_i8 shape");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            let s = scale[j];
            let base_x = i * k;
            let base_q = j * k;
            let mut c = 0;
            while c + 4 <= k {
                acc += x[base_x + c] * (q[base_q + c] as f32)
                    + x[base_x + c + 1] * (q[base_q + c + 1] as f32)
                    + x[base_x + c + 2] * (q[base_q + c + 2] as f32)
                    + x[base_x + c + 3] * (q[base_q + c + 3] as f32);
                c += 4;
            }
            while c < k {
                acc += x[base_x + c] * (q[base_q + c] as f32);
                c += 1;
            }
            out[i * n + j] = acc * s;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_roundtrip_and_matmul_close() {
        let rows = 4;
        let cols = 8;
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.07).sin()).collect();
        let q = QuantMatrix::quantize_int8(&w, rows, cols).unwrap();
        let de = q.dequantize();
        let mut max_err = 0.0f32;
        for (a, b) in w.iter().zip(de.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.05, "max_err {max_err}");

        let m = 2;
        let x: Vec<f32> = (0..m * cols).map(|i| (i as f32) * 0.01).collect();
        let ref_out = reference::matmul_nt(&x, &w, m, cols, rows).unwrap();
        let q_out = q.matmul_nt(&x, m).unwrap();
        for (a, b) in ref_out.iter().zip(q_out.iter()) {
            assert!((a - b).abs() < 0.15, "{a} vs {b}");
        }
    }

    #[test]
    fn int4_roundtrip_reasonable() {
        let rows = 3;
        let cols = 5;
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.1).cos()).collect();
        let q = QuantMatrix::quantize_int4(&w, rows, cols).unwrap();
        assert_eq!(q.data.len(), (rows * cols + 1) / 2);
        let de = q.dequantize();
        let mut max_err = 0.0f32;
        for (a, b) in w.iter().zip(de.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.25, "max_err {max_err}");
    }
}
