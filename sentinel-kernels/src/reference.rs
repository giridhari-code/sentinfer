//! Slice-level reference implementations. These are the oracle every backend is checked against.
//! Semantics follow the `*_ref` functions of the DeepInfer TileLang ops (see NOTICE).

use std::cmp::Ordering;

use crate::error::{ensure, KernelError, Result};

/// `norm_ref`: RMSNorm with optional fused residual add. Returns `(out, residual_out)`.
pub fn rms_norm(
    x: &[f32],
    weight: &[f32],
    residual: Option<&[f32]>,
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    ensure!(hidden > 0, KernelError::Shape, "hidden must be > 0");
    ensure!(x.len() == rows * hidden, KernelError::Shape, "x has {} elements, expected {rows} x {hidden}", x.len());
    ensure!(weight.len() == hidden, KernelError::Shape, "weight has {} elements, expected {hidden}", weight.len());
    if let Some(res) = residual {
        ensure!(res.len() == x.len(), KernelError::Shape, "residual has {} elements, expected {}", res.len(), x.len());
    }

    let mut out = vec![0.0f32; x.len()];
    let mut residual_out = residual.map(|_| vec![0.0f32; x.len()]);
    for r in 0..rows {
        let base = r * hidden;
        let mut sum_sq = 0.0f32;
        for c in 0..hidden {
            let mut v = x[base + c];
            if let Some(res) = residual {
                v += res[base + c];
            }
            if let Some(ro) = residual_out.as_mut() {
                ro[base + c] = v;
            }
            out[base + c] = v;
            sum_sq += v * v;
        }
        let rsqrt = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        for c in 0..hidden {
            out[base + c] = out[base + c] * rsqrt * weight[c];
        }
    }
    Ok((out, residual_out))
}

fn desc_stable(keys: &[f32]) -> impl Fn(&usize, &usize) -> Ordering + '_ {
    // partial_cmp (not total_cmp) so that -0.0 == 0.0, like torch's sort.
    move |&a, &b| keys[b].partial_cmp(&keys[a]).unwrap_or(Ordering::Equal)
}

/// `apply_top_p_top_k_ref`: stable top-p then stable top-k, masking logits in place with `-inf`.
pub fn apply_top_p_top_k(logits: &mut [f32], top_p: &[f32], top_k: &[i32], rows: usize, vocab: usize) -> Result<()> {
    ensure!(vocab > 0, KernelError::Shape, "vocab must be > 0");
    ensure!(logits.len() == rows * vocab, KernelError::Shape, "logits has {} elements, expected {rows} x {vocab}", logits.len());
    ensure!(top_p.len() == rows, KernelError::Shape, "top_p has {} elements, expected {rows}", top_p.len());
    ensure!(top_k.len() == rows, KernelError::Shape, "top_k has {} elements, expected {rows}", top_k.len());

    let mut probs = vec![0.0f32; vocab];
    let mut order: Vec<usize> = Vec::with_capacity(vocab);
    for r in 0..rows {
        let row = &mut logits[r * vocab..(r + 1) * vocab];

        // top-p over softmax probabilities
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (p, &l) in probs.iter_mut().zip(row.iter()) {
            *p = (l - max).exp();
            sum += *p;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        order.clear();
        order.extend(0..vocab);
        order.sort_by(desc_stable(&probs)); // Vec::sort_by is stable
        let mut cum = 0.0f32;
        for &i in &order {
            let p = probs[i];
            cum += p;
            // (cumsum - p) in float32, exactly as the reference computes it
            if cum - p > top_p[r] {
                row[i] = f32::NEG_INFINITY;
            }
        }

        // top-k over the already top-p-masked logits
        let k = top_k[r];
        if k >= 0 {
            order.clear();
            order.extend(0..vocab);
            order.sort_by(desc_stable(row));
            let limit = (k as usize).min(vocab);
            for (rank, &i) in order.iter().enumerate() {
                if rank >= limit {
                    row[i] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(())
}

/// `compute_seqlens_k_zos_ref`.
pub fn compute_seqlens_k_zos(
    seqlens_k: &mut [i32],
    seqlens_q: &[i32],
    remap: &[i32],
    prev_num_accepted_tokens: &[i32],
    nextn: i32,
) -> Result<()> {
    let n = seqlens_k.len();
    ensure!(seqlens_q.len() == n && remap.len() == n, KernelError::Shape, "seqlens_k/q/remap lengths differ ({n}, {}, {})", seqlens_q.len(), remap.len());
    for i in 0..n {
        seqlens_k[i] += seqlens_q[i];
        if remap[i] != -1 {
            // like the reference, other negative values clamp to index 0
            let idx = remap[i].max(0) as usize;
            let accepted = *prev_num_accepted_tokens
                .get(idx)
                .ok_or_else(|| KernelError::Index(format!("remap[{i}]={} outside prev_num_accepted_tokens ({})", remap[i], prev_num_accepted_tokens.len())))?;
            seqlens_k[i] -= nextn - accepted;
        }
    }
    Ok(())
}

/// `get_kv_ca_index_prefill_ref`.
pub fn kv_index_prefill(offset_len: &[i32], n: usize, length: usize) -> Result<Vec<i32>> {
    ensure!(offset_len.len() == n * 2, KernelError::Shape, "offset_len has {} elements, expected {n} x 2", offset_len.len());
    let mut out = vec![-1i32; n * length];
    for i in 0..n {
        let (offset, len) = (offset_len[2 * i], offset_len[2 * i + 1]);
        for j in 0..length {
            if (j as i64) < len as i64 {
                out[i * length + j] = offset + j as i32;
            }
        }
    }
    Ok(out)
}

/// `get_kv_ca_index_decode_ref`.
pub fn kv_index_decode(offset_len: &[i32], n: usize, length: usize, token_per_block: usize, block_tables: &[i32]) -> Result<Vec<i32>> {
    ensure!(offset_len.len() == n * 2, KernelError::Shape, "offset_len has {} elements, expected {n} x 2", offset_len.len());
    ensure!(token_per_block > 0, KernelError::Shape, "token_per_block must be > 0");
    let tpb = token_per_block as i64;
    let mut out = vec![-1i32; n * length];
    for i in 0..n {
        let (offset, len) = (offset_len[2 * i] as i64, offset_len[2 * i + 1] as i64);
        for j in 0..length {
            if (j as i64) >= len {
                continue; // stays -1
            }
            let pos = offset + j as i64;
            ensure!(pos >= 0, KernelError::Index, "negative token position {pos} in row {i}");
            let block = *block_tables
                .get((pos / tpb) as usize)
                .ok_or_else(|| KernelError::Index(format!("block index {} outside block_tables ({})", pos / tpb, block_tables.len())))?;
            let slot = block as i64 * tpb + pos % tpb;
            out[i * length + j] = i32::try_from(slot).map_err(|_| KernelError::Index(format!("slot {slot} does not fit in i32")))?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * (1.0 + b.abs())
    }

    #[test]
    fn rms_norm_known_values() {
        let (out, res) = rms_norm(&[3.0, 4.0], &[1.0, 1.0], None, 1, 2, 0.0).unwrap();
        assert!(res.is_none());
        assert!(close(out[0], 0.848_528_1) && close(out[1], 1.131_370_9));
    }

    #[test]
    fn rms_norm_residual_and_weight() {
        let (out, res) = rms_norm(&[1.0, 1.0], &[2.0, 0.5], Some(&[2.0, 2.0]), 1, 2, 0.0).unwrap();
        assert_eq!(res.unwrap(), vec![3.0, 3.0]);
        // x + r = [3, 3]; rms = 3 -> normalized [1, 1] -> * weight
        assert!(close(out[0], 2.0) && close(out[1], 0.5));
    }

    #[test]
    fn rms_norm_rejects_bad_shapes() {
        assert!(rms_norm(&[1.0; 3], &[1.0; 2], None, 1, 2, 0.0).is_err());
        assert!(rms_norm(&[1.0; 2], &[1.0; 3], None, 1, 2, 0.0).is_err());
        assert!(rms_norm(&[1.0; 2], &[1.0; 2], Some(&[1.0]), 1, 2, 0.0).is_err());
    }

    fn masked(logits: &[f32]) -> Vec<bool> {
        logits.iter().map(|l| *l == f32::NEG_INFINITY).collect()
    }

    #[test]
    fn top_p_keeps_the_nucleus() {
        // softmax([0,1,2,3]) ~ [0.032, 0.087, 0.237, 0.644]
        let mut l = vec![0.0, 1.0, 2.0, 3.0];
        apply_top_p_top_k(&mut l, &[0.5], &[-1], 1, 4).unwrap();
        assert_eq!(masked(&l), vec![true, true, true, false]);

        let mut l = vec![0.0, 1.0, 2.0, 3.0];
        apply_top_p_top_k(&mut l, &[0.7], &[-1], 1, 4).unwrap();
        assert_eq!(masked(&l), vec![true, true, false, false]);
    }

    #[test]
    fn top_p_one_and_top_k_minus_one_disable_filters() {
        let mut l = vec![0.0, 1.0, 2.0, 3.0];
        apply_top_p_top_k(&mut l, &[1.0], &[-1], 1, 4).unwrap();
        assert_eq!(l, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn top_k_is_stable_on_ties() {
        let mut l = vec![1.0, 1.0, 1.0];
        apply_top_p_top_k(&mut l, &[1.0], &[2], 1, 3).unwrap();
        assert_eq!(masked(&l), vec![false, false, true]);
    }

    #[test]
    fn top_k_zero_masks_everything_and_large_k_is_clamped() {
        let mut l = vec![0.5, 0.1];
        apply_top_p_top_k(&mut l, &[1.0], &[0], 1, 2).unwrap();
        assert_eq!(masked(&l), vec![true, true]);
        let mut l = vec![0.5, 0.1];
        apply_top_p_top_k(&mut l, &[1.0], &[99], 1, 2).unwrap();
        assert_eq!(masked(&l), vec![false, false]);
    }

    #[test]
    fn top_p_top_k_work_per_row() {
        let mut l = vec![0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0];
        apply_top_p_top_k(&mut l, &[0.5, 1.0], &[-1, 1], 2, 4).unwrap();
        assert_eq!(masked(&l), vec![true, true, true, false, false, true, true, true]);
    }

    #[test]
    fn seqlens_k_zos_adjusts_only_remapped() {
        let mut k = vec![10, 20];
        compute_seqlens_k_zos(&mut k, &[2, 3], &[-1, 0], &[1], 3).unwrap();
        assert_eq!(k, vec![12, 21]);
    }

    #[test]
    fn seqlens_k_zos_checks_remap_range() {
        let mut k = vec![10];
        assert!(compute_seqlens_k_zos(&mut k, &[1], &[5], &[1], 3).is_err());
        assert!(compute_seqlens_k_zos(&mut k, &[1, 1], &[-1], &[1], 3).is_err());
    }

    #[test]
    fn kv_index_prefill_pads_with_minus_one() {
        let out = kv_index_prefill(&[5, 3, 0, 0], 2, 4).unwrap();
        assert_eq!(out, vec![5, 6, 7, -1, -1, -1, -1, -1]);
    }

    #[test]
    fn kv_index_decode_translates_through_block_table() {
        // tokens 2..7 in blocks of 4: block table [7, 2] -> slots 30, 31, 8, 9, 10
        let out = kv_index_decode(&[2, 5], 1, 6, 4, &[7, 2]).unwrap();
        assert_eq!(out, vec![30, 31, 8, 9, 10, -1]);
    }

    #[test]
    fn kv_index_decode_reports_bad_block_table() {
        assert!(kv_index_decode(&[0, 8], 1, 8, 4, &[1]).is_err());
        assert!(kv_index_decode(&[0, 1], 1, 1, 0, &[1]).is_err());
    }
}

/// `out[m, n] = x[m, k] @ w[n, k]^T` (weights stored row-major as `[out, in]`).
pub fn matmul_nt(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    if x.len() != m * k || w.len() != n * k {
        return Err(KernelError::Shape(format!(
            "matmul_nt: x {} vs {m}x{k}, w {} vs {n}x{k}",
            x.len(),
            w.len()
        )));
    }
    let mut out = vec![0.0f32; m * n];
    // Block over output columns for better cache use on the weight matrix.
    const BN: usize = 16;
    for j0 in (0..n).step_by(BN) {
        let j1 = (j0 + BN).min(n);
        for i in 0..m {
            let xi = &x[i * k..(i + 1) * k];
            for j in j0..j1 {
                let wj = &w[j * k..(j + 1) * k];
                let mut acc = 0.0f32;
                let mut c = 0;
                while c + 4 <= k {
                    acc += xi[c] * wj[c]
                        + xi[c + 1] * wj[c + 1]
                        + xi[c + 2] * wj[c + 2]
                        + xi[c + 3] * wj[c + 3];
                    c += 4;
                }
                while c < k {
                    acc += xi[c] * wj[c];
                    c += 1;
                }
                out[i * n + j] = acc;
            }
        }
    }
    Ok(out)
}

pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// Softmax over the last dimension of a row-major `[rows, cols]` matrix, in place.
pub fn softmax_rows_inplace(v: &mut [f32], rows: usize, cols: usize) -> Result<()> {
    if v.len() != rows * cols {
        return Err(KernelError::Shape(format!(
            "softmax_rows: len {} != {rows}x{cols}",
            v.len()
        )));
    }
    for r in 0..rows {
        let row = &mut v[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in row.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
    Ok(())
}

/// Rotary embedding on `[tokens, heads, head_dim]`, pairs `(i, i + head_dim/2)`.
pub fn rope_inplace(
    x: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    start_pos: usize,
    theta: f32,
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(KernelError::Shape("rope: head_dim must be even".into()));
    }
    if x.len() != tokens * heads * head_dim {
        return Err(KernelError::Shape(format!(
            "rope: len {} != {tokens}x{heads}x{head_dim}",
            x.len()
        )));
    }
    let half = head_dim / 2;
    for t in 0..tokens {
        let pos = (start_pos + t) as f32;
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            for i in 0..half {
                let freq = theta.powf(-((2 * i) as f32) / head_dim as f32);
                let (sin, cos) = (pos * freq).sin_cos();
                let (x0, x1) = (x[base + i], x[base + i + half]);
                x[base + i] = x0 * cos - x1 * sin;
                x[base + i + half] = x1 * cos + x0 * sin;
            }
        }
    }
    Ok(())
}


/// Multi-threaded matmul over output rows (std threads, no extra deps).
pub fn matmul_nt_parallel(x: &[f32], w: &[f32], m: usize, k: usize, n: usize, threads: usize) -> Result<Vec<f32>> {
    if threads <= 1 || m < 4 {
        return matmul_nt(x, w, m, k, n);
    }
    if x.len() != m * k || w.len() != n * k {
        return matmul_nt(x, w, m, k, n);
    }
    let threads = threads.min(m).max(1);
    let mut out = vec![0.0f32; m * n];
    let chunk = (m + threads - 1) / threads;
    std::thread::scope(|scope| {
        for (t, chunk_out) in out.chunks_mut(chunk * n).enumerate() {
            let row0 = t * chunk;
            if row0 >= m {
                break;
            }
            let rows = (chunk_out.len() / n).min(m - row0);
            let x_slice = &x[row0 * k..(row0 + rows) * k];
            let w = w;
            scope.spawn(move || {
                if let Ok(partial) = matmul_nt(x_slice, w, rows, k, n) {
                    chunk_out[..partial.len()].copy_from_slice(&partial);
                }
            });
        }
    });
    Ok(out)
}


/// Causal multi-head attention over a growing KV cache.
///
/// * `q` — `[tokens, num_heads, head_dim]` for the new chunk
/// * `k_cache` / `v_cache` — `[total_pos, num_kv_heads, head_dim]` flattened (`total_pos = start_pos + tokens`)
/// * `group` — `num_heads / num_kv_heads`
/// Returns `attn` — `[tokens, num_heads, head_dim]`
pub fn attention_causal(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    tokens: usize,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::Shape("attention: invalid head grouping".into()));
    }
    let group = num_heads / num_kv_heads;
    let kv_stride = num_kv_heads * head_dim;
    let total = start_pos + tokens;
    if q.len() != tokens * num_heads * head_dim {
        return Err(KernelError::Shape(format!("attention: q len {}", q.len())));
    }
    if k_cache.len() < total * kv_stride || v_cache.len() < total * kv_stride {
        return Err(KernelError::Shape("attention: KV cache shorter than positions".into()));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn = vec![0.0f32; tokens * num_heads * head_dim];
    let mut scores = Vec::with_capacity(total);
    for ti in 0..tokens {
        let pos = start_pos + ti;
        for head in 0..num_heads {
            let kvh = head / group;
            let qv = &q[(ti * num_heads + head) * head_dim..(ti * num_heads + head + 1) * head_dim];
            scores.clear();
            for j in 0..=pos {
                let kj = &k_cache[j * kv_stride + kvh * head_dim..j * kv_stride + (kvh + 1) * head_dim];
                let mut dot = 0.0f32;
                let mut c = 0;
                while c + 4 <= head_dim {
                    dot += qv[c] * kj[c]
                        + qv[c + 1] * kj[c + 1]
                        + qv[c + 2] * kj[c + 2]
                        + qv[c + 3] * kj[c + 3];
                    c += 4;
                }
                while c < head_dim {
                    dot += qv[c] * kj[c];
                    c += 1;
                }
                scores.push(dot * scale);
            }
            // softmax
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for s in scores.iter_mut() {
                *s *= inv;
            }
            let out = &mut attn[(ti * num_heads + head) * head_dim..(ti * num_heads + head + 1) * head_dim];
            for (j, &wgt) in scores.iter().enumerate() {
                let vj = &v_cache[j * kv_stride + kvh * head_dim..j * kv_stride + (kvh + 1) * head_dim];
                for c in 0..head_dim {
                    out[c] += wgt * vj[c];
                }
            }
        }
    }
    Ok(attn)
}


/// Gather rows: `out[i] = src[indices[i]]` for rank-2 `[n_src, cols]`.
pub fn gather_rows(src: &[f32], indices: &[i32], cols: usize) -> Result<Vec<f32>> {
    let n_src = if cols == 0 { 0 } else { src.len() / cols };
    ensure!(src.len() == n_src * cols, KernelError::Shape, "gather_rows: src len");
    let mut out = vec![0.0f32; indices.len() * cols];
    for (i, &idx) in indices.iter().enumerate() {
        if idx < 0 || idx as usize >= n_src {
            return Err(KernelError::Index(format!("gather_rows: index {idx} out of range 0..{n_src}")));
        }
        let s = (idx as usize) * cols;
        out[i * cols..(i + 1) * cols].copy_from_slice(&src[s..s + cols]);
    }
    Ok(out)
}

/// Gather i32 ids by indices (MTP / input_ids_zos style).
pub fn gather_i32(src: &[i32], indices: &[i32]) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx < 0 || idx as usize >= src.len() {
            return Err(KernelError::Index(format!("gather_i32: {idx} vs {}", src.len())));
        }
        out.push(src[idx as usize]);
    }
    Ok(out)
}

/// Concatenate two kv-index tensors along the sequence dimension: `[n, l0]` + `[n, l1]` → `[n, l0+l1]`.
pub fn kv_index_cat(a: &[i32], b: &[i32], n: usize, l0: usize, l1: usize) -> Result<Vec<i32>> {
    ensure!(a.len() == n * l0 && b.len() == n * l1, KernelError::Shape, "kv_index_cat shape");
    let mut out = vec![0i32; n * (l0 + l1)];
    for i in 0..n {
        out[i * (l0 + l1)..i * (l0 + l1) + l0].copy_from_slice(&a[i * l0..(i + 1) * l0]);
        out[i * (l0 + l1) + l0..i * (l0 + l1) + l0 + l1].copy_from_slice(&b[i * l1..(i + 1) * l1]);
    }
    Ok(out)
}

/// Load paged KV cache into contiguous buffer.
/// `block_tables[seq * max_blocks + b]` → physical block; each block holds `block_size` tokens of `stride` floats.
pub fn load_paged_kv_cache(
    paged: &[f32],
    block_tables: &[i32],
    num_seqs: usize,
    max_blocks: usize,
    block_size: usize,
    stride: usize,
    context_lens: &[i32],
) -> Result<Vec<f32>> {
    ensure!(
        block_tables.len() == num_seqs * max_blocks && context_lens.len() == num_seqs,
        KernelError::Shape,
        "load_paged_kv_cache meta"
    );
    let max_ctx = context_lens.iter().map(|&c| c.max(0) as usize).max().unwrap_or(0);
    let mut out = vec![0.0f32; num_seqs * max_ctx * stride];
    for s in 0..num_seqs {
        let ctx = context_lens[s].max(0) as usize;
        for t in 0..ctx {
            let b = t / block_size;
            let off = t % block_size;
            if b >= max_blocks {
                return Err(KernelError::Index(format!("load_paged: block {b} >= {max_blocks}")));
            }
            let phys = block_tables[s * max_blocks + b];
            if phys < 0 {
                return Err(KernelError::Index(format!("load_paged: invalid block {phys}")));
            }
            let src = (phys as usize) * block_size * stride + off * stride;
            let dst = s * max_ctx * stride + t * stride;
            if src + stride > paged.len() {
                return Err(KernelError::Index("load_paged: paged OOB".into()));
            }
            out[dst..dst + stride].copy_from_slice(&paged[src..src + stride]);
        }
    }
    Ok(out)
}

/// Softmax + multinomial sample one token per row. `seed` drives a simple LCG per row.
pub fn sample_from_logits(logits: &[f32], rows: usize, vocab: usize, temperature: f32, seed: u64) -> Result<Vec<i32>> {
    ensure!(logits.len() == rows * vocab, KernelError::Shape, "sample_from_logits shape");
    let mut out = vec![0i32; rows];
    let mut state = seed | 1;
    for r in 0..rows {
        let row = &logits[r * vocab..(r + 1) * vocab];
        let mut mx = f32::NEG_INFINITY;
        for &v in row {
            if v.is_finite() && v > mx {
                mx = v;
            }
        }
        let inv_t = if temperature > 1e-6 { 1.0 / temperature } else { 0.0 };
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; vocab];
        if inv_t == 0.0 {
            // greedy
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            out[r] = best as i32;
            continue;
        }
        for (i, &v) in row.iter().enumerate() {
            let p = if v.is_finite() { ((v - mx) * inv_t).exp() } else { 0.0 };
            probs[i] = p;
            sum += p;
        }
        // LCG
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 33) as f32 / (1u32 << 31) as f32;
        let mut target = u * sum;
        let mut chosen = vocab.saturating_sub(1);
        for (i, &p) in probs.iter().enumerate() {
            target -= p;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        out[r] = chosen as i32;
    }
    Ok(out)
}

/// Fused: apply top-p / top-k then sample.
pub fn fused_sample(
    logits: &mut [f32],
    top_p: &[f32],
    top_k: &[i32],
    rows: usize,
    vocab: usize,
    temperature: f32,
    seed: u64,
) -> Result<Vec<i32>> {
    apply_top_p_top_k(logits, top_p, top_k, rows, vocab)?;
    sample_from_logits(logits, rows, vocab, temperature, seed)
}

#[cfg(test)]
mod phase3a_tests {
    use super::*;

    #[test]
    fn matmul_nt_identity() {
        // x 2x3, w 3x3 identity-ish rows
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let out = matmul_nt(&x, &w, 2, 3, 3).unwrap();
        assert_eq!(out, x);
    }

    #[test]
    fn silu_known() {
        let mut x = vec![0.0f32, 1.0];
        silu_inplace(&mut x);
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 0.731_058_6).abs() < 1e-5);
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let mut v = vec![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0];
        softmax_rows_inplace(&mut v, 2, 3).unwrap();
        let s0: f32 = v[0..3].iter().sum();
        let s1: f32 = v[3..6].iter().sum();
        assert!((s0 - 1.0).abs() < 1e-5);
        assert!((s1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gather_and_kv_cat_and_sample() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let g = gather_rows(&src, &[2, 0], 2).unwrap();
        assert_eq!(g, vec![5.0, 6.0, 1.0, 2.0]);
        let cat = kv_index_cat(&[1, 2, 3, 4], &[5, 6], 2, 2, 1).unwrap();
        assert_eq!(cat, vec![1, 2, 5, 3, 4, 6]);
        let mut logits = vec![0.0f32, 10.0, 0.0,  0.0, 0.0, 5.0];
        let ids = fused_sample(&mut logits, &[1.0, 1.0], &[-1, -1], 2, 3, 0.0, 1).unwrap();
        assert_eq!(ids, vec![1, 2]); // greedy argmax
    }

    #[test]
    fn load_paged_kv_smoke() {
        // 2 physical blocks, block_size=2, stride=2
        let paged = vec![
            1.0, 1.0, 2.0, 2.0, // block 0
            3.0, 3.0, 4.0, 4.0, // block 1
        ];
        let tables = vec![0, 1]; // one seq, two blocks
        let out = load_paged_kv_cache(&paged, &tables, 1, 2, 2, 2, &[3]).unwrap();
        // tokens 0,1 from block0; token 2 from block1 offset0
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn attention_causal_smoke() {
        // 1 token, 2 heads, 1 kv head, dim 4, start 0
        let q = vec![1.0f32, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0f32, 0.0, 0.0, 0.0];
        let v = vec![0.5f32, 0.5, 0.5, 0.5];
        let out = attention_causal(&q, &k, &v, 1, 0, 2, 1, 4).unwrap();
        assert_eq!(out.len(), 8);
        // single key → attn = v for both heads
        for i in 0..4 {
            assert!((out[i] - 0.5).abs() < 1e-5);
            assert!((out[4 + i] - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_at_zero_is_identity() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0]; // 1 token, 1 head, dim 4
        let orig = x.clone();
        rope_inplace(&mut x, 1, 1, 4, 0, 10000.0).unwrap();
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
