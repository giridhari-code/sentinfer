//! Full-sequence forward pass with saved activations and a hand-written backward pass for Seek-V.
//!
//! The forward pass mirrors `SeekV::forward` (same ops, same order), so a model trained here
//! behaves identically when loaded for inference. Correctness of the backward pass is checked
//! against finite differences in the tests below.

use sentinel_core::{Result, SentinelError};
use sentinel_model::ops::{matmul_nt, rope_inplace, rope_inverse_inplace, silu, softmax_inplace};
use sentinel_model::{SeekVConfig, SeekVWeights};

use crate::math::{cross_entropy, linear_bwd, rmsnorm_bwd, rmsnorm_fwd};

struct LayerAct {
    x_in: Vec<f32>,
    r1: Vec<f32>,
    hn: Vec<f32>,
    /// After rotary embedding.
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// `[t, heads, t]`; entries with `j > t` are unused (zero).
    probs: Vec<f32>,
    attn: Vec<f32>,
    x_mid: Vec<f32>,
    r2: Vec<f32>,
    mn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
}

pub struct Activations {
    tokens: Vec<i32>,
    layers: Vec<LayerAct>,
    x_last: Vec<f32>,
    rf: Vec<f32>,
    xf: Vec<f32>,
    /// `[t, vocab]`: logits for every position.
    pub logits: Vec<f32>,
}

fn invalid(msg: String) -> SentinelError {
    SentinelError::InvalidState(msg)
}

pub fn forward(cfg: &SeekVConfig, w: &SeekVWeights, tokens: &[i32]) -> Result<Activations> {
    let t = tokens.len();
    if t == 0 {
        return Err(invalid("empty token sequence".into()));
    }
    if t > cfg.max_positions {
        return Err(invalid(format!("sequence of {t} tokens exceeds max_positions {}", cfg.max_positions)));
    }
    if let Some(bad) = tokens.iter().find(|&&tok| tok < 0 || tok as usize >= cfg.vocab_size) {
        return Err(invalid(format!("token {bad} outside vocabulary of {}", cfg.vocab_size)));
    }
    let (d, h, hk, hd, inter, vocab) =
        (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.intermediate_size, cfg.vocab_size);
    let (eps, theta) = (cfg.rms_eps, cfg.rope_theta);
    let group = h / hk;
    let kv = hk * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut x = vec![0.0f32; t * d];
    for (i, &tok) in tokens.iter().enumerate() {
        let tok = tok as usize;
        x[i * d..(i + 1) * d].copy_from_slice(&w.embed[tok * d..(tok + 1) * d]);
    }

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for lw in &w.layers {
        let x_in = x.clone();
        let (hn, r1) = rmsnorm_fwd(&x_in, &lw.attn_norm, t, d, eps);
        let mut q = matmul_nt(&hn, &lw.wq, t, d, h * hd);
        let mut k = matmul_nt(&hn, &lw.wk, t, d, kv);
        let v = matmul_nt(&hn, &lw.wv, t, d, kv);
        rope_inplace(&mut q, t, h, hd, 0, theta);
        rope_inplace(&mut k, t, hk, hd, 0, theta);

        let mut probs = vec![0.0f32; t * h * t];
        let mut attn = vec![0.0f32; t * h * hd];
        for ti in 0..t {
            for head in 0..h {
                let kvh = head / group;
                let base = ti * h + head;
                let qv = &q[base * hd..(base + 1) * hd];
                let row = &mut probs[base * t..base * t + ti + 1];
                for (j, s) in row.iter_mut().enumerate() {
                    let kj = &k[j * kv + kvh * hd..j * kv + (kvh + 1) * hd];
                    let dot: f32 = qv.iter().zip(kj).map(|(a, b)| a * b).sum();
                    *s = dot * scale;
                }
                softmax_inplace(row);
                let out = &mut attn[base * hd..(base + 1) * hd];
                for (j, &p) in row.iter().enumerate() {
                    let vj = &v[j * kv + kvh * hd..j * kv + (kvh + 1) * hd];
                    for c in 0..hd {
                        out[c] += p * vj[c];
                    }
                }
            }
        }
        let o = matmul_nt(&attn, &lw.wo, t, h * hd, d);
        let x_mid: Vec<f32> = x_in.iter().zip(&o).map(|(a, b)| a + b).collect();

        let (mn, r2) = rmsnorm_fwd(&x_mid, &lw.mlp_norm, t, d, eps);
        let gate = matmul_nt(&mn, &lw.w_gate, t, d, inter);
        let up = matmul_nt(&mn, &lw.w_up, t, d, inter);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        let down = matmul_nt(&act, &lw.w_down, t, inter, d);
        x = x_mid.iter().zip(&down).map(|(a, b)| a + b).collect();

        layers.push(LayerAct { x_in, r1, hn, q, k, v, probs, attn, x_mid, r2, mn, gate, up, act });
    }

    let (xf, rf) = rmsnorm_fwd(&x, &w.final_norm, t, d, eps);
    let logits = matmul_nt(&xf, &w.lm_head, t, d, vocab);
    Ok(Activations { tokens: tokens.to_vec(), layers, x_last: x, rf, xf, logits })
}

/// Back-propagates `dlogits` (`[t, vocab]`) through the network, accumulating into `g`.
pub fn backward(cfg: &SeekVConfig, w: &SeekVWeights, a: &Activations, dlogits: &[f32], g: &mut SeekVWeights) {
    let t = a.tokens.len();
    let (d, h, hk, hd, inter, vocab) =
        (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.intermediate_size, cfg.vocab_size);
    let theta = cfg.rope_theta;
    let group = h / hk;
    let kv = hk * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    let dxf = linear_bwd(dlogits, &a.xf, &w.lm_head, t, d, vocab, &mut g.lm_head);
    let mut dx = rmsnorm_bwd(&dxf, &a.x_last, &w.final_norm, &a.rf, t, d, &mut g.final_norm);

    for l in (0..cfg.num_layers).rev() {
        let lw = &w.layers[l];
        let la = &a.layers[l];
        let lg = &mut g.layers[l];

        // MLP block: x_out = x_mid + down(act(gate, up))
        let dact = linear_bwd(&dx, &la.act, &lw.w_down, t, inter, d, &mut lg.w_down);
        let mut dgate = vec![0.0f32; t * inter];
        let mut dup = vec![0.0f32; t * inter];
        for i in 0..t * inter {
            let gt = la.gate[i];
            let s = 1.0 / (1.0 + (-gt).exp());
            dup[i] = dact[i] * gt * s;
            dgate[i] = dact[i] * la.up[i] * s * (1.0 + gt * (1.0 - s));
        }
        let dmn_gate = linear_bwd(&dgate, &la.mn, &lw.w_gate, t, d, inter, &mut lg.w_gate);
        let dmn_up = linear_bwd(&dup, &la.mn, &lw.w_up, t, d, inter, &mut lg.w_up);
        let dmn: Vec<f32> = dmn_gate.iter().zip(&dmn_up).map(|(a, b)| a + b).collect();
        let dx_norm2 = rmsnorm_bwd(&dmn, &la.x_mid, &lw.mlp_norm, &la.r2, t, d, &mut lg.mlp_norm);
        let dx_mid: Vec<f32> = dx.iter().zip(&dx_norm2).map(|(a, b)| a + b).collect();

        // Attention block: x_mid = x_in + wo(attn(...))
        let dattn = linear_bwd(&dx_mid, &la.attn, &lw.wo, t, h * hd, d, &mut lg.wo);
        let mut dq = vec![0.0f32; t * h * hd];
        let mut dk = vec![0.0f32; t * kv];
        let mut dv = vec![0.0f32; t * kv];
        let mut dp: Vec<f32> = Vec::new();
        for ti in 0..t {
            for head in 0..h {
                let kvh = head / group;
                let base = ti * h + head;
                let p = &la.probs[base * t..base * t + ti + 1];
                let dout = &dattn[base * hd..(base + 1) * hd];
                dp.clear();
                for (j, &pj) in p.iter().enumerate() {
                    let vj = &la.v[j * kv + kvh * hd..j * kv + (kvh + 1) * hd];
                    dp.push(dout.iter().zip(vj).map(|(a, b)| a * b).sum::<f32>());
                    let dvj = &mut dv[j * kv + kvh * hd..j * kv + (kvh + 1) * hd];
                    for c in 0..hd {
                        dvj[c] += pj * dout[c];
                    }
                }
                // softmax backward: ds_j = p_j (dp_j - sum_i p_i dp_i)
                let dot: f32 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
                let qv = &la.q[base * hd..(base + 1) * hd];
                for (j, &pj) in p.iter().enumerate() {
                    let ds = pj * (dp[j] - dot) * scale;
                    let kj = &la.k[j * kv + kvh * hd..j * kv + (kvh + 1) * hd];
                    let dqv = &mut dq[base * hd..(base + 1) * hd];
                    let dkj = &mut dk[j * kv + kvh * hd..j * kv + (kvh + 1) * hd];
                    for c in 0..hd {
                        dqv[c] += ds * kj[c];
                        dkj[c] += ds * qv[c];
                    }
                }
            }
        }
        rope_inverse_inplace(&mut dq, t, h, hd, 0, theta);
        rope_inverse_inplace(&mut dk, t, hk, hd, 0, theta);
        let dhn_q = linear_bwd(&dq, &la.hn, &lw.wq, t, d, h * hd, &mut lg.wq);
        let dhn_k = linear_bwd(&dk, &la.hn, &lw.wk, t, d, kv, &mut lg.wk);
        let dhn_v = linear_bwd(&dv, &la.hn, &lw.wv, t, d, kv, &mut lg.wv);
        let dhn: Vec<f32> = (0..t * d).map(|i| dhn_q[i] + dhn_k[i] + dhn_v[i]).collect();
        let dx_norm1 = rmsnorm_bwd(&dhn, &la.x_in, &lw.attn_norm, &la.r1, t, d, &mut lg.attn_norm);
        dx = dx_mid.iter().zip(&dx_norm1).map(|(a, b)| a + b).collect();
    }

    for (ti, &tok) in a.tokens.iter().enumerate() {
        let tok = tok as usize;
        for c in 0..d {
            g.embed[tok * d + c] += dx[ti * d + c];
        }
    }
}

fn split_window<'a>(cfg: &SeekVConfig, window: &'a [i32]) -> Result<(&'a [i32], &'a [i32])> {
    if window.len() < 2 {
        return Err(invalid("a training window needs at least 2 tokens".into()));
    }
    let (inputs, targets) = (&window[..window.len() - 1], &window[1..]);
    if let Some(bad) = targets.iter().find(|&&tok| tok < 0 || tok as usize >= cfg.vocab_size) {
        return Err(invalid(format!("target token {bad} outside vocabulary of {}", cfg.vocab_size)));
    }
    Ok((inputs, targets))
}

/// Loss of one window of `T + 1` tokens (inputs `[0..T]`, targets `[1..T+1]`), accumulating its
/// gradient into `grads`. Returns the mean per-token loss.
pub fn window_loss_and_grads(cfg: &SeekVConfig, w: &SeekVWeights, window: &[i32], grads: &mut SeekVWeights) -> Result<f32> {
    let (inputs, targets) = split_window(cfg, window)?;
    let acts = forward(cfg, w, inputs)?;
    let (loss, dlogits) = cross_entropy(&acts.logits, targets, inputs.len(), cfg.vocab_size);
    backward(cfg, w, &acts, &dlogits, grads);
    Ok(loss)
}

/// Forward-only loss of one window.
pub fn window_loss(cfg: &SeekVConfig, w: &SeekVWeights, window: &[i32]) -> Result<f32> {
    let (inputs, targets) = split_window(cfg, window)?;
    let acts = forward(cfg, w, inputs)?;
    Ok(cross_entropy(&acts.logits, targets, inputs.len(), cfg.vocab_size).0)
}

/// Runs `f` over contiguous chunks of `items` on up to `threads` threads, in chunk order.
fn parallel_chunks<T: Sync, R: Send, F: Fn(&[T]) -> Result<R> + Sync>(items: &[T], threads: usize, f: F) -> Result<Vec<R>> {
    let threads = threads.clamp(1, items.len().max(1));
    let chunk = ((items.len() + threads - 1) / threads).max(1);
    let f = &f;
    let results: Vec<Result<R>> = std::thread::scope(|s| {
        let handles: Vec<_> = items.chunks(chunk).map(|part| s.spawn(move || f(part))).collect();
        handles.into_iter().map(|h| h.join().expect("worker thread panicked")).collect()
    });
    results.into_iter().collect()
}

/// Mean loss and mean gradient over a batch of windows.
pub fn compute_gradients(cfg: &SeekVConfig, w: &SeekVWeights, windows: &[Vec<i32>], threads: usize) -> Result<(f32, SeekVWeights)> {
    if windows.is_empty() {
        return Err(invalid("empty batch".into()));
    }
    let parts = parallel_chunks(windows, threads, |part| {
        let mut g = SeekVWeights::zeros(cfg);
        let mut sum = 0.0f64;
        for win in part {
            sum += window_loss_and_grads(cfg, w, win, &mut g)? as f64;
        }
        Ok((sum, g))
    })?;
    let mut iter = parts.into_iter();
    let (mut loss_sum, mut grads) = iter.next().ok_or_else(|| invalid("no gradient parts".into()))?;
    for (s, g) in iter {
        loss_sum += s;
        grads.add_assign(&g);
    }
    let n = windows.len();
    grads.scale(1.0 / n as f32);
    Ok(((loss_sum / n as f64) as f32, grads))
}

/// Mean loss over `windows` (no gradients).
pub fn mean_loss(cfg: &SeekVConfig, w: &SeekVWeights, windows: &[Vec<i32>], threads: usize) -> Result<f32> {
    if windows.is_empty() {
        return Err(invalid("empty evaluation set".into()));
    }
    let parts = parallel_chunks(windows, threads, |part| {
        let mut sum = 0.0f64;
        for win in part {
            sum += window_loss(cfg, w, win)? as f64;
        }
        Ok(sum)
    })?;
    Ok((parts.iter().sum::<f64>() / windows.len() as f64) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sentinel_model::ops::SplitMix;
    use sentinel_model::{Model, SeekV, StepInput};

    /// Weights large enough that logits and gradients are not vanishingly small, with
    /// non-trivial norm scales so their gradients are exercised too.
    fn test_weights(cfg: &SeekVConfig, seed: u64) -> SeekVWeights {
        let mut w = SeekVWeights::random(cfg, seed);
        w.scale(4.0);
        let mut rng = SplitMix::new(seed ^ 0xABCD);
        for (name, t) in w.named_tensors_mut() {
            if name.ends_with("norm") {
                for x in t.iter_mut() {
                    *x = 1.0 + rng.uniform(-0.3, 0.3);
                }
            }
        }
        w
    }

    fn window(cfg: &SeekVConfig, len: usize, seed: u64) -> Vec<i32> {
        let mut rng = SplitMix::new(seed);
        (0..len).map(|_| (rng.next_u64() % cfg.vocab_size as u64) as i32).collect()
    }

    fn perturb(w: &mut SeekVWeights, tensor: usize, idx: usize, delta: f32) {
        w.named_tensors_mut()[tensor].1[idx] += delta;
    }

    fn check_gradients(cfg: &SeekVConfig, seed: u64) {
        let w = test_weights(cfg, seed);
        let win = window(cfg, 7, seed + 1);
        let mut g = SeekVWeights::zeros(cfg);
        window_loss_and_grads(cfg, &w, &win, &mut g).unwrap();

        let names: Vec<String> = w.named_tensors().into_iter().map(|(n, _)| n).collect();
        let mut rng = SplitMix::new(seed + 2);
        let h = 1e-2f32;
        let mut max_checked = 0.0f32;
        for (ti, name) in names.iter().enumerate() {
            let len = w.named_tensors()[ti].1.len();
            for _ in 0..6usize.min(len) {
                let idx = (rng.next_u64() % len as u64) as usize;
                let (mut plus, mut minus) = (w.clone(), w.clone());
                perturb(&mut plus, ti, idx, h);
                perturb(&mut minus, ti, idx, -h);
                let numeric = (window_loss(cfg, &plus, &win).unwrap() - window_loss(cfg, &minus, &win).unwrap()) / (2.0 * h);
                let analytic = g.named_tensors()[ti].1[idx];
                assert!(
                    (numeric - analytic).abs() <= 1e-3 + 0.03 * analytic.abs(),
                    "{name}[{idx}]: numeric {numeric}, analytic {analytic}"
                );
                max_checked = max_checked.max(analytic.abs());
            }
        }
        // the check is only meaningful if some gradients are clearly non-zero
        assert!(max_checked > 1e-2, "gradients too small to be a useful check ({max_checked})");
    }

    #[test]
    fn gradients_match_finite_differences_with_grouped_query_attention() {
        check_gradients(&SeekVConfig::micro(11), 1); // 2 query heads share 1 kv head
    }

    #[test]
    fn gradients_match_finite_differences_with_multi_head_attention() {
        let cfg = SeekVConfig {
            hidden_size: 12,
            num_layers: 1,
            num_heads: 3,
            num_kv_heads: 3,
            head_dim: 4,
            intermediate_size: 10,
            ..SeekVConfig::micro(9)
        };
        check_gradients(&cfg, 7);
    }

    #[test]
    fn training_forward_matches_the_inference_model() {
        let cfg = SeekVConfig::tiny();
        let w = SeekVWeights::random(&cfg, 3);
        let tokens = [1, 2, 3, 4, 5, 6];
        let acts = forward(&cfg, &w, &tokens).unwrap();
        let last = &acts.logits[(tokens.len() - 1) * cfg.vocab_size..];

        let mut model = SeekV::new(cfg.clone(), w).unwrap();
        let want = model.forward(&[StepInput { seq_id: 1, tokens: &tokens, start_pos: 0 }]).unwrap();
        for (a, b) in last.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn batch_gradients_do_not_depend_on_the_thread_count() {
        let cfg = SeekVConfig::micro(11);
        let w = test_weights(&cfg, 4);
        let windows: Vec<Vec<i32>> = (0..6).map(|i| window(&cfg, 8, 100 + i)).collect();
        let (l1, g1) = compute_gradients(&cfg, &w, &windows, 1).unwrap();
        let (l3, g3) = compute_gradients(&cfg, &w, &windows, 3).unwrap();
        assert!((l1 - l3).abs() < 1e-5);
        for ((name, a), (_, b)) in g1.named_tensors().into_iter().zip(g3.named_tensors()) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() <= 1e-5 + 1e-4 * x.abs(), "{name}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn batch_gradient_is_the_mean_of_window_gradients() {
        let cfg = SeekVConfig::micro(11);
        let w = test_weights(&cfg, 5);
        let windows: Vec<Vec<i32>> = (0..3).map(|i| window(&cfg, 6, 50 + i)).collect();
        let (_, batch) = compute_gradients(&cfg, &w, &windows, 1).unwrap();
        let mut sum = SeekVWeights::zeros(&cfg);
        for win in &windows {
            window_loss_and_grads(&cfg, &w, win, &mut sum).unwrap();
        }
        sum.scale(1.0 / 3.0);
        for ((name, a), (_, b)) in batch.named_tensors().into_iter().zip(sum.named_tensors()) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() <= 1e-6 + 1e-5 * x.abs(), "{name}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn a_small_gradient_step_lowers_the_loss() {
        let cfg = SeekVConfig::micro(11);
        let mut w = test_weights(&cfg, 6);
        let windows: Vec<Vec<i32>> = (0..4).map(|i| window(&cfg, 8, 200 + i)).collect();
        let (before, mut g) = compute_gradients(&cfg, &w, &windows, 1).unwrap();
        g.scale(-0.05);
        w.add_assign(&g);
        let after = mean_loss(&cfg, &w, &windows, 1).unwrap();
        assert!(after < before, "loss went from {before} to {after}");
    }

    #[test]
    fn rejects_malformed_input() {
        let cfg = SeekVConfig::micro(11);
        let w = SeekVWeights::random(&cfg, 1);
        let mut g = SeekVWeights::zeros(&cfg);
        assert!(window_loss_and_grads(&cfg, &w, &[1], &mut g).is_err()); // too short
        assert!(window_loss_and_grads(&cfg, &w, &[1, 99], &mut g).is_err()); // target outside vocab
        assert!(window_loss_and_grads(&cfg, &w, &[99, 1], &mut g).is_err()); // input outside vocab
        let long = vec![1; cfg.max_positions + 2];
        assert!(window_loss_and_grads(&cfg, &w, &long, &mut g).is_err());
        assert!(compute_gradients(&cfg, &w, &[], 1).is_err());
    }
}
