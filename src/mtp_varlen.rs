//! Variable-length MTP decode planning (port of `csrc/engine/mtp_varlen.hpp`).
//!
//! Decides how many draft tokens each request verifies this step, using the draft
//! accept-rate estimates and a measured (batch size -> forward time) curve, with
//! runtime feedback from recent forward/sampler timings.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::error::{ensure, Result, SentinelError};
use crate::sequence::MtpState;

/// `f64` with a total order so it can key a `BTreeMap` (multiset).
#[derive(Clone, Copy, Debug)]
struct OrdF64(f64);

impl PartialEq for OrdF64 {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Default, Debug)]
struct MultiSet {
    map: BTreeMap<OrdF64, usize>,
    len: usize,
}

impl MultiSet {
    fn insert(&mut self, v: f64) {
        *self.map.entry(OrdF64(v)).or_insert(0) += 1;
        self.len += 1;
    }

    fn remove(&mut self, v: f64) -> bool {
        let key = OrdF64(v);
        let Some(count) = self.map.get_mut(&key) else {
            return false;
        };
        *count -= 1;
        let empty = *count == 0;
        if empty {
            self.map.remove(&key);
        }
        self.len -= 1;
        true
    }

    fn max(&self) -> Option<f64> {
        self.map.keys().next_back().map(|k| k.0)
    }

    fn min(&self) -> Option<f64> {
        self.map.keys().next().map(|k| k.0)
    }

    fn pop_max(&mut self) -> Option<f64> {
        let m = self.max()?;
        self.remove(m);
        Some(m)
    }

    fn pop_min(&mut self) -> Option<f64> {
        let m = self.min()?;
        self.remove(m);
        Some(m)
    }
}

/// Dynamic median with two multisets: `lower` holds the smaller half and is
/// either the same size as `upper` or one larger.
#[derive(Default, Debug)]
struct MedianTracker {
    lower: MultiSet,
    upper: MultiSet,
}

impl MedianTracker {
    fn insert(&mut self, value: f64) {
        match self.lower.max() {
            Some(top) if value > top => self.upper.insert(value),
            _ => self.lower.insert(value),
        }
        self.rebalance();
    }

    fn erase(&mut self, value: f64) {
        if !self.lower.remove(value) {
            let removed = self.upper.remove(value);
            debug_assert!(removed, "erased a value that was never inserted");
        }
        self.rebalance();
    }

    fn size(&self) -> usize {
        self.lower.len + self.upper.len
    }

    /// Requires `size() > 0`.
    fn get(&self) -> f64 {
        let lo = self.lower.max().unwrap_or(0.0);
        if self.lower.len == self.upper.len {
            (lo + self.upper.min().unwrap_or(lo)) * 0.5
        } else {
            lo
        }
    }

    fn rebalance(&mut self) {
        if self.lower.len > self.upper.len + 1 {
            if let Some(m) = self.lower.pop_max() {
                self.upper.insert(m);
            }
        } else if self.lower.len < self.upper.len {
            if let Some(m) = self.upper.pop_min() {
                self.lower.insert(m);
            }
        }
    }
}

/// Samples kept by `SamplerRuntimeEstimator` (a struct field type cannot use `Self::CONST`).
const SAMPLER_WINDOW: usize = 9;

/// Median over a sliding window; used to estimate the sampler part of a decode step.
#[derive(Debug, Default)]
pub struct SamplerRuntimeEstimator {
    samples: [f64; SAMPLER_WINDOW],
    size: usize,
    next: usize,
    median_ms: f64,
}

impl SamplerRuntimeEstimator {
    const WINDOW: usize = SAMPLER_WINDOW;
    const MIN_SAMPLES: usize = 5;

    pub fn update(&mut self, value_ms: f64) {
        self.samples[self.next] = value_ms;
        self.next = (self.next + 1) % Self::WINDOW;
        self.size = (self.size + 1).min(Self::WINDOW);

        let mut sorted = self.samples;
        sorted[..self.size].sort_by(|a, b| a.total_cmp(b));
        let middle = self.size / 2;
        self.median_ms =
            if self.size % 2 == 0 { (sorted[middle - 1] + sorted[middle]) * 0.5 } else { sorted[middle] };
    }

    pub fn get(&self) -> Option<f64> {
        (self.size >= Self::MIN_SAMPLES).then_some(self.median_ms)
    }
}

/// Fits a per-graph-slot slope on top of the forward startup curve (which ignores context
/// length). Input is `(forward_ms - baseline_ms) / graph_size`. Keeps the last `WINDOW`
/// steps, takes the median per graph size, then the median of those medians (graph sizes
/// with at least `MIN_SAMPLES` samples, equally weighted).
#[derive(Debug, Default)]
pub struct ForwardRuntimeEstimator {
    window: VecDeque<(i32, f64)>,
    samples_by_graph: BTreeMap<i32, MedianTracker>,
    graph_medians: MedianTracker,
}

impl ForwardRuntimeEstimator {
    const WINDOW: usize = 5000;
    const MIN_SAMPLES: usize = 10;

    pub fn update(&mut self, graph_size: i32, normalized_residual: f64) {
        if self.window.len() == Self::WINDOW {
            if let Some((old_graph, old_value)) = self.window.pop_front() {
                self.update_graph(old_graph, old_value, false);
            }
        }
        self.window.push_back((graph_size, normalized_residual));
        self.update_graph(graph_size, normalized_residual, true);
    }

    pub fn get(&self) -> f64 {
        if self.graph_medians.size() == 0 {
            0.0
        } else {
            self.graph_medians.get()
        }
    }

    fn update_graph(&mut self, graph_size: i32, value: f64, insert: bool) {
        let samples = self.samples_by_graph.entry(graph_size).or_default();
        if samples.size() >= Self::MIN_SAMPLES {
            self.graph_medians.erase(samples.get());
        }
        if insert {
            samples.insert(value);
        } else {
            samples.erase(value);
        }
        if samples.size() >= Self::MIN_SAMPLES {
            self.graph_medians.insert(samples.get());
        } else if samples.size() == 0 {
            self.samples_by_graph.remove(&graph_size);
        }
    }
}

#[derive(Debug, Default)]
pub struct MtpVarlen {
    /// (batch_size, forward_ms) startup curve for speculative decoding; batch sizes strictly increasing.
    decode_perf_curve: Vec<(usize, f32)>,
    forward_time_slope: ForwardRuntimeEstimator,
    sampler_time_by_num_seqs: HashMap<usize, SamplerRuntimeEstimator>,
    sampler_time_fallback: SamplerRuntimeEstimator,
}

fn push_rates(queue: &mut Vec<(f32, usize)>, rates: &[f32], nd: usize, seq_idx: usize, has_draft: bool) -> Result<()> {
    if rates.is_empty() {
        // No estimate yet: assume the draft is accepted if there is one, otherwise 0.
        let fill = if has_draft { 1.0 } else { 0.0 };
        queue.extend(std::iter::repeat((fill, seq_idx)).take(nd));
        return Ok(());
    }
    // accept_rate is estimated by the MTP confidence head
    ensure!(
        rates.len() == nd,
        SentinelError::InvalidState,
        "accept rates ({}) do not match draft tokens ({nd})",
        rates.len()
    );
    queue.extend(rates.iter().map(|&r| (r, seq_idx)));
    Ok(())
}

fn by_rate_desc(a: &(f32, usize), b: &(f32, usize)) -> Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// Walks the perf curve while the total number of query tokens grows, keeping the
/// batch size with the best expected throughput.
struct BestBatch {
    curve_idx: usize,
    best_throughput: f64,
    best_bsz: usize,
}

impl BestBatch {
    fn update(&mut self, curve: &[(usize, f32)], cur_bsz: usize, cur_expected: f32, slope_ms: f64, sampler_ms: f64) {
        while self.curve_idx < curve.len() && cur_bsz > curve[self.curve_idx].0 {
            self.curve_idx += 1;
        }
        if self.curve_idx == curve.len() {
            return;
        }
        let (graph_size, baseline_ms) = curve[self.curve_idx];
        let time_ms = baseline_ms as f64 + slope_ms * graph_size as f64 + sampler_ms;
        let throughput = cur_expected as f64 / time_ms;
        if throughput > self.best_throughput {
            self.best_throughput = throughput;
            self.best_bsz = cur_bsz;
        }
    }
}

impl MtpVarlen {
    pub fn decode_perf_curve(&self) -> &[(usize, f32)] {
        &self.decode_perf_curve
    }

    pub fn set_decode_perf_curve(&mut self, curve: Vec<(usize, f32)>) -> Result<()> {
        ensure!(
            curve.windows(2).all(|w| w[0].0 < w[1].0),
            SentinelError::Config,
            "decode perf curve batch sizes must be strictly increasing"
        );
        self.decode_perf_curve = curve;
        Ok(())
    }

    pub fn update_runtime(&mut self, num_seqs: usize, graph_size: usize, forward_ms: f64, sampler_ms: f64) -> Result<()> {
        ensure!(graph_size > 0, SentinelError::InvalidState, "graph_size must be > 0");
        let idx = self.decode_perf_curve.partition_point(|p| p.0 < graph_size);
        ensure!(
            idx < self.decode_perf_curve.len(),
            SentinelError::InvalidState,
            "graph_size {graph_size} is beyond the decode perf curve"
        );
        let baseline_ms = self.decode_perf_curve[idx].1 as f64;
        self.forward_time_slope.update(graph_size as i32, (forward_ms - baseline_ms) / graph_size as f64);

        // The sampler has no startup curve: track history per num_seqs and globally.
        self.sampler_time_by_num_seqs.entry(num_seqs).or_default().update(sampler_ms);
        self.sampler_time_fallback.update(sampler_ms);
        Ok(())
    }

    /// Returns `seqlen_q` for each scheduled sequence (1 main token + verified draft tokens).
    pub fn schedule(&self, scheduled: &[&MtpState]) -> Result<Vec<usize>> {
        if self.decode_perf_curve.is_empty() {
            return Ok(scheduled.iter().map(|m| 1 + m.draft_tokens.len()).collect());
        }

        // At round N we use the accept rates of round N-2 to pick the best total batch size
        // (so confidence of later drafts cannot influence whether an earlier draft is verified),
        // and those of round N-1 to split that budget across requests. Per request there are
        // only two cases:
        //   1. schedule_cnt == process_cnt: every scheduled step already produced accept rates;
        //      accept_rates is round N-1 and prev_accept_rates is round N-2.
        //   2. schedule_cnt == process_cnt + 1: the latest forward has no rates yet (async
        //      scheduling); accept_rates is round N-2 and prev_accept_rates is round N-3.
        let num_seqs = scheduled.len();
        let mut acc_for_bsz: Vec<(f32, usize)> = Vec::new(); // (accept_rate, seq_idx)
        let mut acc_for_seqs_q: Vec<(f32, usize)> = Vec::new();

        for (i, mtp) in scheduled.iter().enumerate() {
            let nd = mtp.draft_tokens.len();
            ensure!(
                mtp.process_cnt <= mtp.schedule_cnt && mtp.schedule_cnt <= mtp.process_cnt + 1,
                SentinelError::InvalidState,
                "schedule_cnt: {}, process_cnt: {}; should not happen",
                mtp.schedule_cnt,
                mtp.process_cnt
            );
            let post_process_paired = mtp.process_cnt == mtp.schedule_cnt;
            // A pre-scheduled step always has a draft; otherwise look at has_draft_info.
            let has_draft = mtp.has_draft_info || mtp.schedule_cnt == mtp.process_cnt + 1;

            let bsz_rates = if post_process_paired { &mtp.prev_accept_rates } else { &mtp.accept_rates };
            push_rates(&mut acc_for_bsz, bsz_rates, nd, i, has_draft)?;
            push_rates(&mut acc_for_seqs_q, &mtp.accept_rates, nd, i, has_draft)?;
        }

        acc_for_bsz.sort_by(by_rate_desc);
        acc_for_seqs_q.sort_by(by_rate_desc);

        // Same num_seqs history if it has enough samples, else the global history.
        let estimate = self
            .sampler_time_by_num_seqs
            .get(&num_seqs)
            .and_then(|e| e.get())
            .or_else(|| self.sampler_time_fallback.get());
        let sampler_ms = estimate.unwrap_or(0.0);
        // The startup curve is a lower bound, so a negative slope is clamped to 0.
        let slope_ms = self.forward_time_slope.get().max(0.0);

        let mut best = BestBatch { curve_idx: 0, best_throughput: 0.0, best_bsz: 0 };
        let mut cur_expected = num_seqs as f32;
        let mut cur_bsz = num_seqs;

        best.update(&self.decode_perf_curve, cur_bsz, cur_expected, slope_ms, sampler_ms); // golden tokens only
        for &(rate, _) in &acc_for_bsz {
            cur_expected += rate;
            cur_bsz += 1;
            best.update(&self.decode_perf_curve, cur_bsz, cur_expected, slope_ms, sampler_ms);
        }
        ensure!(
            best.best_bsz >= num_seqs,
            SentinelError::InvalidState,
            "no batch size fits the decode perf curve (best_bsz={}, num_seqs={num_seqs})",
            best.best_bsz
        );

        let mut seqlens_q = vec![1usize; num_seqs];
        for &(_, seq_idx) in acc_for_seqs_q.iter().take(best.best_bsz - num_seqs) {
            seqlens_q[seq_idx] += 1;
        }
        Ok(seqlens_q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mtp(drafts: usize, rates: &[f32], prev: &[f32]) -> MtpState {
        MtpState {
            draft_tokens: vec![1; drafts],
            accept_rates: rates.to_vec(),
            prev_accept_rates: prev.to_vec(),
            has_draft_info: true,
            process_cnt: 3,
            schedule_cnt: 3,
            ..Default::default()
        }
    }

    #[test]
    fn median_tracker_matches_sorted_median() {
        let mut t = MedianTracker::default();
        for v in [5.0, 1.0, 4.0, 2.0, 3.0] {
            t.insert(v);
        }
        assert_eq!(t.get(), 3.0);
        t.erase(3.0);
        assert_eq!(t.get(), 3.0); // (2 + 4) / 2
        t.insert(10.0);
        assert_eq!(t.get(), 4.0);
        assert_eq!(t.size(), 5);
    }

    #[test]
    fn median_tracker_handles_duplicates() {
        let mut t = MedianTracker::default();
        for _ in 0..4 {
            t.insert(2.0);
        }
        t.erase(2.0);
        assert_eq!(t.size(), 3);
        assert_eq!(t.get(), 2.0);
    }

    #[test]
    fn sampler_estimator_needs_min_samples() {
        let mut e = SamplerRuntimeEstimator::default();
        for v in [1.0, 3.0, 2.0, 5.0] {
            e.update(v);
        }
        assert_eq!(e.get(), None);
        e.update(4.0);
        assert_eq!(e.get(), Some(3.0));
    }

    #[test]
    fn sampler_estimator_window_slides() {
        let mut e = SamplerRuntimeEstimator::default();
        for _ in 0..9 {
            e.update(1.0);
        }
        for _ in 0..9 {
            e.update(7.0);
        }
        assert_eq!(e.get(), Some(7.0));
    }

    #[test]
    fn forward_estimator_needs_min_samples_per_graph() {
        let mut e = ForwardRuntimeEstimator::default();
        for _ in 0..9 {
            e.update(8, 0.5);
        }
        assert_eq!(e.get(), 0.0);
        e.update(8, 0.5);
        assert_eq!(e.get(), 0.5);
    }

    #[test]
    fn empty_curve_uses_draft_lengths() {
        let v = MtpVarlen::default();
        let a = mtp(2, &[], &[]);
        let b = mtp(0, &[], &[]);
        assert_eq!(v.schedule(&[&a, &b]).unwrap(), vec![3, 1]);
    }

    #[test]
    fn flat_curve_verifies_all_drafts() {
        let mut v = MtpVarlen::default();
        v.set_decode_perf_curve(vec![(1, 10.0), (2, 10.0), (4, 10.0), (8, 10.0)]).unwrap();
        let a = mtp(2, &[0.9, 0.8], &[0.9, 0.8]);
        let b = mtp(2, &[0.9, 0.8], &[0.9, 0.8]);
        assert_eq!(v.schedule(&[&a, &b]).unwrap(), vec![3, 3]);
    }

    #[test]
    fn expensive_larger_graph_keeps_only_main_tokens() {
        let mut v = MtpVarlen::default();
        v.set_decode_perf_curve(vec![(2, 10.0), (4, 100.0)]).unwrap();
        let a = mtp(2, &[0.9, 0.8], &[0.9, 0.8]);
        let b = mtp(2, &[0.9, 0.8], &[0.9, 0.8]);
        assert_eq!(v.schedule(&[&a, &b]).unwrap(), vec![1, 1]);
    }

    #[test]
    fn budget_goes_to_the_higher_accept_rate() {
        let mut v = MtpVarlen::default();
        // batch of 3 total tokens is the sweet spot: 2 seqs + 1 draft
        v.set_decode_perf_curve(vec![(2, 10.0), (3, 10.0), (4, 1000.0)]).unwrap();
        let a = mtp(1, &[0.3], &[0.3]);
        let b = mtp(1, &[0.9], &[0.9]);
        assert_eq!(v.schedule(&[&a, &b]).unwrap(), vec![1, 2]);
    }

    #[test]
    fn counter_invariant_is_enforced() {
        let v = {
            let mut v = MtpVarlen::default();
            v.set_decode_perf_curve(vec![(1, 1.0)]).unwrap();
            v
        };
        let mut bad = mtp(1, &[0.5], &[0.5]);
        bad.schedule_cnt = 9;
        assert!(v.schedule(&[&bad]).is_err());
    }

    #[test]
    fn curve_must_be_strictly_increasing() {
        let mut v = MtpVarlen::default();
        assert!(v.set_decode_perf_curve(vec![(2, 1.0), (2, 2.0)]).is_err());
    }

    #[test]
    fn update_runtime_validates_graph_size() {
        let mut v = MtpVarlen::default();
        v.set_decode_perf_curve(vec![(4, 10.0)]).unwrap();
        assert!(v.update_runtime(2, 8, 12.0, 1.0).is_err());
        assert!(v.update_runtime(2, 4, 12.0, 1.0).is_ok());
    }
}
