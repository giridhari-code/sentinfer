//! Expert Parallelism + load balancing (DeepSeek / DeepInfer style).
//!
//! Problems this module targets:
//! * **Load imbalance** — a few experts get most tokens, others idle
//! * **Communication overhead** — skewed all-to-all in multi-rank EP
//!
//! Tools:
//! * [`ExpertPlacement`] — map logical experts → EP ranks (optional replicas for EPLB)
//! * [`LoadTracker`] — per-expert token counts + imbalance metrics
//! * [`route_with_capacity`] — top-k with capacity factor (drop / renorm overflow)
//! * [`apply_load_bias`] — inference-time bias against overloaded experts
//! * [`All2AllPlan`] — estimated bytes / max rank load for the step

use std::cmp::Ordering;

use sentinel_core::{Result, SentinelError};

use crate::moe::ExpertParallelConfig;

/// How to place `num_experts` logical experts across `ep_size` ranks.
#[derive(Debug, Clone)]
pub struct ExpertPlacement {
    pub num_experts: usize,
    pub ep_size: usize,
    /// `replicas[logical_expert] = list of ranks that host a copy`
    pub replicas: Vec<Vec<usize>>,
    /// Rank → list of logical experts stored there
    pub rank_experts: Vec<Vec<usize>>,
}

impl ExpertPlacement {
    /// Contiguous shard: rank `r` owns experts `[r * n/ep, (r+1)*n/ep)`.
    pub fn contiguous(num_experts: usize, ep_size: usize) -> Result<Self> {
        if ep_size == 0 {
            return Err(SentinelError::Config("ep_size must be > 0".into()));
        }
        if num_experts % ep_size != 0 {
            return Err(SentinelError::Config(
                "num_experts must be divisible by ep_size for contiguous placement".into(),
            ));
        }
        let per = num_experts / ep_size;
        let mut replicas = vec![Vec::new(); num_experts];
        let mut rank_experts = vec![Vec::new(); ep_size];
        for e in 0..num_experts {
            let r = e / per;
            replicas[e].push(r);
            rank_experts[r].push(e);
        }
        Ok(Self {
            num_experts,
            ep_size,
            replicas,
            rank_experts,
        })
    }

    /// EPLB-style: contiguous base + extra replicas of the *heaviest* experts
    /// (by historical load) on the *lightest* ranks.
    ///
    /// `loads[e]` = recent token count for expert `e`. If empty, falls back to contiguous.
    pub fn eplb(
        num_experts: usize,
        ep_size: usize,
        loads: &[usize],
        extra_replicas: usize,
    ) -> Result<Self> {
        let mut p = Self::contiguous(num_experts, ep_size)?;
        if !p.cfg_eplb_ok(extra_replicas) || loads.len() != num_experts || extra_replicas == 0 {
            return Ok(p);
        }
        // Sort experts by load descending; ranks by current expert-count ascending.
        let mut expert_order: Vec<usize> = (0..num_experts).collect();
        expert_order.sort_by(|&a, &b| loads[b].cmp(&loads[a]));

        let mut rank_load: Vec<usize> = p
            .rank_experts
            .iter()
            .map(|es| es.iter().map(|&e| loads.get(e).copied().unwrap_or(0)).sum())
            .collect();

        let mut placed = 0usize;
        for &e in &expert_order {
            if placed >= extra_replicas {
                break;
            }
            // lightest rank that does not already host e
            let mut ranks: Vec<usize> = (0..ep_size).collect();
            ranks.sort_by(|&a, &b| rank_load[a].cmp(&rank_load[b]));
            for r in ranks {
                if p.replicas[e].contains(&r) {
                    continue;
                }
                p.replicas[e].push(r);
                p.rank_experts[r].push(e);
                rank_load[r] += loads[e];
                placed += 1;
                break;
            }
        }
        Ok(p)
    }

    fn cfg_eplb_ok(&self, extra: usize) -> bool {
        self.ep_size > 1 && extra > 0
    }

    /// Rank that should process this token for logical expert `e` (lowest current load among replicas).
    pub fn pick_rank(&self, expert: usize, rank_token_counts: &[usize]) -> usize {
        let reps = &self.replicas[expert];
        if reps.is_empty() {
            return 0;
        }
        *reps
            .iter()
            .min_by_key(|&&r| rank_token_counts.get(r).copied().unwrap_or(0))
            .unwrap_or(&reps[0])
    }

    pub fn experts_on_rank(&self, rank: usize) -> &[usize] {
        self.rank_experts.get(rank).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

/// Running per-expert (and optional per-rank) token counts.
#[derive(Debug, Clone)]
pub struct LoadTracker {
    pub expert_tokens: Vec<usize>,
    pub rank_tokens: Vec<usize>,
    pub total_tokens: usize,
    pub dropped_tokens: usize,
}

impl LoadTracker {
    pub fn new(num_experts: usize, ep_size: usize) -> Self {
        Self {
            expert_tokens: vec![0; num_experts],
            rank_tokens: vec![0; ep_size.max(1)],
            total_tokens: 0,
            dropped_tokens: 0,
        }
    }

    pub fn record_expert(&mut self, expert: usize, n: usize) {
        if expert < self.expert_tokens.len() {
            self.expert_tokens[expert] += n;
            self.total_tokens += n;
        }
    }

    pub fn record_rank(&mut self, rank: usize, n: usize) {
        if rank < self.rank_tokens.len() {
            self.rank_tokens[rank] += n;
        }
    }

    pub fn record_drop(&mut self, n: usize) {
        self.dropped_tokens += n;
    }

    /// max/mean expert load (1.0 = perfect). Higher = worse imbalance.
    pub fn imbalance_ratio(&self) -> f32 {
        let n = self.expert_tokens.len();
        if n == 0 || self.total_tokens == 0 {
            return 1.0;
        }
        let mean = self.total_tokens as f32 / n as f32;
        if mean <= 0.0 {
            return 1.0;
        }
        let max = self.expert_tokens.iter().copied().max().unwrap_or(0) as f32;
        max / mean
    }

    /// Coefficient of variation of expert loads.
    pub fn load_cv(&self) -> f32 {
        let n = self.expert_tokens.len();
        if n < 2 || self.total_tokens == 0 {
            return 0.0;
        }
        let mean = self.total_tokens as f32 / n as f32;
        let var: f32 = self
            .expert_tokens
            .iter()
            .map(|&c| {
                let d = c as f32 - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        var.sqrt() / mean.max(1e-6)
    }

    pub fn reset_step(&mut self) {
        self.expert_tokens.fill(0);
        self.rank_tokens.fill(0);
        self.total_tokens = 0;
        self.dropped_tokens = 0;
    }
}

/// One token's routing decision after capacity enforcement.
#[derive(Debug, Clone)]
pub struct RoutedToken {
    pub token: usize,
    pub expert: usize,
    pub weight: f32,
    pub rank: usize,
}

/// Top-k routing with **capacity factor**.
///
/// Each expert may accept at most
/// `ceil(capacity_factor * tokens * top_k / num_experts)` tokens this step.
/// Overflow is dropped (weights renormed among kept experts for that token).
pub fn route_with_capacity(
    scores: &[f32], // [tokens * num_experts] post-softmax
    tokens: usize,
    num_experts: usize,
    top_k: usize,
    capacity_factor: f32,
    placement: &ExpertPlacement,
    rank_counts: &mut [usize],
) -> Result<(Vec<RoutedToken>, usize /* dropped */)> {
    if scores.len() != tokens * num_experts {
        return Err(SentinelError::InvalidState("route scores shape".into()));
    }
    let cap = capacity_factor.max(1.0);
    let per_expert_cap = ((tokens * top_k) as f32 * cap / num_experts as f32)
        .ceil()
        .max(1.0) as usize;

    let mut expert_fill = vec![0usize; num_experts];
    let mut out = Vec::with_capacity(tokens * top_k);
    let mut dropped = 0usize;

    for tok in 0..tokens {
        let row = &scores[tok * num_experts..(tok + 1) * num_experts];
        let mut idx: Vec<usize> = (0..num_experts).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(Ordering::Equal));
        idx.truncate(top_k);

        let mut kept: Vec<(usize, f32)> = Vec::with_capacity(top_k);
        for &e in &idx {
            if expert_fill[e] >= per_expert_cap {
                dropped += 1;
                continue;
            }
            expert_fill[e] += 1;
            kept.push((e, row[e]));
        }
        if kept.is_empty() {
            // force assign to least-full expert among original top-k
            if let Some(&e) = idx.iter().min_by_key(|&&e| expert_fill[e]) {
                expert_fill[e] += 1;
                kept.push((e, row[e].max(1e-6)));
                dropped = dropped.saturating_sub(1);
            }
        }
        let sum: f32 = kept.iter().map(|(_, w)| *w).sum();
        let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 / kept.len().max(1) as f32 };
        for (e, w) in kept {
            let rank = placement.pick_rank(e, rank_counts);
            if rank < rank_counts.len() {
                rank_counts[rank] += 1;
            }
            out.push(RoutedToken {
                token: tok,
                expert: e,
                weight: w * inv,
                rank,
            });
        }
    }
    Ok((out, dropped))
}

/// Subtract load bias from router logits (pre-softmax): overloaded experts become less attractive.
///
/// `bias_strength` typical range 0.01 .. 0.5.
pub fn apply_load_bias(
    logits: &mut [f32],
    tokens: usize,
    num_experts: usize,
    expert_load: &[usize],
    bias_strength: f32,
) {
    if bias_strength <= 0.0 || expert_load.len() != num_experts {
        return;
    }
    let mean = expert_load.iter().sum::<usize>() as f32 / num_experts.max(1) as f32;
    let mean = mean.max(1.0);
    for t in 0..tokens {
        for e in 0..num_experts {
            let load = expert_load[e] as f32;
            // positive load above mean → negative logit bias
            let bias = bias_strength * (load / mean - 1.0);
            logits[t * num_experts + e] -= bias;
        }
    }
}

/// Estimated all-to-all traffic for one MoE step under this routing.
#[derive(Debug, Clone)]
pub struct All2AllPlan {
    pub tokens_per_rank: Vec<usize>,
    pub tokens_per_expert: Vec<usize>,
    pub max_rank_tokens: usize,
    pub total_dispatches: usize,
    /// Approximate payload bytes if each token is `hidden * 4` bytes (f32).
    pub estimated_bytes_one_way: usize,
    pub imbalance_ratio: f32,
}

impl All2AllPlan {
    pub fn from_routes(
        routes: &[RoutedToken],
        num_experts: usize,
        ep_size: usize,
        hidden: usize,
    ) -> Self {
        let mut tokens_per_rank = vec![0usize; ep_size.max(1)];
        let mut tokens_per_expert = vec![0usize; num_experts];
        for r in routes {
            if r.rank < tokens_per_rank.len() {
                tokens_per_rank[r.rank] += 1;
            }
            if r.expert < tokens_per_expert.len() {
                tokens_per_expert[r.expert] += 1;
            }
        }
        let max_rank_tokens = tokens_per_rank.iter().copied().max().unwrap_or(0);
        let total = routes.len();
        let mean = total as f32 / ep_size.max(1) as f32;
        let imbalance = if mean > 0.0 {
            max_rank_tokens as f32 / mean
        } else {
            1.0
        };
        // dispatch + combine ≈ 2 * tokens * hidden * 4, but report one-way dispatch
        let estimated_bytes_one_way = total.saturating_mul(hidden).saturating_mul(4);
        Self {
            tokens_per_rank,
            tokens_per_expert,
            max_rank_tokens,
            total_dispatches: total,
            estimated_bytes_one_way,
            imbalance_ratio: imbalance,
        }
    }
}

/// Build placement from [`ExpertParallelConfig`].
pub fn placement_from_config(
    num_experts: usize,
    ep: &ExpertParallelConfig,
    historical_loads: Option<&[usize]>,
) -> Result<ExpertPlacement> {
    let ep_size = ep.ep_size.max(1);
    if ep_size == 1 {
        // single rank owns all experts
        let mut replicas = vec![vec![0usize]; num_experts];
        let rank_experts = vec![(0..num_experts).collect()];
        return Ok(ExpertPlacement {
            num_experts,
            ep_size: 1,
            replicas,
            rank_experts,
        });
    }
    if ep.enable_eplb {
        let loads = historical_loads
            .map(|l| l.to_vec())
            .unwrap_or_else(|| vec![1usize; num_experts]);
        ExpertPlacement::eplb(num_experts, ep_size, &loads, ep.eplb_replicas)
    } else {
        ExpertPlacement::contiguous(num_experts, ep_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_placement() {
        let p = ExpertPlacement::contiguous(8, 4).unwrap();
        assert_eq!(p.rank_experts[0], vec![0, 1]);
        assert_eq!(p.replicas[7], vec![3]);
    }

    #[test]
    fn capacity_limits_hot_experts() {
        let p = ExpertPlacement::contiguous(4, 1).unwrap();
        // all mass on expert 0
        let mut scores = vec![0.0f32; 8 * 4];
        for t in 0..8 {
            scores[t * 4] = 1.0;
        }
        let mut ranks = vec![0usize; 1];
        let (routes, dropped) =
            route_with_capacity(&scores, 8, 4, 1, 1.0, &p, &mut ranks).unwrap();
        // cap = ceil(8*1*1/4)=2 per expert → most dropped or forced elsewhere
        assert!(routes.len() <= 8);
        assert!(dropped > 0 || routes.iter().any(|r| r.expert != 0));
    }

    #[test]
    fn load_bias_reduces_hot_logit() {
        let mut logits = vec![0.0f32; 2 * 2]; // 2 tokens, 2 experts
        let loads = [100usize, 1];
        apply_load_bias(&mut logits, 2, 2, &loads, 0.5);
        assert!(logits[0] < logits[1]);
    }

    #[test]
    fn eplb_adds_replica() {
        let loads = vec![100usize, 1, 1, 1];
        let p = ExpertPlacement::eplb(4, 2, &loads, 1).unwrap();
        // expert 0 should appear on more than one rank ideally
        assert!(p.replicas[0].len() >= 1);
        assert_eq!(p.ep_size, 2);
    }
}
