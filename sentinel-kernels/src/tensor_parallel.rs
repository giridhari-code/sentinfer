//! Tensor parallelism (TP) — shard GEMM across ranks + collectives.
//!
//! ## Model
//! * **Column-parallel:** `W` sharded on output dim → each rank computes partial `Y_r`, then **all-gather**.
//! * **Row-parallel:** `W` sharded on input (K) dim → each rank computes partial product, then **all-reduce sum**.
//!
//! ## Backends
//! * [`InProcessTp`] — shared-memory ranks in one process (CPU multi-“GPU” / multi-thread).
//! * Collectives are pluggable via [`TpCollective`].
//!
//! Real multi-GPU (NCCL/CUDA) is out of tree; wire the same trait over your device later.

use std::sync::{Arc, Barrier, Mutex};

use crate::error::{ensure, KernelError, Result};
use crate::reference;

/// Identity of one tensor-parallel rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpRank {
    pub rank: usize,
    pub world_size: usize,
}

impl TpRank {
    pub fn new(rank: usize, world_size: usize) -> Result<Self> {
        ensure!(world_size > 0, KernelError::Shape, "world_size");
        ensure!(rank < world_size, KernelError::Shape, "rank");
        Ok(Self { rank, world_size })
    }

    /// Split `full` elements across ranks; returns (local_start, local_len).
    pub fn shard(&self, full: usize) -> (usize, usize) {
        let base = full / self.world_size;
        let rem = full % self.world_size;
        let start = self.rank * base + self.rank.min(rem);
        let len = base + if self.rank < rem { 1 } else { 0 };
        (start, len)
    }
}

/// Collective operations required by TP matmuls.
pub trait TpCollective: Send + Sync {
    fn world_size(&self) -> usize;
    fn rank(&self) -> usize;
    /// Sum-reduce `buf` in place across ranks (same length on every rank).
    fn all_reduce_sum(&self, buf: &mut [f32]) -> Result<()>;
    /// All-gather variable-length shards into contiguous `out` ordered by rank.
    /// `local` is this rank's shard; `out` length = sum of shard lens.
    fn all_gather_f32(&self, local: &[f32], out: &mut [f32], shard_lens: &[usize]) -> Result<()>;
}

/// Single-rank no-op collective (world_size = 1).
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityCollective;

impl TpCollective for IdentityCollective {
    fn world_size(&self) -> usize {
        1
    }
    fn rank(&self) -> usize {
        0
    }
    fn all_reduce_sum(&self, _buf: &mut [f32]) -> Result<()> {
        Ok(())
    }
    fn all_gather_f32(&self, local: &[f32], out: &mut [f32], shard_lens: &[usize]) -> Result<()> {
        ensure!(shard_lens.len() == 1 && shard_lens[0] == local.len(), KernelError::Shape, "identity gather");
        ensure!(out.len() == local.len(), KernelError::Shape, "out");
        out.copy_from_slice(local);
        Ok(())
    }
}

/// In-process TP mesh: `world_size` logical ranks synchronized with barriers + shared slots.
pub struct InProcessTp {
    rank: usize,
    world_size: usize,
    barrier: Arc<Barrier>,
    /// Slot per rank for staging vectors during collectives.
    slots: Arc<Mutex<Vec<Option<Vec<f32>>>>>,
}

impl InProcessTp {
    /// Build one mesh and `world_size` rank handles sharing the same barrier/slots.
    pub fn mesh(world_size: usize) -> Result<Vec<Self>> {
        ensure!(world_size > 0, KernelError::Shape, "world_size");
        let barrier = Arc::new(Barrier::new(world_size));
        let slots = Arc::new(Mutex::new(vec![None; world_size]));
        Ok((0..world_size)
            .map(|rank| Self {
                rank,
                world_size,
                barrier: Arc::clone(&barrier),
                slots: Arc::clone(&slots),
            })
            .collect())
    }

    pub fn clone_handle(&self) -> Self {
        Self {
            rank: self.rank,
            world_size: self.world_size,
            barrier: Arc::clone(&self.barrier),
            slots: Arc::clone(&self.slots),
        }
    }
}

impl TpCollective for InProcessTp {
    fn world_size(&self) -> usize {
        self.world_size
    }
    fn rank(&self) -> usize {
        self.rank
    }

    fn all_reduce_sum(&self, buf: &mut [f32]) -> Result<()> {
        let n = buf.len();
        {
            let mut slots = self.slots.lock().map_err(|_| KernelError::Backend("tp lock".into()))?;
            slots[self.rank] = Some(buf.to_vec());
        }
        self.barrier.wait();
        let sum = {
            let slots = self.slots.lock().map_err(|_| KernelError::Backend("tp lock".into()))?;
            let mut acc = vec![0.0f32; n];
            for r in 0..self.world_size {
                let part = slots[r]
                    .as_ref()
                    .ok_or_else(|| KernelError::Backend(format!("tp missing rank {r}")))?;
                ensure!(part.len() == n, KernelError::Shape, "all_reduce len mismatch");
                for i in 0..n {
                    acc[i] += part[i];
                }
            }
            acc
        };
        self.barrier.wait();
        if self.rank == 0 {
            let mut slots = self.slots.lock().map_err(|_| KernelError::Backend("tp lock".into()))?;
            for s in slots.iter_mut() {
                *s = None;
            }
        }
        self.barrier.wait();
        buf.copy_from_slice(&sum);
        Ok(())
    }

    fn all_gather_f32(&self, local: &[f32], out: &mut [f32], shard_lens: &[usize]) -> Result<()> {
        ensure!(shard_lens.len() == self.world_size, KernelError::Shape, "shard_lens");
        ensure!(shard_lens[self.rank] == local.len(), KernelError::Shape, "local len");
        let total: usize = shard_lens.iter().sum();
        ensure!(out.len() == total, KernelError::Shape, "out len");
        {
            let mut slots = self.slots.lock().map_err(|_| KernelError::Backend("tp lock".into()))?;
            slots[self.rank] = Some(local.to_vec());
        }
        self.barrier.wait();
        {
            let slots = self.slots.lock().map_err(|_| KernelError::Backend("tp lock".into()))?;
            let mut off = 0;
            for r in 0..self.world_size {
                let part = slots[r]
                    .as_ref()
                    .ok_or_else(|| KernelError::Backend(format!("gather missing {r}")))?;
                ensure!(part.len() == shard_lens[r], KernelError::Shape, "gather shard");
                out[off..off + part.len()].copy_from_slice(part);
                off += part.len();
            }
        }
        self.barrier.wait();
        if self.rank == 0 {
            let mut slots = self.slots.lock().map_err(|_| KernelError::Backend("tp lock".into()))?;
            for s in slots.iter_mut() {
                *s = None;
            }
        }
        self.barrier.wait();
        Ok(())
    }
}

/// Slice weight rows `[n_full, k]` for column-parallel (shard N).
pub fn shard_rows(w_full: &[f32], n_full: usize, k: usize, tp: TpRank) -> Result<Vec<f32>> {
    ensure!(w_full.len() == n_full * k, KernelError::Shape, "shard_rows");
    let (start, len) = tp.shard(n_full);
    let mut out = vec![0.0f32; len * k];
    for r in 0..len {
        out[r * k..(r + 1) * k].copy_from_slice(&w_full[(start + r) * k..(start + r + 1) * k]);
    }
    Ok(out)
}

/// Slice weight along K for row-parallel: each rank holds `W[:, k_start:k_start+k_len]` as `[n, k_len]`.
pub fn shard_cols(w_full: &[f32], n: usize, k_full: usize, tp: TpRank) -> Result<Vec<f32>> {
    ensure!(w_full.len() == n * k_full, KernelError::Shape, "shard_cols");
    let (start, len) = tp.shard(k_full);
    let mut out = vec![0.0f32; n * len];
    for r in 0..n {
        for c in 0..len {
            out[r * len + c] = w_full[r * k_full + start + c];
        }
    }
    Ok(out)
}

/// Column-parallel matmul: local `X[m,k] @ W_local[n_r,k]^T` then all-gather on N.
pub fn column_parallel_matmul_nt(
    coll: &dyn TpCollective,
    x: &[f32],
    w_local: &[f32],
    m: usize,
    k: usize,
    n_local: usize,
    n_full: usize,
) -> Result<Vec<f32>> {
    ensure!(x.len() == m * k && w_local.len() == n_local * k, KernelError::Shape, "col-parallel");
    let local = reference::matmul_nt(x, w_local, m, k, n_local)?;
    // Gather along N for each row of m
    let mut shard_lens = vec![0usize; coll.world_size()];
    // recompute lens from n_full
    for r in 0..coll.world_size() {
        let (_, len) = TpRank::new(r, coll.world_size())?.shard(n_full);
        shard_lens[r] = len;
    }
    ensure!(shard_lens[coll.rank()] == n_local, KernelError::Shape, "n_local");
    let mut out = vec![0.0f32; m * n_full];
    for i in 0..m {
        let local_row = &local[i * n_local..(i + 1) * n_local];
        let mut gathered = vec![0.0f32; n_full];
        coll.all_gather_f32(local_row, &mut gathered, &shard_lens)?;
        out[i * n_full..(i + 1) * n_full].copy_from_slice(&gathered);
    }
    Ok(out)
}

/// Row-parallel matmul: shard X on K, local W `[n, k_r]`, local product, all-reduce sum → `[m,n]`.
pub fn row_parallel_matmul_nt(
    coll: &dyn TpCollective,
    x_full: &[f32],
    w_local: &[f32],
    m: usize,
    k_full: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let tp = TpRank::new(coll.rank(), coll.world_size())?;
    let (k0, k_len) = tp.shard(k_full);
    ensure!(w_local.len() == n * k_len, KernelError::Shape, "row-parallel W");
    ensure!(x_full.len() == m * k_full, KernelError::Shape, "row-parallel X");
    let mut x_local = vec![0.0f32; m * k_len];
    for i in 0..m {
        for c in 0..k_len {
            x_local[i * k_len + c] = x_full[i * k_full + k0 + c];
        }
    }
    let mut partial = reference::matmul_nt(&x_local, w_local, m, k_len, n)?;
    coll.all_reduce_sum(&mut partial)?;
    Ok(partial)
}

/// Convenience: full weight column-parallel end-to-end (shard + matmul + gather).
pub fn column_parallel_from_full(
    coll: &dyn TpCollective,
    x: &[f32],
    w_full: &[f32],
    m: usize,
    k: usize,
    n_full: usize,
) -> Result<Vec<f32>> {
    let tp = TpRank::new(coll.rank(), coll.world_size())?;
    let (_, n_local) = tp.shard(n_full);
    let w_local = shard_rows(w_full, n_full, k, tp)?;
    column_parallel_matmul_nt(coll, x, &w_local, m, k, n_local, n_full)
}

/// Convenience: full weight row-parallel.
pub fn row_parallel_from_full(
    coll: &dyn TpCollective,
    x: &[f32],
    w_full: &[f32],
    m: usize,
    k_full: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let tp = TpRank::new(coll.rank(), coll.world_size())?;
    let w_local = shard_cols(w_full, n, k_full, tp)?;
    row_parallel_matmul_nt(coll, x, &w_local, m, k_full, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn shard_covers_full() {
        let ws = 3;
        let full = 10;
        let mut covered = 0;
        for r in 0..ws {
            let (_, len) = TpRank::new(r, ws).unwrap().shard(full);
            covered += len;
        }
        assert_eq!(covered, full);
    }

    #[test]
    fn column_parallel_matches_dense() {
        let m = 2;
        let k = 8;
        let n = 6;
        let world = 3;
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.02).collect();
        let dense = reference::matmul_nt(&x, &w, m, k, n).unwrap();

        let mesh = InProcessTp::mesh(world).unwrap();
        let mut handles = vec![];
        let x = Arc::new(x);
        let w = Arc::new(w);
        let results = Arc::new(Mutex::new(vec![None; world]));
        for rank in 0..world {
            let coll = mesh[rank].clone_handle();
            let x = Arc::clone(&x);
            let w = Arc::clone(&w);
            let results = Arc::clone(&results);
            handles.push(thread::spawn(move || {
                let y = column_parallel_from_full(&coll, &x, &w, m, k, n).unwrap();
                results.lock().unwrap()[rank] = Some(y);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let results = results.lock().unwrap();
        for r in 0..world {
            let y = results[r].as_ref().unwrap();
            for (a, b) in dense.iter().zip(y.iter()) {
                assert!((a - b).abs() < 1e-4, "rank {r}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn row_parallel_matches_dense() {
        let m = 2;
        let k = 9;
        let n = 4;
        let world = 3;
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.03).sin()).collect();
        let dense = reference::matmul_nt(&x, &w, m, k, n).unwrap();

        let mesh = InProcessTp::mesh(world).unwrap();
        let mut handles = vec![];
        let x = Arc::new(x);
        let w = Arc::new(w);
        let results = Arc::new(Mutex::new(vec![None; world]));
        for rank in 0..world {
            let coll = mesh[rank].clone_handle();
            let x = Arc::clone(&x);
            let w = Arc::clone(&w);
            let results = Arc::clone(&results);
            handles.push(thread::spawn(move || {
                let y = row_parallel_from_full(&coll, &x, &w, m, k, n).unwrap();
                results.lock().unwrap()[rank] = Some(y);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let results = results.lock().unwrap();
        for r in 0..world {
            let y = results[r].as_ref().unwrap();
            for (a, b) in dense.iter().zip(y.iter()) {
                assert!((a - b).abs() < 1e-4, "rank {r}: {a} vs {b}");
            }
        }
    }
}

