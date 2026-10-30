//! Host-side buffer pool for activations and scratch tensors.
//!
//! Reuses `Vec<f32>` of similar sizes instead of allocating on every forward step.
//! Designed for single-threaded engine use (scheduler / model step). For multi-thread
//! workers, give each worker its own pool.

use std::collections::HashMap;

/// Round capacity up to the next power of two (min 64) so size classes stay coarse.
fn size_class(n: usize) -> usize {
    let n = n.max(64);
    n.next_power_of_two()
}

/// Pool of reusable `Vec<f32>` scratch buffers.
#[derive(Debug, Default)]
pub struct BufferPool {
    /// Keyed by capacity (power-of-two size class).
    free: HashMap<usize, Vec<Vec<f32>>>,
    /// Stats
    hits: u64,
    misses: u64,
    bytes_live: usize,
    peak_bytes: usize,
}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a buffer with at least `len` elements (length set to `len`, capacity may be larger).
    pub fn acquire(&mut self, len: usize) -> Vec<f32> {
        let class = size_class(len);
        if let Some(stack) = self.free.get_mut(&class) {
            if let Some(mut v) = stack.pop() {
                self.hits += 1;
                v.clear();
                v.resize(len, 0.0);
                self.bytes_live += v.capacity() * 4;
                self.peak_bytes = self.peak_bytes.max(self.bytes_live);
                return v;
            }
        }
        self.misses += 1;
        let mut v = Vec::with_capacity(class);
        v.resize(len, 0.0);
        self.bytes_live += v.capacity() * 4;
        self.peak_bytes = self.peak_bytes.max(self.bytes_live);
        v
    }

    /// Return a buffer to the pool. Capacity is kept; length is cleared on next acquire.
    pub fn release(&mut self, mut buf: Vec<f32>) {
        let cap = buf.capacity();
        if cap == 0 {
            return;
        }
        self.bytes_live = self.bytes_live.saturating_sub(cap * 4);
        let class = size_class(cap);
        // Only keep reasonably sized buffers (avoid pinning huge one-off allocations forever).
        if class > 64 * 1024 * 1024 / 4 {
            // > 64 MiB of f32 — drop instead of caching
            return;
        }
        buf.clear();
        self.free.entry(class).or_default().push(buf);
    }

    /// Pre-warm the pool with `count` buffers of size class covering `len`.
    pub fn prewarm(&mut self, len: usize, count: usize) {
        let class = size_class(len);
        let stack = self.free.entry(class).or_default();
        for _ in 0..count {
            let mut v = Vec::with_capacity(class);
            v.resize(len, 0.0);
            v.clear();
            stack.push(v);
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }
    pub fn misses(&self) -> u64 {
        self.misses
    }
    pub fn hit_rate(&self) -> f64 {
        let t = self.hits + self.misses;
        if t == 0 {
            0.0
        } else {
            self.hits as f64 / t as f64
        }
    }
    pub fn bytes_live(&self) -> usize {
        self.bytes_live
    }
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }
    pub fn cached_buffers(&self) -> usize {
        self.free.values().map(|v| v.len()).sum()
    }

    /// Drop all free buffers (keeps stats).
    pub fn clear_free(&mut self) {
        self.free.clear();
    }
}

/// RAII guard: acquires a buffer and returns it to the pool on drop.
pub struct PooledBuf<'a> {
    pool: &'a mut BufferPool,
    buf: Option<Vec<f32>>,
}

impl<'a> PooledBuf<'a> {
    pub fn new(pool: &'a mut BufferPool, len: usize) -> Self {
        let buf = pool.acquire(len);
        Self { pool, buf: Some(buf) }
    }

    pub fn as_slice(&self) -> &[f32] {
        self.buf.as_ref().map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.buf.as_mut().map(|v| v.as_mut_slice()).unwrap_or(&mut [])
    }

    pub fn into_vec(mut self) -> Vec<f32> {
        self.buf.take().unwrap_or_default()
    }
}

impl Drop for PooledBuf<'_> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.release(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_same_size_class() {
        let mut p = BufferPool::new();
        let a = p.acquire(100);
        assert_eq!(a.len(), 100);
        p.release(a);
        let b = p.acquire(80); // same size class as 100 (128)
        assert!(p.hits() >= 1);
        assert_eq!(b.capacity(), 128);
        p.release(b);
    }

    #[test]
    fn pooled_buf_returns_on_drop() {
        let mut p = BufferPool::new();
        {
            let _g = PooledBuf::new(&mut p, 256);
        }
        assert_eq!(p.cached_buffers(), 1);
        assert_eq!(p.hits() + p.misses(), 1);
    }

    #[test]
    fn prewarm_reduces_misses() {
        let mut p = BufferPool::new();
        p.prewarm(1024, 4);
        for _ in 0..4 {
            let v = p.acquire(1000);
            p.release(v);
        }
        assert!(p.hits() >= 4);
        assert_eq!(p.misses(), 0);
    }
}
