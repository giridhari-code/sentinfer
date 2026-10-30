//! KV-cache block manager (port of `csrc/engine/block_manager.hpp`).
//!
//! Block id 0 is reserved as "null"; usable blocks are `1..num_blocks`.
//! Free blocks form an intrusive doubly linked list, so alloc/free are O(1).
//! Locking is not done here: the scheduler owns the managers and takes `&mut`.

use crate::error::{ensure, Result, SentinelError};
use crate::sequence::Sequence;

pub type BlockId = u32;
pub const NULL_BLOCK: BlockId = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStatus {
    Ok,
    Later,
    Never,
}

#[derive(Debug, Clone, Copy, Default)]
struct Block {
    ref_count: i32,
    next: BlockId,
    prev: BlockId,
}

#[derive(Debug)]
pub struct Blocks {
    blocks: Vec<Block>,
    head: BlockId,
    tail: BlockId,
    size: usize,
}

impl Blocks {
    pub fn new(n: usize) -> Result<Self> {
        ensure!(n > 1, SentinelError::BlockManager, "too few GPU blocks: {n}");
        let mut blocks = vec![Block::default(); n];
        for i in 1..n {
            blocks[i].prev = (i - 1) as BlockId;
            blocks[i].next = (i + 1) as BlockId;
        }
        blocks[n - 1].next = NULL_BLOCK;
        Ok(Self { blocks, head: 1, tail: (n - 1) as BlockId, size: n - 1 })
    }

    fn index(&self, id: BlockId) -> Result<usize> {
        let idx = id as usize;
        ensure!(id != NULL_BLOCK && idx < self.blocks.len(), SentinelError::BlockManager, "invalid block id {id}");
        Ok(idx)
    }

    pub fn reset(&self) -> Result<()> {
        ensure!(
            self.size == self.blocks.len() - 1,
            SentinelError::BlockManager,
            "still some blocks in use when reset"
        );
        Ok(())
    }

    fn push_back(&mut self, id: BlockId) -> Result<()> {
        let idx = self.index(id)?;
        let b = self.blocks[idx];
        ensure!(b.next == NULL_BLOCK && b.prev == NULL_BLOCK, SentinelError::BlockManager, "block {id} is still linked");
        ensure!(b.ref_count == 0, SentinelError::BlockManager, "only a free block can be pushed back");
        let tail = self.tail;
        self.blocks[idx].prev = tail;
        if tail != NULL_BLOCK {
            self.blocks[tail as usize].next = id;
        } else {
            self.head = id;
        }
        self.tail = id;
        self.size += 1;
        Ok(())
    }

    fn erase(&mut self, id: BlockId) -> Result<()> {
        let idx = self.index(id)?;
        let b = self.blocks[idx];
        ensure!(b.ref_count == 0, SentinelError::BlockManager, "free block list corrupted: ref_count != 0");
        ensure!(self.size > 0, SentinelError::BlockManager, "free block list is empty");
        if b.next != NULL_BLOCK {
            self.blocks[b.next as usize].prev = b.prev;
        } else {
            self.tail = b.prev;
        }
        if b.prev != NULL_BLOCK {
            self.blocks[b.prev as usize].next = b.next;
        } else {
            self.head = b.next;
        }
        self.blocks[idx].prev = NULL_BLOCK;
        self.blocks[idx].next = NULL_BLOCK;
        self.size -= 1;
        Ok(())
    }

    fn pop_front(&mut self) -> Result<BlockId> {
        ensure!(self.head != NULL_BLOCK, SentinelError::BlockManager, "list is empty, but pop_front is called");
        let id = self.head;
        self.erase(id)?;
        Ok(id)
    }

    pub fn num_free_blocks(&self) -> usize {
        self.size
    }

    pub fn allocate(&mut self) -> Result<BlockId> {
        let id = self.pop_front()?;
        self.blocks[id as usize].ref_count += 1;
        Ok(id)
    }

    pub fn pin_block(&mut self, id: BlockId) -> Result<()> {
        let idx = self.index(id)?;
        ensure!(self.blocks[idx].ref_count > 0, SentinelError::BlockManager, "try to pin free block {id}");
        self.blocks[idx].ref_count += 1;
        Ok(())
    }

    pub fn free(&mut self, id: BlockId) -> Result<()> {
        let idx = self.index(id)?;
        self.blocks[idx].ref_count -= 1;
        ensure!(self.blocks[idx].ref_count >= 0, SentinelError::BlockManager, "ref_count corrupted for block {id}");
        if self.blocks[idx].ref_count == 0 {
            self.push_back(id)?;
        }
        Ok(())
    }

    pub fn ref_count(&self, id: BlockId) -> Result<i32> {
        Ok(self.blocks[self.index(id)?].ref_count)
    }
}

/// Snapshot of KV-block pool usage (for logging / metrics).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockPoolStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub used_blocks: usize,
    pub peak_used: usize,
}

#[derive(Debug)]
pub struct BlockManager {
    num_blocks: usize,
    blocks: Blocks,
    peak_used: usize,
}

impl BlockManager {
    pub fn new(num_blocks: usize) -> Result<Self> {
        Ok(Self {
            num_blocks,
            blocks: Blocks::new(num_blocks)?,
            peak_used: 0,
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn reset(&self) -> Result<()> {
        self.blocks.reset()
    }

    pub fn can_allocate(&self, to_allocate: usize) -> AllocStatus {
        if to_allocate >= self.num_blocks {
            return AllocStatus::Never;
        }
        if self.blocks.num_free_blocks() >= to_allocate {
            AllocStatus::Ok
        } else {
            AllocStatus::Later
        }
    }

    pub fn allocate(&mut self, n: usize) -> Result<Vec<BlockId>> {
        let mut table = Vec::with_capacity(n);
        for _ in 0..n {
            table.push(self.blocks.allocate()?);
        }
        self.touch_peak();
        Ok(table)
    }

    /// Frees in reverse order so the most recently allocated block is reused last.
    pub fn free_blocks(&mut self, ids: &[BlockId]) -> Result<()> {
        for &id in ids.iter().rev() {
            self.blocks.free(id)?;
        }
        Ok(())
    }

    pub fn pin_blocks(&mut self, ids: &[BlockId]) -> Result<()> {
        for &id in ids {
            self.blocks.pin_block(id)?;
        }
        Ok(())
    }

    pub fn pin_block(&mut self, id: BlockId) -> Result<()> {
        self.blocks.pin_block(id)
    }

    pub fn can_allocate_single(&self) -> bool {
        self.blocks.num_free_blocks() >= 1
    }

    pub fn allocate_single(&mut self) -> Result<BlockId> {
        let id = self.blocks.allocate()?;
        self.touch_peak();
        Ok(id)
    }

    pub fn num_free_blocks(&self) -> usize {
        self.blocks.num_free_blocks()
    }

    pub fn num_used_blocks(&self) -> usize {
        // usable blocks are num_blocks - 1 (id 0 reserved)
        (self.num_blocks - 1).saturating_sub(self.blocks.num_free_blocks())
    }

    pub fn peak_used_blocks(&self) -> usize {
        self.peak_used
    }

    pub fn stats(&self) -> BlockPoolStats {
        BlockPoolStats {
            total_blocks: self.num_blocks.saturating_sub(1),
            free_blocks: self.num_free_blocks(),
            used_blocks: self.num_used_blocks(),
            peak_used: self.peak_used,
        }
    }

    fn touch_peak(&mut self) {
        let used = self.num_used_blocks();
        if used > self.peak_used {
            self.peak_used = used;
        }
    }
}

/// Allocate KV blocks (and an optional state slot) for `seq`.
/// The caller must have verified there are enough free resources.
pub fn allocate_seq(bm: &mut BlockManager, sm: Option<&mut BlockManager>, seq: &mut Sequence) -> Result<()> {
    let n = seq.get_num_blocks();
    seq.cache.block_table = bm.allocate(n)?;
    if let Some(sm) = sm {
        seq.cache.state_id = Some(sm.allocate_single()?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_tiny_pool() {
        assert!(BlockManager::new(1).is_err());
        assert!(BlockManager::new(2).is_ok());
    }

    #[test]
    fn alloc_free_roundtrip() {
        let mut bm = BlockManager::new(9).unwrap();
        assert_eq!(bm.num_free_blocks(), 8);
        let t = bm.allocate(5).unwrap();
        assert_eq!(t, vec![1, 2, 3, 4, 5]);
        assert_eq!(bm.num_free_blocks(), 3);
        assert_eq!(bm.num_used_blocks(), 5);
        bm.free_blocks(&t).unwrap();
        assert_eq!(bm.num_free_blocks(), 8);
        assert_eq!(bm.peak_used_blocks(), 5);
        bm.reset().unwrap();
    }

    #[test]
    fn can_allocate_statuses() {
        let mut bm = BlockManager::new(5).unwrap();
        assert_eq!(bm.can_allocate(4), AllocStatus::Ok);
        assert_eq!(bm.can_allocate(5), AllocStatus::Never);
        let _t = bm.allocate(3).unwrap();
        assert_eq!(bm.can_allocate(2), AllocStatus::Later);
    }

    #[test]
    fn pinned_block_survives_one_free() {
        let mut bm = BlockManager::new(4).unwrap();
        let b = bm.allocate_single().unwrap();
        bm.pin_block(b).unwrap();
        bm.free_blocks(&[b]).unwrap();
        assert_eq!(bm.num_free_blocks(), 2);
        bm.free_blocks(&[b]).unwrap();
        assert_eq!(bm.num_free_blocks(), 3);
    }

    #[test]
    fn exhaustion_is_an_error() {
        let mut bm = BlockManager::new(3).unwrap();
        bm.allocate(2).unwrap();
        assert!(bm.allocate_single().is_err());
    }

    #[test]
    fn stats_report_peak() {
        let mut bm = BlockManager::new(10).unwrap();
        let a = bm.allocate(3).unwrap();
        let b = bm.allocate(2).unwrap();
        assert_eq!(bm.stats().used_blocks, 5);
        bm.free_blocks(&b).unwrap();
        assert_eq!(bm.stats().peak_used, 5);
        assert_eq!(bm.stats().used_blocks, 3);
        bm.free_blocks(&a).unwrap();
    }
}
