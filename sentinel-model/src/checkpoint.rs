//! Checkpoint files: weights (`.skv`) and the byte-level helpers the trainer reuses for its
//! resume state.
//!
//! Layout of a weights file (little endian):
//! `"SKVW"`, version u32, config (8 x u64 sizes, rope_theta f32, rms_eps f32, eos i32),
//! tensor count u32, then per tensor: name (u16 length + UTF-8), element count u64, f32 data.
//! The file ends with an FNV-1a 64-bit checksum of everything before it, so truncated or
//! corrupted files are rejected instead of silently loading garbage.

use std::fs;
use std::path::Path;

use sentinel_core::{Result, SentinelError};

use crate::seekv::{SeekVConfig, SeekVWeights};

const WEIGHTS_MAGIC: &[u8; 4] = b"SKVW";
const VERSION: u32 = 1;

pub fn io_err(context: &str, e: std::io::Error) -> SentinelError {
    SentinelError::Io(format!("{context}: {e}"))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[derive(Default)]
pub struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn string(&mut self, s: &str) {
        self.buf.extend_from_slice(&(s.len() as u16).to_le_bytes());
        self.buf.extend_from_slice(s.as_bytes());
    }
    pub fn f32_slice(&mut self, data: &[f32]) {
        self.u64(data.len() as u64);
        self.buf.reserve(data.len() * 4);
        for v in data {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    /// Appends the checksum and returns the finished file contents.
    pub fn finish(mut self) -> Vec<u8> {
        let sum = fnv1a(&self.buf);
        self.buf.extend_from_slice(&sum.to_le_bytes());
        self.buf
    }
}

pub struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    /// Verifies the trailing checksum and reads the rest.
    pub fn checked(data: &'a [u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(SentinelError::Io("file too short to be a checkpoint".into()));
        }
        let (body, trailer) = data.split_at(data.len() - 8);
        let mut stored = [0u8; 8];
        stored.copy_from_slice(trailer);
        if u64::from_le_bytes(stored) != fnv1a(body) {
            return Err(SentinelError::Io("checksum mismatch: file is truncated or corrupted".into()));
        }
        Ok(Self { data: body, pos: 0 })
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.data.len() - self.pos < n {
            return Err(SentinelError::Io("unexpected end of checkpoint".into()));
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }
    pub fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(b))
    }
    pub fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        b.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(b))
    }
    pub fn i32(&mut self) -> Result<i32> {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4)?);
        Ok(i32::from_le_bytes(b))
    }
    pub fn f32(&mut self) -> Result<f32> {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4)?);
        Ok(f32::from_le_bytes(b))
    }
    pub fn string(&mut self) -> Result<String> {
        let mut b = [0u8; 2];
        b.copy_from_slice(self.take(2)?);
        let len = u16::from_le_bytes(b) as usize;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| SentinelError::Io("invalid UTF-8 in tensor name".into()))
    }
    /// Reads a length-prefixed f32 array into `dst`, which must have exactly that length.
    pub fn f32_into(&mut self, dst: &mut [f32]) -> Result<()> {
        let len = self.u64()? as usize;
        if len != dst.len() {
            return Err(SentinelError::Io(format!("tensor has {len} elements, expected {}", dst.len())));
        }
        for (d, chunk) in dst.iter_mut().zip(self.take(len * 4)?.chunks_exact(4)) {
            let mut b = [0u8; 4];
            b.copy_from_slice(chunk);
            *d = f32::from_le_bytes(b);
        }
        Ok(())
    }
    pub fn is_empty(&self) -> bool {
        self.pos == self.data.len()
    }
}

/// Writes through a temporary file and renames, so a crash never leaves a half-written file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| io_err(&format!("creating {}", parent.display()), e))?;
        }
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    fs::write(&tmp, bytes).map_err(|e| io_err(&format!("writing {}", tmp.display()), e))?;
    fs::rename(&tmp, path).map_err(|e| io_err(&format!("renaming to {}", path.display()), e))
}

pub fn save_weights(path: &Path, cfg: &SeekVConfig, weights: &SeekVWeights) -> Result<()> {
    let mut w = ByteWriter::new();
    w.bytes(WEIGHTS_MAGIC);
    w.u32(VERSION);
    for v in [
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.max_positions,
    ] {
        w.u64(v as u64);
    }
    w.f32(cfg.rope_theta);
    w.f32(cfg.rms_eps);
    w.i32(cfg.eos_token_id);
    let tensors = weights.named_tensors();
    w.u32(tensors.len() as u32);
    for (name, data) in tensors {
        w.string(&name);
        w.f32_slice(data);
    }
    write_atomic(path, &w.finish())
}

pub fn load_weights(path: &Path) -> Result<(SeekVConfig, SeekVWeights)> {
    let data = fs::read(path).map_err(|e| io_err(&format!("reading {}", path.display()), e))?;
    let mut r = ByteReader::checked(&data)?;
    if r.bytes(4)? != WEIGHTS_MAGIC {
        return Err(SentinelError::Io("not a Seek-V weights file (bad magic)".into()));
    }
    let version = r.u32()?;
    if version != VERSION {
        return Err(SentinelError::Io(format!("unsupported checkpoint version {version}")));
    }
    let mut sizes = [0usize; 8];
    for s in sizes.iter_mut() {
        *s = r.u64()? as usize;
    }
    let cfg = SeekVConfig {
        vocab_size: sizes[0],
        hidden_size: sizes[1],
        num_layers: sizes[2],
        num_heads: sizes[3],
        num_kv_heads: sizes[4],
        head_dim: sizes[5],
        intermediate_size: sizes[6],
        max_positions: sizes[7],
        rope_theta: r.f32()?,
        rms_eps: r.f32()?,
        eos_token_id: r.i32()?,
    };
    cfg.validate()?;

    let mut weights = SeekVWeights::zeros(&cfg);
    let count = r.u32()? as usize;
    let mut tensors = weights.named_tensors_mut();
    if count != tensors.len() {
        return Err(SentinelError::Io(format!("checkpoint has {count} tensors, expected {}", tensors.len())));
    }
    for (name, dst) in tensors.iter_mut() {
        let got = r.string()?;
        if &got != name {
            return Err(SentinelError::Io(format!("expected tensor `{name}`, found `{got}`")));
        }
        r.f32_into(dst)?;
    }
    if !r.is_empty() {
        return Err(SentinelError::Io("trailing bytes after the last tensor".into()));
    }
    drop(tensors);
    weights.validate(&cfg)?;
    Ok((cfg, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("sentinel-test-{}-{name}", std::process::id()))
    }

    #[test]
    fn weights_roundtrip_exactly() {
        let cfg = SeekVConfig::micro(20);
        let w = SeekVWeights::random(&cfg, 5);
        let path = temp_path("roundtrip.skv");
        save_weights(&path, &cfg, &w).unwrap();
        let (cfg2, w2) = load_weights(&path).unwrap();
        assert_eq!(cfg2.vocab_size, cfg.vocab_size);
        assert_eq!(cfg2.eos_token_id, cfg.eos_token_id);
        assert_eq!(cfg2.rope_theta, cfg.rope_theta);
        assert_eq!(w2.embed, w.embed);
        assert_eq!(w2.layers[1].w_down, w.layers[1].w_down);
        assert_eq!(w2.lm_head, w.lm_head);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn corruption_and_truncation_are_detected() {
        let cfg = SeekVConfig::micro(20);
        let w = SeekVWeights::random(&cfg, 5);
        let path = temp_path("corrupt.skv");
        save_weights(&path, &cfg, &w).unwrap();
        let good = fs::read(&path).unwrap();

        let mut flipped = good.clone();
        let mid = flipped.len() / 2;
        flipped[mid] ^= 0xff;
        fs::write(&path, &flipped).unwrap();
        assert!(load_weights(&path).is_err());

        fs::write(&path, &good[..good.len() - 100]).unwrap();
        assert!(load_weights(&path).is_err());

        fs::write(&path, b"short").unwrap();
        assert!(load_weights(&path).is_err());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn wrong_magic_is_rejected() {
        let mut w = ByteWriter::new();
        w.bytes(b"NOPE");
        w.u32(1);
        let path = temp_path("magic.skv");
        fs::write(&path, w.finish()).unwrap();
        let err = load_weights(&path).err().expect("must fail");
        assert!(err.to_string().contains("magic"), "{err}");
        fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_is_an_io_error() {
        assert!(matches!(load_weights(&temp_path("does-not-exist.skv")), Err(SentinelError::Io(_))));
    }

    #[test]
    fn loaded_model_matches_the_saved_one() {
        use crate::model::{Model, StepInput};
        let cfg = SeekVConfig::tiny();
        let weights = SeekVWeights::random(&cfg, 9);
        let path = temp_path("model.skv");
        save_weights(&path, &cfg, &weights).unwrap();
        let mut original = crate::seekv::SeekV::new(cfg, weights).unwrap();
        let mut loaded = crate::seekv::SeekV::load(&path).unwrap();
        let step = [StepInput { seq_id: 1, tokens: &[1, 2, 3], start_pos: 0 }];
        assert_eq!(original.forward(&step).unwrap(), loaded.forward(&step).unwrap());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn byte_reader_rejects_wrong_tensor_length() {
        let mut w = ByteWriter::new();
        w.f32_slice(&[1.0, 2.0, 3.0]);
        let bytes = w.finish();
        let mut r = ByteReader::checked(&bytes).unwrap();
        let mut dst = [0.0f32; 2];
        assert!(r.f32_into(&mut dst).is_err());
    }
}
