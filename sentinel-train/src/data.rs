//! Token datasets and window sampling.
//!
//! A dataset is one long token stream. Training samples random windows of `seq_len + 1` tokens
//! (inputs are the first `seq_len`, targets the last `seq_len`); evaluation walks the
//! validation stream in order with non-overlapping windows.

use std::fs;
use std::path::Path;

use sentinel_core::{Result, SentinelError};
use sentinel_model::checkpoint::io_err;
use sentinel_model::ops::SplitMix;
use sentinel_model::ByteTokenizer;

#[derive(Debug, Clone, Default)]
pub struct TokenDataset {
    tokens: Vec<i32>,
}

impl TokenDataset {
    pub fn from_tokens(tokens: Vec<i32>) -> Self {
        Self { tokens }
    }

    /// Byte-level tokens of each document, with an end-of-sequence token after every document.
    pub fn from_text_documents(docs: &[&str]) -> Self {
        let tok = ByteTokenizer;
        let mut tokens = Vec::new();
        for doc in docs {
            tokens.extend(tok.encode(doc, false));
            tokens.push(ByteTokenizer::EOS);
        }
        Self { tokens }
    }

    /// Byte-level tokens of a text file. With `split_docs`, blank lines separate documents and
    /// each one gets an end-of-sequence token; otherwise the file is one raw byte stream.
    pub fn from_text_file(path: &Path, split_docs: bool) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| io_err(&format!("reading {}", path.display()), e))?;
        if split_docs {
            let text = String::from_utf8_lossy(&bytes);
            let docs: Vec<&str> = text.split("\n\n").map(str::trim).filter(|d| !d.is_empty()).collect();
            Ok(Self::from_text_documents(&docs))
        } else {
            Ok(Self { tokens: bytes.iter().map(|&b| b as i32).collect() })
        }
    }

    /// Pre-tokenized ids as little-endian `u16` (for vocabularies up to 65536).
    pub fn from_u16_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| io_err(&format!("reading {}", path.display()), e))?;
        if bytes.len() % 2 != 0 {
            return Err(SentinelError::Io(format!("{}: size is not a multiple of 2 bytes", path.display())));
        }
        Ok(Self { tokens: bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as i32).collect() })
    }

    /// Pre-tokenized ids as little-endian `u32`.
    pub fn from_u32_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| io_err(&format!("reading {}", path.display()), e))?;
        if bytes.len() % 4 != 0 {
            return Err(SentinelError::Io(format!("{}: size is not a multiple of 4 bytes", path.display())));
        }
        let mut tokens = Vec::with_capacity(bytes.len() / 4);
        for c in bytes.chunks_exact(4) {
            let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            tokens.push(i32::try_from(v).map_err(|_| SentinelError::Io(format!("{}: token id {v} too large", path.display())))?);
        }
        Ok(Self { tokens })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    pub fn max_token(&self) -> Option<i32> {
        self.tokens.iter().copied().max()
    }

    /// Splits off the last `val_fraction` of the stream as a validation set: `(train, validation)`.
    pub fn split(mut self, val_fraction: f32) -> (Self, Self) {
        let n_val = ((self.tokens.len() as f64) * val_fraction.clamp(0.0, 1.0) as f64) as usize;
        let val = self.tokens.split_off(self.tokens.len() - n_val);
        (self, Self { tokens: val })
    }

    /// `count` windows of `window` tokens starting at uniformly random offsets.
    pub fn sample_windows(&self, rng: &mut SplitMix, count: usize, window: usize) -> Result<Vec<Vec<i32>>> {
        if window == 0 || self.tokens.len() < window {
            return Err(SentinelError::InvalidState(format!(
                "dataset has {} tokens, need at least {window} for one window",
                self.tokens.len()
            )));
        }
        let starts = (self.tokens.len() - window + 1) as u64;
        Ok((0..count)
            .map(|_| {
                let start = (rng.next_u64() % starts) as usize;
                self.tokens[start..start + window].to_vec()
            })
            .collect())
    }

    /// Up to `max_windows` consecutive, non-overlapping windows from the start of the stream.
    pub fn eval_windows(&self, window: usize, max_windows: usize) -> Vec<Vec<i32>> {
        if window == 0 {
            return Vec::new();
        }
        self.tokens.chunks_exact(window).take(max_windows).map(|c| c.to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("sentinel-train-test-{}-{name}", std::process::id()))
    }

    #[test]
    fn documents_are_separated_by_eos() {
        let ds = TokenDataset::from_text_documents(&["ab", "c"]);
        assert_eq!(ds.tokens(), &[97, 98, ByteTokenizer::EOS, 99, ByteTokenizer::EOS]);
    }

    #[test]
    fn text_file_modes() {
        let path = temp_path("text.txt");
        fs::write(&path, "hello\n\nworld\n\n\n\nbye").unwrap();
        let raw = TokenDataset::from_text_file(&path, false).unwrap();
        assert_eq!(raw.len(), "hello\n\nworld\n\n\n\nbye".len());
        let docs = TokenDataset::from_text_file(&path, true).unwrap();
        assert_eq!(docs.tokens().iter().filter(|&&t| t == ByteTokenizer::EOS).count(), 3);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn binary_token_files() {
        let p16 = temp_path("tokens16.bin");
        fs::write(&p16, [1u8, 0, 44, 1, 255, 255]).unwrap();
        assert_eq!(TokenDataset::from_u16_file(&p16).unwrap().tokens(), &[1, 300, 65535]);
        fs::write(&p16, [1u8, 0, 2]).unwrap();
        assert!(TokenDataset::from_u16_file(&p16).is_err());
        fs::remove_file(&p16).ok();

        let p32 = temp_path("tokens32.bin");
        fs::write(&p32, [7u8, 0, 0, 0, 0, 1, 0, 0]).unwrap();
        assert_eq!(TokenDataset::from_u32_file(&p32).unwrap().tokens(), &[7, 256]);
        fs::write(&p32, [255u8, 255, 255, 255]).unwrap();
        assert!(TokenDataset::from_u32_file(&p32).is_err()); // does not fit in i32
        fs::remove_file(&p32).ok();
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(TokenDataset::from_text_file(&temp_path("nope.txt"), false).is_err());
    }

    #[test]
    fn split_keeps_the_tail_for_validation() {
        let ds = TokenDataset::from_tokens((0..100).collect());
        let (train, val) = ds.split(0.1);
        assert_eq!(train.len(), 90);
        assert_eq!(val.tokens(), &(90..100).collect::<Vec<i32>>()[..]);
        let (all, none) = TokenDataset::from_tokens((0..10).collect()).split(0.0);
        assert_eq!((all.len(), none.len()), (10, 0));
    }

    #[test]
    fn sampled_windows_are_valid_and_reproducible() {
        let ds = TokenDataset::from_tokens((0..50).collect());
        let a = ds.sample_windows(&mut SplitMix::new(3), 20, 9).unwrap();
        let b = ds.sample_windows(&mut SplitMix::new(3), 20, 9).unwrap();
        assert_eq!(a, b);
        for w in &a {
            assert_eq!(w.len(), 9);
            // windows are contiguous slices of the stream (consecutive integers here)
            assert!(w.windows(2).all(|p| p[1] == p[0] + 1));
            assert!(*w.last().unwrap() < 50);
        }
        let starts: std::collections::HashSet<i32> = a.iter().map(|w| w[0]).collect();
        assert!(starts.len() > 3, "offsets should vary");
    }

    #[test]
    fn window_can_cover_the_whole_stream() {
        let ds = TokenDataset::from_tokens(vec![1, 2, 3]);
        let w = ds.sample_windows(&mut SplitMix::new(1), 4, 3).unwrap();
        assert!(w.iter().all(|x| x == &[1, 2, 3]));
    }

    #[test]
    fn too_short_dataset_is_an_error() {
        let ds = TokenDataset::from_tokens(vec![1, 2]);
        assert!(ds.sample_windows(&mut SplitMix::new(1), 1, 3).is_err());
        assert!(ds.sample_windows(&mut SplitMix::new(1), 1, 0).is_err());
    }

    #[test]
    fn eval_windows_do_not_overlap() {
        let ds = TokenDataset::from_tokens((0..10).collect());
        assert_eq!(ds.eval_windows(4, 10), vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        assert_eq!(ds.eval_windows(4, 1).len(), 1);
        assert!(ds.eval_windows(0, 5).is_empty());
    }

    #[test]
    fn max_token_scan() {
        assert_eq!(TokenDataset::from_tokens(vec![3, 9, 2]).max_token(), Some(9));
        assert_eq!(TokenDataset::default().max_token(), None);
    }
}
