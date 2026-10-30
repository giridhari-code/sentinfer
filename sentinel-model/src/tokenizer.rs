//! Tokenizers for Seek-V / kLLM.
//!
//! * [`ByteTokenizer`] — default train/serve path (vocab 258)
//! * [`IdTokenizer`] — pre-tokenized ids; optional piece table for decode
//! * [`PieceTokenizer`] — load a vocab file (one piece per line); greedy longest-match encode

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use sentinel_core::{Result, SentinelError};

/// Encode text to token ids and decode ids back to text.
pub trait Tokenizer: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn bos_id(&self) -> Option<i32>;
    fn eos_id(&self) -> Option<i32>;
    fn encode(&self, text: &str, add_bos: bool) -> Vec<i32>;
    fn decode(&self, ids: &[i32]) -> String;
}

/// Byte-level tokenizer: ids 0..=255 are raw bytes, 256 is BOS, 257 is EOS.
#[derive(Debug, Default, Clone, Copy)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub const BOS: i32 = 256;
    pub const EOS: i32 = 257;
    pub const VOCAB_SIZE: usize = 258;
}

impl Tokenizer for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        Self::VOCAB_SIZE
    }
    fn bos_id(&self) -> Option<i32> {
        Some(Self::BOS)
    }
    fn eos_id(&self) -> Option<i32> {
        Some(Self::EOS)
    }
    fn encode(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let mut ids = Vec::with_capacity(text.len() + 1);
        if add_bos {
            ids.push(Self::BOS);
        }
        ids.extend(text.bytes().map(|b| b as i32));
        ids
    }
    fn decode(&self, ids: &[i32]) -> String {
        let bytes: Vec<u8> = ids.iter().filter(|&&i| (0..256).contains(&i)).map(|&i| i as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl ByteTokenizer {
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<i32> {
        Tokenizer::encode(self, text, add_bos)
    }
    pub fn decode(&self, ids: &[i32]) -> String {
        Tokenizer::decode(self, ids)
    }
}

/// Pass-through tokenizer for pre-tokenized id streams.
#[derive(Debug, Clone)]
pub struct IdTokenizer {
    pub vocab_size: usize,
    pub eos_id: Option<i32>,
    pub bos_id: Option<i32>,
    pub pieces: Option<Vec<String>>,
}

impl IdTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size, eos_id: None, bos_id: None, pieces: None }
    }

    /// Load piece table from a vocab file (one UTF-8 piece per line; id = line index).
    pub fn from_vocab_file(path: impl AsRef<Path>, bos_id: Option<i32>, eos_id: Option<i32>) -> Result<Self> {
        let pieces = read_vocab_lines(path.as_ref())?;
        let vocab_size = pieces.len();
        Ok(Self { vocab_size, bos_id, eos_id, pieces: Some(pieces) })
    }
}

impl Tokenizer for IdTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn bos_id(&self) -> Option<i32> {
        self.bos_id
    }
    fn eos_id(&self) -> Option<i32> {
        self.eos_id
    }
    fn encode(&self, _text: &str, _add_bos: bool) -> Vec<i32> {
        Vec::new()
    }
    fn decode(&self, ids: &[i32]) -> String {
        if let Some(pieces) = &self.pieces {
            ids.iter()
                .filter_map(|&i| {
                    if i >= 0 && (i as usize) < pieces.len() {
                        Some(pieces[i as usize].as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("")
        } else {
            ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
        }
    }
}

/// Vocab-file tokenizer with greedy longest-match encoding (Phase D).
///
/// File format: one piece per line, id = 0-based line index.
/// Special lines may be named `<bos>` / `<eos>` / `<unk>` (case-sensitive) to set ids.
#[derive(Debug, Clone)]
pub struct PieceTokenizer {
    pieces: Vec<String>,
    piece_to_id: HashMap<String, i32>,
    /// Pieces sorted by descending length for greedy match.
    by_len: Vec<(String, i32)>,
    bos_id: Option<i32>,
    eos_id: Option<i32>,
    unk_id: Option<i32>,
}

impl PieceTokenizer {
    pub fn from_vocab_file(path: impl AsRef<Path>) -> Result<Self> {
        let pieces = read_vocab_lines(path.as_ref())?;
        Self::from_pieces(pieces)
    }

    pub fn from_pieces(pieces: Vec<String>) -> Result<Self> {
        if pieces.is_empty() {
            return Err(SentinelError::InvalidState("vocab is empty".into()));
        }
        let mut piece_to_id = HashMap::with_capacity(pieces.len());
        let mut bos_id = None;
        let mut eos_id = None;
        let mut unk_id = None;
        for (i, p) in pieces.iter().enumerate() {
            let id = i as i32;
            piece_to_id.insert(p.clone(), id);
            match p.as_str() {
                "<bos>" | "<s>" => bos_id = Some(id),
                "<eos>" | "</s>" => eos_id = Some(id),
                "<unk>" | "<UNK>" => unk_id = Some(id),
                _ => {}
            }
        }
        let mut by_len: Vec<(String, i32)> = piece_to_id.iter().map(|(s, &id)| (s.clone(), id)).collect();
        by_len.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        Ok(Self { pieces, piece_to_id, by_len, bos_id, eos_id, unk_id })
    }

    pub fn piece(&self, id: i32) -> Option<&str> {
        self.pieces.get(id as usize).map(|s| s.as_str())
    }
}

impl Tokenizer for PieceTokenizer {
    fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    fn bos_id(&self) -> Option<i32> {
        self.bos_id
    }
    fn eos_id(&self) -> Option<i32> {
        self.eos_id
    }
    fn encode(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let mut ids = Vec::new();
        if add_bos {
            if let Some(b) = self.bos_id {
                ids.push(b);
            }
        }
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        while i < chars.len() {
            let mut matched = false;
            for (piece, id) in &self.by_len {
                let plen = piece.chars().count();
                if plen == 0 || i + plen > chars.len() {
                    continue;
                }
                let slice: String = chars[i..i + plen].iter().collect();
                if slice == *piece {
                    ids.push(*id);
                    i += plen;
                    matched = true;
                    break;
                }
            }
            if !matched {
                // single char fallback via vocab, else unk, else skip
                let ch: String = chars[i..i + 1].iter().collect();
                if let Some(&id) = self.piece_to_id.get(&ch) {
                    ids.push(id);
                } else if let Some(u) = self.unk_id {
                    ids.push(u);
                }
                i += 1;
            }
        }
        ids
    }
    fn decode(&self, ids: &[i32]) -> String {
        ids.iter()
            .filter_map(|&i| self.piece(i))
            .collect::<Vec<_>>()
            .join("")
    }
}

fn read_vocab_lines(path: &Path) -> Result<Vec<String>> {
    let f = File::open(path).map_err(|e| SentinelError::InvalidState(format!("open {}: {e}", path.display())))?;
    let reader = BufReader::new(f);
    let mut pieces = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| SentinelError::InvalidState(format!("read {} line {}: {e}", path.display(), lineno + 1)))?;
        // keep empty lines as empty pieces (valid id)
        pieces.push(line);
    }
    Ok(pieces)
}

/// Write little-endian token ids (u16 or u32) for training with `--tokens-u16` / `--tokens-u32`.
pub fn write_token_file(path: impl AsRef<Path>, ids: &[i32], bits: u8) -> Result<()> {
    use std::io::Write;
    let mut f = File::create(path.as_ref())
        .map_err(|e| SentinelError::InvalidState(format!("create token file: {e}")))?;
    match bits {
        16 => {
            for &id in ids {
                if id < 0 || id > u16::MAX as i32 {
                    return Err(SentinelError::InvalidState(format!("token {id} does not fit u16")));
                }
                f.write_all(&(id as u16).to_le_bytes())
                    .map_err(|e| SentinelError::InvalidState(format!("write: {e}")))?;
            }
        }
        32 => {
            for &id in ids {
                f.write_all(&(id as u32).to_le_bytes())
                    .map_err(|e| SentinelError::InvalidState(format!("write: {e}")))?;
            }
        }
        _ => return Err(SentinelError::InvalidState("bits must be 16 or 32".into())),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn roundtrip_including_unicode() {
        let t = ByteTokenizer;
        let ids = Tokenizer::encode(&t, "héllo", true);
        assert_eq!(ids[0], ByteTokenizer::BOS);
        assert_eq!(Tokenizer::decode(&t, &ids), "héllo");
    }

    #[test]
    fn decode_skips_special_tokens() {
        assert_eq!(ByteTokenizer.decode(&[104, 105, ByteTokenizer::EOS, -1, 999]), "hi");
    }

    #[test]
    fn id_tokenizer_with_pieces() {
        let t = IdTokenizer {
            vocab_size: 3,
            eos_id: Some(2),
            bos_id: Some(0),
            pieces: Some(vec!["<bos>".into(), "hi".into(), "<eos>".into()]),
        };
        assert_eq!(Tokenizer::decode(&t, &[1, 1]), "hihi");
    }

    #[test]
    fn piece_tokenizer_greedy_longest_match() {
        let pieces = vec![
            "<bos>".into(),
            "a".into(),
            "b".into(),
            "ab".into(),
            "<eos>".into(),
            "<unk>".into(),
        ];
        let t = PieceTokenizer::from_pieces(pieces).unwrap();
        let ids = Tokenizer::encode(&t, "abba", false);
        // "ab" + "b" + "a"
        assert_eq!(ids, vec![3, 2, 1]);
        assert_eq!(Tokenizer::decode(&t, &ids), "abba");
        let with_bos = Tokenizer::encode(&t, "a", true);
        assert_eq!(with_bos[0], 0);
    }

    #[test]
    fn piece_tokenizer_from_file_roundtrip() {
        let dir = std::env::temp_dir().join("sentinel_piece_tok_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("vocab.txt");
        {
            let mut f = File::create(&path).unwrap();
            writeln!(f, "<bos>").unwrap();
            writeln!(f, "hello").unwrap();
            writeln!(f, " ").unwrap();
            writeln!(f, "world").unwrap();
            writeln!(f, "<eos>").unwrap();
        }
        let t = PieceTokenizer::from_vocab_file(&path).unwrap();
        let ids = Tokenizer::encode(&t, "hello world", false);
        assert_eq!(Tokenizer::decode(&t, &ids), "hello world");
    }

    #[test]
    fn write_token_file_u16() {
        let dir = std::env::temp_dir().join("sentinel_tok_write");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ids.bin");
        write_token_file(&path, &[1, 2, 300], 16).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 6);
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 1);
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 300);
    }
}
