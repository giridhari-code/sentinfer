//! Convert text → token id file for `train --tokens-u16` / `--tokens-u32`.
//!
//!   tokenize --vocab vocab.txt --data corpus.txt --out tokens.bin [--bits 16] [--add-bos] [--add-eos]

use std::path::PathBuf;

use sentinel_model::{write_token_file, ByteTokenizer, PieceTokenizer, Tokenizer};

const USAGE: &str = "\
usage: tokenize --data FILE --out FILE [--vocab FILE] [--bits 16|32] [--add-bos] [--add-eos] [--byte]

Without --vocab, uses ByteTokenizer (vocab 258). With --vocab, uses PieceTokenizer.
";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}\n\n{USAGE}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut data = None;
    let mut out = None;
    let mut vocab: Option<PathBuf> = None;
    let mut bits = 16u8;
    let mut add_bos = false;
    let mut add_eos = false;
    let mut force_byte = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--data" => data = Some(it.next().ok_or("--data needs value")?),
            "--out" => out = Some(it.next().ok_or("--out needs value")?),
            "--vocab" => vocab = Some(it.next().ok_or("--vocab needs value")?.into()),
            "--bits" => bits = it.next().ok_or("--bits needs value")?.parse().map_err(|_| "bad --bits")?,
            "--add-bos" => add_bos = true,
            "--add-eos" => add_eos = true,
            "--byte" => force_byte = true,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown `{other}`")),
        }
    }
    let data = data.ok_or("--data required")?;
    let out = out.ok_or("--out required")?;
    let text = std::fs::read_to_string(&data).map_err(|e| e.to_string())?;

    let mut ids = if force_byte || vocab.is_none() {
        Tokenizer::encode(&ByteTokenizer, &text, add_bos)
    } else {
        let tok = PieceTokenizer::from_vocab_file(vocab.as_ref().unwrap()).map_err(|e| e.to_string())?;
        let mut ids = Tokenizer::encode(&tok, &text, add_bos);
        if add_eos {
            if let Some(e) = tok.eos_id() {
                ids.push(e);
            }
        }
        ids
    };
    if add_eos && (force_byte || vocab.is_none()) {
        ids.push(ByteTokenizer::EOS);
    }

    write_token_file(&out, &ids, bits).map_err(|e| e.to_string())?;
    eprintln!("wrote {} tokens → {out} (bits={bits})", ids.len());
    Ok(())
}
