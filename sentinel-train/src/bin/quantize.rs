//! Quantize a Seek-V checkpoint (int8 / int4).
//!
//!   quantize --input model.skv --output model-q8.skv --scheme int8

use sentinel_model::QuantScheme;
use sentinel_model::quantize_checkpoint;

fn main() {
    let mut input = None;
    let mut output = None;
    let mut scheme = QuantScheme::Int8;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" | "-i" => input = it.next(),
            "--output" | "-o" => output = it.next(),
            "--scheme" => {
                let s = it.next().unwrap_or_default();
                scheme = match s.as_str() {
                    "int8" | "q8" => QuantScheme::Int8,
                    "int4" | "q4" => QuantScheme::Int4,
                    _ => {
                        eprintln!("unknown scheme {s} (int8|int4)");
                        std::process::exit(2);
                    }
                };
            }
            "-h" | "--help" => {
                eprintln!("usage: quantize -i model.skv -o out.skv --scheme int8|int4");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown {other}");
                std::process::exit(2);
            }
        }
    }
    let input = input.expect("--input required");
    let output = output.expect("--output required");
    match quantize_checkpoint(&input, &output, scheme) {
        Ok((fp, packed)) => {
            eprintln!(
                "ok scheme={scheme:?} fp32_bytes={fp} packed_est={packed} ratio={:.2}x → {output}",
                fp as f64 / packed.max(1) as f64
            );
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
