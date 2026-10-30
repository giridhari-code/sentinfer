//! Quantize a Seek-V checkpoint.
//!
//!   quantize -i model.skv -o model.skq --scheme int8     # packed SKQ1
//!   quantize -i model.skv -o model-q8.skv --scheme int8  # fp32 rewrite (compat)

use std::path::Path;

use sentinel_model::{quantize_checkpoint, QuantScheme, QuantizedSeekV, SeekV};

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
                eprintln!("usage: quantize -i model.skv -o out.skq|out.skv --scheme int8|int4");
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
    let packed_out = output.ends_with(".skq");
    let model = SeekV::load(Path::new(&input)).unwrap_or_else(|e| {
        eprintln!("load: {e}");
        std::process::exit(1);
    });
    let q = QuantizedSeekV::from_fp32(&model, scheme).unwrap_or_else(|e| {
        eprintln!("quantize: {e}");
        std::process::exit(1);
    });
    let fp: usize = model.weights().named_tensors().iter().map(|(_, t)| t.len() * 4).sum();
    let packed = q.nbytes_packed();
    if packed_out {
        q.save_packed(Path::new(&output)).unwrap_or_else(|e| {
            eprintln!("save skq: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "ok scheme={scheme:?} fp32_bytes={fp} packed={packed} ratio={:.2}x → {output} (SKQ1)",
            fp as f64 / packed.max(1) as f64
        );
    } else {
        match quantize_checkpoint(Path::new(&input), Path::new(&output), scheme) {
            Ok((fp, packed)) => eprintln!(
                "ok scheme={scheme:?} fp32_bytes={fp} packed_est={packed} ratio={:.2}x → {output}",
                fp as f64 / packed.max(1) as f64
            ),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
}
