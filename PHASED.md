# Phase D — tokenizer + release packaging

## Done

| Item | Detail |
| --- | --- |
| `PieceTokenizer` | vocab file, greedy longest-match encode/decode |
| `IdTokenizer::from_vocab_file` | decode table for pre-tokenized models |
| `write_token_file` | emit `u16`/`u32` id files for training |
| `tokenize` CLI | text → token bin for `train --tokens-u*` |
| `scripts/release.sh` | stage `kllm`, `train`, `generate`, `tokenize`, docs → tarball |
| `Makefile` | `make test` / `kllm` / `train` / `bench` / `release` |

## Vocab file format

One piece per line; line index = token id. Optional special names:

```
<bos>
hello
world
<eos>
<unk>
```

```bash
# tokenize corpus
cargo run -p sentinel-train --bin tokenize -- \
  --vocab vocab.txt --data corpus.txt --out tokens.bin --bits 16 --add-bos

# train on ids
cargo run -p sentinel-train --bin train -- \
  --tokens-u16 tokens.bin --vocab-size N --eos-id E --out runs/x
```

Byte path (default) still needs no vocab file.

## Release

```bash
./scripts/release.sh
# → dist/kllm-0.1.0-linux-x86_64.tar.gz
```
