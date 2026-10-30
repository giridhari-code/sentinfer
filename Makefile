# Sentinel / kLLM
.PHONY: test release kllm train bench clean

test:
	./check.sh

kllm:
	cargo build --release -p sentinel-serve --bin kllm

train:
	cargo build --release -p sentinel-train --bin train --bin generate --bin tokenize

bench:
	cargo run --release -p sentinel-model --example bench

release:
	./scripts/release.sh

clean:
	cargo clean
	rm -rf dist
