# cim_compile

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](Cargo.toml)
[![Rust](https://img.shields.io/badge/rust-2024-orange)](rust-toolchain.toml)
[![CI](https://github.com/nealpro/cim_compile/actions/workflows/ci.yml/badge.svg)](https://github.com/nealpro/cim_compile/actions/workflows/ci.yml)
[![Target](https://img.shields.io/badge/target-AIHWKIT%20simulation-lightgrey)](https://aihwkit.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## TL;DR

`cim_compile` is a local compiler prototype for compute-in-memory simulation. The active path is:

```text
ONNX model slice -> NormalizedProgram -> verified cim dialect -> AIHWKIT simulation package
```

The compiler extracts supported static-weight projections, schedules them as `cim.tile.dispatch` operations, emits deterministic full-precision tile payloads, and can invoke a checked-in Python AIHWKIT bridge. Dynamic activation work such as embedding lookup, RMSNorm, rotary math, attention score/softmax/context kernels, residual glue, and LM-head logits remains explicit digital PyTorch fallback for the current tiny-decoder slice.

## Current Scope

- Rust frontend for a deliberately narrow ONNX/prost slice.
- Normalized IR for supported projection, attention, MLP, and token-logits metadata.
- Stable MLIR-like `output.cim` text with parser/printer/verifier tests.
- AIHWKIT package artifacts: `aihwkit_manifest.json`, `aihwkit_weights.bin`, and optional `aihwkit_digital.bin`.
- Python bridge at `python/cim_compile_aihwkit/runner.py` that reconstructs AIHWKIT `AnalogLinearMapped` layers and runs the supported projection, attention, logits, and greedy token-ID generation paths.
- Default AIHWKIT runtime config is an ideal mapped inference configuration so v1 validates compiler plumbing deterministically before realistic device/noise presets are added.

The old hardware/RV32I code is retained as transitional scaffolding. New simulator work should target AIHWKIT.

## Installation

Rust prerequisites:

- Rust stable with edition 2024 support
- Cargo

```bash
cargo build
cargo test
```

Python prerequisites for `--run-aihwkit` and ignored full-model validation:

- Python 3.10+
- PyTorch
- IBM AIHWKIT
- Optional: Hugging Face `transformers` for `--prompt-text`, `--tokenizer`, and `--decode-text`
- Optional: `sentencepiece` when the supplied local tokenizer requires it

The bridge imports AIHWKIT and tokenizer dependencies lazily, so normal Rust checks do not require Python simulator packages.

The SmolLM2 prompt path also requires `onnx` and local tokenizer/config files next to the ONNX file:

```bash
uv pip install --python .venv/bin/python onnx
```

For the local CUDA-enabled AIHWKIT source build used during development, see
[`docs/aihwkit_gpu_build.md`](docs/aihwkit_gpu_build.md).

## Quick Start

Compile a supported real tiny-decoder model:

```bash
cargo run --release -- data/model.onnx -o out --tile-size 128
```

Run token-ID-to-logits inference through AIHWKIT:

```bash
cargo run --release -- data/model.onnx -o out --tile-size 128 --run-aihwkit --input-ids 1,2,3,4 --top-k 5
```

Run greedy token-ID generation:

```bash
cargo run --release -- data/model.onnx -o out --tile-size 128 --run-aihwkit --generate-ids --input-ids 1,2,3,4 --max-new-tokens 8 --top-k 5
```

Run an interactive token-ID session:

```bash
cargo run --release -- data/model.onnx -o out --tile-size 128 --run-aihwkit --interactive-ids --max-new-tokens 4 --top-k 5
```

The prompt accepts comma-separated token IDs. Enter `quit` or press Ctrl-D to exit.

Run text mode with an external compatible tokenizer:

```bash
cargo run --release -- data/model.onnx -o out --tile-size 128 --run-aihwkit --generate-ids --prompt-text "Hello" --tokenizer /path/to/tokenizer --decode-text --max-new-tokens 8
```

Run an interactive text session:

```bash
cargo run --release -- data/model.onnx -o out --tile-size 128 --run-aihwkit --interactive-text --tokenizer /path/to/tokenizer --max-new-tokens 8 --top-k 5
```

No tokenizer artifacts are bundled with `data/model.onnx`; text mode requires a local Hugging Face-compatible `--tokenizer` whose vocabulary matches the emitted manifest.

Run the SmolLM2 prompt-to-text path:

```bash
cargo run --release -- --prompt "Hello" --max-new-tokens 8 --temperature 0.8 data/smolLM2/model_fp16.onnx
```

This path uses the `HuggingFaceTB/SmolLM2-135M-Instruct` ONNX export, specifically [`onnx/model_fp16.onnx`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/onnx/model_fp16.onnx), with tokenizer/config sidecars from the same Hugging Face model repo. It is a separate hybrid runtime path: tokenizer/control-flow operations stay digital in PyTorch, while selected static projection layers use AIHWKIT `AnalogLinearMapped`. The default analog placement is the first decoder layer; use `CIM_COMPILE_SMOLLM2_ANALOG_LAYERS=all` to attempt all decoder layers. SmolLM2 prompt decoding samples by default with `--temperature 0.8`; pass `--temperature 0` for greedy decoding. Run this path outside restricted sandboxes when using the GPU-enabled AIHWKIT build.

Showcase all decoder projection layers through AIHWKIT:

```bash
CIM_COMPILE_SMOLLM2_ANALOG_LAYERS=all \
  cargo run -- --prompt "Explain compute-in-memory in 2 sentences." \
    --max-new-tokens 180 \
    --temperature 0.7 \
    data/smolLM2/model_fp16.onnx
```

Example output:

```text
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.08s
 Running `target/debug/cim_compile --prompt 'Explain compute-in-memory in 2 sentences.' --max-new-tokens 180 --temperature 0.7 data/smolLM2/model_fp16.onnx`
loading ONNX initializers from data/smolLM2/model_fp16.onnx
SmolLM2 runtime ready: device=cuda, active layers=30/30, AIHWKIT analog layers=30/30
In 2 sentences, compute-in-memory is a technique that allows data to be stored and processed in the same memory as the program running on top of it. This allows for efficient data sharing and reduced memory allocation for computations.
```

Generate and compile bundled ONNX fixtures:

```bash
python3 tests/generate_onnx_fixtures.py --output-dir /tmp/cim-fixtures --dim 512
cargo run --release -- /tmp/cim-fixtures/memristor_mha_unrolled.onnx -o out
cargo run --release -- /tmp/cim-fixtures/mha_bfloat16.onnx -o out-fused
```

## CLI

```text
cim_compile [OPTIONS] <ONNX_PATH>

Arguments:
  <ONNX_PATH>              Path to the ONNX model file

Options:
      --prompt <PROMPT>    Prompt text for the SmolLM2 prompt-to-text runtime
  -o, --output-dir <DIR>   Output directory for output.cim and AIHWKIT artifacts [default: .]
      --tile-size <N>      Square crossbar tile dimension [default: 128]
      --run-aihwkit        Run the AIHWKIT bridge after writing artifacts
      --python <PYTHON>    Python executable to use with --run-aihwkit
      --input-ids <CSV>    Comma-separated token IDs, not prompt text, for --run-aihwkit
      --interactive-ids    Let the Python runner prompt interactively for token IDs
      --interactive-text   Let the Python runner prompt interactively for prompt text
      --prompt-text <TEXT> Prompt text for text/tokenizer mode
      --tokenizer <PATH>   Local tokenizer path for text/tokenizer mode
      --decode-text        Decode generated token IDs back to text
      --top-k <N>          Number of next-token candidates to report [default: 5]
      --temperature <T>    Sampling temperature for SmolLM2 prompt mode; 0 selects greedy decoding [default: 0.8]
      --generate-ids       Generate token IDs greedily with the AIHWKIT bridge
      --max-new-tokens <N> Maximum number of token IDs to generate [default: 8]
      --eos-token-id <ID>  Optional token ID that stops generation early
  -h, --help               Print help
  -V, --version            Print version
```

Library entry point:

```rust
let compilation = cim_compile::compile_onnx(path, cim_compile::CompileConfig::square(128))?;
```

`Compilation` contains the normalized IR, verified `cim::Program`, and AIHWKIT package bytes/text.

## `data/model.onnx` Runtime Contract

The checked-in model is a one-layer tiny decoder exported from PyTorch with token-ID inputs. The supported interactive contract is batch size 1, token IDs in `0..31999`, and logits shaped `[1, sequence_length, 32000]`. The runner owns an internal one-layer KV cache during greedy generation; external/non-empty KV-cache inputs are not a supported user-facing API yet.

Text generation is available only when the user supplies a compatible external tokenizer through `--tokenizer`. Without that, the reliable interface is token IDs in and token IDs/top-k IDs out.

## Artifacts

### `output.cim`

Stable text form of the verified `cim` dialect. Example:

```text
cim.tile.dispatch { projection = "wq", tile = [0, 0], matrix_shape = [192, 192], tile_size = [128, 128], weight_offset = 0, order = 0 }
```

`weight_offset` is a byte offset into `aihwkit_weights.bin`.

### `aihwkit_manifest.json`

JSON manifest consumed by the AIHWKIT bridge. It includes:

| Field | Meaning |
|---|---|
| `schema_version` | Manifest schema version. |
| `backend` | Always `aihwkit` for the active backend. |
| `tile_size` | Crossbar tile shape used by compiler scheduling and AIHWKIT mapping. |
| `weight_dtype` | `f32`; AIHWKIT owns analog programming and device simulation. |
| `weights_file` | Relative path to `aihwkit_weights.bin`. |
| `digital_tensors_file` | Relative path to `aihwkit_digital.bin` when digital fallback tensors are needed. |
| `rpu_config` | Default ideal mapped inference config metadata. |
| `projections` | Static-weight projections and tile records. |
| `execution_plan` | CiM versus digital placement records. |
| `simulation_summary` | Runtime placement summary with `aihwkit_stages` and digital stages. |

Each tile record includes `row`, `col`, `matrix_shape`, `tile_size`, `weight_offset`, `byte_len`, and `order`.

### `aihwkit_weights.bin`

Raw `f32` little-endian tile payloads in dispatch order. A tile with order `k` starts at:

```text
k * tile_rows * tile_cols * 4
```

Edge tiles are zero-padded to the selected physical tile shape.

### `aihwkit_digital.bin`

Float32 little-endian tensor payloads for digital fallback. The manifest records each tensor role, shape, byte offset, byte length, and dtype.

## Testing

Default tests generate small ONNX fixtures and do not require AIHWKIT or a local model:

```bash
cargo test
```

CI-equivalent non-test Rust checks:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo check --workspace --all-targets --all-features
```

Ignored full validation tests require a local real model and Python environment with AIHWKIT:

```bash
CIM_COMPILE_REAL_MODEL=data/model.onnx cargo test -- --ignored
CIM_COMPILE_REAL_MODEL=data/model.onnx cargo test --test full -- --ignored
```

Large model files remain local/manual validation assets rather than CI inputs.

## References

See [LINKS.md](LINKS.md) for ONNX, MLIR, AIHWKIT, CiM, and compiler architecture references.
