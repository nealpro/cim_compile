# cim_compile

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](Cargo.toml)
[![Rust](https://img.shields.io/badge/rust-2024-orange)](rust-toolchain.toml)
[![CI](https://github.com/nealpro/cim_compile/actions/workflows/ci.yml/badge.svg)](https://github.com/nealpro/cim_compile/actions/workflows/ci.yml)
[![Target](https://img.shields.io/badge/target-MemTorch%20simulation-lightgrey)](https://memtorch.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## TL;DR

`cim_compile` is a small local compiler that turns a supported ONNX tiny-decoder slice into a verified compute-in-memory simulation package. It extracts the first decoder layer from `data/model.onnx`, lowers CiM-friendly attention and MLP projections into a strict `cim` dialect, and emits the files needed to run those projection layers through MemTorch while digital PyTorch handles embedding lookup, RMSNorm, rotary math, attention score/softmax/context kernels, residual glue, and the final LM-head logits projection. On this MacBook Air, the CPU MemTorch path has been installed and validated for token-ID-to-logits inference on the supported fixture path; CUDA validation belongs on a CUDA machine. The project intentionally supports a narrow slice first, so failures are explicit and the compiler can be tested deeply instead of pretending to handle every ONNX graph.

## Overview

### The Problem

Memristive compute-in-memory systems represent matrix weights as device conductances inside crossbar arrays. That makes the important compiler problem different from normal CPU code generation: the compiler must decide how matrices become crossbar tiles, preserve that schedule, and hand the result to a simulator that understands memristive behavior.

### What This Tool Does

1. **Frontend** — Reads a small ONNX/prost slice and extracts supported projection and digital fallback weights from the first named tiny decoder block.
2. **Normalization** — Converts the supported graph pattern into a local `NormalizedProgram` with explicit attention, MLP, and token-logits metadata.
3. **`cim` dialect** — Lowers CiM-friendly projections into verified `cim.tile.dispatch` operations with explicit tile and scheduling attributes.
4. **MemTorch package** — Writes a manifest, execution plan, and quantized tile payloads.
5. **Simulation run** — Optionally invokes the checked-in Python MemTorch bridge with `--run-memtorch` when Torch and MemTorch are installed in the selected Python environment.

## Key Features

- **One clean target** — The active path is `ONNX -> normalized IR -> cim dialect -> MemTorch package`; the old RV32I hardware backend is no longer part of the compiler.
- **Transformer-aware offload** — The supported decoder slice is normalized explicitly, with CiM assigned to static attention/MLP projections and digital fallback recorded for dynamic kernels.
- **Real tiny-model logits path** — The required `data/model.onnx` fixture compiles as one hybrid decoder layer with 192-wide hidden state, 1024-wide MLP intermediate state, 32k vocabulary logits, and grouped-query attention metadata.
- **Strict dialect verifier** — Rejects invalid tile sizes, out-of-bounds tiles, missing or duplicate orders, inconsistent schedules, bad scales, duplicate coverage, and offset mistakes while allowing padded edge tiles.
- **Stable text IR** — `output.cim` has an MLIR-like textual form with parser/printer round-trip tests.
- **MemTorch-oriented artifacts** — Emits `memtorch_manifest.json`, `memtorch_weights.bin`, optional `memtorch_digital.bin`, and an execution plan that records CiM versus digital placement.
- **Narrow ONNX support** — Accepts the real tiny-model token-ID logits slice, bundled MHA-style projection fixtures, single rank-2 float projection initializers, and linear `MatMul`/`Gemm` nodes with initializer weights.
- **Clear unsupported-op diagnostics** — Unsupported ONNX ops fail with the node name and supported-op list.
- **Tested path** — `cargo test` covers the required real ONNX fixture and runs the MemTorch-backed token-ID-to-logits simulation.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Rust 2024 |
| CLI | `clap` 4 derive API |
| Serialization | `serde` + `serde_json` |
| ONNX protobuf ingestion | `prost` 0.14 |
| Protobuf code generation | `prost-build` + `protoc-bin-vendored` |
| Simulation target | MemTorch via checked-in Python bridge + PyTorch model reconstruction |
| Package manager | Cargo; uv for optional Python simulation dependencies |

## Project Structure

```text
cim_compile/
├── src/
│   ├── lib.rs              ← Public `compile_onnx` API
│   ├── main.rs             ← Thin CLI wrapper
│   ├── frontend.rs         ← Narrow ONNX parser and normalizer
│   ├── ir.rs               ← Normalized internal IR
│   ├── cim.rs              ← Dialect data model, verifier, parser, printer
│   ├── lowering.rs         ← Tiling, schedule order, quantization, tile payloads
│   └── memtorch.rs         ← Manifest, CiM tile bytes, and digital tensor bytes
├── python/
│   └── cim_compile_memtorch/ ← Python MemTorch runtime bridge
├── tests/
│   ├── cim.rs              ← Dialect parser/printer/verifier tests
│   ├── cli.rs              ← End-to-end CLI artifact tests
│   ├── golden.rs           ← Exact tiny-projection golden outputs
│   ├── full.rs             ← Full test for data/model.onnx + MemTorch
│   └── generate_onnx_fixtures.py ← Runtime ONNX fixture generator
├── proto/                  ← Minimal ONNX protobuf schema
├── build.rs                ← Protobuf codegen setup
└── LINKS.md                ← Reference links
```

## Installation

Prerequisites for compiling artifacts:

- Rust stable with edition 2024 support
- Cargo

```bash
git clone <repo-url>
cd cim_compile
cargo build --release
cargo test
```

Python prerequisites for `cargo test`, full fixture generation, and `--run-memtorch`:

- Python 3 in `.venv` or another environment
- PyTorch
- ONNX and `onnxscript` for Torch export
- MemTorch CPU on this laptop, or CUDA MemTorch on a CUDA machine

MemTorch is documented at [memtorch.readthedocs.io](https://memtorch.readthedocs.io/en/latest/). The Rust test suite requires `data/model.onnx` for the real tiny-model fixture, generates minimal ONNX protobuf fixtures with the Python standard library, and runs the MemTorch bridge through `.venv/bin/python` when present. `--run-memtorch` requires Torch and MemTorch.

The local CPU simulation environment used for validation was installed with:

```bash
uv pip install torch onnx onnxscript
uv pip install --no-deps --no-build-isolation memtorch-cpu
uv pip install pandas scipy scikit-learn torchvision matplotlib seaborn ipython lmfit
```

That environment currently imports `torch 2.12.1`, `onnx 1.22.0`, `onnxscript 0.7.0`, and `memtorch 1.1.6-cpu`. `uv pip check` still reports MemTorch's historical `sklearn` package alias as missing; runtime uses `scikit-learn` and the full MemTorch test passes.

## Quick Start

**Compile the required real tiny-model token-logits slice**

```bash
cargo run --release -- data/model.onnx -o out --tile-size 128
```

This extracts the first decoder layer, maps Q/K/V/output and MLP gate/up/down projections to CiM tiles, packages embedding/norm/LM-head tensors for digital fallback, and records score matmul, masking, softmax, grouped-query repeat, context matmul, MLP activation, residuals, and logits projection as digital stages. The default `128 x 128` tile size produces padded edge tiles for the model's `192 x 192`, `96 x 192`, `1024 x 192`, and `192 x 1024` projection matrices. You can also simulate with larger crossbars, for example `--tile-size 192`, when that is the experiment you want to show.

**Run token-ID-to-logits inference through the MemTorch bridge**

```bash
cargo run --release -- data/model.onnx -o out --tile-size 128 --run-memtorch --input-ids 1,2,3,4 --top-k 5
```

The JSON result reports `logits_shape: [1, 4, 32000]` and `next_token_topk` for the last input token.

**Run greedy token-ID generation before tokenizer support**

```bash
cargo run --release -- data/model.onnx -o out --tile-size 128 --run-memtorch --generate-ids --input-ids 1,2,3,4 --max-new-tokens 8 --top-k 5
```

The JSON result reports `generated_ids`, `new_token_ids`, `per_step_topk`, `cache_shapes`, and a `simulation_summary`. This is still not text generation: there is no tokenizer, text prompt handling, non-greedy sampling UI, external KV-cache API, or arbitrary ONNX LLM support in scope.

**Compile the bundled unrolled projection fixture**

```bash
python3 tests/generate_onnx_fixtures.py --output-dir /tmp/cim-fixtures --dim 512
cargo run --release -- /tmp/cim-fixtures/memristor_mha_unrolled.onnx -o out
```

**Compile the fused QKV fixture**

```bash
cargo run --release -- /tmp/cim-fixtures/mha_bfloat16.onnx -o out-fused
```

**Use a smaller simulated crossbar tile**

```bash
python3 tests/generate_onnx_fixtures.py --output-dir /tmp/cim-fixtures-64 --dim 64
cargo run --release -- /tmp/cim-fixtures-64/memristor_mha_unrolled.onnx -o out-64 --tile-size 64
```

**Run the MemTorch bridge**

```bash
cargo run --release -- /tmp/cim-fixtures/memristor_mha_unrolled.onnx -o out --run-memtorch
```

If Python cannot import Torch and MemTorch, the command still writes the compiler artifacts and then reports the missing simulation dependency.

## CLI / API Reference

```text
cim_compile [OPTIONS] <ONNX_PATH>

Arguments:
  <ONNX_PATH>              Path to the ONNX model file

Options:
  -o, --output-dir <DIR>   Output directory for output.cim and MemTorch artifacts [default: .]
      --tile-size <N>      Square crossbar tile dimension [default: 128]
      --bits <N>           Quantization bit-width for tile payloads: 4 or 8 [default: 8]
      --run-memtorch       Run the MemTorch bridge after writing artifacts
      --python <PYTHON>    Python executable to use with --run-memtorch
      --input-ids <CSV>    Comma-separated token IDs for --run-memtorch [default: 1,2,3,4]
      --top-k <N>          Number of next-token candidates to report [default: 5]
      --generate-ids       Generate token IDs greedily with the MemTorch bridge
      --max-new-tokens <N> Maximum number of token IDs to generate [default: 8]
      --eos-token-id <ID>  Optional token ID that stops generation early
  -h, --help               Print help
  -V, --version            Print version
```

Library entry point:

```rust
let compilation = cim_compile::compile_onnx(path, cim_compile::CompileConfig::square(128, 8))?;
```

`Compilation` contains the normalized IR, verified `cim::Program`, and MemTorch package bytes/text.

## Core Algorithm / How It Works

The compiler treats each supported projection matrix as a sheet of weights that must be cut into crossbar-sized tiles. Each tile gets a deterministic schedule order and byte offset. Edge tiles are zero-padded to the selected physical crossbar shape. The checked-in Python MemTorch bridge reconstructs PyTorch `Linear` layers from those tile bytes, asks MemTorch to patch them into a memristive simulation model, and runs token-ID logits or greedy token-ID generation for the supported tiny decoder slice. During generation, the bridge maintains a one-layer digital KV cache while continuing to call MemTorch-patched attention and MLP projections.

Technical flow:

```text
ONNX graph/initializers
  -> NormalizedProgram
  -> verified cim::Program
  -> MemTorch manifest + quantized tile bytes + digital tensor bytes
```

Example `cim` operation:

```text
cim.tile.dispatch { projection = "wq", tile = [0, 0], matrix_shape = [192, 192], tile_size = [128, 128], weight_offset = 0, quant_scale = 0.006597564, order = 0 }
```

The offset is into `memtorch_weights.bin`, which stores tile payloads in dispatch order with no legacy hardware header.

## Configuration

There is no config file. The compiler is configured through CLI flags or `CompileConfig`.

| Option | Meaning |
|---|---|
| `--tile-size <N>` | Uses `N x N` physical crossbar tiles; edge tiles are zero-padded when the matrix shape is not evenly divisible. |
| `--bits <N>` | Quantizes float weights to signed 4-bit or 8-bit ranges, stored as one byte per weight. |
| `--output-dir <DIR>` | Writes generated artifacts into the selected directory. |
| `--run-memtorch` | Runs `python -m cim_compile_memtorch.runner` after compiling. |
| `--python <PYTHON>` | Selects the Python executable for `--run-memtorch`; if omitted, the CLI uses `.venv/bin/python` when present, then falls back to `python3`. The CLI sets `PYTHONPATH` to include the repo's `python/` directory. |
| `--input-ids <CSV>` | Token IDs to feed into the supported logits simulation when `--run-memtorch` is set. |
| `--top-k <N>` | Number of last-token candidates to report from the logits tensor. |
| `--generate-ids` | Runs greedy token-ID generation instead of only reporting prompt logits. |
| `--max-new-tokens <N>` | Maximum number of new IDs to generate in greedy mode. |
| `--eos-token-id <ID>` | Optional generated ID that stops greedy generation early. |

## Output Reference

### `output.cim`

Stable text form of the verified `cim` dialect. This is intended for inspection and parser/verifier round-trip tests.

### `memtorch_manifest.json`

JSON manifest consumed by the checked-in MemTorch bridge.

| Field | Meaning |
|---|---|
| `schema_version` | Manifest schema version, currently `1`. |
| `entry` | Normalized program name. |
| `tile_size` | `[rows, cols]` tile shape used for all dispatches. |
| `quant_bits` | Quantization bit-width requested at compile time. |
| `weights_file` | Relative path to `memtorch_weights.bin`. |
| `digital_tensors_file` | Relative path to `memtorch_digital.bin` when the supported token-logits slice needs digital tensors. |
| `projections` | Projection metadata, bias values if present, and tile records. |
| `digital_tensors` | Float tensor metadata for embedding, norm, and LM-head fallback payloads. |
| `inference_slice` | Narrow supported slice metadata, including token-logits mode and explicit unsupported features. |
| `simulation_summary` | Runtime placement summary: supported modes, MemTorch stages, digital stages, patched projection count, and LM-head placement. |

Each tile record includes `row`, `col`, `matrix_shape`, `tile_size`, `weight_offset`, `quant_scale`, and `order`. Attention manifests also include per-block metadata such as hidden size, Q/KV dimensions, head counts, and whether grouped-query attention is present.

### `memtorch_weights.bin`

Raw signed int8 tile payloads in dispatch order. A tile with order `k` starts at:

```text
k * tile_rows * tile_cols
```

### `memtorch_digital.bin`

Float32 little-endian tensor payloads for digital fallback. The manifest records each tensor role, shape, byte offset, byte length, and dtype.

### Python MemTorch Bridge

`python/cim_compile_memtorch/runner.py` reconstructs PyTorch `Linear` layers from the manifest and weight payloads, loads digital tensors from `memtorch_digital.bin` when present, then uses MemTorch's `patch_model` path when the Python environment has the required packages. The compiler does not emit Python code into the output directory.

## Testing & Validation

Run the default suite:

```bash
cargo test
```

Coverage includes runtime-generated ONNX fixture ingestion, real-model tiny-decoder extraction, normalized lowering, bfloat16/f32 quantization, schedule generation, offset validation, `cim` parser/printer round trips, verifier failures, CLI success and failure cases, unavailable Python bridge diagnostics, MemTorch-backed token-logits and token-ID generation simulations, and exact golden outputs for a tiny projection.

The full MemTorch test requires Torch, MemTorch, and `data/model.onnx`. In the local uv environment, it exercises the real ONNX fixture plus the checked-in MemTorch token-logits/generation bridge:

```bash
cargo test --test full
```

Note: https://huggingface.co/onnx-community/Tiny-LLM-ONNX/resolve/main/onnx/model.onnx is being used for testing.

## Documentation Index

| Document | Contents |
|---|---|
| [LINKS.md](LINKS.md) | Reference links for ONNX, MLIR-style IRs, MemTorch, CiM context, and compiler architecture. |
| [NOTES.md](NOTES.md) | Current milestone state, design decisions, local Python environment, and test counts. |

## License

Released under the [MIT License](LICENSE).
