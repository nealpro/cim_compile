# cim_compile

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Rust](https://img.shields.io/badge/rust-2024-orange)
![Target](https://img.shields.io/badge/target-MemTorch%20simulation-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## TL;DR

`cim_compile` is a small local compiler that turns a supported ONNX model into a verified compute-in-memory simulation package. It reads projection-style weights, lowers them into a strict `cim` dialect, and emits the files needed to run those tiles through MemTorch on a normal laptop. The project intentionally supports a narrow slice first, so failures are explicit and the compiler can be tested deeply instead of pretending to handle every ONNX graph.

## Overview

### The Problem

Memristive compute-in-memory systems represent matrix weights as device conductances inside crossbar arrays. That makes the important compiler problem different from normal CPU code generation: the compiler must decide how matrices become crossbar tiles, preserve that schedule, and hand the result to a simulator that understands memristive behavior.

### What This Tool Does

1. **Frontend** — Reads a small ONNX/prost slice and extracts supported projection, linear, and attention-projection weights.
2. **Normalization** — Converts supported graph patterns into a local `NormalizedProgram` with projection ops and structural reshape/transpose markers.
3. **`cim` dialect** — Lowers projections into verified `cim.tile.dispatch` operations with explicit tile and scheduling attributes.
4. **MemTorch package** — Writes a manifest, quantized tile payloads, and a Python runner that reconstructs a PyTorch model and calls MemTorch patching.
5. **Simulation run** — Optionally invokes the generated runner with `--run-memtorch` when Torch and MemTorch are installed.

## Key Features

- **One clean target** — The active path is `ONNX -> normalized IR -> cim dialect -> MemTorch package`; the old RV32I hardware backend is no longer part of the compiler.
- **Strict dialect verifier** — Rejects invalid tile sizes, non-divisible shapes, out-of-bounds tiles, missing or duplicate orders, inconsistent schedules, bad scales, duplicate coverage, and offset mistakes.
- **Stable text IR** — `output.cim` has an MLIR-like textual form with parser/printer round-trip tests.
- **MemTorch-oriented artifacts** — Emits `memtorch_manifest.json`, `memtorch_weights.bin`, and `run_memtorch.py`.
- **Narrow ONNX support** — Accepts the bundled MHA-style projection fixtures, single rank-2 float projection initializers, and linear `MatMul`/`Gemm` nodes with initializer weights.
- **Clear unsupported-op diagnostics** — Unsupported ONNX ops fail with the node name and supported-op list.
- **Tested path** — `cargo test` currently passes 30 / 30 tests across unit, dialect, CLI, and golden-output coverage.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Rust 2024 |
| CLI | `clap` 4 derive API |
| Serialization | `serde` + `serde_json` |
| ONNX protobuf ingestion | `prost` 0.14 |
| Protobuf code generation | `prost-build` + `protoc-bin-vendored` |
| Simulation target | MemTorch via generated Python + PyTorch model reconstruction |
| Package manager | Cargo |

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
│   └── memtorch.rs         ← Manifest, weight bytes, generated Python runner
├── tests/
│   ├── cim.rs              ← Dialect parser/printer/verifier tests
│   ├── cli.rs              ← End-to-end CLI artifact tests
│   ├── golden.rs           ← Exact tiny-projection golden outputs
│   ├── full.rs             ← Ignored full tests for Torch-generated ONNX
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

Optional prerequisites for full fixture generation or actually running the generated MemTorch simulation:

- Python 3
- PyTorch
- ONNX Python package for Torch export
- MemTorch

MemTorch is documented at [memtorch.readthedocs.io](https://memtorch.readthedocs.io/en/latest/). The default Rust test suite generates minimal ONNX protobuf fixtures with the Python standard library and does not require Torch, ONNX, or MemTorch to be installed; full tests and `--run-memtorch` do.

## Quick Start

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

**Run the generated MemTorch path**

```bash
cargo run --release -- /tmp/cim-fixtures/memristor_mha_unrolled.onnx -o out --run-memtorch
```

If Python cannot import Torch and MemTorch, the command still writes the compiler artifacts and then reports the missing simulation dependency.

## CLI / API Reference

```text
cim_compile <ONNX_PATH> [OPTIONS]

Arguments:
  <ONNX_PATH>              Path to the ONNX model file

Options:
  -o, --output-dir <DIR>   Output directory for output.cim and MemTorch artifacts [default: .]
      --tile-size <N>      Square crossbar tile dimension [default: 128]
      --bits <N>           Quantization bit-width for tile payloads: 4 or 8 [default: 8]
      --run-memtorch       Run the generated MemTorch script after writing artifacts
      --python <PYTHON>    Python executable to use with --run-memtorch [default: python3]
  -h, --help               Print help
  -V, --version            Print version
```

Library entry point:

```rust
let compilation = cim_compile::compile_onnx(path, cim_compile::CompileConfig::square(128, 8))?;
```

`Compilation` contains the normalized IR, verified `cim::Program`, and MemTorch package bytes/text.

## Core Algorithm / How It Works

The compiler treats each supported projection matrix as a sheet of weights that must be cut into crossbar-sized tiles. Each tile gets a deterministic schedule order and byte offset. The generated MemTorch runner reconstructs a PyTorch `Linear` layer from those tile bytes, then asks MemTorch to patch the layer into a memristive simulation model.

Technical flow:

```text
ONNX graph/initializers
  -> NormalizedProgram
  -> verified cim::Program
  -> MemTorch manifest + quantized tile bytes + Python runner
```

Example `cim` operation:

```text
cim.tile.dispatch { projection = "wq", tile = [0, 0], matrix_shape = [512, 512], tile_size = [128, 128], weight_offset = 0, quant_scale = 0.007874016, order = 0 }
```

The offset is into `memtorch_weights.bin`, which stores tile payloads in dispatch order with no legacy hardware header.

## Configuration

There is no config file. The compiler is configured through CLI flags or `CompileConfig`.

| Option | Meaning |
|---|---|
| `--tile-size <N>` | Uses `N x N` tiles; the value must evenly divide each lowered projection matrix. |
| `--bits <N>` | Quantizes float weights to signed 4-bit or 8-bit ranges, stored as one byte per weight. |
| `--output-dir <DIR>` | Writes generated artifacts into the selected directory. |
| `--run-memtorch` | Runs the generated Python script after compiling. |
| `--python <PYTHON>` | Selects the Python executable for `--run-memtorch`. |

## Output Reference

### `output.cim`

Stable text form of the verified `cim` dialect. This is intended for inspection and parser/verifier round-trip tests.

### `memtorch_manifest.json`

JSON manifest consumed by `run_memtorch.py`.

| Field | Meaning |
|---|---|
| `schema_version` | Manifest schema version, currently `1`. |
| `entry` | Normalized program name. |
| `tile_size` | `[rows, cols]` tile shape used for all dispatches. |
| `quant_bits` | Quantization bit-width requested at compile time. |
| `weights_file` | Relative path to `memtorch_weights.bin`. |
| `projections` | Projection metadata, bias values if present, and tile records. |

Each tile record includes `row`, `col`, `matrix_shape`, `tile_size`, `weight_offset`, `quant_scale`, and `order`.

### `memtorch_weights.bin`

Raw signed int8 tile payloads in dispatch order. A tile with order `k` starts at:

```text
k * tile_rows * tile_cols
```

### `run_memtorch.py`

Generated Python runner. It reconstructs PyTorch `Linear` layers from the manifest and weight payloads, then uses MemTorch’s `patch_model` path when the Python environment has the required packages.

## Testing & Validation

The current suite passes 30 / 30 tests:

```bash
cargo test
```

Coverage includes runtime-generated ONNX fixture ingestion, normalized lowering, bfloat16/f32 quantization, schedule generation, offset validation, `cim` parser/printer round trips, verifier failures, CLI success and failure cases, missing MemTorch environment diagnostics, and exact golden outputs for a tiny projection.

Full fixture tests are opt-in because they require Torch and ONNX:

```bash
CIM_COMPILE_FULL_TESTS=1 cargo test --test full -- --ignored
```

## Documentation Index

| Document | Contents |
|---|---|
| [LINKS.md](LINKS.md) | Reference links for ONNX, MLIR-style IRs, MemTorch, CiM context, and compiler architecture. |

## License

Released under the [MIT License](LICENSE).
