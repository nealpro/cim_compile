# cim-compile

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Language](https://img.shields.io/badge/language-Rust-orange)
![ISA](https://img.shields.io/badge/target-RV32I-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## TL;DR

`cim-compile` is a heterogeneous compiler written in Rust that ingests a multi-head attention ONNX model and simultaneously produces two hardware artifacts: RISC-V assembly for CPU orchestration and a binary weight map for a simulated analog memristor crossbar. Standard ML compilers target homogeneous CPU/GPU backends and have no model for analog in-memory compute; this compiler bridges that gap with a multi-level IR, a hardware-constrained tiling pass, and a dual code-generator that keeps the CPU program and the crossbar weight layout in sync. It is a personal project exploring compiler design for emerging compute-in-memory (CiM) hardware.

---

## Overview

### The Problem

Analog Compute-In-Memory (CiM) hardware executes vector-matrix multiplication (VMM) in a single physical step — a weight matrix is encoded as conductance values on a crossbar, and applying an input voltage across the rows produces output currents proportional to the product. This is fundamentally different from a CPU or GPU: the weights live in the hardware itself, and the CPU acts only as an orchestrator dispatching tiles and accumulating partial sums. No existing open compiler toolchain directly targets this execution model for real ONNX models.

### What This Tool Does

1. **Frontend** — Parses an ONNX protobuf file (via prost) and lifts the model into a `HighLevelOp` IR. Extracts four 512×512 bfloat16 projection matrices (W_Q, W_K, W_V, W_O) from the external `.data` file.
2. **Middle-end (Tiling Pass)** — Lowers `HighLevelOp` → `LowLevelOp`. Partitions each 512×512 matrix into a 4×4 grid of 128×128 tiles to match the physical crossbar constraint, yielding 64 `ProjectionTile` operations in dispatch order.
3. **Backend: Assembly Generator** — Walks the `LowLevelOp` list and emits a 10-instruction RV32I loop. Dispatches each tile index to the crossbar via MMIO, reads the ADC-digitized partial sum, and accumulates it in a software buffer.
4. **Backend: Peephole Optimizer** — Applies a pattern-matching pass on the typed instruction AST, folding `Andi + Srli + Slli` sequences into `Andi + net-shift` (3 instructions → 2) without operating on strings.
5. **Backend: Weight Serializer** — Writes a structured binary file (`crossbar_weights.bin`) with a 24-byte header and 64 tiles in the same dispatch order as the assembly loop, ensuring the CPU program and weight map are always consistent.

---

## Key Features

- **Two-level IR** — `HighLevelOp` (abstract operation, e.g. `MultiHeadAttention`) lowered to `LowLevelOp` (`ProjectionTile`), mirroring the frontend/middle-end split in production compilers like MLIR.
- **Hardware-constrained tiling** — The 128×128 crossbar limit drives the partitioning math: four column tiles contribute to each 128-element output slice of a 512×512 projection. Tile size is configurable at the CLI.
- **Typed instruction AST** — `Reg`, `Imm`, and `Instr` enums with `Display` and constructor functions mirror the Cranelift/LLVM instruction builder pattern. The optimizer and renderer operate on structured data, not text.
- **Peephole optimizer** — Pattern-matches a three-instruction shift sequence and collapses it into two instructions using the identity: `new_mask = mask & !((1 << srli_amt) - 1); net_shift = slli_amt − srli_amt`.
- **Coordinated dual output** — Assembly and binary weight file are generated from the same `Vec<LowLevelOp>` in a single pass, guaranteeing tile-order consistency between the CPU loop and the weight map.
- **bfloat16 throughout** — Matches the dynamic range of float32 with half the bandwidth; standard for CiM hardware due to analog noise tolerance.
- **Custom binary format** — `CiMW` magic header, versioned schema, per-tile projection/row/col metadata, row-major bfloat16 weight payload. ~2 MB for 64 tiles.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Rust (edition 2024) |
| ONNX / protobuf ingestion | [prost](https://github.com/tokio-rs/prost) 0.14 |
| CLI argument parsing | [clap](https://github.com/clap-rs/clap) 4 (derive API) |
| Build-time protobuf codegen | prost-build + protoc-bin-vendored |
| ONNX model generation | Python + PyTorch (`data/onnx_file.py`) |

---

## Project Structure

```
cim_compile/
├── src/
│   ├── main.rs              ← CLI entry point (clap); wires all three stages
│   ├── frontend.rs          ← ONNX protobuf → HighLevelOp IR
│   ├── middle.rs            ← Tiling pass: HighLevelOp → Vec<LowLevelOp>
│   ├── hardware.rs          ← CrossbarSpec { tile_rows, tile_cols }
│   ├── backend.rs           ← RV32I codegen + binary weight serializer
│   └── backend/
│       └── optimizer.rs     ← Peephole pass: Andi+Srli+Slli → Andi+net-shift
├── data/
│   ├── onnx_file.py         ← Generates memristor_mha_unrolled.onnx via PyTorch
│   ├── memristor_mha_unrolled.onnx       ← ONNX model (architecture)
│   └── memristor_mha_unrolled.onnx.data  ← External weight tensor data
├── build.rs                 ← prost-build invocation for ONNX proto codegen
├── NOTES.md                 ← Design decisions and milestone log
└── LINKS.md                 ← Reference links (specs, papers)
```

---

## Installation

**Prerequisites**

- Rust stable (1.80+): [rustup.rs](https://rustup.rs)
- Python 3.9+ with PyTorch (only needed to regenerate the ONNX model):
  ```bash
  pip install torch onnx
  ```

**Build**

```bash
git clone <repo>
cd cim_compile
cargo build --release
```

---

## Quick Start

```bash
# 1. Generate the ONNX model (skip if data/memristor_mha_unrolled.onnx already exists)
python data/onnx_file.py

# 2. Compile: ONNX → output.s + crossbar_weights.bin
cargo run --release -- data/memristor_mha_unrolled.onnx
```

Expected output:
```
wrote ./output.s + ./crossbar_weights.bin (64 tiles)
```

```bash
# Write artifacts to a specific directory
cargo run --release -- data/memristor_mha_unrolled.onnx -o out/

# Use a different tile size (must evenly divide embed_dim=512)
cargo run --release -- data/memristor_mha_unrolled.onnx --tile-size 64
```

---

## CLI Reference

```
cim_compile <ONNX_PATH> [OPTIONS]

Arguments:
  <ONNX_PATH>          Path to the ONNX model file

Options:
  -o, --output-dir <DIR>      Output directory for output.s and crossbar_weights.bin
                              [default: .]
      --tile-size <N>         Crossbar tile dimension; must evenly divide embed_dim
                              [default: 128]
  -h, --help                  Print help
  -V, --version               Print version
```

---

## How It Works

### Analog VMM and why tiling is necessary

A memristor crossbar physically implements vector-matrix multiplication: weight values are stored as conductance levels in a grid of cells, and when an input voltage vector is applied across the rows, Ohm's law (I = V × G) drives a current through each cell. Kirchhoff's current law sums the column currents, producing the full matrix-vector product in a single analog step — no multiply-accumulate loop, no memory bandwidth.

The catch is that the physical crossbar is finite. This model has 128×128 cells. A 512×512 projection matrix (W_Q, W_K, W_V, W_O in a multi-head attention layer) is 16× too large to map onto it directly. The solution is tiling: the middle-end pass partitions each 512×512 matrix into a 4×4 grid of 128×128 sub-matrices. The CPU dispatches each tile index to the crossbar controller via MMIO, reads the ADC-digitized partial sum on each return, and accumulates four column-tile contributions to reconstruct each 128-element output slice.

### IR lowering

```
HighLevelOp::MultiHeadAttention { embed_dim: 512, heads: 8, bfloat16 }
  │  middle::build_plan  → 64 (projection, row_tile, col_tile) tuples
  │  middle::slice_tiles → extract_tile() copies strided bfloat16 rows
  ▼
LowLevelOp::ProjectionTile { projection: WQ|WK|WV|WO, row, col, weights: Vec<u8> }
```

`build_plan` enumerates all `(proj, row, col)` combinations: 4 projections × 4 row-tiles × 4 col-tiles = 64 tiles. `extract_tile` performs a strided copy of `tile_size` rows × `tile_size` columns × 2 bytes each from the row-major weight tensor.

### RISC-V code generation and peephole optimization

The backend builds a typed instruction AST (`Vec<Instr>`) using constructor functions, then runs a peephole pass before rendering to text. The key optimization folds this 3-instruction sequence:

```asm
andi  t4, t0, 0xf    # isolate column-tile bits
srli  t4, t4, 2      # right-shift
slli  t4, t4, 8      # left-shift to byte offset
```

into 2 instructions:

```asm
andi  t4, t0, 0xc    # new_mask = 0xf & ~((1<<2)-1) = 0xc
slli  t4, t4, 6      # net shift = 8 - 2 = 6
```

The general rule: `new_mask = mask & !((1 << srli_amt) - 1)`, `net_shift = slli_amt − srli_amt`. The optimizer handles net left-shift, net right-shift, and equal shifts (mask-only, no shift emitted).

**MMIO map:**

| Address | Purpose |
|---|---|
| `0x10000000` | VMM control: write tile index, read ADC partial sum |
| `0x20000000` | Accumulator buffer base |

Target simulators: Spike, QEMU (RV32I).

---

## Output Reference

### `output.s` — RV32I orchestration loop

```asm
.loop:
    sw   t0, 0x0(t2)      # dispatch tile index → crossbar MMIO
    lw   a0, 0x4(t2)      # read ADC partial sum
    andi t4, t0, 0xc      # extract column-tile bits (after peephole)
    slli t4, t4, 6        # byte offset into accumulator buffer (net shift 8-2=6)
    add  t5, t3, t4       # accumulator slot address
    lw   a1, 0x0(t5)      # load running sum
    add  a1, a1, a0       # accumulate partial sum
    sw   a1, 0x0(t5)      # store updated sum
    addi t0, t0, 1        # advance tile counter
    blt  t0, t1, .loop    # loop until all 64 tiles dispatched
```

### `crossbar_weights.bin` — binary weight map

| Offset | Size | Field |
|---|---|---|
| 0 | 4 B | Magic: `CiMW` |
| 4 | 4 B | Version: `1` (u32 LE) |
| 8 | 4 B | dtype: `16` (bfloat16; matches ONNX `data_type` enum) |
| 12 | 4 B | tile_rows: `128` |
| 16 | 4 B | tile_cols: `128` |
| 20 | 4 B | num_tiles: `64` |
| **per tile × 64** | | |
| +0 | 1 B | projection: `0`=W_Q, `1`=W_K, `2`=W_V, `3`=W_O |
| +1 | 1 B | row tile index |
| +2 | 1 B | col tile index |
| +3 | 1 B | pad |
| +4 | 32 768 B | bfloat16 weights, row-major |

Total: `24 + 64 × 32772 = 2 097 432 bytes (~2 MB)`. Tiles appear in dispatch order matching the assembly loop.

**Example row (tile 0):**

| projection | row | col | weight bytes |
|---|---|---|---|
| `0` (W_Q) | `0` | `0` | 32 768 B of bfloat16 values, offset +28 in file |

---

## Documentation Index

| Document | Contents |
|---|---|
| [NOTES.md](NOTES.md) | Design decisions, milestone log, constraint table |
| [LINKS.md](LINKS.md) | Reference links: RISC-V ISA spec, MLIR Toy tutorial, CiM literature |

---

## License

Released under the [MIT License](LICENSE).

---

*Built with Rust. Targets RV32I + analog memristor crossbar hardware.*
