# cim-compile Notes

## Current State (2026-05-24)

### All milestones complete
M6: README.md written — hardware model, pipeline diagram, quick start, output artifact format, source layout.

### Pipeline status
- `src/frontend.rs` — `parse_onnx` decodes ONNX protobuf, reads 4 × [512×512] bfloat16 weight tensors from external `.data` file via prost
- `src/middle.rs` — `tile()` slices each 512×512 matrix into 4×4 grid of 128×128 tiles → 64 `ProjectionTile` ops
- `src/backend.rs` — `emit_asm` / `write_asm` emit `output.s`; `write_weights` emits `crossbar_weights.bin`
- `src/backend/optimizer.rs` — `peephole` pass: folds `Andi + Srli + Slli` into `Andi + net-shift` (3 → 2 instructions)
- `src/main.rs` — calls all three stages, writes both output files

### Output artifacts
- `output.s` — RV32I assembly, 11-instruction loop over 64 tiles, MMIO dispatch + partial-sum accumulation
- `crossbar_weights.bin` — 2MB, magic `CiMW` + 24-byte header, 64 tiles in dispatch order (matching loop)

---

## Design Decisions

### Frontend: ONNX only
Originally had a JSON frontend (`serde` + `serde_json`, `model.json`, `parse_model`). Superseded by ONNX/prost ingestion. `serde` and `serde_json` dependencies removed. JSON schema was internally-tagged enum dispatch; no longer relevant.

### HighLevelOp: MultiHeadAttention
Single op type for the MVP. MHA chosen because:
- Per-head attention ops (QK^T, Attn×V) are naturally crossbar-shaped at seq_len=128, head_dim=64
- Projection matrices (512×512) require tiling, which exercises the middle-end pass

### BFLOAT16 only
16-bit format with 8-exponent bits — same dynamic range as float32, standard for CiM hardware.

### Hardware spec lives in `src/hardware.rs`
`CrossbarSpec` holds `tile_rows` and `tile_cols`. `CrossbarSpec::default_128x128()` passed into `tile()`. Crossbar size is one-place-to-change.

### Crossbar sizing (128×128 constraint)

| Operation          | Matrix shape          | Fits crossbar? |
|--------------------|-----------------------|----------------|
| W_Q/K/V/O proj     | 512×512               | no — needs 4×4 tiling |
| per-head QK^T      | [128×64] × [64×128]   | yes            |
| per-head Attn×V    | [128×128] × [128×64]  | yes            |

### Instruction AST in backend
`Reg`, `Imm`, `Instr` enums with `Display` + constructor functions mirror the Cranelift/LLVM builder pattern. Section functions (`prologue`, `loop_body`, `epilogue`) return `Vec<Instr>`; a single `render` pass produces text. Enables the peephole optimizer to operate on typed data rather than strings.

### Peephole: shift-fold
`andi T4, T0, 0xF + srli T4, T4, 2 + slli T4, T4, 8` → `andi T4, T0, 0xC + slli T4, T4, 6`.
General rule: new\_mask = mask & !((1 << srli\_amt) - 1); net shift = slli\_amt - srli\_amt.
Handles b > a (net slli), b < a (net srli), b == a (mask only). Implemented in `src/backend/optimizer.rs`.

---

## Milestones

| # | Done when | Status |
|---|-----------|--------|
| M1 | `model.json` → `Vec<HighLevelOp>` parses and prints | done |
| M2 | Tiling pass → `Vec<LowLevelOp>` for MHA projections | done |
| M3 | RISC-V assembly text file written for a simple loop | done |
| M4 | Binary weight file written for one 128×128 tile | done |
| M5 | CLI wires frontend → middle-end → backend end-to-end | done |
| M6 | Example `model.json` in repo, README explains the pipeline | done |
