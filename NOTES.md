# cim-compile Notes

## Current State (2026-06-12)

### All milestones complete
M6: README.md written and refreshed for quantized v2 weight output plus both bundled ONNX fixture layouts.

### Pipeline status
- `src/frontend.rs` — `parse_onnx` decodes ONNX protobuf, reads external `.data` tensors via prost, and supports both unrolled 4 × [512×512] projection initializers and fused PyTorch `[1536,512] in_proj_weight` + `[512,512] out_proj.weight`
- `src/middle.rs` — `tile()` slices each 512×512 matrix into 4×4 grid of 128×128 tiles → 64 `ProjectionTile` ops; `quantize()` lowers bfloat16 tile payloads to signed int8 bytes by default with a per-tile scale
- `src/backend.rs` — `emit_asm` / `write_asm` emit `output.s`; `write_weights` emits v2 `crossbar_weights.bin` using the active `CrossbarSpec` tile dimensions
- `src/backend/optimizer.rs` — `peephole` pass: folds `Andi + Srli + Slli` into `Andi + net-shift` (3 → 2 instructions)
- `src/main.rs` — calls all three stages, validates quantization bits, creates the output directory, and reports readable errors

### Output artifacts
- `output.s` — RV32I assembly, 11-instruction loop over 64 tiles, MMIO dispatch + partial-sum accumulation
- `crossbar_weights.bin` — ~1MB for the default 128×128 int8 path, magic `CiMW` + 24-byte v2 header, 64 tiles in dispatch order (matching loop)
- Regression coverage: `cargo test` currently runs 7 unit tests and 2 CLI integration tests covering both bundled ONNX models.

---

## Design Decisions

### Frontend: ONNX only
Originally had a JSON frontend (`serde` + `serde_json`, `model.json`, `parse_model`). Superseded by ONNX/prost ingestion. `serde` and `serde_json` dependencies removed. JSON schema was internally-tagged enum dispatch; no longer relevant.

### HighLevelOp: MultiHeadAttention
Single op type for the MVP. MHA chosen because:
- Per-head attention ops (QK^T, Attn×V) are naturally crossbar-shaped at seq_len=128, head_dim=64
- Projection matrices (512×512) require tiling, which exercises the middle-end pass

### BFLOAT16 frontend, quantized crossbar payload
ONNX weights are ingested as bfloat16. The middle-end now applies symmetric per-tile quantization to 4-bit or 8-bit signed integer values; the backend stores the per-tile scale as f32 so the simulated crossbar can reconstruct approximate weights.

### Hardware spec lives in `src/hardware.rs`
`CrossbarSpec` holds `tile_rows` and `tile_cols`. `CrossbarSpec::new(tile_size)` is passed into tiling and weight serialization, so the binary header reflects the selected tile size.

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
