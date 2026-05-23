# cim-compile Notes

## Current State (2026-05-20)

### Setup complete
- `Cargo.toml`: `serde` + `serde_json` added
- `model.json`: MultiHeadAttention example — 512-dim, 8 heads, seq_len=128, bfloat16
- `LINKS.md`: reference links organized by milestone
- `src/frontend.rs`: `parse_model` implemented with `serde_json::from_reader`
- `src/main.rs`: calls `parse_model`, prints result with `{:?}`

### Active milestone: M3
RISC-V assembly text file — write `output.s` with a loop that iterates over the 64 projection tiles.

### ONNX ingestion complete
`src/frontend.rs` now has `parse_onnx("data/memristor_mha_unrolled.onnx")` which:
- Decodes the protobuf with prost (field numbers match onnx.proto3 exactly)
- Finds the 4 [512×512] bfloat16 initializers (data_type=16) in graph order
- Reads their bytes from the external `.data` file via offset+length
- Populates `MHAWeights { wq, wk, wv, wo }` on `HighLevelOp::MultiHeadAttention`

`src/middle.rs` `tile()` now calls `extract_tile()` per ProjectionTile, slicing the
full 512×512 raw byte matrix into 128×128 chunks (32,768 bytes each, row-major bfloat16).

Infrastructure: `prost = "0.14.3"`, `prost-build`, `protoc-bin-vendored = "3"`,
`build.rs`, `proto/onnx_minimal.proto3`.

---

## Design Decisions

### HighLevelOp: MultiHeadAttention
Single op type for the MVP. MHA chosen because:
- Per-head attention ops (QK^T, Attn×V) are naturally crossbar-shaped at seq_len=128, head_dim=64
- Projection matrices (512×512) require tiling, which exercises the middle-end pass

### BFLOAT16 only
16-bit format with 8-exponent bits — same dynamic range as float32, standard for CiM hardware.
The compiler should hard-reject any other dtype at parse time.

### JSON schema (internally tagged)
```json
{ "type": "MultiHeadAttention", "num_heads": 8, "embed_dim": 512, "seq_len": 128, "dtype": "bfloat16" }
```
Uses serde's `#[serde(tag = "type")]` for enum dispatch. `head_dim` is derived: `embed_dim / num_heads = 64`.

### Hardware spec lives in `src/hardware.rs`
`CrossbarSpec` struct holds `tile_rows` and `tile_cols`. Instantiated with `CrossbarSpec::default_128x128()` and passed into `tile()` as `&CrossbarSpec`. Keeps the hardware constant out of model parameters — changing crossbar size means changing one place.

### Crossbar sizing (128×128 constraint)

| Operation          | Matrix shape          | Fits crossbar? |
|--------------------|-----------------------|----------------|
| W_Q/K/V/O proj     | 512×512               | no — needs 4×4 tiling |
| per-head QK^T      | [128×64] × [64×128]   | yes            |
| per-head Attn×V    | [128×128] × [128×64]  | yes            |

---

## Milestones

| # | Done when | Status |
|---|-----------|--------|
| M1 | `model.json` → `Vec<HighLevelOp>` parses and prints | done |
| M2 | Tiling pass → `Vec<LowLevelOp>` for MHA projections | done |
| M3 | RISC-V assembly text file written for a simple loop | in progress |
| M4 | Binary weight file written for one 128×128 tile | not started |
| M5 | CLI wires frontend → middle-end → backend end-to-end | not started |
| M6 | Example `model.json` in repo, README explains the pipeline | not started |
