use crate::frontend::{HighLevelOp, MHAWeights};
use crate::hardware::CrossbarSpec;

#[derive(Debug, Clone, Copy)]
pub enum Projection {
    WQ,
    WK,
    WV,
    WO,
}

pub enum LowLevelOp {
    ProjectionTile {
        projection: Projection,
        row: u32,
        col: u32,
        weights: Vec<u8>,
        /// Per-tile quantization scale factor. 1.0 means weights are unquantized bfloat16.
        scale: f32,
    },
}

impl std::fmt::Debug for LowLevelOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowLevelOp::ProjectionTile {
                projection,
                row,
                col,
                weights,
                scale,
            } => write!(
                f,
                "ProjectionTile {{ projection: {projection:?}, row: {row}, col: {col}, scale: {scale:.6}, weights: {}B }}",
                weights.len()
            ),
        }
    }
}

pub fn tile(ops: Vec<HighLevelOp>, spec: &CrossbarSpec) -> Result<Vec<LowLevelOp>, String> {
    let mut lowered = Vec::new();

    for op in ops {
        match op {
            HighLevelOp::MultiHeadAttention {
                embed_dim, weights, ..
            } => {
                let weights = weights.ok_or_else(|| {
                    "weights required; use parse_onnx before lowering".to_string()
                })?;
                let plan = build_plan(embed_dim, spec)?;
                lowered.extend(slice_tiles(
                    &plan,
                    &weights,
                    embed_dim as usize,
                    spec.tile_rows as usize,
                )?);
            }
        }
    }

    Ok(lowered)
}

fn build_plan(embed_dim: u32, spec: &CrossbarSpec) -> Result<Vec<(Projection, u32, u32)>, String> {
    if spec.tile_rows == 0 || spec.tile_cols == 0 {
        return Err("tile size must be greater than zero".to_string());
    }
    if embed_dim % spec.tile_rows != 0 || embed_dim % spec.tile_cols != 0 {
        return Err(format!(
            "tile size {}x{} must evenly divide embed_dim {embed_dim}",
            spec.tile_rows, spec.tile_cols
        ));
    }

    let row_tiles = embed_dim / spec.tile_rows;
    let col_tiles = embed_dim / spec.tile_cols;
    let mut plan = Vec::new();
    for proj in [
        Projection::WQ,
        Projection::WK,
        Projection::WV,
        Projection::WO,
    ] {
        for row in 0..row_tiles {
            for col in 0..col_tiles {
                plan.push((proj, row, col));
            }
        }
    }
    Ok(plan)
}

fn slice_tiles(
    plan: &[(Projection, u32, u32)],
    weights: &MHAWeights,
    full_cols: usize,
    tile_size: usize,
) -> Result<Vec<LowLevelOp>, String> {
    plan.iter()
        .map(|&(projection, row, col)| {
            let matrix = match projection {
                Projection::WQ => &weights.wq,
                Projection::WK => &weights.wk,
                Projection::WV => &weights.wv,
                Projection::WO => &weights.wo,
            };
            Ok(LowLevelOp::ProjectionTile {
                projection,
                row,
                col,
                weights: extract_tile(matrix, full_cols, row, col, tile_size)?,
                scale: 1.0,
            })
        })
        .collect()
}

// ── Quantization pass ────────────────────────────────────────────────────────

/// Lowers bfloat16 weights in each tile to `bits`-bit signed integers using
/// per-tile symmetric linear quantization, storing a scale factor alongside.
pub fn quantize(ops: Vec<LowLevelOp>, bits: u32) -> Result<Vec<LowLevelOp>, String> {
    validate_bits(bits)?;

    ops.into_iter()
        .map(|op| match op {
            LowLevelOp::ProjectionTile {
                projection,
                row,
                col,
                weights,
                ..
            } => {
                let (quantized, scale) = quantize_tile(&weights, bits)?;
                Ok(LowLevelOp::ProjectionTile {
                    projection,
                    row,
                    col,
                    weights: quantized,
                    scale,
                })
            }
        })
        .collect()
}

/// Quantize a bfloat16 tile (raw bytes, little-endian) to `bits`-bit signed integers.
/// Returns the quantized bytes (i8 cast to u8) and the per-tile scale factor.
fn quantize_tile(weights: &[u8], bits: u32) -> Result<(Vec<u8>, f32), String> {
    let qmax = validate_bits(bits)?;

    if weights.len() % 2 != 0 {
        return Err(format!(
            "bfloat16 weight buffer has odd byte length {}",
            weights.len()
        ));
    }

    let values: Vec<f32> = weights
        .chunks_exact(2)
        .map(|bytes| {
            let bf16 = u16::from_le_bytes([bytes[0], bytes[1]]);
            f32::from_bits((bf16 as u32) << 16)
        })
        .collect();

    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    if max_abs == 0.0 {
        return Ok((vec![0; values.len()], 1.0));
    }

    let scale = max_abs / qmax as f32;
    let quantized = values
        .into_iter()
        .map(|value| {
            let q = (value / scale).round().clamp(-(qmax as f32), qmax as f32) as i8;
            q as u8
        })
        .collect();

    Ok((quantized, scale))
}

fn validate_bits(bits: u32) -> Result<i32, String> {
    match bits {
        4 | 8 => Ok((1_i32 << (bits - 1)) - 1),
        _ => Err(format!(
            "unsupported quantization bit-width {bits}; expected 4 or 8"
        )),
    }
}

// ── Tiling helpers ───────────────────────────────────────────────────────────

// Extracts a tile_size×tile_size bfloat16 tile from a row-major full_cols-wide matrix.
// Each bfloat16 element is 2 bytes; copies tile_size rows of tile_size elements each.
fn extract_tile(
    matrix: &[u8],
    full_cols: usize,
    tile_row: u32,
    tile_col: u32,
    tile_size: usize,
) -> Result<Vec<u8>, String> {
    let expected_len = full_cols * full_cols * 2;
    if matrix.len() != expected_len {
        return Err(format!(
            "projection matrix has {} bytes, expected {} for {full_cols}x{full_cols} bfloat16",
            matrix.len(),
            expected_len
        ));
    }

    let mut tile = Vec::with_capacity(tile_size * tile_size * 2);
    let row_start = tile_row as usize * tile_size;
    let col_start = tile_col as usize * tile_size;
    for row in row_start..(row_start + tile_size) {
        let src_start = (row * full_cols + col_start) * 2;
        tile.extend_from_slice(&matrix[src_start..src_start + tile_size * 2]);
    }
    Ok(tile)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| {
                let bf16 = (value.to_bits() >> 16) as u16;
                bf16.to_le_bytes()
            })
            .collect()
    }

    fn signed(bytes: &[u8]) -> Vec<i8> {
        bytes.iter().map(|byte| *byte as i8).collect()
    }

    #[test]
    fn quantization_reads_bfloat16_pairs() {
        let weights = bf16_bytes(&[-1.0, 0.0, 1.0]);
        let (quantized, _) = quantize_tile(&weights, 8).unwrap();

        assert_eq!(quantized.len(), 3);
        assert_eq!(signed(&quantized), vec![-127, 0, 127]);
    }

    #[test]
    fn quantization_maps_known_values_to_int8() {
        let weights = bf16_bytes(&[-1.0, 0.0, 0.5, 1.0]);
        let (quantized, scale) = quantize_tile(&weights, 8).unwrap();

        assert!((scale - (1.0 / 127.0)).abs() < f32::EPSILON);
        assert_eq!(signed(&quantized), vec![-127, 0, 64, 127]);
    }

    #[test]
    fn quantization_maps_known_values_to_int4_range() {
        let weights = bf16_bytes(&[-1.0, 0.0, 1.0]);
        let (quantized, scale) = quantize_tile(&weights, 4).unwrap();

        assert!((scale - (1.0 / 7.0)).abs() < f32::EPSILON);
        assert_eq!(signed(&quantized), vec![-7, 0, 7]);
    }

    #[test]
    fn quantization_zero_tile_uses_identity_scale() {
        let weights = bf16_bytes(&[0.0, 0.0, 0.0]);
        let (quantized, scale) = quantize_tile(&weights, 8).unwrap();

        assert_eq!(scale, 1.0);
        assert_eq!(quantized, vec![0, 0, 0]);
    }

    #[test]
    fn quantization_rejects_unsupported_bits() {
        let weights = bf16_bytes(&[1.0]);
        let err = quantize_tile(&weights, 6).unwrap_err();

        assert!(err.contains("expected 4 or 8"));
    }
}
