use crate::CompileConfig;
use crate::cim::{MatrixShape, Program, TileCoord, TileDispatch, TileSize, expected_weight_offset};
use crate::ir::{NormalizedProgram, ProjectionOp};

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredProgram {
    pub program: Program,
    pub tiles: Vec<LoweredTile>,
    pub quant_bits: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredTile {
    pub projection: String,
    pub tile: TileCoord,
    pub matrix_shape: MatrixShape,
    pub tile_size: TileSize,
    pub weight_offset: u64,
    pub quant_scale: f32,
    pub order: u32,
    pub payload: Vec<i8>,
}

pub fn lower_program(
    program: &NormalizedProgram,
    config: CompileConfig,
) -> Result<LoweredProgram, String> {
    validate_bits(config.bits)?;
    let tile_size = TileSize::new(config.tile_rows, config.tile_cols);
    if tile_size.rows == 0 || tile_size.cols == 0 {
        return Err("tile size must be greater than zero".to_string());
    }

    let projections: Vec<&ProjectionOp> = program.projections().collect();
    if projections.is_empty() {
        return Err("normalized program has no supported projection ops".to_string());
    }

    let mut ordered = projections;
    ordered.sort_by_key(|projection| (projection.kind.clone(), projection.name.clone()));

    let mut dispatches = Vec::new();
    let mut tiles = Vec::new();
    let mut order = 0u32;
    for projection in ordered {
        if !projection.rows.is_multiple_of(tile_size.rows)
            || !projection.cols.is_multiple_of(tile_size.cols)
        {
            return Err(format!(
                "tile size {}x{} must evenly divide projection `{}` shape {}x{}",
                tile_size.rows, tile_size.cols, projection.name, projection.rows, projection.cols
            ));
        }

        let row_tiles = projection.rows / tile_size.rows;
        let col_tiles = projection.cols / tile_size.cols;
        for tile_row in 0..row_tiles {
            for tile_col in 0..col_tiles {
                let tile = TileCoord::new(tile_row, tile_col);
                let values = extract_tile(projection, tile, tile_size)?;
                let (payload, scale) = quantize_f32_tile(&values, config.bits)?;
                let weight_offset = expected_weight_offset(order, tile_size);
                let matrix_shape = MatrixShape::new(projection.rows, projection.cols);
                dispatches.push(TileDispatch {
                    projection: projection.kind.clone(),
                    tile,
                    matrix_shape,
                    tile_size,
                    weight_offset,
                    quant_scale: scale,
                    order,
                });
                tiles.push(LoweredTile {
                    projection: projection.kind.to_string(),
                    tile,
                    matrix_shape,
                    tile_size,
                    weight_offset,
                    quant_scale: scale,
                    order,
                    payload,
                });
                order = order
                    .checked_add(1)
                    .ok_or_else(|| "too many tile dispatches".to_string())?;
            }
        }
    }

    let cim = Program::new("cim_compile", "main", dispatches);
    cim.verify()?;

    Ok(LoweredProgram {
        program: cim,
        tiles,
        quant_bits: config.bits,
    })
}

pub fn quantize_f32_tile(values: &[f32], bits: u32) -> Result<(Vec<i8>, f32), String> {
    let qmax = validate_bits(bits)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err("cannot quantize non-finite weight value".to_string());
    }

    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);

    if max_abs == 0.0 {
        return Ok((vec![0; values.len()], 1.0));
    }

    let scale = max_abs / qmax as f32;
    let quantized = values
        .iter()
        .map(|value| (value / scale).round().clamp(-(qmax as f32), qmax as f32) as i8)
        .collect();
    Ok((quantized, scale))
}

pub fn validate_bits(bits: u32) -> Result<i32, String> {
    match bits {
        4 | 8 => Ok((1_i32 << (bits - 1)) - 1),
        _ => Err(format!(
            "unsupported quantization bit-width {bits}; expected 4 or 8"
        )),
    }
}

fn extract_tile(
    projection: &ProjectionOp,
    tile: TileCoord,
    tile_size: TileSize,
) -> Result<Vec<f32>, String> {
    let row_start = tile.row as usize * tile_size.rows as usize;
    let col_start = tile.col as usize * tile_size.cols as usize;
    if row_start + tile_size.rows as usize > projection.rows as usize
        || col_start + tile_size.cols as usize > projection.cols as usize
    {
        return Err(format!(
            "tile [{}, {}] is out of bounds for projection `{}`",
            tile.row, tile.col, projection.name
        ));
    }

    let mut values = Vec::with_capacity(tile_size.rows as usize * tile_size.cols as usize);
    for row in row_start..row_start + tile_size.rows as usize {
        let src_start = row * projection.cols as usize + col_start;
        let src_end = src_start + tile_size.cols as usize;
        values.extend_from_slice(&projection.weights[src_start..src_end]);
    }
    Ok(values)
}

pub fn tile_payload_bytes(tiles: &[LoweredTile]) -> Vec<u8> {
    tiles
        .iter()
        .flat_map(|tile| tile.payload.iter().map(|value| *value as u8))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cim::ProjectionKind;
    use crate::ir::NormalizedOp;

    fn tiny_program() -> NormalizedProgram {
        NormalizedProgram::new(
            "tiny",
            vec![NormalizedOp::Projection(
                ProjectionOp::new(
                    "main",
                    ProjectionKind::Main,
                    2,
                    2,
                    vec![-1.0, 0.0, 0.5, 1.0],
                    None,
                )
                .unwrap(),
            )],
        )
    }

    #[test]
    fn quantization_maps_known_values_to_int8() {
        let (quantized, scale) = quantize_f32_tile(&[-1.0, 0.0, 0.5, 1.0], 8).unwrap();

        assert!((scale - (1.0 / 127.0)).abs() < f32::EPSILON);
        assert_eq!(quantized, vec![-127, 0, 64, 127]);
    }

    #[test]
    fn quantization_maps_known_values_to_int4_range() {
        let (quantized, scale) = quantize_f32_tile(&[-1.0, 0.0, 1.0], 4).unwrap();

        assert!((scale - (1.0 / 7.0)).abs() < f32::EPSILON);
        assert_eq!(quantized, vec![-7, 0, 7]);
    }

    #[test]
    fn lowering_sets_cim_offsets_and_tile_payloads() {
        let lowered = lower_program(&tiny_program(), CompileConfig::square(1, 8)).unwrap();

        assert_eq!(lowered.tiles.len(), 4);
        assert_eq!(lowered.program.entry.dispatches[0].weight_offset, 0);
        assert_eq!(lowered.program.entry.dispatches[1].weight_offset, 1);
        assert_eq!(lowered.tiles[0].payload, vec![-127]);
        assert_eq!(lowered.tiles[1].payload, vec![0]);
        assert_eq!(tile_payload_bytes(&lowered.tiles), vec![129, 0, 127, 127]);
    }

    #[test]
    fn lowering_rejects_non_divisible_tile_size() {
        let err = lower_program(&tiny_program(), CompileConfig::square(3, 8)).unwrap_err();

        assert!(err.contains("must evenly divide"));
    }
}
