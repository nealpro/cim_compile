use crate::frontend::HighLevelOp;
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
            } => write!(
                f,
                "ProjectionTile {{ projection: {projection:?}, row: {row}, col: {col}, weights: {}B }}",
                weights.len()
            ),
        }
    }
}

pub fn tile(ops: Vec<HighLevelOp>, spec: &CrossbarSpec) -> Vec<LowLevelOp> {
    ops.into_iter()
        .flat_map(|op| match op {
            HighLevelOp::MultiHeadAttention {
                embed_dim, weights, ..
            } => {
                let weights = weights.expect("weights required — use parse_onnx, not parse_model");
                let row_tiles = embed_dim / spec.tile_rows;
                let col_tiles = embed_dim / spec.tile_cols;
                let tile_size = spec.tile_rows as usize;
                let full_cols = embed_dim as usize;
                let mut tiles = Vec::new();
                for (proj, matrix) in [
                    (Projection::WQ, &weights.wq),
                    (Projection::WK, &weights.wk),
                    (Projection::WV, &weights.wv),
                    (Projection::WO, &weights.wo),
                ] {
                    for row in 0..row_tiles {
                        for col in 0..col_tiles {
                            tiles.push(LowLevelOp::ProjectionTile {
                                projection: proj,
                                row,
                                col,
                                weights: extract_tile(matrix, full_cols, row, col, tile_size),
                            });
                        }
                    }
                }
                tiles
            }
        })
        .collect()
}

// Extracts a tile_size×tile_size bfloat16 tile from a row-major full_cols-wide matrix.
// Each bfloat16 element is 2 bytes; copies tile_size rows of tile_size elements each.
fn extract_tile(
    matrix: &[u8],
    full_cols: usize,
    tile_row: u32,
    tile_col: u32,
    tile_size: usize,
) -> Vec<u8> {
    let mut tile = Vec::with_capacity(tile_size * tile_size * 2);
    let row_start = tile_row as usize * tile_size;
    let col_start = tile_col as usize * tile_size;
    for row in row_start..(row_start + tile_size) {
        let src_start = (row * full_cols + col_start) * 2;
        tile.extend_from_slice(&matrix[src_start..src_start + tile_size * 2]);
    }
    tile
}
