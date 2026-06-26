use crate::CompileConfig;
use crate::cim::{MatrixShape, Program, TileCoord, TileDispatch, TileSize, expected_weight_offset};
use crate::ir::{
    AttentionBlock, AttentionSliceMetadata, ExecutionTarget, MlpStage, NormalizedOp,
    NormalizedProgram, OperationStage, ProjectionOp, TinyDecoderBlock,
};
use crate::mapping::{choose_attention_target, choose_projection_target};

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredProgram {
    pub program: Program,
    pub tiles: Vec<LoweredTile>,
    pub execution_plan: Vec<OperationExecution>,
    pub attention_blocks: Vec<AttentionBlockPlan>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredTile {
    pub projection: String,
    pub stage: String,
    pub parent: Option<String>,
    pub target: String,
    pub reason: String,
    pub tile: TileCoord,
    pub matrix_shape: MatrixShape,
    pub tile_size: TileSize,
    pub weight_offset: u64,
    pub order: u32,
    pub payload: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OperationExecution {
    pub name: String,
    pub stage: String,
    pub parent: Option<String>,
    pub target: String,
    pub reason: String,
    pub shape: Option<[u32; 2]>,
    pub tile_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionBlockPlan {
    pub name: String,
    pub metadata: Option<AttentionSliceMetadata>,
    pub mode: String,
    pub cim_projections: Vec<String>,
    pub digital_kernels: Vec<String>,
    pub reason: String,
}

struct LoweringContext<'a> {
    tile_size: TileSize,
    dispatches: &'a mut Vec<TileDispatch>,
    tiles: &'a mut Vec<LoweredTile>,
    execution_plan: &'a mut Vec<OperationExecution>,
    attention_blocks: &'a mut Vec<AttentionBlockPlan>,
    order: &'a mut u32,
}

pub fn lower_program(
    program: &NormalizedProgram,
    config: CompileConfig,
) -> Result<LoweredProgram, String> {
    let tile_size = TileSize::new(config.tile_rows, config.tile_cols);
    if tile_size.rows == 0 || tile_size.cols == 0 {
        return Err("tile size must be greater than zero".to_string());
    }

    if program.projection_count() == 0 {
        return Err("normalized program has no supported projection ops".to_string());
    }

    let mut dispatches = Vec::new();
    let mut tiles = Vec::new();
    let mut execution_plan = Vec::new();
    let mut attention_blocks = Vec::new();
    let mut order = 0u32;

    {
        let mut ctx = LoweringContext {
            tile_size,
            dispatches: &mut dispatches,
            tiles: &mut tiles,
            execution_plan: &mut execution_plan,
            attention_blocks: &mut attention_blocks,
            order: &mut order,
        };

        for op in &program.ops {
            match op {
                NormalizedOp::Projection(projection) => {
                    lower_projection(projection, OperationStage::Projection, None, &mut ctx)?
                }
                NormalizedOp::Attention(block) => lower_attention_block(block, &mut ctx)?,
                NormalizedOp::TinyDecoder(block) => lower_tiny_decoder_block(block, &mut ctx)?,
                NormalizedOp::Reshape { name } => {
                    ctx.execution_plan.push(OperationExecution {
                        name: name.clone(),
                        stage: OperationStage::Reshape.as_str().to_string(),
                        parent: None,
                        target: ExecutionTarget::Digital.as_str().to_string(),
                        reason: "structural reshape stays on the digital path".to_string(),
                        shape: None,
                        tile_count: 0,
                    });
                }
                NormalizedOp::Transpose { name, perm } => {
                    ctx.execution_plan.push(OperationExecution {
                        name: name.clone(),
                        stage: OperationStage::Transpose.as_str().to_string(),
                        parent: None,
                        target: ExecutionTarget::Digital.as_str().to_string(),
                        reason: format!(
                            "transpose {:?} is a structural transform, so it stays digital",
                            perm
                        ),
                        shape: None,
                        tile_count: 0,
                    });
                }
            }
        }
    }

    let cim = Program::new("cim_compile", "main", dispatches);
    cim.verify()?;

    Ok(LoweredProgram {
        program: cim,
        tiles,
        execution_plan,
        attention_blocks,
    })
}

fn lower_tiny_decoder_block(
    block: &TinyDecoderBlock,
    ctx: &mut LoweringContext<'_>,
) -> Result<(), String> {
    push_digital_execution(
        ctx.execution_plan,
        "model.embed_tokens",
        OperationStage::EmbeddingLookup,
        Some(block.name.clone()),
        "token ID embedding lookup uses dynamic indices, so it stays digital",
        Some([
            block.metadata.default_sequence_length,
            block.metadata.hidden_size,
        ]),
    );
    push_digital_execution(
        ctx.execution_plan,
        "model.layers.0.input_layernorm",
        OperationStage::Norm("norm.input_layernorm"),
        Some(block.name.clone()),
        "RMSNorm is a reduction and elementwise normalization, so it stays digital",
        Some([
            block.metadata.default_sequence_length,
            block.metadata.hidden_size,
        ]),
    );
    push_digital_execution(
        ctx.execution_plan,
        "model.rotary_emb",
        OperationStage::RotaryEmbedding,
        Some(block.name.clone()),
        "rotary position embedding is dynamic trigonometric glue, so it stays digital",
        Some([
            block.metadata.default_sequence_length,
            block.metadata.head_dim,
        ]),
    );

    lower_attention_block(&block.attention, ctx)?;

    push_digital_execution(
        ctx.execution_plan,
        "model.layers.0.attention_residual",
        OperationStage::Residual("residual.attention"),
        Some(block.name.clone()),
        "residual addition is dynamic activation glue, so it stays digital",
        Some([
            block.metadata.default_sequence_length,
            block.metadata.hidden_size,
        ]),
    );
    push_digital_execution(
        ctx.execution_plan,
        "model.layers.0.post_attention_layernorm",
        OperationStage::Norm("norm.post_attention_layernorm"),
        Some(block.name.clone()),
        "RMSNorm is a reduction and elementwise normalization, so it stays digital",
        Some([
            block.metadata.default_sequence_length,
            block.metadata.hidden_size,
        ]),
    );

    lower_projection(
        &block.mlp_gate_proj,
        OperationStage::Mlp(MlpStage::GateProjection),
        Some(block.name.clone()),
        ctx,
    )?;
    lower_projection(
        &block.mlp_up_proj,
        OperationStage::Mlp(MlpStage::UpProjection),
        Some(block.name.clone()),
        ctx,
    )?;
    push_digital_execution(
        ctx.execution_plan,
        "model.layers.0.mlp.act_fn",
        OperationStage::Mlp(MlpStage::Activation),
        Some(block.name.clone()),
        "SiLU activation is non-linear, so it stays digital",
        Some([
            block.metadata.default_sequence_length,
            block.metadata.intermediate_size,
        ]),
    );
    push_digital_execution(
        ctx.execution_plan,
        "model.layers.0.mlp.multiply",
        OperationStage::Mlp(MlpStage::ElementwiseMultiply),
        Some(block.name.clone()),
        "MLP gated elementwise multiply combines dynamic activations, so it stays digital",
        Some([
            block.metadata.default_sequence_length,
            block.metadata.intermediate_size,
        ]),
    );
    lower_projection(
        &block.mlp_down_proj,
        OperationStage::Mlp(MlpStage::DownProjection),
        Some(block.name.clone()),
        ctx,
    )?;

    push_digital_execution(
        ctx.execution_plan,
        "model.layers.0.mlp_residual",
        OperationStage::Residual("residual.mlp"),
        Some(block.name.clone()),
        "residual addition is dynamic activation glue, so it stays digital",
        Some([
            block.metadata.default_sequence_length,
            block.metadata.hidden_size,
        ]),
    );
    push_digital_execution(
        ctx.execution_plan,
        "model.norm",
        OperationStage::Norm("norm.final"),
        Some(block.name.clone()),
        "final RMSNorm is a reduction and elementwise normalization, so it stays digital",
        Some([
            block.metadata.default_sequence_length,
            block.metadata.hidden_size,
        ]),
    );
    push_digital_execution(
        ctx.execution_plan,
        "lm_head",
        OperationStage::LmHead,
        Some(block.name.clone()),
        "the 32k-vocabulary logits projection stays digital for this token-logits milestone",
        Some([block.metadata.hidden_size, block.metadata.vocab_size]),
    );

    Ok(())
}

fn push_digital_execution(
    execution_plan: &mut Vec<OperationExecution>,
    name: impl Into<String>,
    stage: OperationStage,
    parent: Option<String>,
    reason: impl Into<String>,
    shape: Option<[u32; 2]>,
) {
    execution_plan.push(OperationExecution {
        name: name.into(),
        stage: stage.as_str().to_string(),
        parent,
        target: ExecutionTarget::Digital.as_str().to_string(),
        reason: reason.into(),
        shape,
        tile_count: 0,
    });
}

fn lower_projection(
    projection: &ProjectionOp,
    stage: OperationStage,
    parent: Option<String>,
    ctx: &mut LoweringContext<'_>,
) -> Result<(), String> {
    let decision = choose_projection_target(projection, ctx.tile_size.rows, ctx.tile_size.cols);
    let tile_count =
        projection.rows.div_ceil(ctx.tile_size.rows) * projection.cols.div_ceil(ctx.tile_size.cols);
    ctx.execution_plan.push(OperationExecution {
        name: projection.name.clone(),
        stage: stage.as_str().to_string(),
        parent: parent.clone(),
        target: decision.target.as_str().to_string(),
        reason: decision.reason.clone(),
        shape: Some([projection.rows, projection.cols]),
        tile_count,
    });

    if decision.target != ExecutionTarget::Cim {
        return Ok(());
    }

    let row_tiles = projection.rows.div_ceil(ctx.tile_size.rows);
    let col_tiles = projection.cols.div_ceil(ctx.tile_size.cols);
    for tile_row in 0..row_tiles {
        for tile_col in 0..col_tiles {
            let tile = TileCoord::new(tile_row, tile_col);
            let values = extract_tile(projection, tile, ctx.tile_size)?;
            let weight_offset = expected_weight_offset(*ctx.order, ctx.tile_size);
            let matrix_shape = MatrixShape::new(projection.rows, projection.cols);
            ctx.dispatches.push(TileDispatch {
                projection: projection.kind.clone(),
                tile,
                matrix_shape,
                tile_size: ctx.tile_size,
                weight_offset,
                order: *ctx.order,
            });
            ctx.tiles.push(LoweredTile {
                projection: projection.kind.to_string(),
                stage: stage.as_str().to_string(),
                parent: parent.clone(),
                target: decision.target.as_str().to_string(),
                reason: decision.reason.clone(),
                tile,
                matrix_shape,
                tile_size: ctx.tile_size,
                weight_offset,
                order: *ctx.order,
                payload: values,
            });
            let next_order = (*ctx.order)
                .checked_add(1)
                .ok_or_else(|| "too many tile dispatches".to_string())?;
            *ctx.order = next_order;
        }
    }

    Ok(())
}

fn lower_attention_block(
    block: &AttentionBlock,
    ctx: &mut LoweringContext<'_>,
) -> Result<(), String> {
    let mut cim_projection_names = Vec::new();
    let mut digital_kernel_names = Vec::new();
    let mut block_reason = Vec::new();

    for (stage, projection) in block.projection_entries() {
        let decision = choose_projection_target(projection, ctx.tile_size.rows, ctx.tile_size.cols);
        block_reason.push(decision.reason.clone());
        cim_projection_names.push(projection.name.clone());
        lower_projection(projection, stage, Some(block.name.clone()), ctx)?;
    }

    for kernel in block.kernel_entries() {
        let decision = choose_attention_target(kernel);
        digital_kernel_names.push(kernel.name.clone());
        ctx.execution_plan.push(OperationExecution {
            name: kernel.name.clone(),
            stage: OperationStage::Attention(kernel.stage).as_str().to_string(),
            parent: Some(block.name.clone()),
            target: decision.target.as_str().to_string(),
            reason: decision.reason.clone(),
            shape: kernel.shape,
            tile_count: 0,
        });
        block_reason.push(decision.reason);
    }

    let cim_count = cim_projection_names.len();
    let digital_count = digital_kernel_names.len();
    let mode = if cim_count > 0 && digital_count > 0 {
        "hybrid"
    } else if cim_count > 0 {
        "cim_only"
    } else {
        "digital_only"
    };

    ctx.attention_blocks.push(AttentionBlockPlan {
        name: block.name.clone(),
        metadata: block.metadata.clone(),
        mode: mode.to_string(),
        cim_projections: cim_projection_names,
        digital_kernels: digital_kernel_names,
        reason: block_reason.join("; "),
    });

    Ok(())
}

fn extract_tile(
    projection: &ProjectionOp,
    tile: TileCoord,
    tile_size: TileSize,
) -> Result<Vec<f32>, String> {
    let row_start = tile.row as usize * tile_size.rows as usize;
    let col_start = tile.col as usize * tile_size.cols as usize;
    if row_start >= projection.rows as usize || col_start >= projection.cols as usize {
        return Err(format!(
            "tile [{}, {}] is out of bounds for projection `{}`",
            tile.row, tile.col, projection.name
        ));
    }

    let mut values = Vec::with_capacity(tile_size.rows as usize * tile_size.cols as usize);
    for row in row_start..row_start + tile_size.rows as usize {
        for col in col_start..col_start + tile_size.cols as usize {
            if row < projection.rows as usize && col < projection.cols as usize {
                values.push(projection.weights[row * projection.cols as usize + col]);
            } else {
                values.push(0.0);
            }
        }
    }
    Ok(values)
}

pub fn tile_payload_bytes(tiles: &[LoweredTile]) -> Vec<u8> {
    let byte_len = tiles
        .iter()
        .map(|tile| tile.payload.len() * std::mem::size_of::<f32>())
        .sum();
    let mut bytes = Vec::with_capacity(byte_len);
    for tile in tiles {
        for value in &tile.payload {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        AttentionBlock, AttentionKernel, AttentionKernels, AttentionProjections, AttentionStage,
        NormalizedOp,
    };

    fn tiny_projection_program() -> NormalizedProgram {
        NormalizedProgram::new(
            "tiny",
            vec![NormalizedOp::Projection(
                ProjectionOp::new(
                    "main",
                    crate::cim::ProjectionKind::Main,
                    2,
                    2,
                    vec![-1.0, 0.0, 0.5, 1.0],
                    None,
                )
                .unwrap(),
            )],
        )
    }

    fn tiny_attention_program() -> NormalizedProgram {
        let q = ProjectionOp::new(
            "q_proj",
            crate::cim::ProjectionKind::WQ,
            2,
            2,
            vec![-1.0, 0.0, 0.5, 1.0],
            None,
        )
        .unwrap();
        let k = ProjectionOp::new(
            "k_proj",
            crate::cim::ProjectionKind::WK,
            2,
            2,
            vec![-1.0, 0.0, 0.5, 1.0],
            None,
        )
        .unwrap();
        let v = ProjectionOp::new(
            "v_proj",
            crate::cim::ProjectionKind::WV,
            2,
            2,
            vec![-1.0, 0.0, 0.5, 1.0],
            None,
        )
        .unwrap();
        let o = ProjectionOp::new(
            "out_proj",
            crate::cim::ProjectionKind::WO,
            2,
            2,
            vec![-1.0, 0.0, 0.5, 1.0],
            None,
        )
        .unwrap();

        NormalizedProgram::new(
            "attention",
            vec![NormalizedOp::Attention(Box::new(AttentionBlock::new(
                "mha",
                AttentionProjections {
                    q_proj: q,
                    k_proj: k,
                    v_proj: v,
                    out_proj: o,
                },
                AttentionKernels {
                    score_matmul: AttentionKernel::new(
                        "attn_scores",
                        AttentionStage::ScoreMatMul,
                        None,
                    ),
                    softmax: AttentionKernel::new("attn_softmax", AttentionStage::Softmax, None),
                    context_matmul: AttentionKernel::new(
                        "attn_context",
                        AttentionStage::ContextMatMul,
                        None,
                    ),
                },
                None,
            )))],
        )
    }

    #[test]
    fn lowering_sets_cim_offsets_and_tile_payloads() {
        let lowered = lower_program(&tiny_projection_program(), CompileConfig::square(1)).unwrap();

        assert_eq!(lowered.tiles.len(), 4);
        assert_eq!(lowered.program.entry.dispatches[0].weight_offset, 0);
        assert_eq!(lowered.program.entry.dispatches[1].weight_offset, 4);
        assert_eq!(lowered.tiles[0].payload, vec![-1.0]);
        assert_eq!(lowered.tiles[1].payload, vec![0.0]);
        assert_eq!(
            tile_payload_bytes(&lowered.tiles),
            vec![0, 0, 128, 191, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 128, 63]
        );
        assert_eq!(lowered.execution_plan[0].stage, "projection");
        assert_eq!(lowered.execution_plan[0].target, "cim");
    }

    #[test]
    fn lowering_pads_non_divisible_edge_tiles() {
        let lowered = lower_program(&tiny_projection_program(), CompileConfig::square(3)).unwrap();

        assert_eq!(lowered.tiles.len(), 1);
        assert_eq!(lowered.program.entry.dispatches[0].matrix_shape.rows, 2);
        assert_eq!(lowered.program.entry.dispatches[0].tile_size.rows, 3);
        assert_eq!(
            lowered.tiles[0].payload,
            vec![-1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            tile_payload_bytes(&lowered.tiles).len(),
            9 * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn attention_block_lowers_as_hybrid_plan() {
        let lowered = lower_program(&tiny_attention_program(), CompileConfig::square(1)).unwrap();

        assert_eq!(lowered.tiles.len(), 16);
        assert_eq!(lowered.attention_blocks.len(), 1);
        assert_eq!(lowered.attention_blocks[0].mode, "hybrid");
        assert!(
            lowered.attention_blocks[0]
                .cim_projections
                .contains(&"q_proj".to_string())
        );
        assert!(
            lowered
                .execution_plan
                .iter()
                .any(|entry| entry.name == "attn_softmax" && entry.target == "digital")
        );
    }
}
