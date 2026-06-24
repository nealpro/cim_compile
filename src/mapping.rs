use crate::ir::{AttentionKernel, AttentionStage, ExecutionTarget, ProjectionOp};

#[derive(Debug, Clone, PartialEq)]
pub struct MappingDecision {
    pub target: ExecutionTarget,
    pub cim_cost: f32,
    pub digital_cost: f32,
    pub reason: String,
}

pub fn choose_projection_target(
    projection: &ProjectionOp,
    tile_rows: u32,
    tile_cols: u32,
    bits: u32,
) -> MappingDecision {
    let tiles = (projection.rows.div_ceil(tile_rows) * projection.cols.div_ceil(tile_cols)) as f32;
    let quant_penalty = if bits == 4 { 0.15 } else { 0.05 };
    let cim_cost = tiles * (1.0 + quant_penalty);
    let digital_cost = projection.rows as f32 * projection.cols as f32;
    MappingDecision {
        target: ExecutionTarget::Cim,
        cim_cost,
        digital_cost,
        reason: format!(
            "static-weight projection `{}` is CiM-friendly and maps cleanly to tiles",
            projection.name
        ),
    }
}

pub fn choose_attention_target(kernel: &AttentionKernel) -> MappingDecision {
    let (target, cim_cost, digital_cost, reason) = match kernel.stage {
        AttentionStage::QueryProjection
        | AttentionStage::KeyProjection
        | AttentionStage::ValueProjection
        | AttentionStage::OutputProjection => (
            ExecutionTarget::Cim,
            1.0,
            10.0,
            format!(
                "attention projection `{}` has static weights and is a good CiM candidate",
                kernel.name
            ),
        ),
        AttentionStage::RepeatKv
        | AttentionStage::ScoreMatMul
        | AttentionStage::AttentionMask
        | AttentionStage::ContextMatMul
        | AttentionStage::Softmax => (
            ExecutionTarget::Digital,
            10.0,
            1.0,
            format!(
                "attention kernel `{}` uses dynamic activations or non-linear math, so it stays digital",
                kernel.name
            ),
        ),
    };

    MappingDecision {
        target,
        cim_cost,
        digital_cost,
        reason,
    }
}
