use serde::{Deserialize, Serialize};

use crate::cim::ProjectionKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExecutionTarget {
    Cim,
    Digital,
}

impl ExecutionTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            ExecutionTarget::Cim => "cim",
            ExecutionTarget::Digital => "digital",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AttentionStage {
    QueryProjection,
    KeyProjection,
    ValueProjection,
    RepeatKv,
    ScoreMatMul,
    AttentionMask,
    Softmax,
    ContextMatMul,
    OutputProjection,
}

impl AttentionStage {
    pub fn as_str(self) -> &'static str {
        match self {
            AttentionStage::QueryProjection => "query_projection",
            AttentionStage::KeyProjection => "key_projection",
            AttentionStage::ValueProjection => "value_projection",
            AttentionStage::RepeatKv => "repeat_kv",
            AttentionStage::ScoreMatMul => "score_matmul",
            AttentionStage::AttentionMask => "attention_mask",
            AttentionStage::Softmax => "softmax",
            AttentionStage::ContextMatMul => "context_matmul",
            AttentionStage::OutputProjection => "output_projection",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MlpStage {
    GateProjection,
    UpProjection,
    Activation,
    ElementwiseMultiply,
    DownProjection,
}

impl MlpStage {
    pub fn as_str(self) -> &'static str {
        match self {
            MlpStage::GateProjection => "gate_projection",
            MlpStage::UpProjection => "up_projection",
            MlpStage::Activation => "activation",
            MlpStage::ElementwiseMultiply => "elementwise_multiply",
            MlpStage::DownProjection => "down_projection",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperationStage {
    Projection,
    Attention(AttentionStage),
    Mlp(MlpStage),
    EmbeddingLookup,
    Norm(&'static str),
    RotaryEmbedding,
    Residual(&'static str),
    LmHead,
    Reshape,
    Transpose,
}

impl OperationStage {
    pub fn as_str(self) -> &'static str {
        match self {
            OperationStage::Projection => "projection",
            OperationStage::Attention(AttentionStage::QueryProjection) => {
                "attention.query_projection"
            }
            OperationStage::Attention(AttentionStage::KeyProjection) => "attention.key_projection",
            OperationStage::Attention(AttentionStage::ValueProjection) => {
                "attention.value_projection"
            }
            OperationStage::Attention(AttentionStage::RepeatKv) => "attention.repeat_kv",
            OperationStage::Attention(AttentionStage::ScoreMatMul) => "attention.score_matmul",
            OperationStage::Attention(AttentionStage::AttentionMask) => "attention.attention_mask",
            OperationStage::Attention(AttentionStage::Softmax) => "attention.softmax",
            OperationStage::Attention(AttentionStage::ContextMatMul) => "attention.context_matmul",
            OperationStage::Attention(AttentionStage::OutputProjection) => {
                "attention.output_projection"
            }
            OperationStage::Mlp(MlpStage::GateProjection) => "mlp.gate_projection",
            OperationStage::Mlp(MlpStage::UpProjection) => "mlp.up_projection",
            OperationStage::Mlp(MlpStage::Activation) => "mlp.activation",
            OperationStage::Mlp(MlpStage::ElementwiseMultiply) => "mlp.elementwise_multiply",
            OperationStage::Mlp(MlpStage::DownProjection) => "mlp.down_projection",
            OperationStage::EmbeddingLookup => "embedding.lookup",
            OperationStage::Norm(name) => name,
            OperationStage::RotaryEmbedding => "rotary.embedding",
            OperationStage::Residual(name) => name,
            OperationStage::LmHead => "lm_head.matmul",
            OperationStage::Reshape => "reshape",
            OperationStage::Transpose => "transpose",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedProgram {
    pub name: String,
    pub ops: Vec<NormalizedOp>,
}

impl NormalizedProgram {
    pub fn new(name: impl Into<String>, ops: Vec<NormalizedOp>) -> Self {
        Self {
            name: name.into(),
            ops,
        }
    }

    pub fn projections(&self) -> Vec<&ProjectionOp> {
        let mut projections = Vec::new();
        for op in &self.ops {
            match op {
                NormalizedOp::Projection(projection) => projections.push(projection),
                NormalizedOp::Attention(block) => projections.extend(block.projections()),
                NormalizedOp::TinyDecoder(block) => projections.extend(block.projections()),
                _ => {}
            }
        }
        projections
    }

    pub fn projection_count(&self) -> usize {
        self.projections().len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NormalizedOp {
    Projection(ProjectionOp),
    Attention(AttentionBlock),
    TinyDecoder(TinyDecoderBlock),
    Reshape { name: String },
    Transpose { name: String, perm: Vec<i64> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionKernel {
    pub name: String,
    pub stage: AttentionStage,
    pub shape: Option<[u32; 2]>,
}

impl AttentionKernel {
    pub fn new(name: impl Into<String>, stage: AttentionStage, shape: Option<[u32; 2]>) -> Self {
        Self {
            name: name.into(),
            stage,
            shape,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSliceMetadata {
    pub source_prefix: String,
    pub hidden_size: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub output_dim: u32,
    pub head_dim: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub grouped_query_attention: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TinyDecoderMetadata {
    pub model_kind: String,
    pub inference_mode: String,
    pub source_prefix: String,
    pub decoder_layers: u32,
    pub default_sequence_length: u32,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub head_dim: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub grouped_query_attention: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionBlock {
    pub name: String,
    pub metadata: Option<AttentionSliceMetadata>,
    pub q_proj: ProjectionOp,
    pub k_proj: ProjectionOp,
    pub v_proj: ProjectionOp,
    pub score_matmul: AttentionKernel,
    pub softmax: AttentionKernel,
    pub context_matmul: AttentionKernel,
    pub digital_fallbacks: Vec<AttentionKernel>,
    pub out_proj: ProjectionOp,
    pub residual: Option<String>,
}

impl AttentionBlock {
    pub fn new(
        name: impl Into<String>,
        q_proj: ProjectionOp,
        k_proj: ProjectionOp,
        v_proj: ProjectionOp,
        score_matmul: AttentionKernel,
        softmax: AttentionKernel,
        context_matmul: AttentionKernel,
        out_proj: ProjectionOp,
        residual: Option<String>,
    ) -> Self {
        Self {
            name: name.into(),
            metadata: None,
            q_proj,
            k_proj,
            v_proj,
            score_matmul,
            softmax,
            context_matmul,
            digital_fallbacks: Vec::new(),
            out_proj,
            residual,
        }
    }

    pub fn with_metadata(mut self, metadata: AttentionSliceMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_digital_fallbacks(mut self, fallbacks: Vec<AttentionKernel>) -> Self {
        self.digital_fallbacks = fallbacks;
        self
    }

    pub fn projections(&self) -> Vec<&ProjectionOp> {
        vec![&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj]
    }

    pub fn projection_entries(&self) -> Vec<(OperationStage, &ProjectionOp)> {
        vec![
            (
                OperationStage::Attention(AttentionStage::QueryProjection),
                &self.q_proj,
            ),
            (
                OperationStage::Attention(AttentionStage::KeyProjection),
                &self.k_proj,
            ),
            (
                OperationStage::Attention(AttentionStage::ValueProjection),
                &self.v_proj,
            ),
            (
                OperationStage::Attention(AttentionStage::OutputProjection),
                &self.out_proj,
            ),
        ]
    }

    pub fn kernel_entries(&self) -> Vec<&AttentionKernel> {
        let mut entries = Vec::new();
        entries.extend(
            self.digital_fallbacks
                .iter()
                .filter(|kernel| kernel.stage == AttentionStage::RepeatKv),
        );
        entries.push(&self.score_matmul);
        entries.extend(
            self.digital_fallbacks
                .iter()
                .filter(|kernel| kernel.stage == AttentionStage::AttentionMask),
        );
        entries.push(&self.softmax);
        entries.extend(self.digital_fallbacks.iter().filter(|kernel| {
            !matches!(
                kernel.stage,
                AttentionStage::RepeatKv | AttentionStage::AttentionMask
            )
        }));
        entries.push(&self.context_matmul);
        entries
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DigitalTensor {
    pub name: String,
    pub role: String,
    pub shape: Vec<u32>,
    pub values: Vec<f32>,
}

impl DigitalTensor {
    pub fn new(
        name: impl Into<String>,
        role: impl Into<String>,
        shape: Vec<u32>,
        values: Vec<f32>,
    ) -> Result<Self, String> {
        if shape.is_empty() || shape.contains(&0) {
            return Err("digital tensor shape must contain positive dimensions".to_string());
        }
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim as usize))
            .ok_or_else(|| "digital tensor shape is too large".to_string())?;
        if values.len() != expected {
            return Err(format!(
                "digital tensor has {} values, expected {} for shape {:?}",
                values.len(),
                expected,
                shape
            ));
        }
        Ok(Self {
            name: name.into(),
            role: role.into(),
            shape,
            values,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TinyDecoderBlock {
    pub name: String,
    pub metadata: TinyDecoderMetadata,
    pub attention: AttentionBlock,
    pub mlp_gate_proj: ProjectionOp,
    pub mlp_up_proj: ProjectionOp,
    pub mlp_down_proj: ProjectionOp,
    pub digital_tensors: Vec<DigitalTensor>,
}

impl TinyDecoderBlock {
    pub fn new(
        name: impl Into<String>,
        metadata: TinyDecoderMetadata,
        attention: AttentionBlock,
        mlp_gate_proj: ProjectionOp,
        mlp_up_proj: ProjectionOp,
        mlp_down_proj: ProjectionOp,
        digital_tensors: Vec<DigitalTensor>,
    ) -> Self {
        Self {
            name: name.into(),
            metadata,
            attention,
            mlp_gate_proj,
            mlp_up_proj,
            mlp_down_proj,
            digital_tensors,
        }
    }

    pub fn projections(&self) -> Vec<&ProjectionOp> {
        let mut projections = self.attention.projections();
        projections.push(&self.mlp_gate_proj);
        projections.push(&self.mlp_up_proj);
        projections.push(&self.mlp_down_proj);
        projections
    }

    pub fn mlp_projection_entries(&self) -> Vec<(OperationStage, &ProjectionOp)> {
        vec![
            (
                OperationStage::Mlp(MlpStage::GateProjection),
                &self.mlp_gate_proj,
            ),
            (
                OperationStage::Mlp(MlpStage::UpProjection),
                &self.mlp_up_proj,
            ),
            (
                OperationStage::Mlp(MlpStage::DownProjection),
                &self.mlp_down_proj,
            ),
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionOp {
    pub name: String,
    pub kind: ProjectionKind,
    pub rows: u32,
    pub cols: u32,
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl ProjectionOp {
    pub fn new(
        name: impl Into<String>,
        kind: ProjectionKind,
        rows: u32,
        cols: u32,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, String> {
        let expected = rows as usize * cols as usize;
        if rows == 0 || cols == 0 {
            return Err("projection dimensions must be greater than zero".to_string());
        }
        if weights.len() != expected {
            return Err(format!(
                "projection {}x{} has {} weights, expected {}",
                rows,
                cols,
                weights.len(),
                expected
            ));
        }
        if let Some(bias) = &bias {
            if bias.len() != rows as usize {
                return Err(format!(
                    "projection bias has {} elements, expected {}",
                    bias.len(),
                    rows
                ));
            }
        }
        Ok(Self {
            name: name.into(),
            kind,
            rows,
            cols,
            weights,
            bias,
        })
    }
}
