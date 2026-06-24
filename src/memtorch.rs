use serde::{Deserialize, Serialize};

use crate::ir::NormalizedProgram;
use crate::lowering::{LoweredProgram, tile_payload_bytes};

#[derive(Debug, Clone, PartialEq)]
pub struct MemtorchPackage {
    pub manifest: MemtorchManifest,
    pub weights: Vec<u8>,
    pub digital: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemtorchManifest {
    pub schema_version: u32,
    pub entry: String,
    pub tile_size: [u32; 2],
    pub quant_bits: u32,
    pub weights_file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digital_tensors_file: Option<String>,
    pub projections: Vec<ProjectionManifest>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub digital_tensors: Vec<DigitalTensorManifest>,
    pub execution_plan: Vec<ExecutionManifest>,
    pub attention_blocks: Vec<AttentionBlockManifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_slice: Option<InferenceSliceManifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulation_summary: Option<SimulationSummaryManifest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionManifest {
    pub name: String,
    pub projection: String,
    pub stage: String,
    pub parent: Option<String>,
    pub target: String,
    pub rows: u32,
    pub cols: u32,
    pub bias: Option<Vec<f32>>,
    pub tiles: Vec<TileManifest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileManifest {
    pub row: u32,
    pub col: u32,
    pub stage: String,
    pub parent: Option<String>,
    pub target: String,
    pub reason: String,
    pub matrix_shape: [u32; 2],
    pub tile_size: [u32; 2],
    pub weight_offset: u64,
    pub quant_scale: f32,
    pub order: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionManifest {
    pub name: String,
    pub stage: String,
    pub parent: Option<String>,
    pub target: String,
    pub reason: String,
    pub shape: Option<[u32; 2]>,
    pub tile_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DigitalTensorManifest {
    pub name: String,
    pub role: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub byte_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionBlockManifest {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AttentionSliceManifest>,
    pub mode: String,
    pub cim_projections: Vec<String>,
    pub digital_kernels: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionSliceManifest {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceSliceManifest {
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
    pub cim_projections: Vec<String>,
    pub digital_tensors: Vec<String>,
    pub digital_stages: Vec<String>,
    pub unsupported: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationSummaryManifest {
    pub supported_runtime_modes: Vec<String>,
    pub memtorch_stages: Vec<String>,
    pub digital_stages: Vec<String>,
    pub patched_projection_count: u32,
    pub lm_head_target: String,
}

fn build_projection_manifests(
    normalized: &NormalizedProgram,
    lowered: &LoweredProgram,
) -> Result<Vec<ProjectionManifest>, String> {
    let mut projections = Vec::new();
    for op in &normalized.ops {
        match op {
            crate::ir::NormalizedOp::Projection(projection) => projections.push(
                build_projection_manifest(projection, "projection", None, lowered)?,
            ),
            crate::ir::NormalizedOp::Attention(block) => {
                for (stage, projection) in block.projection_entries() {
                    projections.push(build_projection_manifest(
                        projection,
                        stage.as_str(),
                        Some(block.name.as_str()),
                        lowered,
                    )?)
                }
            }
            crate::ir::NormalizedOp::TinyDecoder(block) => {
                for (stage, projection) in block.attention.projection_entries() {
                    projections.push(build_projection_manifest(
                        projection,
                        stage.as_str(),
                        Some(block.attention.name.as_str()),
                        lowered,
                    )?)
                }
                for (stage, projection) in block.mlp_projection_entries() {
                    projections.push(build_projection_manifest(
                        projection,
                        stage.as_str(),
                        Some(block.name.as_str()),
                        lowered,
                    )?)
                }
            }
            _ => {}
        }
    }
    Ok(projections)
}

fn build_digital_tensors(
    normalized: &NormalizedProgram,
) -> Result<(Vec<DigitalTensorManifest>, Vec<u8>), String> {
    let mut manifest = Vec::new();
    let mut bytes = Vec::new();

    for op in &normalized.ops {
        let crate::ir::NormalizedOp::TinyDecoder(block) = op else {
            continue;
        };
        for tensor in &block.digital_tensors {
            let byte_offset = bytes.len() as u64;
            for value in &tensor.values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let byte_len = bytes.len() as u64 - byte_offset;
            manifest.push(DigitalTensorManifest {
                name: tensor.name.clone(),
                role: tensor.role.clone(),
                dtype: "f32".to_string(),
                shape: tensor.shape.clone(),
                byte_offset,
                byte_len,
            });
        }
    }

    Ok((manifest, bytes))
}

fn build_inference_slice(normalized: &NormalizedProgram) -> Option<InferenceSliceManifest> {
    normalized.ops.iter().find_map(|op| {
        let crate::ir::NormalizedOp::TinyDecoder(block) = op else {
            return None;
        };
        let metadata = &block.metadata;
        Some(InferenceSliceManifest {
            model_kind: metadata.model_kind.clone(),
            inference_mode: metadata.inference_mode.clone(),
            source_prefix: metadata.source_prefix.clone(),
            decoder_layers: metadata.decoder_layers,
            default_sequence_length: metadata.default_sequence_length,
            vocab_size: metadata.vocab_size,
            hidden_size: metadata.hidden_size,
            intermediate_size: metadata.intermediate_size,
            head_dim: metadata.head_dim,
            q_heads: metadata.q_heads,
            kv_heads: metadata.kv_heads,
            grouped_query_attention: metadata.grouped_query_attention,
            cim_projections: block
                .projections()
                .into_iter()
                .map(|projection| projection.name.clone())
                .collect(),
            digital_tensors: block
                .digital_tensors
                .iter()
                .map(|tensor| tensor.role.clone())
                .collect(),
            digital_stages: vec![
                "embedding.lookup".to_string(),
                "norm.input_layernorm".to_string(),
                "rotary.embedding".to_string(),
                "attention.score_matmul".to_string(),
                "attention.attention_mask".to_string(),
                "attention.softmax".to_string(),
                "attention.context_matmul".to_string(),
                "residual.attention".to_string(),
                "norm.post_attention_layernorm".to_string(),
                "mlp.activation".to_string(),
                "mlp.elementwise_multiply".to_string(),
                "residual.mlp".to_string(),
                "norm.final".to_string(),
                "lm_head.matmul".to_string(),
            ],
            unsupported: vec![
                "text tokenization".to_string(),
                "text generation from tokenizer strings".to_string(),
                "non-greedy sampling controls".to_string(),
                "non-empty KV cache decoding".to_string(),
                "arbitrary ONNX transformer graphs".to_string(),
            ],
        })
    })
}

fn build_simulation_summary(
    inference_slice: &Option<InferenceSliceManifest>,
    projections: &[ProjectionManifest],
) -> Option<SimulationSummaryManifest> {
    let inference_slice = inference_slice.as_ref()?;
    Some(SimulationSummaryManifest {
        supported_runtime_modes: vec!["logits".to_string(), "generate_ids".to_string()],
        memtorch_stages: projections
            .iter()
            .map(|projection| projection.stage.clone())
            .collect(),
        digital_stages: inference_slice.digital_stages.clone(),
        patched_projection_count: projections.len() as u32,
        lm_head_target: "digital".to_string(),
    })
}

fn build_projection_manifest(
    projection: &crate::ir::ProjectionOp,
    stage: &str,
    parent: Option<&str>,
    lowered: &LoweredProgram,
) -> Result<ProjectionManifest, String> {
    let tiles = lowered
        .tiles
        .iter()
        .filter(|tile| {
            tile.projection == projection.kind.to_string()
                && tile.stage == stage
                && tile.parent.as_deref() == parent
        })
        .map(|tile| TileManifest {
            row: tile.tile.row,
            col: tile.tile.col,
            stage: tile.stage.clone(),
            parent: tile.parent.clone(),
            target: tile.target.clone(),
            reason: tile.reason.clone(),
            matrix_shape: [tile.matrix_shape.rows, tile.matrix_shape.cols],
            tile_size: [tile.tile_size.rows, tile.tile_size.cols],
            weight_offset: tile.weight_offset,
            quant_scale: tile.quant_scale,
            order: tile.order,
        })
        .collect::<Vec<_>>();

    if tiles.is_empty() {
        return Err(format!(
            "projection `{}` has no lowered MemTorch tiles",
            projection.name
        ));
    }

    Ok(ProjectionManifest {
        name: projection.name.clone(),
        projection: projection.kind.to_string(),
        stage: stage.to_string(),
        parent: parent.map(str::to_string),
        target: "cim".to_string(),
        rows: projection.rows,
        cols: projection.cols,
        bias: projection.bias.clone(),
        tiles,
    })
}

pub fn build_package(
    normalized: &NormalizedProgram,
    lowered: &LoweredProgram,
) -> Result<MemtorchPackage, String> {
    let first_dispatch = lowered
        .program
        .entry
        .dispatches
        .first()
        .ok_or_else(|| "cannot build MemTorch package for empty cim program".to_string())?;
    let projections = build_projection_manifests(normalized, lowered)?;
    let (digital_tensors, digital) = build_digital_tensors(normalized)?;
    let execution_plan = lowered
        .execution_plan
        .iter()
        .map(|execution| ExecutionManifest {
            name: execution.name.clone(),
            stage: execution.stage.clone(),
            parent: execution.parent.clone(),
            target: execution.target.clone(),
            reason: execution.reason.clone(),
            shape: execution.shape,
            tile_count: execution.tile_count,
        })
        .collect::<Vec<_>>();

    let inference_slice = build_inference_slice(normalized);
    let simulation_summary = build_simulation_summary(&inference_slice, &projections);

    let manifest = MemtorchManifest {
        schema_version: 1,
        entry: normalized.name.clone(),
        tile_size: [first_dispatch.tile_size.rows, first_dispatch.tile_size.cols],
        quant_bits: lowered.quant_bits,
        weights_file: "memtorch_weights.bin".to_string(),
        digital_tensors_file: if digital.is_empty() {
            None
        } else {
            Some("memtorch_digital.bin".to_string())
        },
        projections,
        digital_tensors,
        execution_plan,
        attention_blocks: lowered
            .attention_blocks
            .iter()
            .map(|block| AttentionBlockManifest {
                name: block.name.clone(),
                metadata: block
                    .metadata
                    .as_ref()
                    .map(|metadata| AttentionSliceManifest {
                        source_prefix: metadata.source_prefix.clone(),
                        hidden_size: metadata.hidden_size,
                        q_dim: metadata.q_dim,
                        kv_dim: metadata.kv_dim,
                        output_dim: metadata.output_dim,
                        head_dim: metadata.head_dim,
                        q_heads: metadata.q_heads,
                        kv_heads: metadata.kv_heads,
                        grouped_query_attention: metadata.grouped_query_attention,
                    }),
                mode: block.mode.clone(),
                cim_projections: block.cim_projections.clone(),
                digital_kernels: block.digital_kernels.clone(),
                reason: block.reason.clone(),
            })
            .collect(),
        inference_slice,
        simulation_summary,
    };

    Ok(MemtorchPackage {
        manifest,
        weights: tile_payload_bytes(&lowered.tiles),
        digital,
    })
}

pub fn manifest_json(manifest: &MemtorchManifest) -> Result<String, String> {
    serde_json::to_string_pretty(manifest)
        .map(|mut json| {
            json.push('\n');
            json
        })
        .map_err(|err| format!("failed to serialize MemTorch manifest: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompileConfig;
    use crate::cim::ProjectionKind;
    use crate::ir::{NormalizedOp, ProjectionOp};
    use crate::lowering::lower_program;

    #[test]
    fn package_contains_manifest_and_weights() {
        let normalized = NormalizedProgram::new(
            "tiny",
            vec![NormalizedOp::Projection(
                ProjectionOp::new("main", ProjectionKind::Main, 1, 2, vec![1.0, -1.0], None)
                    .unwrap(),
            )],
        );
        let lowered = lower_program(&normalized, CompileConfig::square(1, 8)).unwrap();
        let package = build_package(&normalized, &lowered).unwrap();

        assert_eq!(package.manifest.schema_version, 1);
        assert_eq!(package.manifest.projections[0].tiles.len(), 2);
        assert_eq!(package.weights.len(), 2);
        assert!(
            manifest_json(&package.manifest)
                .unwrap()
                .contains("memtorch_weights.bin")
        );
    }
}
