use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use prost::Message;

use crate::cim::ProjectionKind;
use crate::ir::{
    AttentionBlock, AttentionKernel, AttentionKernels, AttentionProjections,
    AttentionSliceMetadata, AttentionStage, DigitalTensor, NormalizedOp, NormalizedProgram,
    ProjectionOp, TinyDecoderBlock, TinyDecoderMetadata,
};

mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

const FLOAT_DTYPE: i32 = 1;
const INT32_DTYPE: i32 = 6;
const INT64_DTYPE: i32 = 7;
const FLOAT16_DTYPE: i32 = 10;
const BF16_DTYPE: i32 = 16;

#[derive(Debug, Clone)]
struct TensorBlob {
    name: String,
    dims: Vec<i64>,
    data_type: i32,
    raw_data: Vec<u8>,
    float_data: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Node {
    name: String,
    op_type: String,
    input: Vec<String>,
    output: Vec<String>,
    attribute: Vec<onnx_proto::AttributeProto>,
}

pub fn load_onnx_program<P: AsRef<Path>>(onnx_path: P) -> Result<NormalizedProgram, String> {
    let onnx_path = onnx_path.as_ref();
    let onnx_dir = onnx_path.parent().unwrap_or_else(|| Path::new("."));

    let bytes = std::fs::read(onnx_path)
        .map_err(|err| format!("failed to read {}: {err}", onnx_path.display()))?;
    let model_proto = onnx_proto::ModelProto::decode(bytes.as_slice())
        .map_err(|err| format!("failed to decode ONNX protobuf: {err}"))?;
    let graph = model_proto
        .graph
        .ok_or_else(|| "ONNX model has no graph".to_string())?;

    let tensors = graph
        .initializer
        .iter()
        .map(|tensor| {
            Ok(TensorBlob {
                name: tensor.name.clone(),
                dims: tensor.dims.clone(),
                data_type: tensor.data_type,
                raw_data: read_tensor_data(tensor, onnx_dir)?,
                float_data: tensor.float_data.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let nodes = graph
        .node
        .into_iter()
        .map(|node| Node {
            name: if node.name.is_empty() {
                node.output
                    .first()
                    .cloned()
                    .unwrap_or_else(|| node.op_type.clone())
            } else {
                node.name
            },
            op_type: node.op_type,
            input: node.input,
            output: node.output,
            attribute: node.attribute,
        })
        .collect::<Vec<_>>();

    normalize_graph(
        if graph.name.is_empty() {
            "main".to_string()
        } else {
            graph.name
        },
        &tensors,
        &nodes,
    )
}

fn normalize_graph(
    graph_name: String,
    tensors: &[TensorBlob],
    nodes: &[Node],
) -> Result<NormalizedProgram, String> {
    let tensor_by_name = tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect::<BTreeMap<_, _>>();
    if let Some(block) = build_tiny_decoder_block(&graph_name, nodes, &tensor_by_name)? {
        return Ok(NormalizedProgram::new(
            graph_name,
            vec![NormalizedOp::TinyDecoder(Box::new(block))],
        ));
    }

    if let Some(block) = build_named_self_attention_block(&graph_name, nodes, &tensor_by_name)? {
        return Ok(NormalizedProgram::new(
            graph_name,
            vec![NormalizedOp::Attention(Box::new(block))],
        ));
    }

    let attention_projections = projections_from_initializers(tensors).ok();
    let is_attention_candidate = attention_projections
        .as_ref()
        .is_some_and(|projections| projections.len() == 4)
        && nodes.iter().any(|node| node.op_type == "Softmax");

    if is_attention_candidate {
        let mut ops = Vec::new();
        let block = build_attention_block(
            graph_name.clone(),
            nodes,
            attention_projections.expect("attention candidate must have projections"),
        )?;
        ops.push(NormalizedOp::Attention(Box::new(block)));

        for node in nodes {
            match node.op_type.as_str() {
                "MatMul" | "Gemm" | "Softmax" => {}
                "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" | "Identity" | "Shape"
                | "Gather" | "Concat" | "Constant" | "Split" | "Slice" | "Cast" | "Expand"
                | "Mul" | "Div" | "Add" | "Sub" => {
                    ops.push(NormalizedOp::Reshape {
                        name: node.name.clone(),
                    });
                }
                "Transpose" => {
                    ops.push(NormalizedOp::Transpose {
                        name: node.name.clone(),
                        perm: ints_attr(node, "perm").unwrap_or_default(),
                    });
                }
                other => {
                    return Err(format!(
                        "unsupported ONNX op `{other}` in node `{}`; supported ops are MatMul, Gemm, Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Identity, Add, Sub, Mul, Div, Softmax, and attention-style projection graphs",
                        node.name
                    ));
                }
            }
        }

        return Ok(NormalizedProgram::new(graph_name, ops));
    }

    let mut ops = Vec::new();
    if !nodes.is_empty() {
        let mut projection_index = 0usize;
        for node in nodes {
            match node.op_type.as_str() {
                "MatMul" => {
                    if tensor_by_name.contains_key(&node.input.get(1).cloned().unwrap_or_default())
                    {
                        ops.push(NormalizedOp::Projection(projection_from_matmul(
                            node,
                            &tensor_by_name,
                            projection_index,
                        )?));
                        projection_index += 1;
                    } else {
                        ops.push(NormalizedOp::Reshape {
                            name: node.name.clone(),
                        });
                    }
                }
                "Gemm" => {
                    ops.push(NormalizedOp::Projection(projection_from_gemm(
                        node,
                        &tensor_by_name,
                        projection_index,
                    )?));
                    projection_index += 1;
                }
                "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" | "Identity" | "Shape"
                | "Gather" | "Concat" | "Constant" | "Split" | "Slice" | "Cast" | "Expand"
                | "Mul" | "Div" | "Add" | "Sub" | "Softmax" => {
                    ops.push(NormalizedOp::Reshape {
                        name: node.name.clone(),
                    });
                }
                "Transpose" => {
                    ops.push(NormalizedOp::Transpose {
                        name: node.name.clone(),
                        perm: ints_attr(node, "perm").unwrap_or_default(),
                    });
                }
                other => {
                    return Err(format!(
                        "unsupported ONNX op `{other}` in node `{}`; supported ops are MatMul, Gemm, Reshape, Transpose, Flatten, Squeeze, Unsqueeze, and Identity",
                        node.name
                    ));
                }
            }
        }
        if ops
            .iter()
            .any(|op| matches!(op, NormalizedOp::Projection(_)))
        {
            if let Ok(fallback) = projections_from_initializers(tensors) {
                let node_projection_count = ops
                    .iter()
                    .filter(|op| matches!(op, NormalizedOp::Projection(_)))
                    .count();
                if fallback.len() > node_projection_count || fallback.len() == 4 {
                    return Ok(NormalizedProgram::new(
                        graph_name,
                        fallback
                            .into_iter()
                            .map(NormalizedOp::Projection)
                            .collect::<Vec<_>>(),
                    ));
                }
            }
            return Ok(NormalizedProgram::new(graph_name, ops));
        }
    }

    let fallback = projections_from_initializers(tensors)?;
    Ok(NormalizedProgram::new(
        graph_name,
        fallback
            .into_iter()
            .map(NormalizedOp::Projection)
            .collect::<Vec<_>>(),
    ))
}

fn build_tiny_decoder_block(
    graph_name: &str,
    nodes: &[Node],
    tensors: &BTreeMap<String, &TensorBlob>,
) -> Result<Option<TinyDecoderBlock>, String> {
    let Some(attention) = build_named_self_attention_block(graph_name, nodes, tensors)? else {
        return Ok(None);
    };
    let Some(attention_metadata) = attention.metadata.as_ref() else {
        return Ok(None);
    };
    let source_prefix = attention_metadata.source_prefix.clone();
    let Some(gate_node) = find_named_projection_node(nodes, &source_prefix, "mlp", "gate_proj")
    else {
        return Ok(None);
    };
    let up_node = find_named_projection_node(nodes, &source_prefix, "mlp", "up_proj")
        .ok_or_else(|| "tiny decoder slice is missing mlp up_proj MatMul".to_string())?;
    let down_node = find_named_projection_node(nodes, &source_prefix, "mlp", "down_proj")
        .ok_or_else(|| "tiny decoder slice is missing mlp down_proj MatMul".to_string())?;

    let gate_proj = projection_from_rhs_initializer_matmul(
        gate_node,
        tensors,
        ProjectionKind::Named("mlp0_gate".to_string()),
        "MLP gate projection",
    )?;
    let up_proj = projection_from_rhs_initializer_matmul(
        up_node,
        tensors,
        ProjectionKind::Named("mlp1_up".to_string()),
        "MLP up projection",
    )?;
    let down_proj = projection_from_rhs_initializer_matmul(
        down_node,
        tensors,
        ProjectionKind::Named("mlp2_down".to_string()),
        "MLP down projection",
    )?;

    if gate_proj.cols != attention_metadata.hidden_size
        || up_proj.cols != attention_metadata.hidden_size
    {
        return Err(format!(
            "tiny decoder MLP gate/up input dims must match hidden size {}; got gate={} up={}",
            attention_metadata.hidden_size, gate_proj.cols, up_proj.cols
        ));
    }
    if gate_proj.rows != up_proj.rows || down_proj.cols != gate_proj.rows {
        return Err(format!(
            "tiny decoder MLP dimensions are inconsistent: gate={}x{}, up={}x{}, down={}x{}",
            gate_proj.rows,
            gate_proj.cols,
            up_proj.rows,
            up_proj.cols,
            down_proj.rows,
            down_proj.cols
        ));
    }
    if down_proj.rows != attention_metadata.hidden_size {
        return Err(format!(
            "tiny decoder MLP down projection output dim {} must match hidden size {}",
            down_proj.rows, attention_metadata.hidden_size
        ));
    }

    let embedding =
        digital_tensor_from_initializer(tensors, "model.embed_tokens.weight", "token_embedding")?;
    let input_norm = digital_tensor_from_initializer(
        tensors,
        "model.layers.0.input_layernorm.weight",
        "input_layernorm_weight",
    )?;
    let post_attention_norm = digital_tensor_from_initializer(
        tensors,
        "model.layers.0.post_attention_layernorm.weight",
        "post_attention_layernorm_weight",
    )?;
    let final_norm =
        digital_tensor_from_initializer(tensors, "model.norm.weight", "final_norm_weight")?;
    let lm_head = digital_tensor_from_initializer(tensors, "onnx::MatMul_481", "lm_head_weight")?;

    let vocab_size = *embedding
        .shape
        .first()
        .ok_or_else(|| "embedding tensor is missing vocab dimension".to_string())?;
    let embedding_hidden = *embedding
        .shape
        .get(1)
        .ok_or_else(|| "embedding tensor is missing hidden dimension".to_string())?;
    if embedding_hidden != attention_metadata.hidden_size {
        return Err(format!(
            "embedding hidden size {} must match attention hidden size {}",
            embedding_hidden, attention_metadata.hidden_size
        ));
    }
    if lm_head.shape.as_slice() != [attention_metadata.hidden_size, vocab_size] {
        return Err(format!(
            "lm_head shape {:?} must be [hidden={}, vocab={}]",
            lm_head.shape, attention_metadata.hidden_size, vocab_size
        ));
    }

    let metadata = TinyDecoderMetadata {
        model_kind: "tiny_decoder_v1".to_string(),
        inference_mode: "token_ids_to_logits".to_string(),
        source_prefix: source_prefix.clone(),
        decoder_layers: 1,
        default_sequence_length: 4,
        vocab_size,
        hidden_size: attention_metadata.hidden_size,
        intermediate_size: gate_proj.rows,
        head_dim: attention_metadata.head_dim,
        q_heads: attention_metadata.q_heads,
        kv_heads: attention_metadata.kv_heads,
        grouped_query_attention: attention_metadata.grouped_query_attention,
    };

    Ok(Some(TinyDecoderBlock::new(
        "tiny_decoder_v1",
        metadata,
        attention,
        gate_proj,
        up_proj,
        down_proj,
        vec![
            embedding,
            input_norm,
            post_attention_norm,
            final_norm,
            lm_head,
        ],
    )))
}

fn build_named_self_attention_block(
    graph_name: &str,
    nodes: &[Node],
    tensors: &BTreeMap<String, &TensorBlob>,
) -> Result<Option<AttentionBlock>, String> {
    let Some(q_node) = find_self_attention_projection_node(nodes, "q_proj") else {
        return Ok(None);
    };
    let k_node = find_self_attention_projection_node(nodes, "k_proj")
        .ok_or_else(|| "named self-attention slice is missing k_proj MatMul".to_string())?;
    let v_node = find_self_attention_projection_node(nodes, "v_proj")
        .ok_or_else(|| "named self-attention slice is missing v_proj MatMul".to_string())?;
    let out_node = find_self_attention_projection_node(nodes, "o_proj")
        .ok_or_else(|| "named self-attention slice is missing o_proj MatMul".to_string())?;

    let q_index = node_index(nodes, q_node);
    let softmax_index = nodes
        .iter()
        .position(|node| node.op_type == "Softmax" && node.name.contains("/self_attn/"))
        .ok_or_else(|| "named self-attention slice is missing Softmax".to_string())?;
    let score_index = nodes[..softmax_index]
        .iter()
        .rposition(|node| {
            node.op_type == "MatMul"
                && node.name.contains("/self_attn/")
                && !node
                    .input
                    .get(1)
                    .is_some_and(|name| tensors.contains_key(name))
        })
        .ok_or_else(|| "named self-attention slice is missing dynamic score MatMul".to_string())?;
    let context_index = nodes[softmax_index + 1..]
        .iter()
        .position(|node| {
            node.op_type == "MatMul"
                && node.name.contains("/self_attn/")
                && !node
                    .input
                    .get(1)
                    .is_some_and(|name| tensors.contains_key(name))
        })
        .map(|relative| softmax_index + 1 + relative)
        .ok_or_else(|| {
            "named self-attention slice is missing dynamic context MatMul".to_string()
        })?;

    let q_proj = projection_from_attention_matmul(q_node, tensors, ProjectionKind::WQ)?;
    let k_proj = projection_from_attention_matmul(k_node, tensors, ProjectionKind::WK)?;
    let v_proj = projection_from_attention_matmul(v_node, tensors, ProjectionKind::WV)?;
    let out_proj = projection_from_attention_matmul(out_node, tensors, ProjectionKind::WO)?;

    if q_proj.cols != k_proj.cols || q_proj.cols != v_proj.cols {
        return Err(format!(
            "self-attention projections must share an input hidden size; got q={} k={} v={}",
            q_proj.cols, k_proj.cols, v_proj.cols
        ));
    }
    if k_proj.rows != v_proj.rows {
        return Err(format!(
            "self-attention k/v projections must share output size; got k={} v={}",
            k_proj.rows, v_proj.rows
        ));
    }
    if out_proj.cols != q_proj.rows {
        return Err(format!(
            "self-attention output projection input size {} must match q projection output size {}",
            out_proj.cols, q_proj.rows
        ));
    }

    let source_prefix =
        self_attention_source_prefix(&q_node.name).unwrap_or_else(|| graph_name.to_string());
    let block_name = if source_prefix.ends_with("/self_attn") {
        source_prefix.clone()
    } else {
        format!("{source_prefix}/self_attn")
    };
    let head_dim = infer_head_dim(nodes, q_node, k_node).unwrap_or_else(|| {
        if q_proj.rows.is_multiple_of(k_proj.rows) {
            k_proj.rows
        } else {
            gcd(q_proj.rows, k_proj.rows)
        }
    });
    let q_heads = checked_heads(q_proj.rows, head_dim, "q")?;
    let kv_heads = checked_heads(k_proj.rows, head_dim, "kv")?;
    let grouped_query_attention = q_heads != kv_heads;

    let mut digital_fallbacks = Vec::new();
    if grouped_query_attention {
        digital_fallbacks.push(AttentionKernel::new(
            format!("{block_name}/repeat_kv"),
            AttentionStage::RepeatKv,
            Some([q_heads, kv_heads]),
        ));
    }
    if has_attention_mask_glue(&nodes[score_index..softmax_index]) {
        digital_fallbacks.push(AttentionKernel::new(
            format!("{block_name}/attention_mask"),
            AttentionStage::AttentionMask,
            None,
        ));
    }

    let metadata = AttentionSliceMetadata {
        source_prefix,
        hidden_size: q_proj.cols,
        q_dim: q_proj.rows,
        kv_dim: k_proj.rows,
        output_dim: out_proj.rows,
        head_dim,
        q_heads,
        kv_heads,
        grouped_query_attention,
    };

    let residual = nodes
        .iter()
        .skip(node_index(nodes, out_node))
        .find(|node| node.op_type == "Add")
        .map(|node| node.name.clone());

    let block = AttentionBlock::new(
        block_name,
        AttentionProjections {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        },
        AttentionKernels {
            score_matmul: AttentionKernel::new(
                nodes[score_index].name.clone(),
                AttentionStage::ScoreMatMul,
                None,
            ),
            softmax: AttentionKernel::new(
                nodes[softmax_index].name.clone(),
                AttentionStage::Softmax,
                None,
            ),
            context_matmul: AttentionKernel::new(
                nodes[context_index].name.clone(),
                AttentionStage::ContextMatMul,
                None,
            ),
        },
        residual,
    )
    .with_metadata(metadata)
    .with_digital_fallbacks(digital_fallbacks);

    if q_index > softmax_index {
        return Err("self-attention q projection appears after Softmax".to_string());
    }

    Ok(Some(block))
}

fn find_self_attention_projection_node<'a>(
    nodes: &'a [Node],
    projection: &str,
) -> Option<&'a Node> {
    let marker = format!("/self_attn/{projection}/");
    nodes.iter().find(|node| {
        node.op_type == "MatMul" && node.name.contains(&marker) && node.input.len() >= 2
    })
}

fn find_named_projection_node<'a>(
    nodes: &'a [Node],
    source_prefix: &str,
    block: &str,
    projection: &str,
) -> Option<&'a Node> {
    let marker = format!("{source_prefix}/{block}/{projection}/");
    nodes.iter().find(|node| {
        node.op_type == "MatMul" && node.name.contains(&marker) && node.input.len() >= 2
    })
}

fn projection_from_attention_matmul(
    node: &Node,
    tensors: &BTreeMap<String, &TensorBlob>,
    kind: ProjectionKind,
) -> Result<ProjectionOp, String> {
    projection_from_rhs_initializer_matmul(node, tensors, kind, "attention projection")
}

fn projection_from_rhs_initializer_matmul(
    node: &Node,
    tensors: &BTreeMap<String, &TensorBlob>,
    kind: ProjectionKind,
    context: &str,
) -> Result<ProjectionOp, String> {
    if node.input.len() < 2 {
        return Err(format!("MatMul node `{}` must have two inputs", node.name));
    }
    let rhs = tensors.get(&node.input[1]).ok_or_else(|| {
        format!(
            "MatMul node `{}` is supported as a {context} only when its RHS input is an initializer",
            node.name,
        )
    })?;
    let (rows, cols, values) = matrix_values(rhs)?;
    let values = transpose(&values, rows, cols);
    ProjectionOp::new(&node.name, kind, cols, rows, values, None)
}

fn digital_tensor_from_initializer(
    tensors: &BTreeMap<String, &TensorBlob>,
    name: &str,
    role: &str,
) -> Result<DigitalTensor, String> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| format!("tiny decoder slice is missing initializer `{name}`"))?;
    let shape = tensor_shape(tensor)?;
    let values = tensor_values(tensor)?;
    DigitalTensor::new(display_tensor_name(tensor), role, shape, values)
}

fn self_attention_source_prefix(name: &str) -> Option<String> {
    name.split_once("/self_attn/")
        .map(|(prefix, _)| prefix.to_string())
}

fn node_index(nodes: &[Node], needle: &Node) -> usize {
    nodes
        .iter()
        .position(|node| std::ptr::eq(node, needle))
        .expect("node reference came from nodes")
}

fn checked_heads(dim: u32, head_dim: u32, label: &str) -> Result<u32, String> {
    if head_dim == 0 || !dim.is_multiple_of(head_dim) {
        return Err(format!(
            "cannot infer {label} head count from dim {dim} and head_dim {head_dim}"
        ));
    }
    Ok(dim / head_dim)
}

fn infer_head_dim(nodes: &[Node], q_node: &Node, k_node: &Node) -> Option<u32> {
    infer_head_dim_from_projection_reshape(nodes, q_node)
        .or_else(|| infer_head_dim_from_projection_reshape(nodes, k_node))
}

fn infer_head_dim_from_projection_reshape(nodes: &[Node], projection_node: &Node) -> Option<u32> {
    let constants = constant_int_outputs(nodes);
    let projection_output = projection_node.output.first()?;
    let reshape = nodes
        .iter()
        .find(|node| node.op_type == "Reshape" && node.input.first() == Some(projection_output))?;
    let shape_input = reshape.input.get(1)?;
    if let Some(values) = constants.get(shape_input)
        && let Some(value) = values.iter().rev().find(|value| **value > 0)
    {
        return u32::try_from(*value).ok();
    }

    let concat = nodes.iter().find(|node| {
        node.op_type == "Concat" && node.output.iter().any(|output| output == shape_input)
    })?;
    concat.input.iter().rev().find_map(|input| {
        constants.get(input).and_then(|values| {
            values
                .iter()
                .rev()
                .find(|value| **value > 0)
                .and_then(|value| u32::try_from(*value).ok())
        })
    })
}

fn constant_int_outputs(nodes: &[Node]) -> BTreeMap<String, Vec<i64>> {
    let mut constants = BTreeMap::new();
    for node in nodes {
        if node.op_type != "Constant" {
            continue;
        }
        let Some(output) = node.output.first() else {
            continue;
        };
        let Some(values) = node
            .attribute
            .iter()
            .find(|attr| attr.name == "value")
            .and_then(|attr| attr.t.as_ref())
            .and_then(tensor_int_values)
        else {
            continue;
        };
        constants.insert(output.clone(), values);
    }
    constants
}

fn tensor_int_values(tensor: &onnx_proto::TensorProto) -> Option<Vec<i64>> {
    match tensor.data_type {
        INT64_DTYPE => {
            if tensor.raw_data.is_empty() {
                None
            } else {
                Some(
                    tensor
                        .raw_data
                        .chunks_exact(8)
                        .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("chunk length")))
                        .collect(),
                )
            }
        }
        INT32_DTYPE => {
            if !tensor.raw_data.is_empty() {
                Some(
                    tensor
                        .raw_data
                        .chunks_exact(4)
                        .map(|bytes| {
                            i32::from_le_bytes(bytes.try_into().expect("chunk length")) as i64
                        })
                        .collect(),
                )
            } else {
                Some(
                    tensor
                        .int32_data
                        .iter()
                        .map(|value| *value as i64)
                        .collect(),
                )
            }
        }
        _ => None,
    }
}

fn has_attention_mask_glue(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| {
        node.name.contains("/self_attn/") && matches!(node.op_type.as_str(), "Where" | "Add")
    })
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.max(1)
}

fn build_attention_block(
    graph_name: String,
    nodes: &[Node],
    projections: Vec<ProjectionOp>,
) -> Result<AttentionBlock, String> {
    let mut projection_map = BTreeMap::new();
    for projection in projections {
        projection_map.insert(projection.kind.clone(), projection);
    }

    let q_proj = projection_map
        .remove(&ProjectionKind::WQ)
        .ok_or_else(|| "attention graph is missing a q projection".to_string())?;
    let k_proj = projection_map
        .remove(&ProjectionKind::WK)
        .ok_or_else(|| "attention graph is missing a k projection".to_string())?;
    let v_proj = projection_map
        .remove(&ProjectionKind::WV)
        .ok_or_else(|| "attention graph is missing a v projection".to_string())?;
    let out_proj = projection_map
        .remove(&ProjectionKind::WO)
        .ok_or_else(|| "attention graph is missing an out projection".to_string())?;

    let score_matmul = AttentionKernel::new(
        format!("{graph_name}_score_matmul"),
        AttentionStage::ScoreMatMul,
        None,
    );
    let softmax = AttentionKernel::new(
        format!("{graph_name}_softmax"),
        AttentionStage::Softmax,
        None,
    );
    let context_matmul = AttentionKernel::new(
        format!("{graph_name}_context_matmul"),
        AttentionStage::ContextMatMul,
        None,
    );
    let residual = nodes
        .iter()
        .find(|node| node.op_type == "Add")
        .map(|node| node.name.clone());

    Ok(AttentionBlock::new(
        graph_name,
        AttentionProjections {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        },
        AttentionKernels {
            score_matmul,
            softmax,
            context_matmul,
        },
        residual,
    ))
}

fn projection_from_matmul(
    node: &Node,
    tensors: &BTreeMap<String, &TensorBlob>,
    index: usize,
) -> Result<ProjectionOp, String> {
    if node.input.len() < 2 {
        return Err(format!("MatMul node `{}` must have two inputs", node.name));
    }
    let rhs = tensors.get(&node.input[1]).ok_or_else(|| {
        format!(
            "MatMul node `{}` is supported only when its RHS input is an initializer",
            node.name
        )
    })?;
    let (rows, cols, values) = matrix_values(rhs)?;
    let values = transpose(&values, rows, cols);
    ProjectionOp::new(
        &node.name,
        infer_projection_kind(&node.name, index),
        cols,
        rows,
        values,
        None,
    )
}

fn projection_from_gemm(
    node: &Node,
    tensors: &BTreeMap<String, &TensorBlob>,
    index: usize,
) -> Result<ProjectionOp, String> {
    if node.input.len() < 2 {
        return Err(format!(
            "Gemm node `{}` must have at least two inputs",
            node.name
        ));
    }
    let alpha = float_attr(node, "alpha").unwrap_or(1.0);
    let beta = float_attr(node, "beta").unwrap_or(1.0);
    if alpha != 1.0 || beta != 1.0 {
        return Err(format!(
            "Gemm node `{}` uses alpha={alpha} beta={beta}; only alpha=1 beta=1 is supported",
            node.name
        ));
    }
    let weight = tensors.get(&node.input[1]).ok_or_else(|| {
        format!(
            "Gemm node `{}` is supported only when its weight input is an initializer",
            node.name
        )
    })?;
    let (rows, cols, values) = matrix_values(weight)?;
    let trans_b = int_attr(node, "transB").unwrap_or(0);
    let (out_rows, out_cols, values) = if trans_b == 0 {
        (cols, rows, transpose(&values, rows, cols))
    } else {
        (rows, cols, values)
    };
    let bias = if let Some(name) = node.input.get(2) {
        if name.is_empty() {
            None
        } else {
            Some(vector_values(tensors.get(name).ok_or_else(|| {
                format!(
                    "Gemm node `{}` bias input `{name}` is not an initializer",
                    node.name
                )
            })?)?)
        }
    } else {
        None
    };
    ProjectionOp::new(
        &node.name,
        infer_projection_kind(&node.name, index),
        out_rows,
        out_cols,
        values,
        bias,
    )
}

fn projections_from_initializers(tensors: &[TensorBlob]) -> Result<Vec<ProjectionOp>, String> {
    let rank2: Vec<&TensorBlob> = tensors
        .iter()
        .filter(|tensor| is_supported_float_dtype(tensor.data_type) && tensor.dims.len() == 2)
        .collect();

    if let Some(projections) = fused_qkv_projection(&rank2)? {
        return Ok(projections);
    }

    let square = rank2
        .iter()
        .copied()
        .filter(|tensor| dims2(tensor).is_ok_and(|(rows, cols)| rows == cols))
        .collect::<Vec<_>>();
    if square.len() == 4 {
        let kinds = [
            ProjectionKind::WQ,
            ProjectionKind::WK,
            ProjectionKind::WV,
            ProjectionKind::WO,
        ];
        return square
            .into_iter()
            .zip(kinds)
            .map(|(tensor, kind)| {
                let (rows, cols, values) = matrix_values(tensor)?;
                ProjectionOp::new(display_tensor_name(tensor), kind, rows, cols, values, None)
            })
            .collect();
    }

    if rank2.len() == 1 {
        let (rows, cols, values) = matrix_values(rank2[0])?;
        return Ok(vec![ProjectionOp::new(
            display_tensor_name(rank2[0]),
            ProjectionKind::Main,
            rows,
            cols,
            values,
            None,
        )?]);
    }

    Err(unsupported_initializer_error(tensors))
}

fn fused_qkv_projection(tensors: &[&TensorBlob]) -> Result<Option<Vec<ProjectionOp>>, String> {
    let fused = tensors.iter().copied().find(|tensor| {
        dims2(tensor).is_ok_and(|(rows, cols)| rows == cols * 3)
            && tensor.name.ends_with("in_proj_weight")
    });
    let Some(fused) = fused else {
        return Ok(None);
    };
    let (_, cols, values) = matrix_values(fused)?;
    let out = tensors
        .iter()
        .copied()
        .find(|tensor| {
            dims2(tensor).ok() == Some((cols, cols)) && tensor.name.ends_with("out_proj.weight")
        })
        .ok_or_else(|| unsupported_initializer_error_from_refs(tensors))?;
    let (_, _, out_values) = matrix_values(out)?;
    let one = (cols * cols) as usize;

    Ok(Some(vec![
        ProjectionOp::new(
            "wq",
            ProjectionKind::WQ,
            cols,
            cols,
            values[0..one].to_vec(),
            None,
        )?,
        ProjectionOp::new(
            "wk",
            ProjectionKind::WK,
            cols,
            cols,
            values[one..one * 2].to_vec(),
            None,
        )?,
        ProjectionOp::new(
            "wv",
            ProjectionKind::WV,
            cols,
            cols,
            values[one * 2..one * 3].to_vec(),
            None,
        )?,
        ProjectionOp::new("wo", ProjectionKind::WO, cols, cols, out_values, None)?,
    ]))
}

fn matrix_values(tensor: &TensorBlob) -> Result<(u32, u32, Vec<f32>), String> {
    let (rows, cols) = dims2(tensor)?;
    let values = tensor_values(tensor)?;
    let expected = rows as usize * cols as usize;
    if values.len() != expected {
        return Err(format!(
            "initializer `{}` with dims {:?} has {} values, expected {}",
            display_tensor_name(tensor),
            tensor.dims,
            values.len(),
            expected
        ));
    }
    Ok((rows, cols, values))
}

fn vector_values(tensor: &TensorBlob) -> Result<Vec<f32>, String> {
    if tensor.dims.len() != 1 {
        return Err(format!(
            "initializer `{}` must be rank-1 bias, found dims {:?}",
            display_tensor_name(tensor),
            tensor.dims
        ));
    }
    tensor_values(tensor)
}

fn tensor_values(tensor: &TensorBlob) -> Result<Vec<f32>, String> {
    match tensor.data_type {
        FLOAT_DTYPE => {
            if !tensor.raw_data.is_empty() {
                if !tensor.raw_data.len().is_multiple_of(4) {
                    return Err(format!(
                        "float initializer `{}` has invalid raw byte length {}",
                        display_tensor_name(tensor),
                        tensor.raw_data.len()
                    ));
                }
                Ok(tensor
                    .raw_data
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("chunk length")))
                    .collect())
            } else {
                Ok(tensor.float_data.clone())
            }
        }
        BF16_DTYPE => {
            if !tensor.raw_data.len().is_multiple_of(2) {
                return Err(format!(
                    "bfloat16 initializer `{}` has invalid raw byte length {}",
                    display_tensor_name(tensor),
                    tensor.raw_data.len()
                ));
            }
            Ok(tensor
                .raw_data
                .chunks_exact(2)
                .map(|bytes| {
                    let bf16 = u16::from_le_bytes([bytes[0], bytes[1]]);
                    f32::from_bits((bf16 as u32) << 16)
                })
                .collect())
        }
        FLOAT16_DTYPE => {
            if !tensor.raw_data.len().is_multiple_of(2) {
                return Err(format!(
                    "float16 initializer `{}` has invalid raw byte length {}",
                    display_tensor_name(tensor),
                    tensor.raw_data.len()
                ));
            }
            Ok(tensor
                .raw_data
                .chunks_exact(2)
                .map(|bytes| f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
                .collect())
        }
        other => Err(format!(
            "initializer `{}` uses unsupported data_type {}; expected float32, float16, or bfloat16",
            display_tensor_name(tensor),
            other
        )),
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = (bits & 0x03ff) as u32;
    let f32_bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac = frac;
                let mut exp = -14i32;
                while (frac & 0x0400) == 0 {
                    frac <<= 1;
                    exp -= 1;
                }
                frac &= 0x03ff;
                sign | (((exp + 127) as u32) << 23) | (frac << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | (((exp as u32) + 112) << 23) | (frac << 13),
    };
    f32::from_bits(f32_bits)
}

fn transpose(values: &[f32], rows: u32, cols: u32) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for row in 0..rows as usize {
        for col in 0..cols as usize {
            out[col * rows as usize + row] = values[row * cols as usize + col];
        }
    }
    out
}

fn dims2(tensor: &TensorBlob) -> Result<(u32, u32), String> {
    if tensor.dims.len() != 2 {
        return Err(format!(
            "initializer `{}` has rank {}, expected rank 2",
            display_tensor_name(tensor),
            tensor.dims.len()
        ));
    }
    let rows = u32::try_from(tensor.dims[0]).map_err(|_| {
        format!(
            "initializer `{}` has invalid row dimension {}",
            display_tensor_name(tensor),
            tensor.dims[0]
        )
    })?;
    let cols = u32::try_from(tensor.dims[1]).map_err(|_| {
        format!(
            "initializer `{}` has invalid column dimension {}",
            display_tensor_name(tensor),
            tensor.dims[1]
        )
    })?;
    Ok((rows, cols))
}

fn tensor_shape(tensor: &TensorBlob) -> Result<Vec<u32>, String> {
    tensor
        .dims
        .iter()
        .map(|dim| {
            u32::try_from(*dim).map_err(|_| {
                format!(
                    "initializer `{}` has invalid dimension {}",
                    display_tensor_name(tensor),
                    dim
                )
            })
        })
        .collect()
}

fn is_supported_float_dtype(dtype: i32) -> bool {
    matches!(dtype, FLOAT_DTYPE | FLOAT16_DTYPE | BF16_DTYPE)
}

fn infer_projection_kind(name: &str, index: usize) -> ProjectionKind {
    let lower = name.to_ascii_lowercase();
    if lower.contains("q_proj") || lower.ends_with(".q") || lower.contains("query") {
        ProjectionKind::WQ
    } else if lower.contains("k_proj") || lower.ends_with(".k") || lower.contains("key") {
        ProjectionKind::WK
    } else if lower.contains("v_proj") || lower.ends_with(".v") || lower.contains("value") {
        ProjectionKind::WV
    } else if lower.contains("out_proj") || lower.contains("output") {
        ProjectionKind::WO
    } else if index == 0 {
        ProjectionKind::Main
    } else {
        ProjectionKind::Named(format!("linear_{index}"))
    }
}

fn int_attr(node: &Node, name: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .map(|attr| attr.i)
}

fn float_attr(node: &Node, name: &str) -> Option<f32> {
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .map(|attr| attr.f)
}

fn ints_attr(node: &Node, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .map(|attr| attr.ints.clone())
}

fn unsupported_initializer_error_from_refs(tensors: &[&TensorBlob]) -> String {
    let owned = tensors.iter().copied().cloned().collect::<Vec<_>>();
    unsupported_initializer_error(&owned)
}

fn unsupported_initializer_error(tensors: &[TensorBlob]) -> String {
    format!(
        "unsupported ONNX initializer layout; expected one rank-2 float projection tensor, four square projection tensors, or fused [3N,N] in_proj_weight plus [N,N] out_proj.weight. Found float initializers: {}",
        tensors
            .iter()
            .filter(|tensor| is_supported_float_dtype(tensor.data_type))
            .map(|tensor| format!("{}{:?}", display_tensor_name(tensor), tensor.dims))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn display_tensor_name(tensor: &TensorBlob) -> &str {
    if tensor.name.is_empty() {
        "<unnamed>"
    } else {
        &tensor.name
    }
}

fn read_tensor_data(tensor: &onnx_proto::TensorProto, onnx_dir: &Path) -> Result<Vec<u8>, String> {
    if tensor.data_location == 1 {
        let mut location = String::new();
        let mut offset: u64 = 0;
        let mut length: u64 = 0;

        for entry in &tensor.external_data {
            match entry.key.as_str() {
                "location" => location = entry.value.clone(),
                "offset" => {
                    offset = entry.value.parse().map_err(|err| {
                        format!("invalid external data offset for `{}`: {err}", tensor.name)
                    })?
                }
                "length" => {
                    length = entry.value.parse().map_err(|err| {
                        format!("invalid external data length for `{}`: {err}", tensor.name)
                    })?
                }
                _ => {}
            }
        }

        if location.is_empty() {
            return Err(format!(
                "external initializer `{}` is missing a location",
                tensor.name
            ));
        }

        let data_path = onnx_dir.join(&location);
        let mut file = std::fs::File::open(&data_path)
            .map_err(|err| format!("failed to open {}: {err}", data_path.display()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|err| format!("failed to seek {}: {err}", data_path.display()))?;

        let mut buf = vec![0u8; length as usize];
        file.read_exact(&mut buf)
            .map_err(|err| format!("failed to read {}: {err}", data_path.display()))?;
        Ok(buf)
    } else {
        Ok(tensor.raw_data.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_path(name: &str) -> std::path::PathBuf {
        fixture_dir().join(name)
    }

    fn required_data_model_path() -> std::path::PathBuf {
        if let Ok(path) = std::env::var("CIM_COMPILE_REAL_MODEL") {
            return Path::new(&path).to_path_buf();
        }
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("model.onnx")
    }

    fn fixture_dir() -> std::path::PathBuf {
        use std::process::Command;
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!(
            "cim_compile_frontend_fixtures_{}_{}",
            std::process::id(),
            now
        ));
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("generate_onnx_fixtures.py");
        let output = Command::new(test_python())
            .arg(&script)
            .arg("--output-dir")
            .arg(&out_dir)
            .output()
            .expect("failed to run fixture generator");
        assert!(
            output.status.success(),
            "fixture generation failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        out_dir
    }

    fn test_python() -> String {
        if let Ok(python) = std::env::var("CIM_COMPILE_PYTHON") {
            return python;
        }
        let local = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".venv")
            .join("bin")
            .join("python");
        if local.exists() {
            local.display().to_string()
        } else {
            "python3".to_string()
        }
    }

    #[test]
    fn parses_unrolled_projection_initializers() {
        let program = load_onnx_program(data_path("memristor_mha_unrolled.onnx")).unwrap();

        assert_eq!(program.projection_count(), 4);
        let projections = program.projections();
        let first = projections.first().unwrap();
        assert_eq!(first.kind, ProjectionKind::WQ);
        assert_eq!(first.rows, 512);
        assert_eq!(first.cols, 512);
        assert_eq!(first.weights.len(), 512 * 512);
    }

    #[test]
    fn parses_fused_qkv_initializer() {
        let program = load_onnx_program(data_path("mha_bfloat16.onnx")).unwrap();
        let kinds = program
            .projections()
            .into_iter()
            .map(|projection| projection.kind.clone())
            .collect::<Vec<_>>();

        assert_eq!(
            kinds,
            vec![
                ProjectionKind::WQ,
                ProjectionKind::WK,
                ProjectionKind::WV,
                ProjectionKind::WO
            ]
        );
    }

    #[test]
    #[ignore = "requires a local real ONNX model; run with CIM_COMPILE_REAL_MODEL=/path/to/model.onnx cargo test -- --ignored"]
    fn extracts_real_tiny_model_token_logits_slice() {
        let path = required_data_model_path();
        assert!(
            path.exists(),
            "real-model fixture is missing: {}. Set CIM_COMPILE_REAL_MODEL=/path/to/model.onnx when running ignored full-model tests.",
            path.display()
        );
        let program = load_onnx_program(path).unwrap();

        assert_eq!(program.ops.len(), 1);
        let NormalizedOp::TinyDecoder(decoder) = &program.ops[0] else {
            panic!("expected real model to normalize as one tiny decoder block");
        };
        let block = &decoder.attention;
        assert_eq!(block.name, "/model/layers.0/self_attn");
        assert_eq!(block.q_proj.rows, 192);
        assert_eq!(block.q_proj.cols, 192);
        assert_eq!(block.k_proj.rows, 96);
        assert_eq!(block.k_proj.cols, 192);
        assert_eq!(block.v_proj.rows, 96);
        assert_eq!(block.v_proj.cols, 192);
        assert_eq!(block.out_proj.rows, 192);
        assert_eq!(block.out_proj.cols, 192);

        let metadata = block.metadata.as_ref().expect("missing attention metadata");
        assert_eq!(metadata.hidden_size, 192);
        assert_eq!(metadata.q_dim, 192);
        assert_eq!(metadata.kv_dim, 96);
        assert_eq!(metadata.output_dim, 192);
        assert_eq!(metadata.head_dim, 96);
        assert_eq!(metadata.q_heads, 2);
        assert_eq!(metadata.kv_heads, 1);
        assert!(metadata.grouped_query_attention);
        assert!(
            block
                .kernel_entries()
                .iter()
                .any(|kernel| kernel.stage == AttentionStage::RepeatKv)
        );
        assert!(
            block
                .kernel_entries()
                .iter()
                .any(|kernel| kernel.stage == AttentionStage::AttentionMask)
        );

        assert_eq!(decoder.metadata.model_kind, "tiny_decoder_v1");
        assert_eq!(decoder.metadata.inference_mode, "token_ids_to_logits");
        assert_eq!(decoder.metadata.vocab_size, 32000);
        assert_eq!(decoder.metadata.hidden_size, 192);
        assert_eq!(decoder.metadata.intermediate_size, 1024);
        assert_eq!(decoder.metadata.decoder_layers, 1);
        assert_eq!(decoder.mlp_gate_proj.rows, 1024);
        assert_eq!(decoder.mlp_gate_proj.cols, 192);
        assert_eq!(decoder.mlp_up_proj.rows, 1024);
        assert_eq!(decoder.mlp_up_proj.cols, 192);
        assert_eq!(decoder.mlp_down_proj.rows, 192);
        assert_eq!(decoder.mlp_down_proj.cols, 1024);
        assert_eq!(decoder.digital_tensors.len(), 5);
        assert!(
            decoder
                .digital_tensors
                .iter()
                .any(|tensor| tensor.role == "token_embedding" && tensor.shape == [32000, 192])
        );
        assert!(
            decoder
                .digital_tensors
                .iter()
                .any(|tensor| tensor.role == "lm_head_weight" && tensor.shape == [192, 32000])
        );
    }
}
