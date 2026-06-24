use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use prost::Message;

use crate::cim::ProjectionKind;
use crate::ir::{NormalizedOp, NormalizedProgram, ProjectionOp};

mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

const FLOAT_DTYPE: i32 = 1;
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
        let first = program.projections().next().unwrap();
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
}
