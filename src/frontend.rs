use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use prost::Message;

mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Debug)]
pub enum DType {
    BFloat16,
}

pub struct MHAWeights {
    pub wq: Vec<u8>,
    pub wk: Vec<u8>,
    pub wv: Vec<u8>,
    pub wo: Vec<u8>,
}

impl std::fmt::Debug for MHAWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MHAWeights {{ wq: {}B, wk: {}B, wv: {}B, wo: {}B }}",
            self.wq.len(),
            self.wk.len(),
            self.wv.len(),
            self.wo.len()
        )
    }
}

#[derive(Debug)]
pub enum HighLevelOp {
    MultiHeadAttention {
        _num_heads: u32,
        embed_dim: u32,
        _seq_len: u32,
        _dtype: DType,
        weights: Option<MHAWeights>,
    },
}

#[derive(Debug)]
pub struct Model {
    pub ops: Vec<HighLevelOp>,
}

const EMBED_DIM: usize = 512;
const BF16_BYTES: usize = 2;
const PROJECTION_BYTES: usize = EMBED_DIM * EMBED_DIM * BF16_BYTES;

struct TensorBlob {
    name: String,
    dims: Vec<i64>,
    data_type: i32,
    data: Vec<u8>,
}

pub fn parse_onnx<P: AsRef<Path>>(onnx_path: P) -> Result<Model, String> {
    let onnx_path = onnx_path.as_ref();
    let onnx_dir = onnx_path.parent().unwrap_or_else(|| Path::new("."));

    let bytes = std::fs::read(onnx_path)
        .map_err(|err| format!("failed to read {}: {err}", onnx_path.display()))?;
    let model_proto = onnx_proto::ModelProto::decode(bytes.as_slice())
        .map_err(|err| format!("failed to decode ONNX protobuf: {err}"))?;

    let graph = model_proto
        .graph
        .ok_or_else(|| "ONNX model has no graph".to_string())?;

    let tensors: Vec<TensorBlob> = graph
        .initializer
        .iter()
        .map(|tensor| {
            Ok(TensorBlob {
                name: tensor.name.clone(),
                dims: tensor.dims.clone(),
                data_type: tensor.data_type,
                data: read_tensor_data(tensor, onnx_dir)?,
            })
        })
        .collect::<Result<_, String>>()?;

    let weights = extract_mha_weights(&tensors)?;

    Ok(Model {
        ops: vec![HighLevelOp::MultiHeadAttention {
            _num_heads: 8,
            embed_dim: EMBED_DIM as u32,
            _seq_len: 128,
            _dtype: DType::BFloat16,
            weights: Some(weights),
        }],
    })
}

fn extract_mha_weights(tensors: &[TensorBlob]) -> Result<MHAWeights, String> {
    let mut square_weights: Vec<&TensorBlob> = tensors
        .iter()
        .filter(|t| t.dims.as_slice() == [512, 512] && t.data_type == 16)
        .collect();

    if square_weights.len() == 4 {
        for tensor in &square_weights {
            validate_len(tensor, PROJECTION_BYTES)?;
        }

        // The unrolled export emits WQ, WK, WV, WO in graph initializer order.
        return Ok(MHAWeights {
            wq: square_weights.remove(0).data.clone(),
            wk: square_weights.remove(0).data.clone(),
            wv: square_weights.remove(0).data.clone(),
            wo: square_weights.remove(0).data.clone(),
        });
    }

    let fused_qkv = find_tensor(tensors, [1536, 512], "in_proj_weight")?;
    let out_proj = square_weights
        .iter()
        .copied()
        .find(|t| t.name.ends_with("out_proj.weight"))
        .or_else(|| (square_weights.len() == 1).then(|| square_weights[0]))
        .ok_or_else(|| unsupported_initializer_error(tensors))?;

    validate_len(fused_qkv, PROJECTION_BYTES * 3)?;
    validate_len(out_proj, PROJECTION_BYTES)?;

    Ok(MHAWeights {
        wq: fused_qkv.data[0..PROJECTION_BYTES].to_vec(),
        wk: fused_qkv.data[PROJECTION_BYTES..PROJECTION_BYTES * 2].to_vec(),
        wv: fused_qkv.data[PROJECTION_BYTES * 2..PROJECTION_BYTES * 3].to_vec(),
        wo: out_proj.data.clone(),
    })
}

fn find_tensor<'a>(
    tensors: &'a [TensorBlob],
    dims: [i64; 2],
    name_suffix: &str,
) -> Result<&'a TensorBlob, String> {
    let candidates: Vec<&TensorBlob> = tensors
        .iter()
        .filter(|t| t.dims.as_slice() == dims && t.data_type == 16)
        .collect();

    candidates
        .iter()
        .copied()
        .find(|t| t.name.ends_with(name_suffix))
        .or_else(|| (candidates.len() == 1).then(|| candidates[0]))
        .ok_or_else(|| unsupported_initializer_error(tensors))
}

fn validate_len(tensor: &TensorBlob, expected: usize) -> Result<(), String> {
    if tensor.data.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "initializer `{}` with dims {:?} has {} bytes, expected {}",
            display_tensor_name(tensor),
            tensor.dims,
            tensor.data.len(),
            expected
        ))
    }
}

fn unsupported_initializer_error(tensors: &[TensorBlob]) -> String {
    format!(
        "unsupported ONNX initializer layout; expected either four [512,512] bfloat16 projection tensors or fused [1536,512] in_proj_weight plus [512,512] out_proj.weight. Found bfloat16 initializers: {}",
        tensors
            .iter()
            .filter(|t| t.data_type == 16)
            .map(|t| format!("{}{:?}", display_tensor_name(t), t.dims))
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
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join(name)
    }

    fn mha_weights(model: Model) -> MHAWeights {
        let mut ops = model.ops.into_iter();
        match ops.next().expect("expected one op") {
            HighLevelOp::MultiHeadAttention { weights, .. } => {
                weights.expect("expected MHA weights")
            }
        }
    }

    #[test]
    fn parses_unrolled_projection_initializers() {
        let model = parse_onnx(data_path("memristor_mha_unrolled.onnx")).unwrap();
        let weights = mha_weights(model);

        assert_eq!(weights.wq.len(), PROJECTION_BYTES);
        assert_eq!(weights.wk.len(), PROJECTION_BYTES);
        assert_eq!(weights.wv.len(), PROJECTION_BYTES);
        assert_eq!(weights.wo.len(), PROJECTION_BYTES);
    }

    #[test]
    fn parses_fused_qkv_initializer() {
        let model = parse_onnx(data_path("mha_bfloat16.onnx")).unwrap();
        let weights = mha_weights(model);

        assert_eq!(weights.wq.len(), PROJECTION_BYTES);
        assert_eq!(weights.wk.len(), PROJECTION_BYTES);
        assert_eq!(weights.wv.len(), PROJECTION_BYTES);
        assert_eq!(weights.wo.len(), PROJECTION_BYTES);
    }
}
