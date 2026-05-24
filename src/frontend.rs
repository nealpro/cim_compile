use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use prost::Message;
use serde::Deserialize;

mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Deserialize, Debug)]
pub enum DType {
    #[serde(rename = "bfloat16")]
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

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum HighLevelOp {
    MultiHeadAttention {
        _num_heads: u32,
        embed_dim: u32,
        _seq_len: u32,
        _dtype: DType,
        #[serde(skip)]
        weights: Option<MHAWeights>,
    },
}

#[derive(Deserialize, Debug)]
pub struct Model {
    pub ops: Vec<HighLevelOp>,
}

pub fn _parse_model(path: &str) -> Model {
    let model_file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => panic!("File could not be opened: {e}"),
    };
    match serde_json::from_reader(model_file) {
        Ok(m) => m,
        Err(e) => panic!("could not match Model data structure: {e}"),
    }
}

pub fn parse_onnx(onnx_path: &str) -> Model {
    let onnx_dir = Path::new(onnx_path)
        .parent()
        .expect("onnx path has no parent directory");

    let bytes = std::fs::read(onnx_path).expect("failed to read .onnx file");
    let model_proto =
        onnx_proto::ModelProto::decode(bytes.as_slice()).expect("failed to decode ONNX protobuf");

    let graph = model_proto.graph.expect("ONNX model has no graph");

    // Find all [512, 512] bfloat16 (data_type=16) initializers — the 4 projection weights.
    // Initializers appear in graph execution order: WQ, WK, WV, WO.
    let mut weight_blobs: Vec<Vec<u8>> = graph
        .initializer
        .iter()
        .filter(|t| t.dims == [512i64, 512i64] && t.data_type == 16)
        .map(|t| read_tensor_data(t, onnx_dir))
        .collect();

    assert_eq!(
        weight_blobs.len(),
        4,
        "expected exactly 4 [512×512] bfloat16 initializers, found {}",
        weight_blobs.len()
    );

    let wo = weight_blobs.remove(3);
    let wv = weight_blobs.remove(2);
    let wk = weight_blobs.remove(1);
    let wq = weight_blobs.remove(0);

    Model {
        ops: vec![HighLevelOp::MultiHeadAttention {
            _num_heads: 8,
            embed_dim: 512,
            _seq_len: 128,
            _dtype: DType::BFloat16,
            weights: Some(MHAWeights { wq, wk, wv, wo }),
        }],
    }
}

fn read_tensor_data(tensor: &onnx_proto::TensorProto, onnx_dir: &Path) -> Vec<u8> {
    if tensor.data_location == 1 {
        let mut location = String::new();
        let mut offset: u64 = 0;
        let mut length: u64 = 0;

        for entry in &tensor.external_data {
            match entry.key.as_str() {
                "location" => location = entry.value.clone(),
                "offset" => offset = entry.value.parse().expect("invalid external data offset"),
                "length" => length = entry.value.parse().expect("invalid external data length"),
                _ => {}
            }
        }

        let data_path = onnx_dir.join(&location);
        let mut file = std::fs::File::open(&data_path).expect("failed to open external data file");
        file.seek(SeekFrom::Start(offset)).expect("seek failed");

        let mut buf = vec![0u8; length as usize];
        file.read_exact(&mut buf)
            .expect("failed to read external data");
        buf
    } else {
        tensor.raw_data.clone()
    }
}
