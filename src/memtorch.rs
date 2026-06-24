use serde::{Deserialize, Serialize};

use crate::ir::NormalizedProgram;
use crate::lowering::{LoweredProgram, tile_payload_bytes};

#[derive(Debug, Clone, PartialEq)]
pub struct MemtorchPackage {
    pub manifest: MemtorchManifest,
    pub weights: Vec<u8>,
    pub runner: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemtorchManifest {
    pub schema_version: u32,
    pub entry: String,
    pub tile_size: [u32; 2],
    pub quant_bits: u32,
    pub weights_file: String,
    pub projections: Vec<ProjectionManifest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionManifest {
    pub name: String,
    pub projection: String,
    pub rows: u32,
    pub cols: u32,
    pub bias: Option<Vec<f32>>,
    pub tiles: Vec<TileManifest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileManifest {
    pub row: u32,
    pub col: u32,
    pub matrix_shape: [u32; 2],
    pub tile_size: [u32; 2],
    pub weight_offset: u64,
    pub quant_scale: f32,
    pub order: u32,
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
    let mut projections = Vec::new();
    for projection in normalized.projections() {
        let tiles = lowered
            .tiles
            .iter()
            .filter(|tile| tile.projection == projection.kind.to_string())
            .map(|tile| TileManifest {
                row: tile.tile.row,
                col: tile.tile.col,
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
        projections.push(ProjectionManifest {
            name: projection.name.clone(),
            projection: projection.kind.to_string(),
            rows: projection.rows,
            cols: projection.cols,
            bias: projection.bias.clone(),
            tiles,
        });
    }

    let manifest = MemtorchManifest {
        schema_version: 1,
        entry: normalized.name.clone(),
        tile_size: [first_dispatch.tile_size.rows, first_dispatch.tile_size.cols],
        quant_bits: lowered.quant_bits,
        weights_file: "memtorch_weights.bin".to_string(),
        projections,
    };

    Ok(MemtorchPackage {
        manifest,
        weights: tile_payload_bytes(&lowered.tiles),
        runner: runner_script(),
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

fn runner_script() -> String {
    r#"#!/usr/bin/env python3
import argparse
import copy
import json
import os
import struct
from pathlib import Path


def _safe_key(name):
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _configure_local_caches(anchor):
    cache_root = anchor / ".cim_compile_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def _load_tile(raw, tile):
    rows, cols = tile["tile_size"]
    count = rows * cols
    offset = tile["weight_offset"]
    payload = raw[offset:offset + count]
    if len(payload) != count:
        raise ValueError(f"tile at offset {offset} has {len(payload)} bytes, expected {count}")
    values = struct.unpack(f"{count}b", payload)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc
    return torch.tensor(values, dtype=torch.float32).reshape(rows, cols) * float(tile["quant_scale"])


def build_torch_model(manifest, raw_weights):
    import torch
    import torch.nn as nn

    class ProjectionBank(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict()
            self.key_by_name = {}

        def forward_projection(self, name, x):
            return self.layers[self.key_by_name[name]](x)

    model = ProjectionBank()
    for projection in manifest["projections"]:
        key = _safe_key(projection["name"])
        model.key_by_name[projection["name"]] = key
        layer = nn.Linear(projection["cols"], projection["rows"], bias=projection.get("bias") is not None)
        full = torch.zeros((projection["rows"], projection["cols"]), dtype=torch.float32)
        for tile in projection["tiles"]:
            tile_tensor = _load_tile(raw_weights, tile)
            row0 = tile["row"] * tile["tile_size"][0]
            col0 = tile["col"] * tile["tile_size"][1]
            row1 = min(row0 + tile["tile_size"][0], projection["rows"])
            col1 = min(col0 + tile["tile_size"][1], projection["cols"])
            full[row0:row1, col0:col1] = tile_tensor[:row1-row0, :col1-col0]
        with torch.no_grad():
            layer.weight.copy_(full)
            if projection.get("bias") is not None:
                layer.bias.copy_(torch.tensor(projection["bias"], dtype=torch.float32))
        model.layers[key] = layer
    return model


def patch_with_memtorch(model, manifest):
    try:
        import torch
        import memtorch
        from memtorch.mn.Module import patch_model
        from memtorch.map.Parameter import naive_map
        from memtorch.map.Input import naive_scale
        from memtorch.bh.crossbar.Program import naive_program
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc

    memristor_model = memtorch.bh.memristor.VTEAM
    memristor_model_params = {}
    return patch_model(
        copy.deepcopy(model),
        memristor_model=memristor_model,
        memristor_model_params=memristor_model_params,
        module_parameters_to_patch=[torch.nn.Linear],
        mapping_routine=naive_map,
        transistor=True,
        programming_routine=naive_program,
        tile_shape=tuple(manifest["tile_size"]),
        max_input_voltage=0.3,
        scaling_routine=naive_scale,
        ADC_resolution=8,
        ADC_overflow_rate=0.0,
        quant_method="linear",
    )


def main():
    parser = argparse.ArgumentParser(description="Run cim_compile output through MemTorch.")
    parser.add_argument("--manifest", default="memtorch_manifest.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-patch", action="store_true", help="Run the reconstructed PyTorch model without MemTorch patching.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    _configure_local_caches(manifest_path.parent)
    manifest = json.loads(manifest_path.read_text())
    weights_path = Path(args.weights) if args.weights else manifest_path.with_name(manifest["weights_file"])
    raw_weights = weights_path.read_bytes()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc
    torch.manual_seed(args.seed)
    model = build_torch_model(manifest, raw_weights)
    sim_model = model if args.no_patch else patch_with_memtorch(model, manifest)

    results = {}
    for projection in manifest["projections"]:
        x = torch.randn(args.batch_size, projection["cols"], dtype=torch.float32)
        with torch.no_grad():
            y = sim_model.forward_projection(projection["name"], x)
        results[projection["name"]] = {
            "shape": list(y.shape),
            "sum": float(y.sum().item()),
            "mean": float(y.mean().item()),
        }
    print(json.dumps({"entry": manifest["entry"], "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompileConfig;
    use crate::cim::ProjectionKind;
    use crate::ir::{NormalizedOp, ProjectionOp};
    use crate::lowering::lower_program;

    #[test]
    fn package_contains_manifest_weights_and_runner() {
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
        assert!(package.runner.contains("patch_model"));
        assert!(
            manifest_json(&package.manifest)
                .unwrap()
                .contains("memtorch_weights.bin")
        );
    }
}
