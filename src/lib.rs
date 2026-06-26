pub mod aihwkit;
pub mod cim;
pub mod frontend;
pub mod ir;
pub mod lowering;
pub mod mapping;

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileConfig {
    pub tile_rows: u32,
    pub tile_cols: u32,
}

impl CompileConfig {
    pub fn square(tile_size: u32) -> Self {
        Self {
            tile_rows: tile_size,
            tile_cols: tile_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Compilation {
    pub normalized: ir::NormalizedProgram,
    pub cim: cim::Program,
    pub aihwkit: aihwkit::AihwkitPackage,
}

pub fn compile_onnx<P: AsRef<Path>>(
    onnx_path: P,
    config: CompileConfig,
) -> Result<Compilation, String> {
    let normalized = frontend::load_onnx_program(onnx_path)?;
    let lowered = lowering::lower_program(&normalized, config)?;
    let aihwkit = aihwkit::build_package(&normalized, &lowered)?;

    Ok(Compilation {
        normalized,
        cim: lowered.program,
        aihwkit,
    })
}
