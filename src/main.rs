mod backend;
mod frontend;
mod hardware;
mod middle;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "CiM compiler: ONNX → RISC-V assembly + crossbar weight map")]
struct Cli {
    /// Path to the ONNX model file
    onnx_path: PathBuf,

    /// Output directory for output.s and crossbar_weights.bin
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// Crossbar tile dimension — must evenly divide embed_dim (default 128×128)
    #[arg(long, default_value_t = 128)]
    tile_size: u32,
}

fn main() {
    let cli = Cli::parse();

    let spec = hardware::CrossbarSpec::new(cli.tile_size);
    let model = frontend::parse_onnx(
        cli.onnx_path
            .to_str()
            .expect("ONNX path is not valid UTF-8"),
    );
    let ops = middle::tile(model.ops, &spec);

    let asm_path = cli.output_dir.join("output.s");
    let weights_path = cli.output_dir.join("crossbar_weights.bin");

    backend::write_asm(&ops, &asm_path).expect("failed to write output.s");
    backend::write_weights(&ops, &weights_path).expect("failed to write crossbar_weights.bin");

    println!(
        "wrote {} + {} ({} tiles)",
        asm_path.display(),
        weights_path.display(),
        ops.len()
    );
}
