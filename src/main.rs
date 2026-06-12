mod backend;
mod frontend;
mod hardware;
mod middle;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about = "CiM compiler: ONNX → RISC-V assembly + crossbar weight map"
)]
struct Cli {
    /// Path to the ONNX model file
    onnx_path: PathBuf,

    /// Output directory for output.s and crossbar_weights.bin
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// Crossbar tile dimension — must evenly divide embed_dim (default 128×128)
    #[arg(long, default_value_t = 128)]
    tile_size: u32,

    /// Quantization bit-width for crossbar weights (4 or 8; RRAM devices support ≤8 bits)
    #[arg(long, default_value_t = 8)]
    bits: u32,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();

    if cli.bits != 4 && cli.bits != 8 {
        return Err(format!(
            "unsupported quantization bit-width {}; expected 4 or 8",
            cli.bits
        ));
    }

    let spec = hardware::CrossbarSpec::new(cli.tile_size);
    let model = frontend::parse_onnx(&cli.onnx_path)?;
    let ops = middle::tile(model.ops, &spec)?;
    let ops = middle::quantize(ops, cli.bits)?;

    std::fs::create_dir_all(&cli.output_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            cli.output_dir.display()
        )
    })?;

    let asm_path = cli.output_dir.join("output.s");
    let weights_path = cli.output_dir.join("crossbar_weights.bin");

    backend::write_asm(&ops, &asm_path)
        .map_err(|err| format!("failed to write {}: {err}", asm_path.display()))?;
    backend::write_weights(&ops, &weights_path, &spec)
        .map_err(|err| format!("failed to write {}: {err}", weights_path.display()))?;

    println!(
        "wrote {} + {} ({} tiles)",
        asm_path.display(),
        weights_path.display(),
        ops.len()
    );

    Ok(())
}
