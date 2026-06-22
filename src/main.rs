use clap::Parser;
use std::path::PathBuf;
use std::process::Command;

use cim_compile::{CompileConfig, compile_onnx, memtorch};

#[derive(Parser)]
#[command(
    version,
    about = "Dialect-first CiM compiler: ONNX -> cim dialect -> MemTorch simulation package"
)]
struct Cli {
    /// Path to the ONNX model file
    onnx_path: PathBuf,

    /// Output directory for output.cim and MemTorch simulation artifacts
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// Crossbar tile dimension; must evenly divide each projection matrix shape
    #[arg(long, default_value_t = 128)]
    tile_size: u32,

    /// Quantization bit-width for MemTorch tile payloads (4 or 8)
    #[arg(long, default_value_t = 8)]
    bits: u32,

    /// Run the generated MemTorch script after writing artifacts
    #[arg(long)]
    run_memtorch: bool,

    /// Python executable to use with --run-memtorch
    #[arg(long, default_value = "python3")]
    python: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let compilation = compile_onnx(
        &cli.onnx_path,
        CompileConfig::square(cli.tile_size, cli.bits),
    )?;

    std::fs::create_dir_all(&cli.output_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            cli.output_dir.display()
        )
    })?;

    let cim_path = cli.output_dir.join("output.cim");
    let manifest_path = cli.output_dir.join("memtorch_manifest.json");
    let weights_path = cli.output_dir.join("memtorch_weights.bin");
    let runner_path = cli.output_dir.join("run_memtorch.py");

    std::fs::write(&cim_path, compilation.cim.to_text())
        .map_err(|err| format!("failed to write {}: {err}", cim_path.display()))?;
    std::fs::write(
        &manifest_path,
        memtorch::manifest_json(&compilation.memtorch.manifest)?,
    )
    .map_err(|err| format!("failed to write {}: {err}", manifest_path.display()))?;
    std::fs::write(&weights_path, compilation.memtorch.weights)
        .map_err(|err| format!("failed to write {}: {err}", weights_path.display()))?;
    std::fs::write(&runner_path, compilation.memtorch.runner)
        .map_err(|err| format!("failed to write {}: {err}", runner_path.display()))?;

    println!(
        "wrote {} + {} + {} + {} ({} tile dispatches)",
        cim_path.display(),
        manifest_path.display(),
        weights_path.display(),
        runner_path.display(),
        compilation.cim.entry.dispatches.len()
    );

    if cli.run_memtorch {
        let output = Command::new(&cli.python)
            .arg(&runner_path)
            .arg("--manifest")
            .arg(&manifest_path)
            .output()
            .map_err(|err| format!("failed to run {}: {err}", cli.python))?;
        if !output.status.success() {
            return Err(format!(
                "MemTorch runner failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        print!("{}", String::from_utf8_lossy(&output.stdout));
    }

    Ok(())
}
