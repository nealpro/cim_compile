use clap::Parser;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

use cim_compile::{CompileConfig, compile_onnx, memtorch};

#[derive(Parser)]
#[command(
    version,
    about = "Transformer CiM compiler: ONNX tiny-decoder slice -> cim dialect -> MemTorch simulation package"
)]
struct Cli {
    /// Path to the ONNX model file
    onnx_path: PathBuf,

    /// Output directory for output.cim and MemTorch simulation artifacts
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// Crossbar tile dimension; edge tiles are zero-padded when needed
    #[arg(long, default_value_t = 128)]
    tile_size: u32,

    /// Quantization bit-width for MemTorch tile payloads (4 or 8)
    #[arg(long, default_value_t = 8)]
    bits: u32,

    /// Run the MemTorch simulation bridge after writing artifacts
    #[arg(long)]
    run_memtorch: bool,

    /// Python executable to use with --run-memtorch
    #[arg(long)]
    python: Option<String>,

    /// Comma-separated token IDs for --run-memtorch token-logits inference
    #[arg(long, default_value = "1,2,3,4")]
    input_ids: String,

    /// Number of next-token candidates to report from --run-memtorch
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Generate token IDs greedily with the MemTorch bridge
    #[arg(long)]
    generate_ids: bool,

    /// Maximum number of token IDs to generate with --generate-ids
    #[arg(long, default_value_t = 8)]
    max_new_tokens: usize,

    /// Optional token ID that stops --generate-ids early
    #[arg(long)]
    eos_token_id: Option<i64>,
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
    let digital_path = cli.output_dir.join("memtorch_digital.bin");

    std::fs::write(&cim_path, compilation.cim.to_text())
        .map_err(|err| format!("failed to write {}: {err}", cim_path.display()))?;
    std::fs::write(
        &manifest_path,
        memtorch::manifest_json(&compilation.memtorch.manifest)?,
    )
    .map_err(|err| format!("failed to write {}: {err}", manifest_path.display()))?;
    std::fs::write(&weights_path, &compilation.memtorch.weights)
        .map_err(|err| format!("failed to write {}: {err}", weights_path.display()))?;
    let wrote_digital = !compilation.memtorch.digital.is_empty();
    if wrote_digital {
        std::fs::write(&digital_path, &compilation.memtorch.digital)
            .map_err(|err| format!("failed to write {}: {err}", digital_path.display()))?;
    }

    if wrote_digital {
        println!(
            "wrote {} + {} + {} + {} ({} tile dispatches)",
            cim_path.display(),
            manifest_path.display(),
            weights_path.display(),
            digital_path.display(),
            compilation.cim.entry.dispatches.len()
        );
    } else {
        println!(
            "wrote {} + {} + {} ({} tile dispatches)",
            cim_path.display(),
            manifest_path.display(),
            weights_path.display(),
            compilation.cim.entry.dispatches.len()
        );
    }

    if cli.run_memtorch {
        let python = cli.python.clone().unwrap_or_else(default_python);
        let mut command = Command::new(&python);
        command
            .arg("-m")
            .arg("cim_compile_memtorch.runner")
            .arg("--manifest")
            .arg(&manifest_path)
            .arg("--input-ids")
            .arg(&cli.input_ids)
            .arg("--top-k")
            .arg(cli.top_k.to_string())
            .arg("--max-new-tokens")
            .arg(cli.max_new_tokens.to_string())
            .env("PYTHONPATH", python_path_env()?);
        if cli.generate_ids {
            command.arg("--generate-ids");
        }
        if let Some(eos_token_id) = cli.eos_token_id {
            command.arg("--eos-token-id").arg(eos_token_id.to_string());
        }
        let output = command
            .output()
            .map_err(|err| format!("failed to run {python}: {err}"))?;
        if !output.status.success() {
            return Err(format!(
                "MemTorch bridge failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        print!("{}", String::from_utf8_lossy(&output.stdout));
    }

    Ok(())
}

fn default_python() -> String {
    let local_venv = PathBuf::from(".venv").join("bin").join("python");
    if local_venv.exists() {
        local_venv.display().to_string()
    } else {
        "python3".to_string()
    }
}

fn python_path_env() -> Result<OsString, String> {
    let mut paths = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")];
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths).map_err(|err| format!("failed to build PYTHONPATH: {err}"))
}
