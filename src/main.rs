use clap::Parser;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

use cim_compile::{CompileConfig, aihwkit, compile_onnx};

#[derive(Parser)]
#[command(
    version,
    about = "Transformer CiM compiler: ONNX tiny-decoder slice -> cim dialect -> AIHWKIT simulation package"
)]
struct Cli {
    /// Path to the ONNX model file
    onnx_path: PathBuf,

    /// Output directory for output.cim and AIHWKIT simulation artifacts
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// Crossbar tile dimension; edge tiles are zero-padded when needed
    #[arg(long, default_value_t = 128)]
    tile_size: u32,

    /// Run the AIHWKIT simulation bridge after writing artifacts
    #[arg(long)]
    run_aihwkit: bool,

    /// Python executable to use with --run-aihwkit
    #[arg(long)]
    python: Option<String>,

    /// Comma-separated token IDs, not prompt text, for --run-aihwkit token-logits inference
    #[arg(long)]
    input_ids: Option<String>,

    /// Let the Python runner prompt interactively for token IDs in token-ID mode
    #[arg(long)]
    interactive_ids: bool,

    /// Let the Python runner prompt interactively for prompt text in text/tokenizer mode
    #[arg(long)]
    interactive_text: bool,

    /// Prompt text for the Python runner's text/tokenizer mode
    #[arg(long)]
    prompt_text: Option<String>,

    /// Local tokenizer path for Python runner text/tokenizer mode
    #[arg(long)]
    tokenizer: Option<String>,

    /// Decode generated token IDs back to text in Python runner text/tokenizer mode
    #[arg(long)]
    decode_text: bool,

    /// Number of next-token candidates to report from --run-aihwkit
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Generate token IDs greedily with the AIHWKIT bridge
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
    if cli.interactive_ids && cli.interactive_text {
        return Err("--interactive-ids and --interactive-text cannot be combined".to_string());
    }
    if cli.prompt_text.is_some() && cli.input_ids.is_some() {
        return Err("--prompt-text and --input-ids cannot be combined".to_string());
    }
    if cli.interactive_text && cli.input_ids.is_some() {
        return Err("--interactive-text and --input-ids cannot be combined".to_string());
    }
    if cli.interactive_text && cli.prompt_text.is_some() {
        return Err("--interactive-text and --prompt-text cannot be combined".to_string());
    }

    let compilation = compile_onnx(&cli.onnx_path, CompileConfig::square(cli.tile_size))?;

    std::fs::create_dir_all(&cli.output_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            cli.output_dir.display()
        )
    })?;

    let cim_path = cli.output_dir.join("output.cim");
    let manifest_path = cli.output_dir.join("aihwkit_manifest.json");
    let weights_path = cli.output_dir.join("aihwkit_weights.bin");
    let digital_path = cli.output_dir.join("aihwkit_digital.bin");

    std::fs::write(&cim_path, compilation.cim.to_text())
        .map_err(|err| format!("failed to write {}: {err}", cim_path.display()))?;
    std::fs::write(
        &manifest_path,
        aihwkit::manifest_json(&compilation.aihwkit.manifest)?,
    )
    .map_err(|err| format!("failed to write {}: {err}", manifest_path.display()))?;
    std::fs::write(&weights_path, &compilation.aihwkit.weights)
        .map_err(|err| format!("failed to write {}: {err}", weights_path.display()))?;
    let wrote_digital = !compilation.aihwkit.digital.is_empty();
    if wrote_digital {
        std::fs::write(&digital_path, &compilation.aihwkit.digital)
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

    if cli.run_aihwkit {
        let python = cli.python.clone().unwrap_or_else(default_python);
        let mut command = Command::new(&python);
        command
            .arg("-m")
            .arg("cim_compile_aihwkit.runner")
            .arg("--manifest")
            .arg(&manifest_path)
            .arg("--top-k")
            .arg(cli.top_k.to_string())
            .arg("--max-new-tokens")
            .arg(cli.max_new_tokens.to_string())
            .env("PYTHONPATH", python_path_env()?);
        if let Some(input_ids) = &cli.input_ids {
            command.arg("--input-ids").arg(input_ids);
        }
        if cli.generate_ids {
            command.arg("--generate-ids");
        }
        if cli.interactive_ids {
            command.arg("--interactive-ids");
        }
        if cli.interactive_text {
            command.arg("--interactive-text");
        }
        if let Some(prompt_text) = &cli.prompt_text {
            command.arg("--prompt-text").arg(prompt_text);
        }
        if let Some(tokenizer) = &cli.tokenizer {
            command.arg("--tokenizer").arg(tokenizer);
        }
        if cli.decode_text {
            command.arg("--decode-text");
        }
        if let Some(eos_token_id) = cli.eos_token_id {
            command.arg("--eos-token-id").arg(eos_token_id.to_string());
        }
        if cli.interactive_ids || cli.interactive_text {
            let status = command
                .status()
                .map_err(|err| format!("failed to run {python}: {err}"))?;
            if !status.success() {
                return Err(format!("AIHWKIT bridge failed with status {status}"));
            }
        } else {
            let output = command
                .output()
                .map_err(|err| format!("failed to run {python}: {err}"))?;
            if !output.status.success() {
                return Err(format!(
                    "AIHWKIT bridge failed\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }
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
