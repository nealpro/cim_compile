use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_EXPECTED_TILES: u32 = 64;
const DEFAULT_TILE_SIZE: u32 = 128;

fn repo_path(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn temp_output_dir(model_name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cim_compile_cli_{}_{}_{}",
        model_name,
        std::process::id(),
        now
    ))
}

fn test_python() -> String {
    if let Ok(python) = std::env::var("CIM_COMPILE_PYTHON") {
        return python;
    }
    let local = repo_path(".venv/bin/python");
    if local.exists() {
        local.display().to_string()
    } else {
        "python3".to_string()
    }
}

fn generated_fixture_path(file_name: &str) -> PathBuf {
    let out_dir = temp_output_dir("onnx_fixtures");
    let script = repo_path("tests/generate_onnx_fixtures.py");
    let output = Command::new(test_python())
        .arg(script)
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

    out_dir.join(file_name)
}

fn run_model_with_args(model_path: &str, extra_args: &[&str]) -> PathBuf {
    let out_dir = temp_output_dir(
        Path::new(model_path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("model"),
    );

    let mut command = Command::new(env!("CARGO_BIN_EXE_cim_compile"));
    command.arg(model_path).arg("-o").arg(&out_dir);
    command.args(extra_args);
    let output = command.output().expect("failed to run cim_compile");

    assert!(
        output.status.success(),
        "cim_compile failed for {model_path}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    out_dir
}

fn run_model(model_path: &str) -> PathBuf {
    run_model_with_args(model_path, &[])
}

fn run_model_expect_failure(model_path: &str, extra_args: &[&str]) -> String {
    let out_dir = temp_output_dir(
        Path::new(model_path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("model"),
    );

    let mut command = Command::new(env!("CARGO_BIN_EXE_cim_compile"));
    command.arg(model_path).arg("-o").arg(&out_dir);
    command.args(extra_args);
    let output = command.output().expect("failed to run cim_compile");

    assert!(
        !output.status.success(),
        "cim_compile unexpectedly succeeded for {model_path}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_outputs(out_dir: &Path, expected_tiles: u32, tile_size: u32) {
    let cim_path = out_dir.join("output.cim");
    let manifest_path = out_dir.join("memtorch_manifest.json");
    let weights_path = out_dir.join("memtorch_weights.bin");
    let runner_path = out_dir.join("run_memtorch.py");
    let payload_bytes = tile_size as usize * tile_size as usize;
    let expected_weight_file_bytes = expected_tiles as usize * payload_bytes;

    assert!(cim_path.exists(), "missing {}", cim_path.display());
    assert!(
        manifest_path.exists(),
        "missing {}",
        manifest_path.display()
    );
    assert!(weights_path.exists(), "missing {}", weights_path.display());
    assert!(runner_path.exists(), "missing {}", runner_path.display());

    let weights = fs::read(&weights_path).expect("failed to read weights file");
    assert_eq!(weights.len(), expected_weight_file_bytes);

    let cim_text = fs::read_to_string(&cim_path).expect("failed to read output.cim");
    assert!(cim_text.starts_with("cim.module @cim_compile"));
    assert!(cim_text.contains("cim.tile.dispatch"));

    let manifest = fs::read_to_string(&manifest_path).expect("failed to read manifest");
    assert!(manifest.contains("\"schema_version\": 1"));
    assert!(manifest.contains("\"weights_file\": \"memtorch_weights.bin\""));
    assert!(manifest.contains(&format!("\"order\": {}", expected_tiles - 1)));

    let runner = fs::read_to_string(&runner_path).expect("failed to read runner");
    assert!(runner.contains("patch_model"));
    assert!(runner.contains("MemTorch simulation requires torch and memtorch"));
}

#[test]
fn cli_compiles_unrolled_projection_model() {
    let fixture = generated_fixture_path("memristor_mha_unrolled.onnx");
    let out_dir = run_model(fixture.to_str().unwrap());

    assert_outputs(&out_dir, DEFAULT_EXPECTED_TILES, DEFAULT_TILE_SIZE);
}

#[test]
fn cli_compiles_fused_mha_model() {
    let fixture = generated_fixture_path("mha_bfloat16.onnx");
    let out_dir = run_model(fixture.to_str().unwrap());

    assert_outputs(&out_dir, DEFAULT_EXPECTED_TILES, DEFAULT_TILE_SIZE);
}

#[test]
fn cli_compiles_custom_tile_size() {
    let out_dir = run_model_with_args(
        generated_fixture_path("memristor_mha_unrolled.onnx")
            .to_str()
            .unwrap(),
        &["--tile-size", "64"],
    );

    assert_outputs(&out_dir, 256, 64);
}

#[test]
fn cli_rejects_invalid_quantization_bits() {
    let stderr = run_model_expect_failure(
        generated_fixture_path("memristor_mha_unrolled.onnx")
            .to_str()
            .unwrap(),
        &["--bits", "6"],
    );

    assert!(stderr.contains("unsupported quantization bit-width 6"));
}

#[test]
fn cli_rejects_non_divisible_tile_size() {
    let stderr = run_model_expect_failure(
        generated_fixture_path("memristor_mha_unrolled.onnx")
            .to_str()
            .unwrap(),
        &["--tile-size", "100"],
    );

    assert!(stderr.contains("must evenly divide"));
}

#[test]
fn cli_rejects_unavailable_memtorch_python_executable_when_asked_to_run() {
    let missing_python = temp_output_dir("missing_python").join("python");
    let stderr = run_model_expect_failure(
        generated_fixture_path("memristor_mha_unrolled.onnx")
            .to_str()
            .unwrap(),
        &[
            "--run-memtorch",
            "--python",
            missing_python.to_str().unwrap(),
        ],
    );

    assert!(stderr.contains("failed to run"));
}
