use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

const DEFAULT_EXPECTED_TILES: u32 = 64;
const DEFAULT_TILE_SIZE: u32 = 128;
const REAL_MODEL_EXPECTED_TILES: u32 = 60;

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

fn required_data_model_path() -> PathBuf {
    if let Ok(path) = std::env::var("CIM_COMPILE_REAL_MODEL") {
        return PathBuf::from(path);
    }
    repo_path("data/model.onnx")
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

fn run_model_with_env(model_path: &str, extra_args: &[&str], envs: &[(&str, &Path)]) -> PathBuf {
    let out_dir = temp_output_dir(
        Path::new(model_path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("model"),
    );

    let mut command = Command::new(env!("CARGO_BIN_EXE_cim_compile"));
    command.arg(model_path).arg("-o").arg(&out_dir);
    command.args(extra_args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command.output().expect("failed to run cim_compile");

    assert!(
        output.status.success(),
        "cim_compile failed for {model_path}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    out_dir
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
    assert_outputs_with_digital(out_dir, expected_tiles, tile_size, false);
}

fn assert_outputs_with_digital(
    out_dir: &Path,
    expected_tiles: u32,
    tile_size: u32,
    expect_digital: bool,
) {
    let cim_path = out_dir.join("output.cim");
    let manifest_path = out_dir.join("aihwkit_manifest.json");
    let weights_path = out_dir.join("aihwkit_weights.bin");
    let digital_path = out_dir.join("aihwkit_digital.bin");
    let runner_path = out_dir.join("run_aihwkit.py");
    let payload_bytes = tile_size as usize * tile_size as usize * std::mem::size_of::<f32>();
    let expected_weight_file_bytes = expected_tiles as usize * payload_bytes;

    assert!(cim_path.exists(), "missing {}", cim_path.display());
    assert!(
        manifest_path.exists(),
        "missing {}",
        manifest_path.display()
    );
    assert!(weights_path.exists(), "missing {}", weights_path.display());
    if expect_digital {
        assert!(digital_path.exists(), "missing {}", digital_path.display());
    } else {
        assert!(
            !digital_path.exists(),
            "unexpected {}",
            digital_path.display()
        );
    }
    assert!(
        !runner_path.exists(),
        "compiler should not emit generated Python: {}",
        runner_path.display()
    );

    let weights = fs::read(&weights_path).expect("failed to read weights file");
    assert_eq!(weights.len(), expected_weight_file_bytes);

    let cim_text = fs::read_to_string(&cim_path).expect("failed to read output.cim");
    assert!(cim_text.starts_with("cim.module @cim_compile"));
    assert!(cim_text.contains("cim.tile.dispatch"));

    let manifest = fs::read_to_string(&manifest_path).expect("failed to read manifest");
    assert!(manifest.contains("\"schema_version\": 1"));
    assert!(manifest.contains("\"backend\": \"aihwkit\""));
    assert!(manifest.contains("\"weight_dtype\": \"f32\""));
    assert!(manifest.contains("\"weights_file\": \"aihwkit_weights.bin\""));
    if expect_digital {
        assert!(manifest.contains("\"digital_tensors_file\": \"aihwkit_digital.bin\""));
    }
    assert!(manifest.contains(&format!("\"order\": {}", expected_tiles - 1)));
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
fn cli_compiles_non_divisible_tile_size_with_padding() {
    let out_dir = run_model_with_args(
        generated_fixture_path("memristor_mha_unrolled.onnx")
            .to_str()
            .unwrap(),
        &["--tile-size", "100"],
    );

    assert_outputs(&out_dir, 144, 100);
}

#[test]
fn cli_help_distinguishes_token_id_and_text_modes() {
    let output = Command::new(env!("CARGO_BIN_EXE_cim_compile"))
        .arg("--help")
        .output()
        .expect("failed to run cim_compile --help");

    assert!(
        output.status.success(),
        "help failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--input-ids <INPUT_IDS>"));
    assert!(stdout.contains("not prompt text"));
    assert!(stdout.contains("--interactive-ids"));
    assert!(stdout.contains("token IDs in token-ID mode"));
    assert!(stdout.contains("--interactive-text"));
    assert!(stdout.contains("prompt text in text/tokenizer mode"));
    assert!(stdout.contains("--prompt-text <PROMPT_TEXT>"));
    assert!(stdout.contains("text/tokenizer mode"));
    assert!(stdout.contains("--tokenizer <TOKENIZER>"));
    assert!(stdout.contains("--decode-text"));
}

#[test]
fn cli_prompt_text_without_run_aihwkit_only_writes_artifacts() {
    let missing_python = temp_output_dir("missing_python").join("python");
    let out_dir = run_model_with_args(
        generated_fixture_path("memristor_mha_unrolled.onnx")
            .to_str()
            .unwrap(),
        &[
            "--prompt-text",
            "hello from text mode",
            "--tokenizer",
            "local-tokenizer-or-name",
            "--decode-text",
            "--python",
            missing_python.to_str().unwrap(),
        ],
    );

    assert_outputs(&out_dir, DEFAULT_EXPECTED_TILES, DEFAULT_TILE_SIZE);
}

#[cfg(unix)]
#[test]
fn cli_forwards_one_shot_text_options_to_python_runner_when_asked_to_run() {
    use std::os::unix::fs::PermissionsExt;

    let script_dir = temp_output_dir("fake_python");
    fs::create_dir_all(&script_dir).expect("failed to create fake python directory");
    let fake_python = script_dir.join("python");
    let captured_args = script_dir.join("args.txt");
    fs::write(
        &fake_python,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CIM_COMPILE_CAPTURE_ARGS\"\nexit 0\n",
    )
    .expect("failed to write fake python");
    let mut permissions = fs::metadata(&fake_python)
        .expect("failed to stat fake python")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_python, permissions).expect("failed to chmod fake python");

    run_model_with_env(
        generated_fixture_path("memristor_mha_unrolled.onnx")
            .to_str()
            .unwrap(),
        &[
            "--run-aihwkit",
            "--python",
            fake_python.to_str().unwrap(),
            "--generate-ids",
            "--prompt-text",
            "hello from text mode",
            "--tokenizer",
            "local-tokenizer-or-name",
            "--decode-text",
        ],
        &[("CIM_COMPILE_CAPTURE_ARGS", &captured_args)],
    );

    let args = fs::read_to_string(&captured_args).expect("failed to read captured args");
    assert!(args.contains("-m\ncim_compile_aihwkit.runner\n"));
    assert!(args.contains("--generate-ids\n"));
    assert!(args.contains("--prompt-text\nhello from text mode\n"));
    assert!(args.contains("--tokenizer\nlocal-tokenizer-or-name\n"));
    assert!(args.contains("--decode-text\n"));
}

#[cfg(unix)]
#[test]
fn cli_forwards_interactive_text_to_python_runner_when_asked_to_run() {
    use std::os::unix::fs::PermissionsExt;

    let script_dir = temp_output_dir("fake_python_interactive_text");
    fs::create_dir_all(&script_dir).expect("failed to create fake python directory");
    let fake_python = script_dir.join("python");
    let captured_args = script_dir.join("args.txt");
    fs::write(
        &fake_python,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CIM_COMPILE_CAPTURE_ARGS\"\nexit 0\n",
    )
    .expect("failed to write fake python");
    let mut permissions = fs::metadata(&fake_python)
        .expect("failed to stat fake python")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_python, permissions).expect("failed to chmod fake python");

    run_model_with_env(
        generated_fixture_path("memristor_mha_unrolled.onnx")
            .to_str()
            .unwrap(),
        &[
            "--run-aihwkit",
            "--python",
            fake_python.to_str().unwrap(),
            "--interactive-text",
            "--tokenizer",
            "local-tokenizer-or-name",
            "--decode-text",
        ],
        &[("CIM_COMPILE_CAPTURE_ARGS", &captured_args)],
    );

    let args = fs::read_to_string(&captured_args).expect("failed to read captured args");
    assert!(args.contains("-m\ncim_compile_aihwkit.runner\n"));
    assert!(args.contains("--interactive-text\n"));
    assert!(args.contains("--tokenizer\nlocal-tokenizer-or-name\n"));
    assert!(args.contains("--decode-text\n"));
}

#[test]
#[ignore = "requires a local real ONNX model; run with CIM_COMPILE_REAL_MODEL=/path/to/model.onnx cargo test -- --ignored"]
fn cli_compiles_required_real_tiny_model_token_logits_slice() {
    let fixture = required_data_model_path();
    assert!(
        fixture.exists(),
        "real-model fixture is missing: {}. Set CIM_COMPILE_REAL_MODEL=/path/to/model.onnx when running ignored full-model tests.",
        fixture.display()
    );
    let out_dir = run_model_with_args(fixture.to_str().unwrap(), &["--tile-size", "128"]);

    assert_outputs_with_digital(&out_dir, REAL_MODEL_EXPECTED_TILES, DEFAULT_TILE_SIZE, true);

    let manifest_path = out_dir.join("aihwkit_manifest.json");
    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).expect("failed to read manifest"))
            .expect("manifest should be valid JSON");
    let attention_blocks = manifest["attention_blocks"]
        .as_array()
        .expect("attention_blocks should be an array");
    assert_eq!(attention_blocks.len(), 1);
    let block = &attention_blocks[0];
    assert_eq!(block["mode"], "hybrid");
    assert_eq!(block["metadata"]["hidden_size"], 192);
    assert_eq!(block["metadata"]["q_dim"], 192);
    assert_eq!(block["metadata"]["kv_dim"], 96);
    assert_eq!(block["metadata"]["head_dim"], 96);
    assert_eq!(block["metadata"]["q_heads"], 2);
    assert_eq!(block["metadata"]["kv_heads"], 1);
    assert_eq!(block["metadata"]["grouped_query_attention"], true);
    assert_eq!(
        block["cim_projections"].as_array().unwrap().len(),
        4,
        "expected Q/K/V/O on CiM"
    );
    let digital = block["digital_kernels"].as_array().unwrap();
    assert!(
        digital
            .iter()
            .any(|name| name.as_str().unwrap().contains("repeat_kv"))
    );
    assert!(
        digital
            .iter()
            .any(|name| name.as_str().unwrap().contains("Softmax"))
    );

    let plan = manifest["execution_plan"].as_array().unwrap();
    assert!(plan.iter().any(|entry| {
        entry["stage"] == "attention.score_matmul" && entry["target"] == "digital"
    }));
    assert!(plan.iter().any(|entry| {
        entry["stage"] == "attention.context_matmul" && entry["target"] == "digital"
    }));
    assert!(plan.iter().any(|entry| {
        entry["stage"] == "attention.query_projection"
            && entry["target"] == "cim"
            && entry["tile_count"] == 4
    }));
    assert!(plan.iter().any(|entry| {
        entry["stage"] == "attention.key_projection"
            && entry["target"] == "cim"
            && entry["tile_count"] == 2
    }));
    assert!(plan.iter().any(|entry| {
        entry["stage"] == "mlp.gate_projection"
            && entry["target"] == "cim"
            && entry["tile_count"] == 16
    }));
    assert!(plan.iter().any(|entry| {
        entry["stage"] == "mlp.down_projection"
            && entry["target"] == "cim"
            && entry["tile_count"] == 16
    }));
    assert!(
        plan.iter()
            .any(|entry| { entry["stage"] == "lm_head.matmul" && entry["target"] == "digital" })
    );
    let lm_head = plan
        .iter()
        .find(|entry| entry["stage"] == "lm_head.matmul")
        .expect("execution plan should include lm_head.matmul");
    let lm_head_reason = lm_head["reason"]
        .as_str()
        .expect("lm_head.matmul should include a reason");
    assert!(!lm_head_reason.contains("32k-vocabulary"));
    assert!(lm_head_reason.contains("vocab_size = 32000"));

    let inference = &manifest["inference_slice"];
    assert_eq!(inference["model_kind"], "tiny_decoder_v1");
    assert_eq!(inference["inference_mode"], "token_ids_to_logits");
    assert_eq!(inference["vocab_size"], 32000);
    assert_eq!(inference["hidden_size"], 192);
    assert_eq!(inference["intermediate_size"], 1024);
    assert_eq!(inference["decoder_layers"], 1);
    assert_eq!(inference["grouped_query_attention"], true);
    assert_eq!(manifest["digital_tensors"].as_array().unwrap().len(), 5);

    let topology = &manifest["model_topology"];
    assert_eq!(topology["model_kind"], "tiny_decoder_v1");
    assert_eq!(topology["decoder_layers"], 1);
    assert_eq!(topology["vocab_size"], 32000);
    assert_eq!(topology["hidden_size"], 192);

    let summary = &manifest["simulation_summary"];
    assert_eq!(summary["analog_projection_count"], 7);
    assert_eq!(summary["decoder_layers"], 1);
    assert_eq!(summary["vocab_size"], 32000);
    assert_eq!(summary["lm_head_target"], "digital");
    let modes = summary["supported_runtime_modes"].as_array().unwrap();
    assert!(modes.iter().any(|mode| mode == "logits"));
    assert!(modes.iter().any(|mode| mode == "generate_ids"));
    let aihwkit_stages = summary["aihwkit_stages"].as_array().unwrap();
    assert_eq!(aihwkit_stages.len(), 7);
    assert!(
        aihwkit_stages
            .iter()
            .any(|stage| stage == "attention.query_projection")
    );
    assert!(
        aihwkit_stages
            .iter()
            .any(|stage| stage == "mlp.gate_projection")
    );
    assert!(
        aihwkit_stages
            .iter()
            .any(|stage| stage == "mlp.down_projection")
    );
    let digital_stages = summary["digital_stages"].as_array().unwrap();
    assert!(digital_stages.iter().any(|stage| stage == "lm_head.matmul"));
}

#[test]
fn cli_rejects_unavailable_aihwkit_python_executable_when_asked_to_run() {
    let missing_python = temp_output_dir("missing_python").join("python");
    let stderr = run_model_expect_failure(
        generated_fixture_path("memristor_mha_unrolled.onnx")
            .to_str()
            .unwrap(),
        &[
            "--run-aihwkit",
            "--python",
            missing_python.to_str().unwrap(),
        ],
    );

    assert!(stderr.contains("failed to run"));
}
