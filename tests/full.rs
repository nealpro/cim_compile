use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

fn temp_dir(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cim_compile_full_{}_{}_{}",
        name,
        std::process::id(),
        now
    ))
}

fn repo_path(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn real_model_path() -> PathBuf {
    if let Ok(path) = std::env::var("CIM_COMPILE_REAL_MODEL") {
        return PathBuf::from(path);
    }
    repo_path("data/model.onnx")
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

#[test]
#[ignore = "requires a local real ONNX model and MemTorch; run with CIM_COMPILE_REAL_MODEL=/path/to/model.onnx cargo test --test full -- --ignored"]
fn full_real_tiny_model_memtorch_token_logits_runs() {
    let fixture = real_model_path();
    assert!(
        fixture.exists(),
        "real-model fixture is missing: {}. Set CIM_COMPILE_REAL_MODEL=/path/to/model.onnx when running ignored full-model tests.",
        fixture.display()
    );

    let out_dir = temp_dir("out");
    let output = Command::new(env!("CARGO_BIN_EXE_cim_compile"))
        .arg(&fixture)
        .arg("-o")
        .arg(&out_dir)
        .arg("--tile-size")
        .arg("128")
        .arg("--run-memtorch")
        .arg("--python")
        .arg(test_python())
        .arg("--input-ids")
        .arg("1,2,3,4")
        .arg("--top-k")
        .arg("5")
        .output()
        .expect("failed to run cim_compile");
    assert!(
        output.status.success(),
        "cim_compile failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out_dir.join("output.cim").exists());
    assert!(out_dir.join("memtorch_manifest.json").exists());
    assert!(out_dir.join("memtorch_weights.bin").exists());
    assert!(out_dir.join("memtorch_digital.bin").exists());
    assert!(!out_dir.join("run_memtorch.py").exists());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"results\""));
    assert!(stdout.contains("\"execution_plan\""));
    assert!(stdout.contains("\"attention_blocks\""));
    assert!(stdout.contains("\"attention_slice\""));
    assert!(stdout.contains("\"token_logits\""));
    assert!(stdout.contains("\"next_token_topk\""));
    assert!(stdout.contains("\"output_shape\""));

    let json_start = stdout
        .find("{\n  \"attention_blocks\"")
        .expect("expected MemTorch JSON result in stdout");
    let value: Value =
        serde_json::from_str(&stdout[json_start..]).expect("stdout JSON should parse");
    assert_eq!(value["mode"], json!("token_ids_to_logits"));
    assert_eq!(value["input_ids"], json!([1, 2, 3, 4]));
    assert_eq!(value["logits_shape"], json!([1, 4, 32000]));
    assert_eq!(value["next_token_topk"].as_array().unwrap().len(), 5);
    for entry in value["next_token_topk"].as_array().unwrap() {
        assert!(entry["token_id"].as_i64().unwrap() >= 0);
        assert!(entry["score"].as_f64().unwrap().is_finite());
    }
    assert_eq!(value["attention_slice"]["output_shape"], json!([1, 4, 192]));
    assert_eq!(
        value["attention_blocks"][0]["metadata"]["grouped_query_attention"],
        true
    );
    assert!(
        value["execution_plan"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| { entry["stage"] == "mlp.gate_projection" && entry["target"] == "cim" })
    );
    assert!(
        value["execution_plan"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| { entry["stage"] == "lm_head.matmul" && entry["target"] == "digital" })
    );
}

#[test]
#[ignore = "requires a local real ONNX model and MemTorch; run with CIM_COMPILE_REAL_MODEL=/path/to/model.onnx cargo test --test full -- --ignored"]
fn full_real_tiny_model_memtorch_token_generation_runs() {
    let fixture = real_model_path();
    assert!(
        fixture.exists(),
        "real-model fixture is missing: {}. Set CIM_COMPILE_REAL_MODEL=/path/to/model.onnx when running ignored full-model tests.",
        fixture.display()
    );

    let out_dir = temp_dir("generate");
    let output = Command::new(env!("CARGO_BIN_EXE_cim_compile"))
        .arg(&fixture)
        .arg("-o")
        .arg(&out_dir)
        .arg("--tile-size")
        .arg("128")
        .arg("--run-memtorch")
        .arg("--python")
        .arg(test_python())
        .arg("--generate-ids")
        .arg("--input-ids")
        .arg("1,2,3,4")
        .arg("--max-new-tokens")
        .arg("2")
        .arg("--top-k")
        .arg("5")
        .output()
        .expect("failed to run cim_compile");
    assert!(
        output.status.success(),
        "cim_compile failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out_dir.join("output.cim").exists());
    assert!(out_dir.join("memtorch_manifest.json").exists());
    assert!(out_dir.join("memtorch_weights.bin").exists());
    assert!(out_dir.join("memtorch_digital.bin").exists());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"generated_ids\""));
    assert!(stdout.contains("\"per_step_topk\""));
    assert!(stdout.contains("\"cache_shapes\""));
    assert!(stdout.contains("\"simulation_summary\""));

    let json_start = stdout
        .find("{\n  \"attention_blocks\"")
        .expect("expected MemTorch JSON result in stdout");
    let value: Value =
        serde_json::from_str(&stdout[json_start..]).expect("stdout JSON should parse");
    assert_eq!(value["mode"], json!("generate_ids"));
    assert_eq!(value["prompt_ids"], json!([1, 2, 3, 4]));
    assert_eq!(value["generated_ids"].as_array().unwrap().len(), 6);
    assert_eq!(value["new_token_ids"].as_array().unwrap().len(), 2);
    assert_eq!(value["decode_steps"], json!(2));
    assert_eq!(value["stop_reason"], json!("max_new_tokens"));
    assert_eq!(value["cache_shapes"]["key"], json!([1, 1, 6, 96]));
    assert_eq!(value["cache_shapes"]["value"], json!([1, 1, 6, 96]));
    assert_eq!(value["simulation_summary"]["memtorch_patched"], json!(true));
    assert_eq!(
        value["simulation_summary"]["patched_projection_count"],
        json!(7)
    );

    let per_step = value["per_step_topk"].as_array().unwrap();
    assert_eq!(per_step.len(), 2);
    for step in per_step {
        let topk = step["topk"].as_array().unwrap();
        assert_eq!(topk.len(), 5);
        for entry in topk {
            assert!(entry["token_id"].as_i64().unwrap() >= 0);
            assert!(entry["score"].as_f64().unwrap().is_finite());
        }
    }
}
