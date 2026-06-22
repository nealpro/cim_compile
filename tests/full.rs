use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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

#[test]
#[ignore = "requires torch+onnx; run with CIM_COMPILE_FULL_TESTS=1 cargo test --test full -- --ignored"]
fn full_torch_export_fixture_compiles() {
    if std::env::var("CIM_COMPILE_FULL_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping full test; set CIM_COMPILE_FULL_TESTS=1 to require torch+onnx");
        return;
    }

    let fixture_dir = temp_dir("fixtures");
    let python = std::env::var("CIM_COMPILE_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let generator = Command::new(&python)
        .arg(repo_path("tests/generate_onnx_fixtures.py"))
        .arg("--mode")
        .arg("all")
        .arg("--dim")
        .arg("64")
        .arg("--seq-len")
        .arg("8")
        .arg("--output-dir")
        .arg(&fixture_dir)
        .output()
        .expect("failed to run full fixture generator");
    assert!(
        generator.status.success(),
        "full fixture generation failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&generator.stdout),
        String::from_utf8_lossy(&generator.stderr)
    );

    let out_dir = temp_dir("out");
    let output = Command::new(env!("CARGO_BIN_EXE_cim_compile"))
        .arg(fixture_dir.join("memristor_mha_unrolled.onnx"))
        .arg("-o")
        .arg(&out_dir)
        .arg("--tile-size")
        .arg("64")
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
}
