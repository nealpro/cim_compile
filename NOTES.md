# cim_compile Notes

## Current State (2026-06-24)

The active compiler path is:

```text
ONNX -> NormalizedProgram -> verified cim::Program -> MemTorch manifest + CiM tile payloads + digital tensor payloads -> Python MemTorch bridge
```

This replaces the earlier RV32I/hardware-artifact path for now. The project is a narrow local-first compiler for one real tiny-decoder token-logits/token-ID-generation slice and a MemTorch simulation package.

CI is configured in `.github/workflows/ci.yml` as focused jobs for formatting, Clippy with `-D warnings`, `cargo check`, the default Rust test suite, rustdoc warnings, and llvm-cov coverage artifact generation. Default CI is hermetic and must not require `data/model.onnx` or any model download.

## Milestone State

- The local `data/model.onnx` fixture normalizes into one explicit `tiny_decoder_v1` block when ignored real-model validation tests are run manually.
- The supported real-model slice includes token embedding lookup, input RMSNorm, Q/K/V projections, rotary position math, score matmul, softmax, context matmul, output projection, attention residual, post-attention RMSNorm, MLP gate/up/down projections, MLP activation/multiply, MLP residual, final RMSNorm, and LM-head logits projection.
- CiM/MemTorch placement covers Q/K/V/output and MLP gate/up/down projections.
- Digital PyTorch fallback covers embedding lookup, RMSNorms, rotary math, grouped-query repeat, attention mask/score/softmax/context kernels, MLP activation/multiply, residuals, and the final 32k-vocabulary LM head.
- The Python bridge can run greedy pre-tokenizer token-ID generation with a one-layer digital KV cache. Inputs and outputs are token IDs only.
- Normalized projections lower into verified `cim.tile.dispatch` operations.
- The `cim` dialect prints/parses stable MLIR-like text and round-trips in tests.
- Lowering emits deterministic quantized tile payload offsets, including zero-padded edge tiles for non-divisible matrix shapes.
- The CLI writes `output.cim`, `memtorch_manifest.json`, `memtorch_weights.bin`, and `memtorch_digital.bin` for the real token-logits slice.
- `--run-memtorch` invokes the checked-in `python/cim_compile_memtorch/runner.py` module through `python -m cim_compile_memtorch.runner` and emits JSON with `logits_shape`, `next_token_topk`, and optionally `generated_ids`.

## Design Decisions

- The `cim` dialect is the compiler boundary. Each `cim.tile.dispatch` carries `projection`, `tile`, `matrix_shape`, `tile_size`, `weight_offset`, `quant_scale`, and `order`.
- `weight_offset` points into `memtorch_weights.bin`, which has no legacy hardware header. For fixed-size tiles, `payload_offset = order * tile_rows * tile_cols`.
- `memtorch_digital.bin` stores float32 little-endian digital fallback tensors. The manifest records role, shape, dtype, byte offset, and byte length for each tensor.
- Public-facing wording should describe the project as a narrow tiny-decoder/token-logits CiM compiler, not a general LLM inference engine.
- ONNX support remains intentionally narrow: the first named decoder layer in `data/model.onnx`, projection-producing `MatMul`/`Gemm` nodes, projection initializer patterns, and structural transformer-adjacent ops needed to reach those weights.
- `data/model.onnx` is an ignored local validation fixture, not a CI input. Tests that require it are marked `#[ignore]` and accept `CIM_COMPILE_REAL_MODEL=/path/to/model.onnx`.
- `tests/generate_onnx_fixtures.py` generates minimal fixtures at test runtime for the mandatory CI regression path.
- The default crossbar tile is still `128 x 128`, but non-divisible projection shapes are accepted by emitting padded edge tiles. The real token-logits slice emits 60 CiM tile dispatches at `128 x 128`.
- The manifest includes a simulation summary listing supported runtime modes (`logits`, `generate_ids`), seven MemTorch-patched projection stages, digital fallback stages, and the digital LM-head placement.
- The LM head is intentionally digital for this milestone. It produces logits and greedy token IDs, but the project still does not claim tokenizer support, text generation from strings, non-greedy sampling controls, external KV-cache APIs, or arbitrary LLM inference.
- MemTorch remains a Python runtime dependency rather than a Rust crate dependency.
- On this MacBook Air, the validated local path is `memtorch-cpu`; CUDA MemTorch should be validated on a CUDA machine.
- `NormalizedOp::Attention` and `NormalizedOp::TinyDecoder` use boxed payloads so the enum stays small enough for Clippy's `large_enum_variant` check under CI's `-D warnings` policy.
- Lowering helper state is grouped into a private context struct so CI can keep `clippy::too_many_arguments` enabled without weakening the workflow.

## Local Python Environment

The project-local uv environment now imports:

- `torch 2.12.1`
- `onnx 1.22.0`
- `onnxscript 0.7.0`
- `memtorch 1.1.6-cpu`

The CPU package was installed without its stale `sklearn` dependency alias, then its actual runtime dependencies were installed explicitly.
`uv pip check` still reports that historical alias as missing, but runtime import and the full MemTorch test pass with `scikit-learn` installed.

## CLI Behavior

- Default compile writes only artifacts.
- `--run-memtorch` invokes the Python MemTorch bridge without emitting generated Python into the output directory.
- `--python <PYTHON>` selects the bridge interpreter.
- `--input-ids <CSV>` selects token IDs for the supported token-logits simulation; the default is `1,2,3,4`.
- `--top-k <N>` selects how many last-token logits candidates the bridge reports; the default is `5`.
- `--generate-ids` switches the bridge into greedy token-ID generation mode.
- `--max-new-tokens <N>` controls greedy generation length; the default is `8`.
- `--eos-token-id <ID>` optionally stops generation when the selected ID appears.
- If `--python` is omitted, the CLI uses `.venv/bin/python` when present, otherwise `python3`.
- The bridge keeps Matplotlib/cache files under the output directory to avoid unwritable home-cache issues in sandboxed runs.

## Regression Coverage

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo check --workspace --all-targets --all-features`, `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items`, and `cargo test --workspace --all-targets --all-features` pass locally after the CI cleanup. Default tests pass with 31 executed tests and 4 ignored real-model tests, including when `data/model.onnx` is temporarily hidden. The CI coverage steps also pass locally: `cargo llvm-cov --workspace --all-targets --all-features --no-report`, `cargo llvm-cov report --html --output-dir target/llvm-cov`, and `cargo llvm-cov report --lcov --output-path target/llvm-cov/lcov.info`.

`cargo test` covers generated minimal fixtures, frontend parsing, normalized lowering, tiling and quantization, manifest generation, CLI behavior, `cim` parser/printer/verifier behavior, and golden outputs without requiring `data/model.onnx`.

Ignored manual validation tests exercise `data/model.onnx` with the checked-in MemTorch token-logits and token-ID-generation bridge when run with `CIM_COMPILE_REAL_MODEL=data/model.onnx cargo test -- --ignored` or `CIM_COMPILE_REAL_MODEL=data/model.onnx cargo test --test full -- --ignored`. Both commands pass in the local uv/MemTorch CPU environment.
