# cim_compile Notes

## Current State (2026-06-25)

The active compiler path is:

```text
ONNX -> NormalizedProgram -> verified cim::Program -> AIHWKIT manifest + f32 tile payloads + digital tensor payloads -> Python AIHWKIT bridge
```

This starts the backend pivot to IBM AIHWKIT. The compiler still supports a narrow local tiny-decoder token-logits/token-ID-generation slice, but simulator artifacts are now AIHWKIT-first instead of quantized package bytes for a patching backend.

## Milestone State

- The local `data/model.onnx` fixture normalizes into one explicit `tiny_decoder_v1` block when ignored real-model validation tests are run manually.
- CiM placement covers static Q/K/V/output and MLP gate/up/down projections.
- Digital PyTorch fallback covers embedding lookup, RMSNorms, rotary math, grouped-query repeat, attention mask/score/softmax/context kernels, MLP activation/multiply, residuals, and the final vocabulary LM head.
- The Python AIHWKIT bridge reconstructs `AnalogLinearMapped` projection layers, runs projection smoke checks, attention-slice checks, token-ID logits, and greedy token-ID generation.
- The runner supports interactive token-ID generation, one-shot prompt-text generation, and interactive prompt-text generation when an external local Hugging Face-compatible tokenizer is supplied with `--tokenizer`.
- `output.cim` prints/parses stable MLIR-like text with f32 byte offsets into `aihwkit_weights.bin`.
- The CLI writes `output.cim`, `aihwkit_manifest.json`, `aihwkit_weights.bin`, and optional `aihwkit_digital.bin`.
- `--run-aihwkit` invokes `python -m cim_compile_aihwkit.runner` and emits JSON with logits/top-k results and optional generated token IDs.
- `--prompt <TEXT> data/smolLM2/model_fp16.onnx` invokes a separate SmolLM2 prompt runtime that loads local tokenizer/config sidecars, runs greedy text generation, and uses AIHWKIT `AnalogLinearMapped` for selected static decoder projections.

## Design Decisions

- AIHWKIT owns analog programming, mapping, and non-ideality simulation. The compiler emits deterministic f32 tile payloads, not int4/int8 quantized simulator payloads.
- The ONNX-to-MLIR direction is tracked in project notes and `LINKS.md`: keep the thin Rust frontend for `data/model.onnx` near term, add an out-of-tree MLIR dialect spike, and prefer ONNX-MLIR as a tool/output boundary long term over building a full ONNX dialect here.
- The current custom `cim` text is transitional. It is a deterministic compiler boundary for today, but parser/printer/verifier/schema ownership should move to MLIR ODS/TableGen once the dialect spike is proven.
- Each `cim.tile.dispatch` carries `projection`, `tile`, `matrix_shape`, `tile_size`, `weight_offset`, and `order`.
- `weight_offset` is a byte offset. For fixed-size f32 tiles, `payload_offset = order * tile_rows * tile_cols * 4`.
- The default AIHWKIT runtime config is an ideal mapped inference setup. Realistic PCM/ReRAM/noise presets should be added as explicit manifest/CLI options after this plumbing is stable.
- Public-facing wording should describe the project as a narrow CiM compiler prototype, not a general LLM inference engine.
- ONNX support remains intentionally narrow: the first named decoder layer in `data/model.onnx`, projection-producing `MatMul`/`Gemm` nodes, projection initializer patterns, and structural transformer-adjacent ops needed to reach those weights.
- `data/model.onnx` is an ignored local validation fixture, not a CI input. Tests that require it are marked `#[ignore]` and accept `CIM_COMPILE_REAL_MODEL=/path/to/model.onnx`.
- The default crossbar tile is `128 x 128`; non-divisible projection shapes are accepted by emitting padded edge tiles.
- The LM head is intentionally digital for this milestone. The repo does not bundle tokenizer artifacts; text generation from strings requires an explicit local `--tokenizer` with a vocabulary compatible with the emitted manifest. There is still no non-greedy sampling control, external KV-cache API, or arbitrary LLM inference claim.
- `NormalizedOp::Attention` and `NormalizedOp::TinyDecoder` use boxed payloads so the enum stays small enough for Clippy's `large_enum_variant` check.

## `data/model.onnx` Runtime Contract

- The checked-in `data/model.onnx` is a one-layer decoder slice with graph input `input_ids: int64 [batch_size, sequence_length]`, attention mask and position IDs, and layer-0 KV-cache inputs.
- The current project runner exposes the narrower supported contract: batch size 1, token IDs in `0..31999`, logits shaped `[1, N, 32000]`, and internal one-layer KV-cache use during greedy generation.
- Important tensors are `model.embed_tokens.weight [32000, 192]`, layer norm weights `[192]`, Q/K/V/O projections `[192,192]`, `[96,192]`, `[96,192]`, `[192,192]` after normalization, MLP gate/up/down `[1024,192]`, `[1024,192]`, `[192,1024]`, and LM head `[192,32000]`.
- No tokenizer files are present in the repo. Text mode is therefore an optional bridge feature, not part of the emitted compiler package. The bridge loads tokenizer artifacts with `local_files_only=True` and rejects tokenizer/model vocabulary mismatches before running inference.

## CLI Behavior

- Default compile writes only artifacts.
- `--run-aihwkit` invokes the Python AIHWKIT bridge without emitting generated Python into the output directory.
- `--python <PYTHON>` selects the bridge interpreter.
- `--input-ids <CSV>` selects token IDs for the supported token-logits simulation; the default is `1,2,3,4`.
- `--interactive-ids` streams an interactive token-ID generation loop through the Python runner.
- `--interactive-text --tokenizer <PATH>` streams an interactive prompt-text generation loop through the Python runner.
- `--prompt-text <TEXT> --tokenizer <PATH>` asks the Python runner to encode prompt text through `transformers.AutoTokenizer` with `local_files_only=True` and emits compact generation JSON.
- `--prompt <TEXT>` is reserved for the SmolLM2 path and prints generated text directly rather than package JSON. SmolLM2 prompt decoding samples by default with `--temperature 0.8`; `--temperature 0` selects greedy decoding.
- `--decode-text` decodes generated IDs through the same explicit tokenizer; prompt-text mode can also return decoded generated text because the tokenizer is already required for encoding.
- `--top-k <N>` selects how many last-token logits candidates the bridge reports; the default is `5`.
- `--generate-ids` switches the bridge into greedy token-ID generation mode.
- `--max-new-tokens <N>` controls greedy generation length; the default is `8`.
- `--eos-token-id <ID>` optionally stops generation when the selected ID appears.
- If `--python` is omitted, the CLI uses `.venv/bin/python` when present, otherwise `python3`.
- The bridge keeps Matplotlib/cache files under the output directory to avoid unwritable home-cache issues in sandboxed runs.
- A local CUDA-enabled AIHWKIT build is documented in `docs/aihwkit_gpu_build.md`. The proven Fedora 44 path uses GCC 15, CUDA arch `89` for RTX 4080, C++20 CUDA compilation, `RPU_USE_TORCH_BUFFERS=OFF`, and small CUDA 13 compatibility patches in the nested `aihwkit/` checkout.

## Regression Coverage

Default tests cover generated minimal fixtures, frontend parsing, normalized lowering, f32 tiling/padding, manifest generation, CLI behavior, `cim` parser/printer/verifier behavior, and golden outputs without requiring `data/model.onnx` or AIHWKIT.

CI-equivalent non-test Rust checks:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo check --workspace --all-targets --all-features
```

Ignored manual validation tests exercise `data/model.onnx` with the checked-in AIHWKIT token-logits and token-ID-generation bridge when run with:

```bash
CIM_COMPILE_REAL_MODEL=data/model.onnx cargo test -- --ignored
CIM_COMPILE_REAL_MODEL=data/model.onnx cargo test --test full -- --ignored
```
