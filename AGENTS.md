# AGENTS.md

## Project Goal

Build `cim_compile` into a real domain compiler for compute-in-memory simulation. The target pipeline is:

```text
ONNX model -> MLIR-based normalized model IR -> reusable CiM/circuit dialect -> AIHWKIT simulation package
```

The end goal is more important than the current narrow implementation. Existing tiny-decoder and MemTorch paths are transitional scaffolding; do not preserve them for compatibility.

## MLIR Direction

- Creating a project MLIR dialect is a priority and should be done as soon as the repo has enough researched design clarity to do it responsibly.
- Near-term work should prepare for MLIR rather than deepen custom parser/printer/verifier infrastructure that MLIR can own.
- The project dialect should model what is unique here: analog/CiM placement, crossbar tiling, quantization metadata, device and circuit constraints, backend capabilities, and AIHWKIT package emission.
- Do not create a full ONNX dialect from scratch. Study ONNX-MLIR first and decide whether to consume its output, reuse its approach, or integrate with it as a tool boundary.
- Prefer an out-of-tree MLIR dialect first. Prove the dialect, pass pipeline, and tests before attempting deep Rust/MLIR binding integration.
- Use TableGen/ODS for operation definitions so op syntax, traits, attributes, verifiers, builders, parsers, and printers have one source of truth.
- The first dialect milestone should support one static-weight operation and one tiled dispatch operation, with parser/printer round trips and a lowering pass.

## Architecture Direction

- Keep Rust as the compiler driver where practical: CLI behavior, orchestration, artifact packaging, and compatibility with current tests belong in Rust until an MLIR tool boundary replaces specific pieces.
- Use MLIR for normalized model representation, compiler passes, dialect verification, lowering, tiling, quantization metadata, and backend-specific legalization as soon as feasible.
- Keep Python as the simulator bridge: AIHWKIT/PyTorch integration, runtime model reconstruction, and simulation execution belong under `python/`.
- Treat `cim` as the compiler boundary. It should become a real MLIR dialect rather than remaining only a hand-rolled MLIR-like text format.
- Prefer general model abstractions over transformer-only abstractions. Add reusable IR concepts for operators, tensors, weights, execution placement, and backend capabilities before adding model-specific shortcuts.
- Model-specific recognizers are allowed only as adapters that lower into generic IR. Do not let tiny-decoder assumptions leak into shared lowering, packaging, or simulation interfaces.

## Backend Migration

- New simulation work should target AIHWKIT, not MemTorch.
- Do not expand `memtorch.rs` or `python/cim_compile_memtorch/` unless needed to preserve current compatibility while the AIHWKIT path is being introduced.
- Introduce AIHWKIT types and files with backend-neutral naming where practical. Prefer `simulation`, `backend`, or `analog` concepts in shared IR, and reserve `aihwkit` names for Python bridge code or final backend package emitters.
- The desired backend shape is `ONNX -> MLIR model IR -> backend-independent CiM/circuit dialect -> AIHWKIT runner`.
- Keep backend manifests explicit and versioned. Any file consumed by Python should include schema/version fields and deterministic payload offsets.

## Research And Validation Roadmap

- Survey adjacent projects before major architecture changes. Record findings in `LINKS.md` or a dedicated research note.
- Start with ONNX-MLIR. It is the main reference for ONNX-to-MLIR import, ONNX dialect design, shape handling, and lowering strategy.
- Study CIMFlow and CIMFlow-Compiler as the closest CIM compiler reference. Identify which ideas transfer to analog simulation and which are specific to digital SRAM CIM, ISA generation, and SystemC simulation.
- Study Cinnamon for heterogeneous compute-in-memory and compute-near-memory compiler architecture.
- Study CIRCT for circuit/hardware IR patterns, scheduling concepts, and how MLIR-based hardware projects organize dialects and passes.
- Compare AIHWKIT and CrossSim as analog simulation backends. AIHWKIT is the primary target; CrossSim is a useful reference for analog crossbar simulation behavior and validation questions.
- Review FINN, hls4ml, IREE, StableHLO, and torch-mlir for model-to-hardware compiler architecture, pass organization, testing style, and reusable MLIR conventions.
- Run an MLIR feasibility spike: build a minimal out-of-tree `cim` or `analog` dialect with one static-weight op, one tiled dispatch op, one lowering pass, and parser/printer tests.
- Run an AIHWKIT backend spike: prove that emitted metadata can reconstruct and simulate at least `Linear` and `Conv` modules using AIHWKIT analog layers.
- Decide the ONNX path only after research: consume ONNX-MLIR output, call ONNX-MLIR as a tool, integrate with its libraries, or keep a temporary thin Rust ONNX frontend until the MLIR import path is ready.
- Make the MLIR path the main compiler path once it can round-trip one generated ONNX fixture and produce an AIHWKIT-runnable package.

## Related Projects And Viability

- ONNX-MLIR is the strongest reference for ONNX and MLIR frontend work. Do not duplicate its ONNX dialect unless research proves there is no practical integration route.
- CIMFlow is the closest project by domain. Its existence validates the MLIR+CIM direction, but this project should differ by targeting analog/CiM simulation packages for AIHWKIT rather than digital CIM instruction streams and SystemC simulation.
- Cinnamon validates that MLIR can be used for heterogeneous compute-in-memory and compute-near-memory compilation, but it is not an ONNX-to-AIHWKIT simulation pipeline.
- CIRCT is relevant for hardware and circuit IR structure, but it is not a neural-network analog simulation backend.
- CrossSim is relevant as an analog crossbar simulator, but it is not an ONNX-to-MLIR compiler.
- FINN and hls4ml are useful model-to-hardware compiler references, but they target FPGA/HLS/QNN workflows rather than analog AIHWKIT simulation.
- IREE, StableHLO, and torch-mlir are useful references for modern MLIR compiler architecture, but they are not CiM simulation compilers.
- Viability conclusion: this project is viable if it does not try to become another ONNX-MLIR or CIMFlow. Its distinct purpose should be an MLIR-based analog/CiM compiler path that emits AIHWKIT simulations.

## Generalization Rules

- Do not hardcode `data/model.onnx`, layer `0`, hidden size `192`, vocab size `32000`, QKV names, token IDs, or any single-model assumption in generic code.
- When supporting an ONNX pattern, represent the discovered structure in normalized IR using operator/tensor semantics rather than source-model names.
- Static-weight linear algebra should be represented as reusable projection, matmul, linear, and convolution units that can serve CNNs, MLPs, attention blocks, and other model families.
- Dynamic activation operations, reductions, softmax, normalization, indexing, and shape transforms should be represented explicitly even if they remain digital simulation stages.
- Prefer capability-based placement decisions: decide whether an op maps to analog/CiM based on weights, shape, data dependency, backend support, and cost metadata, not on model family.

## Testing Expectations

- Default tests do not need to be optimized for CI safety. It is acceptable for meaningful tests to depend on local models, AIHWKIT, LLVM/MLIR tooling, simulator packages, CUDA, or other research dependencies when that better validates the compiler.
- CI should be limited to non-test checks such as formatting, Clippy, type checking, and documentation builds.
- Tests should document their prerequisites clearly, especially when they require model files, simulator installs, accelerator hardware, external tools, or environment variables.
- MLIR integration should still start with small parser/printer or FileCheck-style tests, then expand to real models and simulator-backed validation as soon as practical.
- When changing IR, manifests, payload ordering, or textual `cim` output, update focused golden tests.
- Run CI-equivalent non-test Rust checks through:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo check --workspace --all-targets --all-features
```

## Coding Guidelines

- Preserve deterministic output ordering for tiles, manifests, execution plans, MLIR text, and generated artifacts.
- Keep error messages explicit: include unsupported ONNX op names, node names, tensor names, dialect legalization failures, and backend capability gaps where relevant.
- Do not add broad dependencies without a clear compiler or simulation boundary reason.
- Keep Python bridge code import-lazy for heavy simulator dependencies so normal Rust tests and compile-only workflows do not require AIHWKIT.
- Avoid generated Python in output directories; checked-in bridge modules should consume emitted manifests and payloads.
- Update `README.md`, `NOTES.md`, and research notes when user-facing behavior, supported model scope, backend expectations, MLIR architecture, or validation commands change.
