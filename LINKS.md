# Reference Links

## For M1 — Frontend / ONNX Parsing

| Resource | URL | Notes |
|----------|-----|-------|
| ONNX Operator Spec | https://onnx.ai/onnx/operators/ | MatMul and Conv ops — industry vocabulary for describing ML ops |
| ONNX-MLIR | https://github.com/onnx/onnx-mlir | Primary reference for ONNX import, ONNX dialect handling, shape handling, and possible tool/output boundary |
| prost docs | https://docs.rs/prost | Protobuf decode — `Message::decode`, generated types |

## Simulation Backend

| Resource | URL | Notes |
|----------|-----|-------|
| IBM AIHWKIT docs | https://aihwkit.readthedocs.io/en/latest/ | Active analog/CiM simulation backend target |
| IBM AIHWKIT repository | https://github.com/IBM/aihwkit | Local clone is available under `aihwkit/` for API reference |
| CrossSim | https://github.com/sandialabs/cross-sim | Analog crossbar simulation reference for validation questions and backend comparison |

## Local Validation Fixtures

| Resource | URL | Notes |
|----------|-----|-------|
| Tiny LLM ONNX model | https://huggingface.co/onnx-community/Tiny-LLM-ONNX/resolve/main/onnx/model.onnx | Source for the local ignored `data/model.onnx` validation fixture |

## Compiler Architecture / IR Design

| Resource | URL | Notes |
|----------|-----|-------|
| MLIR Toy Tutorial | https://mlir.llvm.org/docs/Tutorials/Toy/ | Conceptual model for IRs and lowering passes — no install needed |
| MLIR Linalg Dialect Rationale | https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/ | Why structured ops like MatMul are represented the way they are |
| CIRCT | https://circt.llvm.org/ | MLIR-based hardware/circuit IR reference for dialect and pass organization |
| torch-mlir | https://github.com/llvm/torch-mlir | Model-to-MLIR frontend reference for pass organization and test style |
| IREE | https://iree.dev/ | MLIR compiler/runtime reference for lowering pipelines and backend boundaries |
| StableHLO | https://github.com/openxla/stablehlo | Stable ML operation dialect reference for portable model IR conventions |
| CIM-E reference compiler | https://github.com/rpelke/CIM-E | Open-source CiM compiler — study only, not a dependency |
