# Reference Links

## For M1 — Frontend / ONNX Parsing

| Resource | URL | Notes |
|----------|-----|-------|
| ONNX Operator Spec | https://onnx.ai/onnx/operators/ | MatMul and Conv ops — industry vocabulary for describing ML ops |
| prost docs | https://docs.rs/prost | Protobuf decode — `Message::decode`, generated types |

## Compiler Architecture / IR Design

| Resource | URL | Notes |
|----------|-----|-------|
| MLIR Toy Tutorial | https://mlir.llvm.org/docs/Tutorials/Toy/ | Conceptual model for IRs and lowering passes — no install needed |
| MLIR Linalg Dialect Rationale | https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/ | Why structured ops like MatMul are represented the way they are |
| CIM-E reference compiler | https://github.com/rpelke/CIM-E | Open-source CiM compiler — study only, not a dependency |

## For M3 — RISC-V Backend

| Resource | URL | Notes |
|----------|-----|-------|
| RISC-V ISA Spec | https://github.com/riscv/riscv-isa-manual | Chapter 2 for RV32I base — branch, load/store, registers |
