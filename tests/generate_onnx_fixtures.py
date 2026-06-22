#!/usr/bin/env python3
"""Generate ONNX fixtures for cim_compile tests.

Default mode is pure Python and writes a minimal protobuf subset matching
proto/onnx_minimal.proto3. Torch modes use PyTorch export and are intended for
local/full testing when torch+onnx are installed.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


BF16 = 16
FLOAT = 1


def key(field: int, wire_type: int) -> bytes:
    return varint((field << 3) | wire_type)


def varint(value: int) -> bytes:
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def len_field(field: int, payload: bytes) -> bytes:
    return key(field, 2) + varint(len(payload)) + payload


def varint_field(field: int, value: int) -> bytes:
    return key(field, 0) + varint(value)


def string_field(field: int, value: str) -> bytes:
    return len_field(field, value.encode())


def bytes_field(field: int, value: bytes) -> bytes:
    return len_field(field, value)


def bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
    return struct.pack("<H", bits)


def tensor(name: str, dims: list[int], data_type: int, raw_data: bytes) -> bytes:
    out = bytearray()
    for dim in dims:
        out += varint_field(1, dim)
    out += varint_field(2, data_type)
    out += string_field(8, name)
    out += bytes_field(9, raw_data)
    return bytes(out)


def graph(name: str, initializers: list[bytes]) -> bytes:
    out = bytearray()
    out += string_field(2, name)
    for initializer in initializers:
        out += len_field(5, initializer)
    return bytes(out)


def model(graph_payload: bytes) -> bytes:
    return len_field(7, graph_payload)


def patterned_bf16_matrix(rows: int, cols: int, seed: int) -> bytes:
    values = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]
    out = bytearray()
    for index in range(rows * cols):
        out += bf16(values[(index + seed) % len(values)])
    return bytes(out)


def write_minimal_unrolled(out_dir: Path, dim: int) -> Path:
    initializers = [
        tensor("q_proj.weight", [dim, dim], BF16, patterned_bf16_matrix(dim, dim, 0)),
        tensor("k_proj.weight", [dim, dim], BF16, patterned_bf16_matrix(dim, dim, 1)),
        tensor("v_proj.weight", [dim, dim], BF16, patterned_bf16_matrix(dim, dim, 2)),
        tensor("out_proj.weight", [dim, dim], BF16, patterned_bf16_matrix(dim, dim, 3)),
    ]
    path = out_dir / "memristor_mha_unrolled.onnx"
    path.write_bytes(model(graph("minimal_unrolled", initializers)))
    return path


def write_minimal_fused(out_dir: Path, dim: int) -> Path:
    fused = (
        patterned_bf16_matrix(dim, dim, 0)
        + patterned_bf16_matrix(dim, dim, 1)
        + patterned_bf16_matrix(dim, dim, 2)
    )
    initializers = [
        tensor("mha.in_proj_weight", [dim * 3, dim], BF16, fused),
        tensor("mha.out_proj.weight", [dim, dim], BF16, patterned_bf16_matrix(dim, dim, 3)),
    ]
    path = out_dir / "mha_bfloat16.onnx"
    path.write_bytes(model(graph("minimal_fused", initializers)))
    return path


def write_minimal_linear(out_dir: Path) -> Path:
    raw = b"".join(struct.pack("<f", value) for value in [1.0, -1.0, 0.5, 2.0])
    path = out_dir / "linear_float32.onnx"
    path.write_bytes(model(graph("minimal_linear", [tensor("linear.weight", [2, 2], FLOAT, raw)])))
    return path


def generate_minimal(out_dir: Path, dim: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        write_minimal_unrolled(out_dir, dim),
        write_minimal_fused(out_dir, dim),
        write_minimal_linear(out_dir),
    ]:
        print(path)


def generate_torch_unrolled_mha(out_dir: Path, dim: int, seq_len: int) -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("full fixture generation requires torch and onnx installed") from exc

    class MemristorTargetMHA(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        def forward(self, x):
            batch, seq, embed = x.shape
            q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).contiguous().view(batch, seq, embed)
            return self.out_proj(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    heads = 8 if dim % 8 == 0 else 1
    model_obj = MemristorTargetMHA(dim, heads).bfloat16().eval()
    dummy_input = torch.randn(1, seq_len, dim, dtype=torch.bfloat16)
    torch.onnx.export(
        model_obj,
        dummy_input,
        str(out_dir / "memristor_mha_unrolled.onnx"),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_seq"],
        output_names=["output_seq"],
    )
    print(out_dir / "memristor_mha_unrolled.onnx")


def generate_torch_linear(out_dir: Path, dim: int) -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit("torch-linear fixture generation requires torch and onnx installed") from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    class LinearOnly(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.linear = nn.Linear(dim, dim, bias=True)

        def forward(self, x):
            return self.linear(x)

    model_obj = LinearOnly(dim).float().eval()
    dummy_input = torch.randn(2, dim, dtype=torch.float32)
    path = out_dir / "linear_torch.onnx"
    torch.onnx.export(
        model_obj,
        dummy_input,
        str(path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["minimal", "torch-mha", "torch-linear", "all"],
        default="minimal",
    )
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.mode == "minimal":
        generate_minimal(out_dir, args.dim)
    elif args.mode == "torch-mha":
        generate_torch_unrolled_mha(out_dir, args.dim, args.seq_len)
    elif args.mode == "torch-linear":
        generate_torch_linear(out_dir, args.dim)
    else:
        generate_minimal(out_dir, args.dim)
        generate_torch_linear(out_dir, args.dim)
        generate_torch_unrolled_mha(out_dir, args.dim, args.seq_len)


if __name__ == "__main__":
    main()
