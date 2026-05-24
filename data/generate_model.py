from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemristorTargetMHA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, S, E = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scaling = 1.0 / (self.head_dim**0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, E)
        return self.out_proj(out)


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    onnx_path = out_dir / "memristor_mha_unrolled.onnx"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MemristorTargetMHA(embed_dim=512, num_heads=8).to(device).bfloat16()
    model.eval()

    dummy_input = torch.randn(1, 128, 512, dtype=torch.bfloat16, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_seq"],
        output_names=["output_seq"],
    )

    data_path = Path(str(onnx_path) + ".data")
    print(f"wrote {onnx_path}")
    if data_path.exists():
        print(f"wrote {data_path}")
