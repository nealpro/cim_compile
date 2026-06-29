import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


LAYER_WEIGHT_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _require_file(path, description):
    if not path.exists():
        raise SystemExit(f"missing {description}: {path}")
    return path


def _load_json(path):
    _require_file(path, path.name)
    return json.loads(path.read_text())


def _load_tokenizer(model_dir):
    _require_file(model_dir / "tokenizer.json", "SmolLM2 tokenizer.json")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("SmolLM2 prompt mode requires transformers in the Python environment.") from exc

    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def _load_onnx_initializers(model_path):
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise SystemExit(
            "SmolLM2 prompt mode requires onnx. Install it with: uv pip install --python .venv/bin/python onnx"
        ) from exc

    model = onnx.load(model_path, load_external_data=True)
    return {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}


def _runtime_device(requested):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA, but torch.cuda.is_available() is false")
    return device


def _validate_temperature(temperature):
    if not math.isfinite(temperature):
        raise SystemExit("--temperature must be finite")
    if temperature < 0.0:
        raise SystemExit("--temperature must be greater than or equal to 0")
    return float(temperature)


def _select_next_token(next_token_logits, temperature):
    import torch

    if temperature == 0.0:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)
    probabilities = torch.softmax(next_token_logits.float() / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def _runtime_tensor(initializers, name, device):
    import torch

    if name not in initializers:
        raise KeyError(f"ONNX initializer is missing required tensor {name!r}")
    return torch.from_numpy(initializers[name].copy()).to(device=device, dtype=torch.float32)


def _build_rpu_config(tile_size):
    try:
        from aihwkit.simulator.configs import TorchInferenceRPUConfig
    except ImportError as exc:
        raise SystemExit("SmolLM2 analog mode requires aihwkit in the Python environment.") from exc

    rpu_config = TorchInferenceRPUConfig()
    rpu_config.mapping.max_input_size = tile_size
    rpu_config.mapping.max_output_size = tile_size
    rpu_config.mapping.digital_bias = True
    if hasattr(rpu_config.forward, "is_perfect"):
        rpu_config.forward.is_perfect = True
    if hasattr(rpu_config, "drift_compensation"):
        rpu_config.drift_compensation = None
    return rpu_config


class HybridLinear:
    def __init__(self, name, onnx_weight, analog, device, rpu_config):
        self.name = name
        self.analog = analog
        self.weight = None
        self.layer = None

        in_features, out_features = [int(dim) for dim in onnx_weight.shape]
        if analog:
            try:
                from aihwkit.nn import AnalogLinearMapped
            except ImportError as exc:
                raise SystemExit("SmolLM2 analog mode requires aihwkit in the Python environment.") from exc

            layer = AnalogLinearMapped(
                in_features,
                out_features,
                bias=False,
                rpu_config=rpu_config,
            )
            layer.set_weights(onnx_weight.t().contiguous().cpu(), None)
            self.layer = layer.to(device).eval()
        else:
            self.weight = onnx_weight

    def __call__(self, x):
        if self.layer is None:
            return x.matmul(self.weight)

        leading = x.shape[:-1]
        y = self.layer(x.reshape(-1, x.shape[-1]).float())
        return y.reshape(*leading, y.shape[-1])


def _parse_analog_layers(raw, layer_count):
    if raw is None or raw.strip() == "":
        raw = os.environ.get("CIM_COMPILE_SMOLLM2_ANALOG_LAYERS", "1")
    raw = raw.strip().lower()
    if raw in {"all", "*"}:
        return set(range(layer_count))
    if raw in {"none", "0"}:
        return set()
    if raw.isdigit():
        return set(range(min(int(raw), layer_count)))

    selected = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    invalid = [layer for layer in selected if layer < 0 or layer >= layer_count]
    if invalid:
        raise SystemExit(
            f"invalid analog layer index {invalid[0]}; expected 0..{layer_count - 1}"
        )
    return selected


def _rms_norm(x, weight, eps):
    return x * torch_rsqrt_mean_square(x, eps) * weight


def torch_rsqrt_mean_square(x, eps):
    import torch

    return torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)


def _rotate_half(x):
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q, k, position_ids, theta):
    import torch

    head_dim = q.shape[-1]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=q.dtype, device=q.device) / head_dim)
    )
    freqs = torch.einsum("bs,d->bsd", position_ids.to(dtype=q.dtype), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class SmolLM2Runtime:
    def __init__(self, model_path, device, analog_layers=None, tile_size=128, layer_limit=None):
        import torch

        self.torch = torch
        self.model_path = Path(model_path)
        self.model_dir = self.model_path.parent
        self.config = _load_json(self.model_dir / "config.json")
        self.tokenizer = _load_tokenizer(self.model_dir)
        self.device = device
        self.eos_token_id = int(self.config.get("eos_token_id", self.tokenizer.eos_token_id))
        self.hidden_size = int(self.config["hidden_size"])
        configured_layer_count = int(self.config["num_hidden_layers"])
        self.layer_count = configured_layer_count
        if layer_limit is not None:
            self.layer_count = min(int(layer_limit), configured_layer_count)
            if self.layer_count <= 0:
                raise SystemExit("--layer-limit must be greater than zero")
        self.q_heads = int(self.config["num_attention_heads"])
        self.kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = self.hidden_size // self.q_heads
        self.intermediate_size = int(self.config["intermediate_size"])
        self.rms_eps = float(self.config.get("rms_norm_eps", 1.0e-5))
        self.rope_theta = float(self.config.get("rope_theta", 10000.0))
        self.analog_layers = _parse_analog_layers(analog_layers, self.layer_count)

        if self.hidden_size % self.q_heads != 0:
            raise SystemExit("SmolLM2 config hidden_size must be divisible by num_attention_heads")
        if self.q_heads % self.kv_heads != 0:
            raise SystemExit("SmolLM2 config num_attention_heads must be divisible by num_key_value_heads")

        _require_file(self.model_path, "SmolLM2 ONNX model")
        sys.stderr.write(f"loading ONNX initializers from {self.model_path}\n")
        initializers = _load_onnx_initializers(self.model_path)
        self._validate_initializers(initializers)

        rpu_config = _build_rpu_config(tile_size)
        self.embed = _runtime_tensor(initializers, "model.embed_tokens.weight", device)
        final_norm_name = f"model.layers.{configured_layer_count}.final_norm_layernorm.weight"
        self.final_norm = _runtime_tensor(initializers, final_norm_name, device)
        self.layers = [self._load_layer(initializers, index, device, rpu_config) for index in range(self.layer_count)]
        del initializers

        sys.stderr.write(
            "SmolLM2 runtime ready: "
            f"device={device}, active layers={self.layer_count}/{configured_layer_count}, "
            f"AIHWKIT analog layers={len(self.analog_layers)}/{self.layer_count}\n"
        )

    def _validate_initializers(self, initializers):
        missing = []
        required = [
            "model.embed_tokens.weight",
            f"model.layers.{self.config['num_hidden_layers']}.final_norm_layernorm.weight",
        ]
        for layer in range(self.layer_count):
            prefix = f"model.layers.{layer}"
            required.extend(
                [
                    f"{prefix}.input_layernorm.weight",
                    f"{prefix}.attn.q_proj.MatMul.weight",
                    f"{prefix}.attn.k_proj.MatMul.weight",
                    f"{prefix}.attn.v_proj.MatMul.weight",
                    f"{prefix}.attn.o_proj.MatMul.weight",
                    f"{prefix}.post_attention_layernorm.weight",
                    f"{prefix}.mlp.gate_proj.MatMul.weight",
                    f"{prefix}.mlp.up_proj.MatMul.weight",
                    f"{prefix}.mlp.down_proj.MatMul.weight",
                ]
            )
        for name in required:
            if name not in initializers:
                missing.append(name)
        if missing:
            raise SystemExit(f"SmolLM2 ONNX is missing required initializer {missing[0]!r}")

    def _load_layer(self, initializers, index, device, rpu_config):
        prefix = f"model.layers.{index}"
        analog = index in self.analog_layers
        return {
            "input_norm": _runtime_tensor(initializers, f"{prefix}.input_layernorm.weight", device),
            "post_norm": _runtime_tensor(initializers, f"{prefix}.post_attention_layernorm.weight", device),
            "q": HybridLinear(
                f"{prefix}.attn.q_proj",
                _runtime_tensor(initializers, f"{prefix}.attn.q_proj.MatMul.weight", device),
                analog,
                device,
                rpu_config,
            ),
            "k": HybridLinear(
                f"{prefix}.attn.k_proj",
                _runtime_tensor(initializers, f"{prefix}.attn.k_proj.MatMul.weight", device),
                analog,
                device,
                rpu_config,
            ),
            "v": HybridLinear(
                f"{prefix}.attn.v_proj",
                _runtime_tensor(initializers, f"{prefix}.attn.v_proj.MatMul.weight", device),
                analog,
                device,
                rpu_config,
            ),
            "o": HybridLinear(
                f"{prefix}.attn.o_proj",
                _runtime_tensor(initializers, f"{prefix}.attn.o_proj.MatMul.weight", device),
                analog,
                device,
                rpu_config,
            ),
            "gate": HybridLinear(
                f"{prefix}.mlp.gate_proj",
                _runtime_tensor(initializers, f"{prefix}.mlp.gate_proj.MatMul.weight", device),
                analog,
                device,
                rpu_config,
            ),
            "up": HybridLinear(
                f"{prefix}.mlp.up_proj",
                _runtime_tensor(initializers, f"{prefix}.mlp.up_proj.MatMul.weight", device),
                analog,
                device,
                rpu_config,
            ),
            "down": HybridLinear(
                f"{prefix}.mlp.down_proj",
                _runtime_tensor(initializers, f"{prefix}.mlp.down_proj.MatMul.weight", device),
                analog,
                device,
                rpu_config,
            ),
        }

    def encode_prompt(self, prompt, use_chat_template=True, max_context=128):
        rendered = prompt
        if use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.shape[1] > max_context:
            sys.stderr.write(
                f"prompt encoded to {input_ids.shape[1]} tokens; keeping the last {max_context}\n"
            )
            input_ids = input_ids[:, -max_context:]
        if input_ids.numel() == 0:
            raise SystemExit("--prompt encoded to zero token IDs")
        return input_ids

    def _attention(self, layer, x, position_ids, cache):
        import torch

        batch_size, seq_len, _ = x.shape
        q = layer["q"](x).view(batch_size, seq_len, self.q_heads, self.head_dim).transpose(1, 2)
        k = layer["k"](x).view(batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = layer["v"](x).view(batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, position_ids, self.rope_theta)

        if cache is not None:
            k = torch.cat((cache["key"], k), dim=2)
            v = torch.cat((cache["value"], v), dim=2)
        next_cache = {"key": k, "value": v}

        if self.q_heads != self.kv_heads:
            repeat = self.q_heads // self.kv_heads
            k_for_attention = k.repeat_interleave(repeat, dim=1)
            v_for_attention = v.repeat_interleave(repeat, dim=1)
        else:
            k_for_attention = k
            v_for_attention = v

        scores = torch.matmul(q, k_for_attention.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_len = int(k_for_attention.shape[2])
        past_len = key_len - seq_len
        query_positions = torch.arange(
            past_len, past_len + seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(-1)
        key_positions = torch.arange(key_len, dtype=torch.long, device=self.device).unsqueeze(0)
        causal_mask = key_positions > query_positions
        scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, key_len), float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        context = torch.matmul(weights, v_for_attention)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return layer["o"](context), next_cache

    def forward(self, input_ids, caches=None, start_pos=0):
        import torch
        import torch.nn.functional as F

        hidden = self.embed[input_ids]
        seq_len = int(input_ids.shape[1])
        position_ids = torch.arange(start_pos, start_pos + seq_len, device=self.device).unsqueeze(0)
        next_caches = []

        for index, layer in enumerate(self.layers):
            cache = caches[index] if caches is not None else None
            residual = hidden
            normed = _rms_norm(hidden, layer["input_norm"], self.rms_eps)
            attn_output, next_cache = self._attention(layer, normed, position_ids, cache)
            hidden = residual + attn_output

            residual = hidden
            normed = _rms_norm(hidden, layer["post_norm"], self.rms_eps)
            gated = F.silu(layer["gate"](normed)) * layer["up"](normed)
            hidden = residual + layer["down"](gated)
            next_caches.append(next_cache)

        hidden = _rms_norm(hidden, self.final_norm, self.rms_eps)
        logits = hidden.matmul(self.embed.t())
        return logits, next_caches

    def generate(
        self,
        prompt,
        max_new_tokens=16,
        max_context=128,
        use_chat_template=True,
        temperature=0.8,
    ):
        import torch

        input_ids = self.encode_prompt(prompt, use_chat_template=use_chat_template, max_context=max_context)
        generated = input_ids
        caches = None
        next_input = input_ids
        start_pos = 0
        new_tokens = []

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits, caches = self.forward(next_input, caches=caches, start_pos=start_pos)
                start_pos += int(next_input.shape[1])
                next_token = _select_next_token(logits[:, -1, :], temperature)
                token_id = int(next_token.item())
                new_tokens.append(token_id)
                generated = torch.cat((generated, next_token), dim=1)
                next_input = next_token
                if token_id == self.eos_token_id:
                    break

        return self.tokenizer.decode(new_tokens, skip_special_tokens=True), generated[0].tolist()


def main():
    parser = argparse.ArgumentParser(description="Run SmolLM2 prompt generation with AIHWKIT projections.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-context", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--analog-layers", default=None)
    parser.add_argument("--layer-limit", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    temperature = _validate_temperature(args.temperature)

    device = _runtime_device(args.device)
    runtime = SmolLM2Runtime(
        Path(args.model),
        device=device,
        analog_layers=args.analog_layers,
        tile_size=args.tile_size,
        layer_limit=args.layer_limit,
    )
    if args.inspect_only:
        print(
            json.dumps(
                {
                    "device": str(device),
                    "layers": runtime.layer_count,
                    "analog_layers": sorted(runtime.analog_layers),
                    "vocab_size": int(runtime.embed.shape[0]),
                    "hidden_size": int(runtime.hidden_size),
                },
                sort_keys=True,
            )
        )
        return
    text, _generated_ids = runtime.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_context=args.max_context,
        use_chat_template=not args.raw_prompt,
        temperature=temperature,
    )
    print(text)


if __name__ == "__main__":
    main()
