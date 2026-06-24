import argparse
import copy
import json
import math
import os
import struct
from pathlib import Path


def _safe_key(name):
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _configure_local_caches(anchor):
    cache_root = anchor / ".cim_compile_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def _load_tile(raw, tile):
    rows, cols = tile["tile_size"]
    count = rows * cols
    offset = tile["weight_offset"]
    payload = raw[offset : offset + count]
    if len(payload) != count:
        raise ValueError(f"tile at offset {offset} has {len(payload)} bytes, expected {count}")
    values = struct.unpack(f"{count}b", payload)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc
    return torch.tensor(values, dtype=torch.float32).reshape(rows, cols) * float(
        tile["quant_scale"]
    )


def _load_digital_tensors(manifest, digital_path):
    if not manifest.get("digital_tensors"):
        return {}
    if digital_path is None:
        raise ValueError("manifest has digital tensors but no digital tensor payload path")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc

    raw = digital_path.read_bytes()
    tensors = {}
    for tensor in manifest["digital_tensors"]:
        if tensor["dtype"] != "f32":
            raise ValueError(f"unsupported digital tensor dtype {tensor['dtype']}")
        shape = [int(dim) for dim in tensor["shape"]]
        count = math.prod(shape)
        offset = int(tensor["byte_offset"])
        byte_len = int(tensor["byte_len"])
        expected = count * 4
        if byte_len != expected:
            raise ValueError(
                f"digital tensor {tensor['role']} has {byte_len} bytes, expected {expected}"
            )
        payload = bytearray(raw[offset : offset + byte_len])
        if len(payload) != byte_len:
            raise ValueError(
                f"digital tensor {tensor['role']} payload is truncated at offset {offset}"
            )
        value = torch.frombuffer(payload, dtype=torch.float32, count=count).clone().reshape(shape)
        tensors[tensor["role"]] = value
    return tensors


def build_torch_model(manifest, raw_weights):
    import torch
    import torch.nn as nn

    class ProjectionBank(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict()
            self.key_by_name = {}

        def forward_projection(self, name, x):
            layer = self.layers[self.key_by_name[name]]
            if x.dim() <= 2:
                return layer(x)
            leading = x.shape[:-1]
            y = layer(x.reshape(-1, x.shape[-1]))
            return y.reshape(*leading, y.shape[-1])

    model = ProjectionBank()
    for projection in manifest["projections"]:
        key = _safe_key(projection["name"])
        model.key_by_name[projection["name"]] = key
        layer = nn.Linear(
            projection["cols"],
            projection["rows"],
            bias=projection.get("bias") is not None,
        )
        full = torch.zeros((projection["rows"], projection["cols"]), dtype=torch.float32)
        for tile in projection["tiles"]:
            tile_tensor = _load_tile(raw_weights, tile)
            row0 = tile["row"] * tile["tile_size"][0]
            col0 = tile["col"] * tile["tile_size"][1]
            row1 = min(row0 + tile["tile_size"][0], projection["rows"])
            col1 = min(col0 + tile["tile_size"][1], projection["cols"])
            full[row0:row1, col0:col1] = tile_tensor[: row1 - row0, : col1 - col0]
        with torch.no_grad():
            layer.weight.copy_(full)
            if projection.get("bias") is not None:
                layer.bias.copy_(torch.tensor(projection["bias"], dtype=torch.float32))
        model.layers[key] = layer
    return model


def patch_with_memtorch(model, manifest):
    try:
        import torch
        import memtorch
        from memtorch.bh.crossbar.Program import naive_program
        from memtorch.map.Input import naive_scale
        from memtorch.map.Parameter import naive_map
        from memtorch.mn.Module import patch_model
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc

    return patch_model(
        copy.deepcopy(model),
        memristor_model=memtorch.bh.memristor.VTEAM,
        memristor_model_params={},
        module_parameters_to_patch=[torch.nn.Linear],
        mapping_routine=naive_map,
        transistor=True,
        programming_routine=naive_program,
        tile_shape=tuple(manifest["tile_size"]),
        max_input_voltage=0.3,
        scaling_routine=naive_scale,
        ADC_resolution=8,
        ADC_overflow_rate=0.0,
        quant_method="linear",
    )


def _projection_by_stage(manifest, stage):
    for projection in manifest["projections"]:
        if projection["stage"] == stage:
            return projection
    raise ValueError(f"manifest is missing projection stage {stage}")


def _digital_tensor(digital_tensors, role):
    try:
        return digital_tensors[role]
    except KeyError as exc:
        raise ValueError(f"manifest digital tensors are missing role {role}") from exc


def _rms_norm(x, weight, eps=1.0e-6):
    import torch

    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _rotate_half(x):
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(q, k, position_ids):
    import torch

    head_dim = q.shape[-1]
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, dtype=q.dtype, device=q.device) / head_dim)
    )
    freqs = torch.einsum("bt,d->btd", position_ids.to(dtype=q.dtype), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _run_attention_core(sim_model, manifest, x, position_ids=None, kv_cache=None, use_cache=False):
    import torch

    block = next(
        (block for block in manifest.get("attention_blocks", []) if block.get("metadata")),
        None,
    )
    if block is None:
        raise ValueError("manifest has no attention block metadata")

    metadata = block["metadata"]
    q_proj = _projection_by_stage(manifest, "attention.query_projection")
    k_proj = _projection_by_stage(manifest, "attention.key_projection")
    v_proj = _projection_by_stage(manifest, "attention.value_projection")
    out_proj = _projection_by_stage(manifest, "attention.output_projection")

    batch_size, seq_len, _ = x.shape
    head_dim = int(metadata["head_dim"])
    q_heads = int(metadata["q_heads"])
    kv_heads = int(metadata["kv_heads"])
    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)

    past_len = 0
    if kv_cache is not None:
        past_len = int(kv_cache["key"].shape[2])

    q = sim_model.forward_projection(q_proj["name"], x)
    k_current = sim_model.forward_projection(k_proj["name"], x)
    v_current = sim_model.forward_projection(v_proj["name"], x)

    q = q.reshape(batch_size, seq_len, q_heads, head_dim).transpose(1, 2)
    k_current = k_current.reshape(batch_size, seq_len, kv_heads, head_dim).transpose(1, 2)
    v_current = v_current.reshape(batch_size, seq_len, kv_heads, head_dim).transpose(1, 2)
    q, k_current = _apply_rotary(q, k_current, position_ids)

    if kv_cache is None:
        k_for_attention = k_current
        v_for_attention = v_current
    else:
        k_for_attention = torch.cat((kv_cache["key"], k_current), dim=2)
        v_for_attention = torch.cat((kv_cache["value"], v_current), dim=2)

    next_cache = None
    if use_cache:
        next_cache = {"key": k_for_attention, "value": v_for_attention}

    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError(f"q_heads={q_heads} must be a multiple of kv_heads={kv_heads}")
        repeat = q_heads // kv_heads
        k = k_for_attention.repeat_interleave(repeat, dim=1)
        v = v_for_attention.repeat_interleave(repeat, dim=1)
    else:
        k = k_for_attention
        v = v_for_attention

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    key_len = int(k.shape[2])
    query_positions = torch.arange(
        past_len, past_len + seq_len, dtype=torch.long, device=scores.device
    ).unsqueeze(-1)
    key_positions = torch.arange(key_len, dtype=torch.long, device=scores.device).unsqueeze(0)
    causal_mask = key_positions > query_positions
    scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, key_len), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    context = torch.matmul(weights, v)
    context = context.transpose(1, 2).contiguous().reshape(batch_size, seq_len, q_heads * head_dim)
    output = sim_model.forward_projection(out_proj["name"], context)
    debug = {
        "block": block["name"],
        "metadata": metadata,
        "input_shape": list(x.shape),
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "scores_shape": list(scores.shape),
        "context_shape": list(context.shape),
        "output_shape": list(output.shape),
        "cache_key_shape": list(next_cache["key"].shape) if next_cache else None,
        "cache_value_shape": list(next_cache["value"].shape) if next_cache else None,
        "output_sum": float(output.sum().item()),
        "output_mean": float(output.mean().item()),
    }
    return output, debug, next_cache


def _run_projection_smoke(sim_model, manifest, batch_size):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc

    results = {}
    for projection in manifest["projections"]:
        x = torch.randn(batch_size, projection["cols"], dtype=torch.float32)
        with torch.no_grad():
            y = sim_model.forward_projection(projection["name"], x)
        results[projection["name"]] = {
            "shape": list(y.shape),
            "sum": float(y.sum().item()),
            "mean": float(y.mean().item()),
        }
    return results


def run_attention_slice(sim_model, manifest, batch_size, seq_len):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc

    if not any(block.get("metadata") for block in manifest.get("attention_blocks", [])):
        return None

    metadata = next(block["metadata"] for block in manifest["attention_blocks"] if block.get("metadata"))
    hidden = int(metadata["hidden_size"])
    x = torch.randn(batch_size, seq_len, hidden, dtype=torch.float32)

    with torch.no_grad():
        _, debug, _ = _run_attention_core(sim_model, manifest, x)
    return debug


def _decoder_tensors(digital_tensors):
    return {
        "embedding": _digital_tensor(digital_tensors, "token_embedding"),
        "input_norm_weight": _digital_tensor(digital_tensors, "input_layernorm_weight"),
        "post_norm_weight": _digital_tensor(
            digital_tensors, "post_attention_layernorm_weight"
        ),
        "final_norm_weight": _digital_tensor(digital_tensors, "final_norm_weight"),
        "lm_head": _digital_tensor(digital_tensors, "lm_head_weight"),
    }


def _topk_entries(logits, top_k):
    import torch

    k = max(1, min(int(top_k), logits.shape[-1]))
    top_values, top_indices = torch.topk(logits[:, -1, :], k=k, dim=-1)
    return [
        {"token_id": int(index), "score": float(score)}
        for index, score in zip(top_indices[0].tolist(), top_values[0].tolist())
    ]


def _run_decoder_tokens(
    sim_model,
    manifest,
    tensors,
    token_ids,
    start_position=0,
    kv_cache=None,
    use_cache=False,
):
    import torch

    ids = torch.tensor([token_ids], dtype=torch.long)
    if ids.numel() == 0:
        raise ValueError("token_ids must contain at least one token ID")
    embedding = tensors["embedding"]
    if ids.min().item() < 0 or ids.max().item() >= embedding.shape[0]:
        raise ValueError(
            f"token_ids must be in [0, {embedding.shape[0] - 1}], got {token_ids}"
        )

    gate_proj = _projection_by_stage(manifest, "mlp.gate_projection")
    up_proj = _projection_by_stage(manifest, "mlp.up_projection")
    down_proj = _projection_by_stage(manifest, "mlp.down_projection")

    embedding_output = embedding[ids]
    normed = _rms_norm(embedding_output, tensors["input_norm_weight"])
    position_ids = torch.arange(
        start_position, start_position + ids.shape[1], dtype=torch.long
    ).unsqueeze(0)
    attn_output, attention_debug, next_cache = _run_attention_core(
        sim_model,
        manifest,
        normed,
        position_ids=position_ids,
        kv_cache=kv_cache,
        use_cache=use_cache,
    )
    hidden = embedding_output + attn_output
    mlp_input = _rms_norm(hidden, tensors["post_norm_weight"])
    gate = sim_model.forward_projection(gate_proj["name"], mlp_input)
    up = sim_model.forward_projection(up_proj["name"], mlp_input)
    gated = (gate * torch.sigmoid(gate)) * up
    mlp_output = sim_model.forward_projection(down_proj["name"], gated)
    hidden = hidden + mlp_output
    final = _rms_norm(hidden, tensors["final_norm_weight"])
    logits = torch.matmul(final, tensors["lm_head"])
    debug = {
        "input_ids": token_ids,
        "position_ids": position_ids[0].tolist(),
        "logits_shape": list(logits.shape),
        "stage_outputs": {
            "embedding_shape": list(embedding_output.shape),
            "mlp_gate_shape": list(gate.shape),
            "mlp_up_shape": list(up.shape),
            "mlp_output_shape": list(mlp_output.shape),
        },
        "attention": attention_debug,
    }
    return logits, next_cache, debug


def run_token_logits(sim_model, manifest, digital_tensors, input_ids, top_k):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc

    if not manifest.get("inference_slice"):
        return None
    tensors = _decoder_tensors(digital_tensors)

    with torch.no_grad():
        logits, _, debug = _run_decoder_tokens(
            sim_model, manifest, tensors, input_ids, use_cache=False
        )

    return {
        "mode": manifest["inference_slice"]["inference_mode"],
        "input_ids": input_ids,
        "logits_shape": list(logits.shape),
        "next_token_topk": _topk_entries(logits, top_k),
        "stage_outputs": debug["stage_outputs"],
        "attention": debug["attention"],
    }


def _cache_shapes(kv_cache):
    if kv_cache is None:
        return None
    return {
        "layers": 1,
        "key": list(kv_cache["key"].shape),
        "value": list(kv_cache["value"].shape),
    }


def run_token_generation(
    sim_model,
    manifest,
    digital_tensors,
    input_ids,
    max_new_tokens,
    top_k,
    eos_token_id=None,
):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc

    if not manifest.get("inference_slice"):
        return None
    if max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be greater than or equal to zero")

    tensors = _decoder_tensors(digital_tensors)
    generated_ids = list(input_ids)
    new_token_ids = []
    per_step_topk = []
    kv_cache = None
    next_input = list(input_ids)
    next_position = 0
    stop_reason = "max_new_tokens"

    with torch.no_grad():
        if max_new_tokens == 0:
            _, kv_cache, _ = _run_decoder_tokens(
                sim_model,
                manifest,
                tensors,
                next_input,
                start_position=next_position,
                kv_cache=None,
                use_cache=True,
            )
        for step in range(max_new_tokens):
            logits, kv_cache, debug = _run_decoder_tokens(
                sim_model,
                manifest,
                tensors,
                next_input,
                start_position=next_position,
                kv_cache=kv_cache,
                use_cache=True,
            )
            next_position += len(next_input)
            topk = _topk_entries(logits, top_k)
            next_token_id = int(topk[0]["token_id"])
            new_token_ids.append(next_token_id)
            generated_ids.append(next_token_id)
            per_step_topk.append(
                {
                    "step": step,
                    "input_ids": list(next_input),
                    "selected_token_id": next_token_id,
                    "topk": topk,
                    "cache_shapes_after_step": _cache_shapes(kv_cache),
                    "attention": debug["attention"],
                }
            )
            next_input = [next_token_id]
            if eos_token_id is not None and next_token_id == int(eos_token_id):
                stop_reason = "eos_token_id"
                break

        if new_token_ids and stop_reason == "max_new_tokens":
            _, kv_cache, _ = _run_decoder_tokens(
                sim_model,
                manifest,
                tensors,
                next_input,
                start_position=next_position,
                kv_cache=kv_cache,
                use_cache=True,
            )

    return {
        "mode": "generate_ids",
        "prompt_ids": list(input_ids),
        "generated_ids": generated_ids,
        "new_token_ids": new_token_ids,
        "per_step_topk": per_step_topk,
        "stop_reason": stop_reason,
        "decode_steps": len(new_token_ids),
        "cache_shapes": _cache_shapes(kv_cache),
        "eos_token_id": eos_token_id,
    }


def _parse_input_ids(raw):
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--input-ids must contain at least one comma-separated integer")
    return [int(value) for value in values]


def _runtime_simulation_summary(manifest, no_patch):
    summary = dict(manifest.get("simulation_summary") or {})
    memtorch_stages = summary.get(
        "memtorch_stages", [projection["stage"] for projection in manifest.get("projections", [])]
    )
    digital_stages = summary.get("digital_stages", [])
    summary.update(
        {
            "memtorch_patched": not no_patch,
            "patched_projection_count": len(manifest.get("projections", [])),
            "memtorch_stages": memtorch_stages,
            "digital_stages": digital_stages,
            "supported_runtime_modes": summary.get(
                "supported_runtime_modes", ["logits"]
            ),
        }
    )
    return summary


def run(
    manifest_path,
    weights_path=None,
    digital_path=None,
    batch_size=1,
    seq_len=4,
    seed=0,
    no_patch=False,
    input_ids=None,
    top_k=5,
    generate_ids=False,
    max_new_tokens=8,
    eos_token_id=None,
):
    _configure_local_caches(manifest_path.parent)
    manifest = json.loads(manifest_path.read_text())
    weights_path = weights_path or manifest_path.with_name(manifest["weights_file"])
    if digital_path is None and manifest.get("digital_tensors_file"):
        digital_path = manifest_path.with_name(manifest["digital_tensors_file"])
    raw_weights = weights_path.read_bytes()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "MemTorch simulation requires torch and memtorch. Install them in this Python environment first."
        ) from exc
    torch.manual_seed(seed)
    model = build_torch_model(manifest, raw_weights)
    sim_model = model if no_patch else patch_with_memtorch(model, manifest)
    digital_tensors = _load_digital_tensors(manifest, digital_path)

    results = _run_projection_smoke(sim_model, manifest, batch_size)
    attention_slice = run_attention_slice(sim_model, manifest, batch_size, seq_len)
    input_ids = input_ids or [1, 2, 3, 4]
    token_logits = run_token_logits(sim_model, manifest, digital_tensors, input_ids, top_k)
    token_generation = (
        run_token_generation(
            sim_model,
            manifest,
            digital_tensors,
            input_ids,
            max_new_tokens,
            top_k,
            eos_token_id=eos_token_id,
        )
        if generate_ids
        else None
    )
    mode = "generate_ids" if token_generation else token_logits["mode"] if token_logits else "attention_slice"
    simulation_summary = _runtime_simulation_summary(manifest, no_patch)
    return {
        "mode": mode,
        "entry": manifest["entry"],
        "execution_plan": manifest.get("execution_plan", []),
        "attention_blocks": manifest.get("attention_blocks", []),
        "inference_slice": manifest.get("inference_slice"),
        "simulation_summary": simulation_summary,
        "input_ids": input_ids if token_logits else None,
        "logits_shape": token_logits["logits_shape"] if token_logits else None,
        "next_token_topk": token_logits["next_token_topk"] if token_logits else [],
        "attention_slice": attention_slice,
        "token_logits": token_logits,
        "token_generation": token_generation,
        "generated_ids": token_generation["generated_ids"] if token_generation else None,
        "prompt_ids": token_generation["prompt_ids"] if token_generation else None,
        "new_token_ids": token_generation["new_token_ids"] if token_generation else None,
        "per_step_topk": token_generation["per_step_topk"] if token_generation else [],
        "stop_reason": token_generation["stop_reason"] if token_generation else None,
        "decode_steps": token_generation["decode_steps"] if token_generation else 0,
        "cache_shapes": token_generation["cache_shapes"] if token_generation else None,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run cim_compile output through MemTorch.")
    parser.add_argument("--manifest", default="memtorch_manifest.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--digital", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-ids", default="1,2,3,4")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate-ids", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Run the reconstructed PyTorch model without MemTorch patching.",
    )
    args = parser.parse_args()

    result = run(
        Path(args.manifest),
        Path(args.weights) if args.weights else None,
        Path(args.digital) if args.digital else None,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        no_patch=args.no_patch,
        input_ids=_parse_input_ids(args.input_ids),
        top_k=args.top_k,
        generate_ids=args.generate_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=args.eos_token_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
