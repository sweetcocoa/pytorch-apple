"""Extract IR from Qwen2.5-1.5B-Instruct for NPU compilation.

Generates two IR JSON files:
  - qwen2_prefill_ir.json: Prefill phase (seq_len=128)
  - qwen2_decode_ir.json:  Decode phase (seq_len=1, KV cache=max_cache_len)

Decode uses StaticCache-style index_copy for KV updates (no cat, fixed shapes).
KV cache shape is (B, H, max_cache_len, D), enabling generation up to max_cache_len tokens.

Also generates qwen2_kv_mapping.json for explicit prefill↔decode KV tensor linking.

Requires: pip install transformers

Usage:
    python examples/extract_qwen_ir.py [--max-cache-len 2048] [--prefill-seq-len 128]
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn

# Add torch_to_ir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "torch_to_ir"))
from torch_ir import extract_ir  # noqa: E402


def get_qwen_config():
    """Get Qwen2.5-1.5B-Instruct config."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return config


class QwenPrefillWrapper(nn.Module):
    """Wraps Qwen2 model for prefill IR extraction.

    Flattens past_key_values from (layer, 2, B, H, S, D) to individual tensor args.
    Returns flattened outputs including updated KV cache.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, cache_position):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = outputs.logits

        # Flatten KV cache: list of (key, value) tuples → flat list
        result = [logits]
        for layer_kv in outputs.past_key_values:
            result.append(layer_kv[0])  # key
            result.append(layer_kv[1])  # value
        return tuple(result)


class _IndexCopyCache:
    """Minimal KV cache using out-of-place index_copy instead of cat.

    Compatible with transformers' cache interface (update, get_seq_length).
    Returns the full pre-allocated buffer on update(), so attention operates
    on the fixed max_cache_len dimension.
    """

    def __init__(self, kv_flat, num_layers):
        self.key_cache = [kv_flat[2 * i] for i in range(num_layers)]
        self.value_cache = [kv_flat[2 * i + 1] for i in range(num_layers)]
        self._seen_tokens = self.key_cache[0].shape[-2] if num_layers > 0 else 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_position = cache_kwargs["cache_position"]
        self.key_cache[layer_idx] = self.key_cache[layer_idx].index_copy(2, cache_position, key_states)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].index_copy(2, cache_position, value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for i in range(len(self)):
            yield self.key_cache[i], self.value_cache[i]

    def __getitem__(self, idx):
        return self.key_cache[idx], self.value_cache[idx]


class QwenDecodeWrapper(nn.Module):
    """Wraps Qwen2 model for decode (single token) IR extraction.

    Uses _IndexCopyCache for StaticCache-like behavior:
    - KV buffers have fixed shape (B, H, max_cache_len, D)
    - New KV entries are written via index_copy at cache_position
    - No cat operations — attention operates on the full buffer
    """

    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids, cache_position, *past_kv_flat):
        cache = _IndexCopyCache(past_kv_flat, self.num_layers)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )

        result = [outputs.logits]
        for i in range(self.num_layers):
            result.append(cache.key_cache[i])
            result.append(cache.value_cache[i])
        return tuple(result)


def _clean_ir_attrs(ir):
    """Convert non-JSON-serializable attr values (e.g. torch.device) to strings."""
    import json as _json
    for node in ir.nodes:
        for k, v in list(node.attrs.items()):
            try:
                _json.dumps(v)
            except (TypeError, ValueError):
                node.attrs[k] = str(v)


def save_kv_mapping(prefill_ir, decode_ir, num_layers, prefill_seq_len, max_cache_len,
                    path="qwen2_kv_mapping.json"):
    """Save explicit prefill output ↔ decode input/output KV tensor mapping."""
    prefill_kv = prefill_ir.graph_outputs[1:]   # [0] is logits
    _FIXED_INPUTS = {"input_ids", "attention_mask", "position_ids", "cache_position"}
    decode_kv_in = [s for s in decode_ir.graph_inputs if s.name not in _FIXED_INPUTS]
    decode_kv_out = decode_ir.graph_outputs[1:]

    layers = []
    for i in range(num_layers):
        layers.append({
            "layer": i,
            "prefill_key_output": prefill_kv[2 * i].name,
            "prefill_value_output": prefill_kv[2 * i + 1].name,
            "decode_key_input": decode_kv_in[2 * i].name,
            "decode_value_input": decode_kv_in[2 * i + 1].name,
            "decode_key_output": decode_kv_out[2 * i].name,
            "decode_value_output": decode_kv_out[2 * i + 1].name,
        })

    mapping = {
        "num_layers": num_layers,
        "prefill_seq_len": prefill_seq_len,
        "max_cache_len": max_cache_len,
        "layers": layers,
    }
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    return mapping


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Qwen2.5 IR for NPU compilation")
    parser.add_argument("--prefill-seq-len", type=int, default=128, help="Prefill sequence length")
    parser.add_argument("--max-cache-len", type=int, default=2048, help="Max KV cache length for decode")
    args = parser.parse_args()

    prefill_seq_len = args.prefill_seq_len
    max_cache_len = args.max_cache_len

    print("=" * 60)
    print("Qwen2.5-1.5B-Instruct IR Extraction")
    print("=" * 60)

    config = get_qwen_config()
    num_layers = config.num_hidden_layers
    num_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    print("\nModel config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_kv_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  vocab_size: {config.vocab_size}")
    print("\nCompilation config:")
    print(f"  prefill_seq_len: {prefill_seq_len}")
    print(f"  max_cache_len: {max_cache_len}")

    # Patch: transformers calls torch.is_autocast_enabled("meta") in rotary
    # embedding, which fails on meta device. Replace with a no-op context manager.
    from contextlib import nullcontext

    import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod
    _qwen2_mod.maybe_autocast = lambda **_kwargs: nullcontext()

    from transformers import AutoModelForCausalLM

    # Create model on meta device (no memory needed for 1.5B params)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.eval()

    # ── Prefill IR ──
    print("\n--- Prefill IR Extraction ---")
    batch_size = 1

    wrapper = QwenPrefillWrapper(model)
    meta_input = (
        torch.randint(0, config.vocab_size, (batch_size, prefill_seq_len), device="meta"),  # input_ids
        torch.zeros(batch_size, 1, prefill_seq_len, prefill_seq_len, device="meta"),  # 4D causal mask
        torch.arange(prefill_seq_len, device="meta").unsqueeze(0),                           # position_ids
        torch.arange(prefill_seq_len, device="meta"),                                        # cache_position
    )

    print(f"  Input shape: ({batch_size}, {prefill_seq_len})")
    prefill_ir = extract_ir(wrapper, meta_input, model_name="Qwen2.5_Prefill")

    print(f"  Nodes: {len(prefill_ir.nodes)}")
    print(f"  Weights: {len(prefill_ir.weights)}")

    # Show op distribution
    op_counts = {}
    for node in prefill_ir.nodes:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print("  Op distribution:")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {op}: {cnt}")

    _clean_ir_attrs(prefill_ir)
    prefill_path = "qwen2_prefill_ir.json"
    prefill_ir.save(prefill_path)
    print(f"  Saved: {prefill_path}")

    # ── Decode IR ──
    print("\n--- Decode IR Extraction ---")
    decode_wrapper = QwenDecodeWrapper(model, num_layers)

    decode_input_ids = torch.randint(0, config.vocab_size, (batch_size, 1), device="meta")
    # Decode mask covers the full max_cache_len (StaticCache-style)
    decode_attn_mask = torch.zeros(batch_size, 1, 1, max_cache_len, device="meta")
    decode_position_ids = torch.tensor([[prefill_seq_len]], device="meta")
    decode_cache_position = torch.tensor([prefill_seq_len], device="meta")

    # KV cache tensors: (B, H, max_cache_len, D) — full static buffer
    past_kv_args = []
    for _ in range(num_layers):
        past_kv_args.append(torch.randn(batch_size, num_heads, max_cache_len, head_dim, device="meta"))
        past_kv_args.append(torch.randn(batch_size, num_heads, max_cache_len, head_dim, device="meta"))

    decode_inputs = (decode_input_ids, decode_attn_mask, decode_position_ids, decode_cache_position, *past_kv_args)

    print(f"  Input: 1 token + {num_layers * 2} KV cache tensors (max_cache_len={max_cache_len})")
    decode_ir = extract_ir(decode_wrapper, decode_inputs, model_name="Qwen2.5_Decode")

    _clean_ir_attrs(decode_ir)
    print(f"  Nodes: {len(decode_ir.nodes)}")
    decode_path = "qwen2_decode_ir.json"
    decode_ir.save(decode_path)
    print(f"  Saved: {decode_path}")

    # Show op distribution for decode
    op_counts = {}
    for node in decode_ir.nodes:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print("  Op distribution:")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {op}: {cnt}")

    # ── KV Mapping ──
    print("\n--- KV Mapping ---")
    kv_mapping_path = "qwen2_kv_mapping.json"
    mapping = save_kv_mapping(prefill_ir, decode_ir, num_layers, prefill_seq_len, max_cache_len, kv_mapping_path)
    print(f"  Saved: {kv_mapping_path} ({len(mapping['layers'])} layers)")

    # Check for unsupported ops
    from npu_compiler.constraint_checker import SUPPORTED_OPS
    all_ops = set()
    for node in prefill_ir.nodes:
        all_ops.add(node.op_type)
    for node in decode_ir.nodes:
        all_ops.add(node.op_type)

    unsupported = all_ops - SUPPORTED_OPS
    if unsupported:
        print(f"\n⚠ Unsupported ops found: {sorted(unsupported)}")
        print("  Add these to SUPPORTED_OPS and implement codegen handlers.")
    else:
        print("\n✓ All ops are supported!")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
