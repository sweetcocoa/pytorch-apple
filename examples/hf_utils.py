"""HuggingFace weight loading utilities for examples/tests."""

from __future__ import annotations

import glob
import json
import os

import ml_dtypes  # noqa: F401
import numpy as np

from npu_compiler.compiled_program import CompiledProgram
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device
from npu_runtime.weight_loader import load_weights


def load_weights_from_hf(
    model_id: str,
    program: CompiledProgram,
    device: Device,
) -> dict[str, NPUBuffer]:
    """Load weights from a HuggingFace model (downloads if not cached).

    Supports both single-file and multi-shard safetensors models.
    Only loads weight keys that the program actually needs.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
                  or local path to a model directory.
        program: CompiledProgram with weight specs and name mapping.
        device: Metal device.

    Returns:
        Dict mapping placeholder name → NPUBuffer.
    """
    import torch
    from safetensors.torch import load_file as _load_file_torch

    compute_dtype = program.compute_dtype

    def load_file(path):
        """Load safetensors via torch, return numpy dict.

        For bfloat16 compute: zero-copy bitcast via uint16 view.
        For float16 compute: convert to fp16 as before.
        """
        tensors = _load_file_torch(path, device="cpu")
        result = {}
        for k, v in tensors.items():
            if compute_dtype == "bfloat16":
                if v.dtype == torch.bfloat16:
                    result[k] = v.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
                else:
                    result[k] = v.to(torch.bfloat16).view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            else:
                result[k] = v.half().numpy()
        return result

    # Resolve model directory: local path or HuggingFace hub download
    if os.path.isdir(model_id):
        model_dir = model_id
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.json"],
        )

    # Determine which state_dict keys we need
    needed_keys: set[str] = set()
    for w_spec in program.weight_specs:
        needed_keys.add(w_spec.name)

    # Try to use weight index for efficient loading
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        # Group needed keys by shard file
        shard_keys: dict[str, list[str]] = {}
        for key in needed_keys:
            shard_file = weight_map.get(key)
            if shard_file:
                shard_keys.setdefault(shard_file, []).append(key)

        state_dict: dict[str, np.ndarray] = {}
        for shard_file, keys in shard_keys.items():
            shard_path = os.path.join(model_dir, shard_file)
            shard_data = load_file(shard_path)
            for key in keys:
                if key in shard_data:
                    state_dict[key] = shard_data[key]
    else:
        # Single safetensors file or glob all shards
        st_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
        state_dict = {}
        for f in sorted(st_files):
            data = load_file(f)
            state_dict.update(data)

    # Wrapper may add a prefix to weight names (e.g., "model." from self.model).
    # Build a mapping from safetensors keys to IR weight names by trying prefixes.
    raw_keys = set(state_dict.keys())
    remapped = {}
    for ir_key in needed_keys:
        if ir_key in raw_keys:
            remapped[ir_key] = state_dict[ir_key]
        else:
            # Try stripping "model." prefix from IR key to match safetensors key
            for prefix in ("model.",):
                stripped = ir_key[len(prefix):] if ir_key.startswith(prefix) else None
                if stripped and stripped in raw_keys:
                    remapped[ir_key] = state_dict[stripped]
                    break
    state_dict = remapped

    # Handle weight tying (e.g., lm_head.weight == embed_tokens.weight)
    for key in needed_keys:
        if key not in state_dict and "lm_head.weight" in key:
            embed_key = key.replace("lm_head.weight", "model.embed_tokens.weight")
            if embed_key in state_dict:
                state_dict[key] = state_dict[embed_key]

    # Generate missing computed weights
    np_compute_dtype = ml_dtypes.bfloat16 if compute_dtype == "bfloat16" else np.float16
    for w_spec in program.weight_specs:
        if w_spec.name in state_dict:
            continue
        # inv_freq: computed from config, not stored in safetensors
        if "inv_freq" in w_spec.name:
            dim = w_spec.shape[0]
            inv_freq = 1.0 / (10000.0 ** (np.arange(dim, dtype=np.float32) / dim))
            state_dict[w_spec.name] = inv_freq.astype(np_compute_dtype)
        # lifted_tensor: empty KV cache init (shape [0])
        elif "lifted_tensor" in w_spec.name and w_spec.shape == [0]:
            state_dict[w_spec.name] = np.array([], dtype=np_compute_dtype)

    return load_weights(state_dict, program, device)
