"""Weight loader: load weights from safetensors/state_dict with recipe-based transforms."""

from __future__ import annotations

import numpy as np

from npu_compiler.compiled_program import CompiledProgram
from npu_compiler.graph_optimizer import WeightTransformRecipe
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device


def load_weights(
    state_dict: dict[str, np.ndarray],
    program: CompiledProgram,
    device: Device,
) -> dict[str, NPUBuffer]:
    """Load weights from a state dict and apply transform recipes.

    Activation 패딩은 buffer.py/executor.py에서 수행. Weight는 dense 유지 (정상).
    커널이 logical channel 범위만 순회하므로 weight 패딩 불필요.

    Args:
        state_dict: Maps state_dict key → numpy array (FP32).
        program: CompiledProgram with weight specs and recipes.
        device: Metal device.

    Returns:
        Dict mapping placeholder name → NPUBuffer (FP16).
    """
    # Apply BN folding recipes first
    transformed = dict(state_dict)  # shallow copy
    for recipe in program.weight_recipes:
        if recipe.transform == "bn_fold":
            _apply_bn_fold(transformed, recipe)

    # Build reverse mapping: state_dict key → placeholder name
    reverse_map: dict[str, str] = {}
    for placeholder, sd_key in program.weight_name_mapping.items():
        reverse_map[sd_key] = placeholder

    # Convert to NPU buffers
    buffers: dict[str, NPUBuffer] = {}
    for w_spec in program.weight_specs:
        sd_key = w_spec.name
        placeholder = reverse_map.get(sd_key, sd_key)

        # Skip zero-size weights (e.g., empty KV cache init tensors)
        if w_spec.shape == [0]:
            continue

        if sd_key in transformed:
            arr = transformed[sd_key]
            buffers[placeholder] = NPUBuffer.from_numpy(arr, device)
        else:
            raise KeyError(f"Weight '{sd_key}' not found in state dict. "
                           f"Available: {sorted(transformed.keys())}")

    return buffers


def load_weights_from_safetensors(
    path: str,
    program: CompiledProgram,
    device: Device,
) -> dict[str, NPUBuffer]:
    """Load weights from a safetensors file."""
    from safetensors.numpy import load_file
    state_dict = load_file(path)
    return load_weights(state_dict, program, device)



def _apply_bn_fold(state_dict: dict[str, np.ndarray], recipe: WeightTransformRecipe):
    """Fold BN parameters into conv weight and bias in-place."""
    p = recipe.params
    eps = p.get("eps", 1e-5)

    conv_w = state_dict[p["conv_weight"]]
    conv_b = state_dict.get(p["conv_bias"]) if p["conv_bias"] else np.zeros(conv_w.shape[0], dtype=np.float32)

    bn_gamma = state_dict[p["bn_gamma"]] if p["bn_gamma"] else np.ones(conv_w.shape[0], dtype=np.float32)
    bn_beta = state_dict[p["bn_beta"]] if p["bn_beta"] else np.zeros(conv_w.shape[0], dtype=np.float32)
    bn_mean = state_dict[p["bn_mean"]] if p["bn_mean"] else np.zeros(conv_w.shape[0], dtype=np.float32)
    bn_var = state_dict[p["bn_var"]] if p["bn_var"] else np.ones(conv_w.shape[0], dtype=np.float32)

    std = np.sqrt(bn_var + eps)
    scale = bn_gamma / std

    # Fold into conv weights and bias
    state_dict[p["conv_weight"]] = conv_w * scale.reshape(-1, 1, 1, 1)
    folded_bias = (conv_b - bn_mean) * scale + bn_beta

    if p["conv_bias"]:
        state_dict[p["conv_bias"]] = folded_bias
    else:
        # Create a bias entry if conv didn't have one
        # Use a derived key
        state_dict[p["conv_weight"].replace(".weight", ".bias")] = folded_bias
