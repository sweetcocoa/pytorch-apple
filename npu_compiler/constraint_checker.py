"""Validate NPU constraints on an IR graph."""

from __future__ import annotations

from dataclasses import dataclass

from npu_compiler.ir_reader import IRGraph
from npu_compiler.target_config import METAL_GPU, pad_channels, padded_shape_4d  # noqa: F401

# Auto-derived from codegen.HANDLED_OPS — the single source of truth for which
# ops the compiler can process. Adding a new op only requires updating codegen.py
# (handler) and the corresponding .metal shader; this file needs no changes.
from npu_compiler.codegen import HANDLED_OPS as SUPPORTED_OPS  # noqa: F401

# Metal GPU requires 64-byte aligned access for coalesced reads; for FP16/BF16
# (2 bytes each), this means channels must be padded to multiples of 32 elements.
CHANNEL_ALIGNMENT = METAL_GPU.channel_alignment_bytes  # bytes (32 FP16 elements)
CHANNEL_TILE = METAL_GPU.channel_tile                   # FP16 elements per tile (64 bytes / 2 bytes)


@dataclass
class ConstraintViolation:
    node_name: str
    message: str


def check_constraints(graph: IRGraph) -> list[ConstraintViolation]:
    """Check NPU constraints. Returns list of violations (empty = OK)."""
    violations = []

    for node in graph.nodes:
        if node.op_type not in SUPPORTED_OPS:
            violations.append(ConstraintViolation(
                node.name,
                f"Unsupported op: {node.op_type}. "
                f"Supported: {sorted(SUPPORTED_OPS)}"
            ))

        # Ops that pass through zero-size tensors (empty KV cache)
        _ZERO_SHAPE_OK = {"aten.cat.default", "aten.detach_.default", "aten.alias.default"}

        for tensor in node.inputs + node.outputs:
            if any(d < 0 for d in tensor.shape):
                violations.append(ConstraintViolation(
                    node.name,
                    f"Dynamic/invalid shape in tensor '{tensor.name}': {tensor.shape}. "
                    "NPU requires static shapes."
                ))
            elif (any(d == 0 for d in tensor.shape)
                  and node.op_type not in _ZERO_SHAPE_OK):
                violations.append(ConstraintViolation(
                    node.name,
                    f"Zero-size tensor '{tensor.name}': {tensor.shape}. "
                    "NPU requires static shapes."
                ))

    return violations
