"""Fused kernel codegen: registry + handlers for multi-op fused kernels.

Each handler maps a FusedGroup.kernel_type to a KernelCall. New fused patterns
can be added via register_fused_codegen() without modifying this file.
"""

from __future__ import annotations

import numpy as np

# Avoid circular import — KernelCall is defined in codegen_core
from npu_compiler.codegen_core import KernelCall
from npu_compiler.fusion_patterns import FusedGroup
from npu_compiler.ir_reader import IRGraph
from npu_compiler.target_config import padded_shape_4d

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FUSED_CODEGEN_REGISTRY: dict[str, callable] = {}


def register_fused_codegen(kernel_type: str, handler: callable):
    """Register a codegen handler for a fused kernel type.

    Handler signature: (group: FusedGroup, graph: IRGraph) -> KernelCall | None
    """
    _FUSED_CODEGEN_REGISTRY[kernel_type] = handler


def generate_fused_kernel_call(group: FusedGroup, graph: IRGraph) -> KernelCall | None:
    handler = _FUSED_CODEGEN_REGISTRY.get(group.kernel_type)
    if handler is not None:
        return handler(group, graph)
    return None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _gen_add_relu_kernel(group: FusedGroup, graph: IRGraph) -> KernelCall:
    add_node = group.nodes[0]
    last_node = group.nodes[-1]
    out_shape = add_node.outputs[0].shape
    padded = padded_shape_4d(out_shape)
    total = int(np.prod(padded))
    return KernelCall(
        kernel_name="add_relu_kernel",
        kernel_source="add_relu.metal",
        input_buffers=[add_node.inputs[0].name, add_node.inputs[1].name],
        output_buffers=[last_node.outputs[0].name],
        param_buffers=[],
        params={},
        dispatch_type="1d",
        total_threads=total,
    )


def _gen_conv_fused_kernel(group: FusedGroup, graph: IRGraph) -> KernelCall:
    # Import here to avoid circular dependency
    from npu_compiler.codegen_ops import gen_conv_kernel

    first_node = group.nodes[0]
    last_node = group.nodes[-1]
    return gen_conv_kernel(
        first_node, graph, has_relu="relu" in group.kernel_type, output_name=last_node.outputs[0].name
    )


def _gen_rmsnorm_kernel(group: FusedGroup, graph: IRGraph | None = None) -> KernelCall:
    """Generate fused RMSNorm kernel from pow->mean->add(eps)->rsqrt->mul(x)->mul(weight) chain."""
    pow_node = group.nodes[0]
    last_node = group.nodes[-1]

    x_name = pow_node.inputs[0].name
    x_shape = pow_node.inputs[0].shape

    add_node = None
    for n in group.nodes:
        if n.op_type == "aten.add.Tensor" and len(n.inputs) == 1:
            add_node = n
            break
    eps = float(add_node.attrs.get("other", 1e-6)) if add_node else 1e-6

    weight_name = None
    for inp in last_node.inputs:
        group_output_names = set()
        for n in group.nodes:
            for out in n.outputs:
                group_output_names.add(out.name)
        if inp.name not in group_output_names:
            weight_name = inp.name
            break
    if weight_name is None:
        weight_name = last_node.inputs[1].name

    rows = int(np.prod(x_shape[:-1])) if len(x_shape) > 1 else 1
    cols = x_shape[-1]

    return KernelCall(
        kernel_name="rmsnorm_kernel",
        kernel_source="rmsnorm.metal",
        input_buffers=[x_name, weight_name],
        output_buffers=[last_node.outputs[0].name],
        param_buffers=["rmsnorm_params"],
        params={"rows": rows, "cols": cols, "eps": eps},
        dispatch_type="1d",
        total_threads=rows,
    )


def _gen_silu_mul_kernel(group: FusedGroup, graph: IRGraph | None = None) -> KernelCall:
    """Generate fused SiLU * gate kernel: silu(gate) * up -> 1 dispatch."""
    silu_node = group.nodes[0]
    mul_node = group.nodes[1]

    gate_name = silu_node.inputs[0].name
    silu_out_name = silu_node.outputs[0].name
    up_name = None
    for inp in mul_node.inputs:
        if inp.name != silu_out_name:
            up_name = inp.name
            break
    if up_name is None:
        up_name = mul_node.inputs[1].name

    total = int(np.prod(mul_node.outputs[0].shape))
    return KernelCall(
        kernel_name="silu_mul_kernel",
        kernel_source="elementwise_extended.metal",
        input_buffers=[gate_name, up_name],
        output_buffers=[mul_node.outputs[0].name],
        param_buffers=[],
        params={},
        dispatch_type="1d",
        total_threads=total,
    )


def _gen_masked_softmax_kernel(group: FusedGroup, graph: IRGraph | None = None) -> KernelCall:
    """Generate fused masked softmax: add(scores, mask) + softmax -> 1 dispatch."""
    from npu_compiler.codegen_core import _MAX_NDIM
    from npu_compiler.codegen_ops import compute_broadcast_strides

    add_node = group.nodes[0]
    softmax_node = group.nodes[1]

    scores_name = add_node.inputs[0].name
    mask_name = add_node.inputs[1].name
    out_shape = softmax_node.outputs[0].shape

    dim = softmax_node.attrs.get("dim", -1)
    if dim < 0:
        dim += len(out_shape)
    cols = out_shape[dim]
    rows = int(np.prod(out_shape)) // cols

    mask_shape = add_node.inputs[1].shape
    needs_broadcast = list(mask_shape) != list(out_shape)

    if needs_broadcast:
        ndim = len(out_shape)
        mask_strides = compute_broadcast_strides(mask_shape, out_shape)
        return KernelCall(
            kernel_name="masked_softmax_broadcast_kernel",
            kernel_source="softmax.metal",
            input_buffers=[scores_name, mask_name],
            output_buffers=[softmax_node.outputs[0].name],
            param_buffers=["masked_softmax_broadcast_params"],
            params={
                "rows": rows,
                "cols": cols,
                "ndim": ndim,
                "mask_strides": mask_strides + [0] * (_MAX_NDIM - ndim),
                "out_shape": list(out_shape) + [0] * (_MAX_NDIM - ndim),
            },
            dispatch_type="1d",
            total_threads=rows,
        )
    else:
        return KernelCall(
            kernel_name="masked_softmax_kernel",
            kernel_source="softmax.metal",
            input_buffers=[scores_name, mask_name],
            output_buffers=[softmax_node.outputs[0].name],
            param_buffers=["masked_softmax_params"],
            params={"rows": rows, "cols": cols},
            dispatch_type="1d",
            total_threads=rows,
        )


def _gen_fused_decode_attention_kernel(group: FusedGroup, graph: IRGraph | None = None) -> KernelCall:
    """Generate fused decode attention kernel.

    Supports GQA: when gqa_ratio > 1, K/V are (B*Hkv, S, D) and the kernel maps
    each Q head to its KV head internally, eliminating the expand operation.
    """
    meta = group.metadata
    q_name = meta["q_name"]
    k_notrans_name = meta["k_notrans_name"]
    v_name = meta["v_name"]
    mask_name = meta["mask_name"]
    scale = float(meta["scale"])
    B, H, S, D = meta["B"], meta["H"], meta["S"], meta["D"]
    gqa_ratio = meta.get("gqa_ratio", 1)
    final_output_name = meta["final_output_name"]

    return KernelCall(
        kernel_name="fused_decode_attention_kernel",
        kernel_source="fused_decode_attention.metal",
        input_buffers=[q_name, k_notrans_name, v_name, mask_name, "cache_position"],
        output_buffers=[final_output_name],
        param_buffers=["fused_decode_attn_params"],
        params={"batch_heads": B * H, "head_dim": D, "max_seq_len": S, "scale": scale, "gqa_ratio": gqa_ratio},
        dispatch_type="1d",
        total_threads=B * H,
    )


def _gen_rope_rotate_kernel(group: FusedGroup, graph: IRGraph | None = None) -> KernelCall:
    """Generate fused RoPE rotation kernel: output = x*cos + rotate_half(x)*sin."""
    meta = group.metadata
    shape = meta["shape"]
    total = 1
    for s in shape:
        total *= s
    seq_len = shape[2] if len(shape) >= 4 else 1
    return KernelCall(
        kernel_name="rope_rotate_kernel",
        kernel_source="rope_rotate.metal",
        input_buffers=[meta["x_name"], meta["cos_name"], meta["sin_name"]],
        output_buffers=[meta["output_name"]],
        param_buffers=["rope_rotate_params"],
        params={"total_elements": total, "head_dim": meta["head_dim"], "seq_len": seq_len},
        dispatch_type="1d",
        total_threads=total,
    )


# ---------------------------------------------------------------------------
# Register all built-in fused codegen handlers
# ---------------------------------------------------------------------------
register_fused_codegen("conv_bn_relu", _gen_conv_fused_kernel)
register_fused_codegen("conv_bn", _gen_conv_fused_kernel)
register_fused_codegen("conv_relu", _gen_conv_fused_kernel)
register_fused_codegen("add_relu", _gen_add_relu_kernel)
register_fused_codegen("rmsnorm", _gen_rmsnorm_kernel)
register_fused_codegen("silu_mul", _gen_silu_mul_kernel)
register_fused_codegen("masked_softmax", _gen_masked_softmax_kernel)
register_fused_codegen("rope_rotate", _gen_rope_rotate_kernel)
register_fused_codegen("decode_attention", _gen_fused_decode_attention_kernel)
