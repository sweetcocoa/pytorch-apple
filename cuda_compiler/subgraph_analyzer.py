"""Subgraph analyzer: converts an IRGraph into a list of ExecNodes.

This is the core of the CUDA subgraph compiler. It:
1. Classifies each op into a category (BLAS, elementwise, reduction, etc.)
2. Greedily fuses consecutive elementwise ops into FusionGroups
3. Outputs a list of ExecNode objects for the codegen phase

Fusion rule: An ELEMENTWISE node joins the chain of its input[0] producer
if and only if:
  - The producer is in the same chain (elementwise)
  - The producer has exactly one consumer (single-consumer)
  - The shapes match (no broadcast within chain)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cuda_compiler.cuda_codegen import compute_total_elements, generate_fused_kernel
from cuda_compiler.cuda_program import (
    AliasStep,
    CUBLASStep,
    ExecStep,
    FusedKernelStep,
    ReductionKernelStep,
    SpecialKernelStep,
)
from cuda_compiler.cuda_templates import TEMPLATE_MAP
from cuda_compiler.op_classify import OpCategory, classify_op
from npu_compiler.ir_reader import IRGraph, OpNode

# Ops that are zero-cost passthrough during pattern matching
_PASSTHROUGH_OPS = {
    "aten.expand.default",
    "aten._assert_tensor_metadata.default",
    "aten.to.dtype",
    "aten.dropout.default",
}

# ---------------------------------------------------------------------------
# Internal fusion data structures
# ---------------------------------------------------------------------------


@dataclass
class _FusionChain:
    """A chain of elementwise ops that will become a single CUDA kernel."""

    nodes: list[OpNode] = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    output_name: str = ""
    shape: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Consumer count analysis
# ---------------------------------------------------------------------------


def _compute_consumer_count(graph: IRGraph) -> dict[str, int]:
    """Count how many nodes consume each tensor."""
    counts: dict[str, int] = {}
    for node in graph.nodes:
        for inp in node.inputs:
            counts[inp.name] = counts.get(inp.name, 0) + 1
    return counts


def _compute_producer_map(graph: IRGraph) -> dict[str, OpNode]:
    """Map tensor name -> the OpNode that produces it."""
    producers: dict[str, OpNode] = {}
    for node in graph.nodes:
        for out in node.outputs:
            producers[out.name] = node
    return producers


# ---------------------------------------------------------------------------
# GQA attention pattern matching
# ---------------------------------------------------------------------------


def _try_match_gqa_attention(
    expand_node: OpNode,
    consumers_map: dict[str, list[OpNode]],
    fused_nodes: set[str],
) -> tuple | None:
    """Match expand → reshape → [transpose] → matmul for GQA attention.

    Pattern 1 (QK^T): expand [1,kv,1,S,D]->[1,kv,qpkv,S,D] → reshape [1,H,S,D] → transpose [1,H,D,S] → matmul
    Pattern 2 (SV):   expand [1,kv,1,S,D]->[1,kv,qpkv,S,D] → reshape [1,H,S,D] → matmul

    Returns (expand, reshape, transpose_or_None, matmul, kv_heads, q_per_kv, head_dim, seq_len, has_transpose)
    or None if pattern doesn't match.
    """
    in_shape = expand_node.inputs[0].shape
    out_shape = expand_node.outputs[0].shape

    # Must be 5D expand: [1, kv, 1, S, D] → [1, kv, qpkv, S, D]
    if len(out_shape) != 5 or len(in_shape) != 5:
        return None
    if in_shape[2] != 1 or out_shape[2] <= 1:
        return None
    kv_heads = in_shape[1]
    q_per_kv = out_shape[2]
    seq_len = in_shape[3]
    head_dim = in_shape[4]

    # Check expand has single consumer: reshape
    expand_out = expand_node.outputs[0].name
    reshape_candidates = consumers_map.get(expand_out, [])
    if len(reshape_candidates) != 1:
        return None
    reshape_node = reshape_candidates[0]
    if reshape_node.op_type != "aten.reshape.default" or reshape_node.name in fused_nodes:
        return None

    # Reshape must produce [1, H, S, D] where H = kv * qpkv
    r_shape = reshape_node.outputs[0].shape
    expected_h = kv_heads * q_per_kv
    if r_shape != [1, expected_h, seq_len, head_dim]:
        return None

    # Check reshape's single consumer: transpose or matmul
    reshape_out = reshape_node.outputs[0].name
    next_candidates = consumers_map.get(reshape_out, [])
    if len(next_candidates) != 1:
        return None
    next_node = next_candidates[0]

    if next_node.op_type == "aten.transpose.int" and next_node.name not in fused_nodes:
        # QK^T pattern: reshape → transpose → matmul
        transpose_node = next_node
        t_out = transpose_node.outputs[0].name
        mm_candidates = consumers_map.get(t_out, [])
        if len(mm_candidates) != 1:
            return None
        matmul_node = mm_candidates[0]
        if matmul_node.op_type != "aten.matmul.default" or matmul_node.name in fused_nodes:
            return None
        # Verify Q shape: [1, H, 1, D] (decode, M=1)
        q_shape = matmul_node.inputs[0].shape
        if len(q_shape) != 4 or q_shape[2] != 1:
            return None
        return (expand_node, reshape_node, transpose_node, matmul_node,
                kv_heads, q_per_kv, head_dim, seq_len, True)

    elif next_node.op_type == "aten.matmul.default" and next_node.name not in fused_nodes:
        # Score×V pattern: reshape → matmul (no transpose)
        matmul_node = next_node
        # Verify scores shape: [1, H, 1, S] (decode, M=1)
        scores_shape = matmul_node.inputs[0].shape
        if len(scores_shape) != 4 or scores_shape[2] != 1:
            return None
        return (expand_node, reshape_node, None, matmul_node,
                kv_heads, q_per_kv, head_dim, seq_len, False)

    return None


# ---------------------------------------------------------------------------
# BLAS step generation
# ---------------------------------------------------------------------------


def _make_blas_step(node: OpNode) -> CUBLASStep:
    """Create a CUBLASStep for matmul/linear/addmm/conv2d."""
    op = node.op_type

    if op in ("aten.linear.default", "aten.addmm.default"):
        if op == "aten.addmm.default":
            # addmm: bias + input @ weight.T
            bias, inp, weight = node.inputs[0], node.inputs[1], node.inputs[2]
            M = inp.shape[0]
            K = inp.shape[1]
            N = weight.shape[0]
            return CUBLASStep(
                blas_type="gemm",
                input_buffer_names=[inp.name, weight.name, bias.name],
                output_buffer_name=node.outputs[0].name,
                params={"M": M, "N": N, "K": K, "transpose_b": True, "has_bias": True},
            )
        # linear: input @ weight.T + optional bias
        inp = node.inputs[0]
        weight = node.inputs[1]
        has_bias = len(node.inputs) > 2
        M = inp.shape[0] if len(inp.shape) == 2 else int(np.prod(inp.shape[:-1]))
        K = inp.shape[-1]
        N = weight.shape[0]
        input_bufs = [inp.name, weight.name]
        if has_bias:
            input_bufs.append(node.inputs[2].name)
        return CUBLASStep(
            blas_type="gemm",
            input_buffer_names=input_bufs,
            output_buffer_name=node.outputs[0].name,
            params={"M": M, "N": N, "K": K, "transpose_b": True, "has_bias": has_bias},
        )

    if op == "aten.matmul.default":
        a_shape = node.inputs[0].shape
        b_shape = node.inputs[1].shape

        if len(a_shape) >= 3:
            batch = int(np.prod(a_shape[:-2]))
            M, K = a_shape[-2], a_shape[-1]
            N = b_shape[-1]
            return CUBLASStep(
                blas_type="gemm_batched",
                input_buffer_names=[node.inputs[0].name, node.inputs[1].name],
                output_buffer_name=node.outputs[0].name,
                params={"batch": batch, "M": M, "N": N, "K": K, "transpose_b": False},
            )

        M, K = a_shape[0], a_shape[1]
        N = b_shape[1]
        return CUBLASStep(
            blas_type="gemm",
            input_buffer_names=[node.inputs[0].name, node.inputs[1].name],
            output_buffer_name=node.outputs[0].name,
            params={"M": M, "N": N, "K": K, "transpose_b": False, "has_bias": False},
        )

    if op == "aten.conv2d.default":
        inp = node.inputs[0]
        weight = node.inputs[1]
        has_bias = len(node.inputs) > 2
        N, C_in, H, W = inp.shape
        C_out, _, KH, KW = weight.shape
        stride = node.attrs.get("stride", [1, 1])
        padding = node.attrs.get("padding", [0, 0])
        groups = node.attrs.get("groups", 1)
        out_h = (H + 2 * padding[0] - KH) // stride[0] + 1
        out_w = (W + 2 * padding[1] - KW) // stride[1] + 1

        input_bufs = [inp.name, weight.name]
        if has_bias:
            input_bufs.append(node.inputs[2].name)

        return CUBLASStep(
            blas_type="conv2d",
            input_buffer_names=input_bufs,
            output_buffer_name=node.outputs[0].name,
            params={
                "batch": N, "in_channels": C_in, "in_h": H, "in_w": W,
                "out_channels": C_out, "out_h": out_h, "out_w": out_w,
                "kernel_h": KH, "kernel_w": KW,
                "stride_h": stride[0], "stride_w": stride[1],
                "pad_h": padding[0], "pad_w": padding[1],
                "has_bias": has_bias, "groups": groups,
            },
        )

    raise ValueError(f"Unknown BLAS op: {op}")


# ---------------------------------------------------------------------------
# Reduction step generation
# ---------------------------------------------------------------------------


def _make_reduction_step(node: OpNode) -> ReductionKernelStep | SpecialKernelStep:
    """Create a step for reduction/pool/batchnorm ops."""
    op = node.op_type

    if op == "aten.softmax.int":
        shape = node.inputs[0].shape
        dim = node.attrs.get("dim", -1)
        if dim < 0:
            dim += len(shape)
        cols = shape[dim]
        rows = int(np.prod(shape)) // cols
        return ReductionKernelStep(
            kernel_name="softmax_kernel",
            source_code=TEMPLATE_MAP["softmax_kernel"],
            input_buffer_names=[node.inputs[0].name],
            output_buffer_name=node.outputs[0].name,
            params={"rows": rows, "cols": cols},
        )

    if op == "aten.mean.dim":
        in_shape = node.inputs[0].shape
        dims = node.attrs.get("dim", [-1])
        if dims == [-1] or dims == [len(in_shape) - 1]:
            cols = in_shape[-1]
            rows = int(np.prod(in_shape[:-1]))
            return ReductionKernelStep(
                kernel_name="mean_last_dim_kernel",
                source_code=TEMPLATE_MAP["mean_last_dim_kernel"],
                input_buffer_names=[node.inputs[0].name],
                output_buffer_name=node.outputs[0].name,
                params={"rows": rows, "cols": cols},
            )
        # 4D mean → adaptive avg pool
        if len(in_shape) == 4:
            N, C, H, W = in_shape
            out_shape = node.outputs[0].shape
            out_h, out_w = out_shape[2] if len(out_shape) > 2 else 1, out_shape[3] if len(out_shape) > 3 else 1
            return ReductionKernelStep(
                kernel_name="adaptive_avg_pool2d_kernel",
                source_code=TEMPLATE_MAP["adaptive_avg_pool2d_kernel"],
                input_buffer_names=[node.inputs[0].name],
                output_buffer_name=node.outputs[0].name,
                params={
                    "batch": N, "channels": C, "in_h": H, "in_w": W,
                    "out_h": out_h, "out_w": out_w,
                },
            )

    if op == "aten.max_pool2d.default":
        inp = node.inputs[0]
        N, C, H, W = inp.shape
        out_shape = node.outputs[0].shape
        out_h, out_w = out_shape[2], out_shape[3]
        kernel_size = node.attrs.get("kernel_size", [2, 2])
        stride = node.attrs.get("stride", kernel_size)
        padding = node.attrs.get("padding", [0, 0])
        return ReductionKernelStep(
            kernel_name="max_pool2d_kernel",
            source_code=TEMPLATE_MAP["max_pool2d_kernel"],
            input_buffer_names=[inp.name],
            output_buffer_name=node.outputs[0].name,
            params={
                "batch": N, "channels": C, "in_h": H, "in_w": W,
                "out_h": out_h, "out_w": out_w,
                "kernel_h": kernel_size[0], "kernel_w": kernel_size[1],
                "stride_h": stride[0], "stride_w": stride[1],
                "pad_h": padding[0] if isinstance(padding, list) else padding,
                "pad_w": padding[1] if isinstance(padding, list) else padding,
            },
        )

    if op == "aten.adaptive_avg_pool2d.default":
        inp = node.inputs[0]
        N, C, H, W = inp.shape
        out_shape = node.outputs[0].shape
        out_h, out_w = out_shape[2], out_shape[3]
        return ReductionKernelStep(
            kernel_name="adaptive_avg_pool2d_kernel",
            source_code=TEMPLATE_MAP["adaptive_avg_pool2d_kernel"],
            input_buffer_names=[inp.name],
            output_buffer_name=node.outputs[0].name,
            params={
                "batch": N, "channels": C, "in_h": H, "in_w": W,
                "out_h": out_h, "out_w": out_w,
            },
        )

    if op == "aten.batch_norm.default":
        inp = node.inputs[0]
        gamma = node.inputs[1]
        beta = node.inputs[2]
        running_mean = node.inputs[3]
        running_var = node.inputs[4]
        shape = inp.shape
        N, C = shape[0], shape[1]
        spatial = int(np.prod(shape[2:])) if len(shape) > 2 else 1
        eps = float(node.attrs.get("eps", 1e-5))
        return ReductionKernelStep(
            kernel_name="batch_norm_kernel",
            source_code=TEMPLATE_MAP["batch_norm_kernel"],
            input_buffer_names=[inp.name, gamma.name, beta.name, running_mean.name, running_var.name],
            output_buffer_name=node.outputs[0].name,
            params={"batch": N, "channels": C, "spatial": spatial, "eps": eps},
        )

    raise ValueError(f"Unknown reduction op: {op}")


# ---------------------------------------------------------------------------
# Special kernel step generation
# ---------------------------------------------------------------------------


def _make_special_step(node: OpNode) -> SpecialKernelStep | AliasStep:
    """Create a step for special ops (embedding, rope, index_copy, etc.)."""
    op = node.op_type

    if op == "aten.embedding.default":
        weight = node.inputs[0]
        indices = node.inputs[1]
        embed_dim = weight.shape[1]
        seq_len = int(np.prod(indices.shape))
        return SpecialKernelStep(
            kernel_name="embedding_kernel",
            source_code=TEMPLATE_MAP["embedding_kernel"],
            input_buffer_names=[indices.name, weight.name],
            output_buffer_name=node.outputs[0].name,
            params={"seq_len": seq_len, "embed_dim": embed_dim},
            grid_dim=(max(1, (embed_dim + 255) // 256), seq_len),
        )

    if op == "wrap_with_set_grad_enabled":  # RoPE
        inv_freq = node.inputs[0]
        positions = node.inputs[1]
        out_shape = node.outputs[0].shape
        seq_len = out_shape[-2] if len(out_shape) >= 2 else out_shape[0]
        head_dim = out_shape[-1]
        return SpecialKernelStep(
            kernel_name="rope_kernel",
            source_code=TEMPLATE_MAP["rope_kernel"],
            input_buffer_names=[inv_freq.name, positions.name],
            output_buffer_name=node.outputs[0].name,
            output_buffer_names=[node.outputs[0].name, node.outputs[1].name],
            params={"seq_len": seq_len, "head_dim": head_dim},
            grid_dim=(max(1, (head_dim + 255) // 256), seq_len),
        )

    if op == "aten.index_copy.default":
        in_shape = node.inputs[0].shape
        source_shape = node.inputs[2].shape
        dim = node.attrs.get("dim", 0)
        ndim = len(in_shape)
        outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
        dim_size = in_shape[dim]
        inner = int(np.prod(in_shape[dim + 1:])) if dim + 1 < ndim else 1
        num_indices = source_shape[dim]
        total = int(np.prod(in_shape))
        return SpecialKernelStep(
            kernel_name="index_copy_kernel",
            source_code=TEMPLATE_MAP["index_copy_kernel"],
            input_buffer_names=[node.inputs[0].name, node.inputs[2].name, node.inputs[1].name],
            output_buffer_name=node.outputs[0].name,
            params={"outer_size": outer, "dim_size": dim_size, "inner_size": inner, "num_indices": num_indices},
            grid_dim=(max(1, (total + 255) // 256),),
        )

    if op == "aten.full.default":
        out_shape = node.outputs[0].shape
        fill_value = float(node.attrs.get("fill_value", 0.0))
        total = int(np.prod(out_shape))
        return SpecialKernelStep(
            kernel_name="full_kernel",
            source_code=TEMPLATE_MAP["full_kernel"],
            input_buffer_names=[],
            output_buffer_name=node.outputs[0].name,
            params={"fill_value": fill_value, "total": total},
            grid_dim=(max(1, (total + 255) // 256),),
        )

    raise ValueError(f"Unknown special op: {op}")


# ---------------------------------------------------------------------------
# Shape alias step generation
# ---------------------------------------------------------------------------

def _needs_real_transpose(node: OpNode) -> bool:
    """Check if a transpose/permute op needs a real data movement kernel."""
    if node.op_type == "aten.t.default":
        return len(node.inputs[0].shape) == 2
    if node.op_type == "aten.transpose.int":
        dim0 = node.attrs.get("dim0", 0)
        dim1 = node.attrs.get("dim1", 1)
        in_shape = node.inputs[0].shape
        # Zero-cost if either dimension is 1
        return in_shape[dim0] != 1 and in_shape[dim1] != 1
    return False


def _make_alias_step(node: OpNode) -> AliasStep | SpecialKernelStep:
    """Create a step for zero-cost shape ops, or a real kernel for non-trivial ones."""
    op = node.op_type
    in_shape = node.inputs[0].shape
    out_shape = node.outputs[0].shape

    # Transpose ops that actually need data movement
    if op in ("aten.t.default", "aten.transpose.int") and _needs_real_transpose(node):
        from npu_compiler.codegen_ops import compute_strides

        ndim = len(in_shape)
        if op == "aten.t.default":
            dim0, dim1 = 0, 1
        else:
            dim0 = node.attrs.get("dim0", 0)
            dim1 = node.attrs.get("dim1", 1)
        total = int(np.prod(out_shape))
        strides_in = compute_strides(in_shape)
        strides_out = compute_strides(out_shape)
        return SpecialKernelStep(
            kernel_name="transpose_kernel",
            source_code=TEMPLATE_MAP["transpose_kernel"],
            input_buffer_names=[node.inputs[0].name],
            output_buffer_name=node.outputs[0].name,
            params={
                "ndim": ndim, "dim0": dim0, "dim1": dim1, "total": total,
                "in_shape": list(in_shape), "in_strides": strides_in, "out_strides": strides_out,
            },
            grid_dim=(max(1, (total + 255) // 256),),
        )

    # Cat needs a real kernel
    if op == "aten.cat.default":
        if node.inputs[0].shape == [0]:
            return AliasStep(
                input_buffer_name=node.inputs[1].name,
                output_buffer_name=node.outputs[0].name,
                input_shape=node.inputs[1].shape,
                output_shape=out_shape,
            )
        from npu_compiler.codegen_ops import compute_strides

        ndim = len(out_shape)
        axis = node.attrs.get("dim", 0)
        if axis < 0:
            axis += ndim
        total = int(np.prod(out_shape))
        in0_axis_size = node.inputs[0].shape[axis]
        strides = compute_strides(out_shape)
        return SpecialKernelStep(
            kernel_name="cat_2_kernel",
            source_code=TEMPLATE_MAP["cat_2_kernel"],
            input_buffer_names=[node.inputs[0].name, node.inputs[1].name],
            output_buffer_name=node.outputs[0].name,
            params={
                "axis": axis, "ndim": ndim, "total": total,
                "in0_axis_size": in0_axis_size,
                "out_shape": list(out_shape), "out_strides": strides,
            },
            grid_dim=(max(1, (total + 255) // 256),),
        )

    # Slice needs a real kernel
    if op == "aten.slice.Tensor":
        from npu_compiler.codegen_ops import compute_strides

        ndim = len(in_shape)
        dim = node.attrs.get("dim", 0)
        dim_size = in_shape[dim]
        start = node.attrs.get("start", 0)
        if start < 0:
            start = max(0, dim_size + start)
        end_raw = node.attrs.get("end", dim_size)
        end = min(end_raw, dim_size) if end_raw >= 0 else max(0, dim_size + end_raw)
        step = node.attrs.get("step", 1)
        total = int(np.prod(out_shape))
        # If slice is a no-op (entire dim), alias
        if start == 0 and end >= dim_size and step == 1:
            return AliasStep(
                input_buffer_name=node.inputs[0].name,
                output_buffer_name=node.outputs[0].name,
                input_shape=in_shape,
                output_shape=out_shape,
            )
        in_strides = compute_strides(in_shape)
        return SpecialKernelStep(
            kernel_name="slice_kernel",
            source_code=TEMPLATE_MAP["slice_kernel"],
            input_buffer_names=[node.inputs[0].name],
            output_buffer_name=node.outputs[0].name,
            params={
                "dim": dim, "start": start, "step": step,
                "ndim": ndim, "total": total,
                "in_shape": list(in_shape), "in_strides": in_strides, "out_shape": list(out_shape),
            },
            grid_dim=(max(1, (total + 255) // 256),),
        )

    # Expand needs a real kernel
    if op == "aten.expand.default":
        from npu_compiler.codegen_ops import compute_strides

        ndim = len(out_shape)
        total = int(np.prod(out_shape))
        in_strides = compute_strides(in_shape)
        pad_len = ndim - len(in_shape)
        padded_in = [1] * pad_len + list(in_shape)
        padded_strides = [0] * pad_len + in_strides
        for i in range(ndim):
            if padded_in[i] == 1 and out_shape[i] != 1:
                padded_strides[i] = 0
        return SpecialKernelStep(
            kernel_name="expand_kernel",
            source_code=TEMPLATE_MAP["expand_kernel"],
            input_buffer_names=[node.inputs[0].name],
            output_buffer_name=node.outputs[0].name,
            params={
                "ndim": ndim, "total": total,
                "in_shape": padded_in, "out_shape": list(out_shape), "in_strides": padded_strides,
            },
            grid_dim=(max(1, (total + 255) // 256),),
        )

    # getitem
    if op == "<built-in function getitem>":
        idx = node.attrs.get("index", 0)
        in_name = node.inputs[idx].name if idx < len(node.inputs) else node.inputs[0].name
        return AliasStep(
            input_buffer_name=in_name,
            output_buffer_name=node.outputs[0].name,
            input_shape=out_shape,
            output_shape=out_shape,
        )

    # Default: zero-cost alias
    return AliasStep(
        input_buffer_name=node.inputs[0].name,
        output_buffer_name=node.outputs[0].name,
        input_shape=in_shape,
        output_shape=out_shape,
    )


# ---------------------------------------------------------------------------
# Fusion pattern matching (RMSNorm, SiLU+Mul, Masked Softmax)
# ---------------------------------------------------------------------------


def _follow_single_consumer(
    tensor_name: str,
    consumers: dict[str, list[OpNode]],
    expected_op: str,
    fused: set[str],
) -> OpNode | None:
    """Follow a single consumer of tensor_name if it matches expected_op."""
    next_nodes = [n for n in consumers.get(tensor_name, []) if n.name not in fused]
    if len(next_nodes) == 1 and next_nodes[0].op_type == expected_op:
        return next_nodes[0]
    return None


def _follow_skip_passthrough(
    tensor_name: str,
    consumers: dict[str, list[OpNode]],
    expected_op: str,
    fused: set[str],
) -> tuple[OpNode | None, list[OpNode]]:
    """Follow consumer chain, skipping passthrough ops (expand, assert, to.dtype)."""
    skipped: list[OpNode] = []
    current = tensor_name
    for _ in range(4):
        next_nodes = [n for n in consumers.get(current, []) if n.name not in fused]
        if not next_nodes:
            return None, []
        for c in next_nodes:
            if c.op_type == expected_op:
                return c, skipped
        non_assert = [n for n in next_nodes if n.op_type != "aten._assert_tensor_metadata.default"]
        assert_nodes = [n for n in next_nodes if n.op_type == "aten._assert_tensor_metadata.default"]
        for a in assert_nodes:
            skipped.append(a)
        if len(non_assert) != 1:
            return None, []
        node = non_assert[0]
        if node.op_type in _PASSTHROUGH_OPS:
            skipped.append(node)
            if node.outputs:
                current = node.outputs[0].name
            else:
                return None, []
        else:
            return None, []
    return None, []


def _build_consumer_map(graph: IRGraph) -> dict[str, list[OpNode]]:
    """Build tensor_name -> list of consumer OpNodes."""
    consumers: dict[str, list[OpNode]] = {}
    for node in graph.nodes:
        for inp in node.inputs:
            consumers.setdefault(inp.name, []).append(node)
    return consumers


def _try_match_rmsnorm(
    pow_node: OpNode,
    graph: IRGraph,
    consumers: dict[str, list[OpNode]],
    fused: set[str],
) -> tuple[list[OpNode], str, str, list[int], float] | None:
    """Try to match RMSNorm pattern: pow(x,2) → mean → add(eps) → rsqrt → [expand] → mul(x) → [expand] → mul(weight).

    Returns (chain_nodes, input_name, weight_name, shape, eps) or None.
    """
    exp = pow_node.attrs.get("exponent", 0)
    if exp != 2.0 and exp != 2:
        return None

    chain = [pow_node]
    fused_names = {pow_node.name}
    rmsnorm_input = pow_node.inputs[0].name

    # pow → mean
    mean_node = _follow_single_consumer(pow_node.outputs[0].name, consumers, "aten.mean.dim", fused)
    if mean_node is None:
        return None
    chain.append(mean_node)
    fused_names.add(mean_node.name)

    # mean → add(eps)
    add_node = _follow_single_consumer(mean_node.outputs[0].name, consumers, "aten.add.Tensor", fused)
    if add_node is None:
        return None
    chain.append(add_node)
    fused_names.add(add_node.name)

    # Extract eps from add node's second input
    eps = 1e-6
    if len(add_node.inputs) >= 2:
        eps_input = add_node.inputs[1]
        # eps may be a scalar constant — check attrs
        if "other" in add_node.attrs:
            eps = float(add_node.attrs["other"])
        elif eps_input.shape == [1] or eps_input.shape == []:
            eps = 1e-6  # default

    # add → rsqrt
    rsqrt_node = _follow_single_consumer(add_node.outputs[0].name, consumers, "aten.rsqrt.default", fused)
    if rsqrt_node is None:
        return None
    chain.append(rsqrt_node)
    fused_names.add(rsqrt_node.name)

    # rsqrt → [passthrough] → mul(x)
    mul1_node, skip1 = _follow_skip_passthrough(rsqrt_node.outputs[0].name, consumers, "aten.mul.Tensor", fused)
    if mul1_node is None or len(mul1_node.inputs) != 2:
        return None
    # Verify one input is rmsnorm_input (x) or an alias of it
    mul1_input_names = {inp.name for inp in mul1_node.inputs}
    valid_x_names = {rmsnorm_input}
    for node in graph.nodes:
        if node.op_type in ("aten.to.dtype", "aten.expand.default") and node.inputs:
            if node.inputs[0].name in valid_x_names and node.outputs:
                valid_x_names.add(node.outputs[0].name)
    if not mul1_input_names & valid_x_names:
        return None
    chain.extend(skip1)
    chain.append(mul1_node)
    fused_names.add(mul1_node.name)
    for e in skip1:
        fused_names.add(e.name)

    # mul(x*rsqrt) → [passthrough] → mul(weight)
    mul2_node, skip2 = _follow_skip_passthrough(mul1_node.outputs[0].name, consumers, "aten.mul.Tensor", fused)
    if mul2_node is None or len(mul2_node.inputs) != 2:
        return None
    chain.extend(skip2)
    chain.append(mul2_node)
    fused_names.add(mul2_node.name)
    for e in skip2:
        fused_names.add(e.name)

    # Identify weight: the input of mul2 that is NOT the output of mul1 (or passthrough)
    mul2_input_names = {inp.name for inp in mul2_node.inputs}
    chain_outputs = {n.outputs[0].name for n in chain if n.outputs}
    weight_candidates = mul2_input_names - chain_outputs
    if not weight_candidates:
        return None
    weight_name = weight_candidates.pop()

    # Shape from the input x
    input_shape = pow_node.inputs[0].shape

    return chain, rmsnorm_input, weight_name, list(input_shape), eps


def _try_match_masked_softmax(
    add_node: OpNode,
    consumers: dict[str, list[OpNode]],
    fused: set[str],
) -> tuple[str, str, str, list[int]] | None:
    """Try to match add + softmax → masked_softmax.

    Returns (input_name, mask_name, output_name, shape) or None.
    """
    if add_node.op_type != "aten.add.Tensor" or len(add_node.inputs) != 2:
        return None

    # add → softmax (single consumer)
    softmax_node = _follow_single_consumer(add_node.outputs[0].name, consumers, "aten.softmax.int", fused)
    if softmax_node is None:
        return None

    # Only fuse if softmax dim is the last dim
    shape = add_node.inputs[0].shape
    dim = softmax_node.attrs.get("dim", -1)
    if dim < 0:
        dim += len(shape)
    if dim != len(shape) - 1:
        return None

    return (
        add_node.inputs[0].name,
        add_node.inputs[1].name,
        softmax_node.outputs[0].name,
        list(shape),
    )


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze_subgraph(graph: IRGraph) -> list[ExecStep]:
    """Analyze an IRGraph and produce a list of execution steps.

    This is the core of the subgraph compiler:
    1. Classify each op
    2. Greedily fuse elementwise chains
    3. Produce execution steps

    Returns:
        List of ExecStep (CUBLASStep, FusedKernelStep, etc.)
    """
    consumer_count = _compute_consumer_count(graph)
    producer_map = _compute_producer_map(graph)
    consumers_map = _build_consumer_map(graph)

    steps: list[ExecStep] = []

    # Track which nodes have been consumed by a fusion chain
    fused_nodes: set[str] = set()

    # --- Pre-pass 0: detect GQA expand → reshape → [transpose] → matmul ---
    # Fuses expand + reshape + optional transpose into the matmul, producing gemm_gqa.
    for node in graph.nodes:
        if node.op_type != "aten.expand.default" or node.name in fused_nodes:
            continue
        result = _try_match_gqa_attention(node, consumers_map, fused_nodes)
        if result is None:
            continue
        (expand_node, reshape_node, transpose_node, matmul_node,
         kv_heads, q_per_kv, head_dim, seq_len, has_transpose) = result
        # Mark expand, reshape, transpose (if present), and matmul as fused
        fused_nodes.add(expand_node.name)
        fused_nodes.add(reshape_node.name)
        if transpose_node is not None:
            fused_nodes.add(transpose_node.name)
        fused_nodes.add(matmul_node.name)
        # Emit gemm_gqa step: takes Q and unexpanded KV directly
        q_name = matmul_node.inputs[0].name
        kv_name = expand_node.inputs[0].name  # unexpanded KV
        if has_transpose:
            # QK^T pattern: Q=[1,H,1,D] @ K^T=[1,H,D,S] -> [1,H,1,S]
            out_shape = matmul_node.outputs[0].shape
            steps.append(CUBLASStep(
                blas_type="gemm_gqa",
                input_buffer_names=[q_name, kv_name],
                output_buffer_name=matmul_node.outputs[0].name,
                params={
                    "kv_heads": kv_heads, "q_per_kv": q_per_kv,
                    "M": 1, "K": head_dim, "N": seq_len,
                    "transpose_kv": True, "out_shape": out_shape,
                },
            ))
        else:
            # Score×V pattern: scores=[1,H,1,S] @ V=[1,H,S,D] -> [1,H,1,D]
            out_shape = matmul_node.outputs[0].shape
            steps.append(CUBLASStep(
                blas_type="gemm_gqa",
                input_buffer_names=[q_name, kv_name],
                output_buffer_name=matmul_node.outputs[0].name,
                params={
                    "kv_heads": kv_heads, "q_per_kv": q_per_kv,
                    "M": 1, "K": seq_len, "N": head_dim,
                    "transpose_kv": False, "out_shape": out_shape,
                },
            ))

    # --- Pre-pass 1: detect RMSNorm and Masked Softmax patterns ---
    # These patterns span ELEMENTWISE + REDUCTION ops and must be matched first.
    for node in graph.nodes:
        # RMSNorm: pow(x,2) → mean → add(eps) → rsqrt → mul(x) → mul(weight)
        if node.op_type == "aten.pow.Tensor_Scalar" and node.name not in fused_nodes:
            result = _try_match_rmsnorm(node, graph, consumers_map, fused_nodes)
            if result is not None:
                chain, input_name, weight_name, shape, eps = result
                rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
                cols = shape[-1]
                output_name = chain[-1].outputs[0].name
                for n in chain:
                    fused_nodes.add(n.name)
                steps.append(ReductionKernelStep(
                    kernel_name="rmsnorm_kernel",
                    source_code=TEMPLATE_MAP["rmsnorm_kernel"],
                    input_buffer_names=[input_name, weight_name],
                    output_buffer_name=output_name,
                    params={"rows": rows, "cols": cols, "eps": eps},
                ))

        # Masked Softmax: add(scores, mask) → softmax
        if node.op_type == "aten.add.Tensor" and node.name not in fused_nodes:
            result = _try_match_masked_softmax(node, consumers_map, fused_nodes)
            if result is not None:
                input_name, mask_name, output_name, shape = result
                dim = len(shape) - 1
                cols = shape[dim]
                rows = int(np.prod(shape)) // cols
                # Mark both add and softmax as fused
                fused_nodes.add(node.name)
                softmax_candidates = [
                    n for n in consumers_map.get(node.outputs[0].name, [])
                    if n.op_type == "aten.softmax.int"
                ]
                if softmax_candidates:
                    fused_nodes.add(softmax_candidates[0].name)
                steps.append(ReductionKernelStep(
                    kernel_name="masked_softmax_kernel",
                    source_code=TEMPLATE_MAP["masked_softmax_kernel"],
                    input_buffer_names=[input_name, mask_name],
                    output_buffer_name=output_name,
                    params={"rows": rows, "cols": cols},
                ))

    # Track active fusion chains: node_name -> chain
    active_chains: dict[str, _FusionChain] = {}

    for node in graph.nodes:
        if node.name in fused_nodes:
            continue

        category = classify_op(node.op_type)

        if category == OpCategory.NOOP:
            continue

        if category == OpCategory.ANCHOR_BLAS:
            # Flush any pending chain, then emit BLAS step
            steps.append(_make_blas_step(node))
            continue

        if category == OpCategory.REDUCTION:
            steps.append(_make_reduction_step(node))
            continue

        if category == OpCategory.SPECIAL:
            steps.append(_make_special_step(node))
            continue

        if category == OpCategory.SHAPE_ALIAS:
            steps.append(_make_alias_step(node))
            continue

        if category == OpCategory.ELEMENTWISE:
            # Try to extend an existing chain or start a new one
            chain = _try_extend_chain(node, active_chains, consumer_count, producer_map)
            if chain is None:
                # Start new chain
                chain = _FusionChain(
                    nodes=[node],
                    input_names=_collect_external_inputs(node, set()),
                    output_name=node.outputs[0].name,
                    shape=list(node.outputs[0].shape),
                )
                active_chains[node.outputs[0].name] = chain
            else:
                # Extended existing chain — update output
                chain.nodes.append(node)
                chain.output_name = node.outputs[0].name
                # Add any new external inputs from this node
                chain_internal = {n.outputs[0].name for n in chain.nodes}
                for inp in node.inputs:
                    if inp.name not in chain_internal and inp.name not in chain.input_names:
                        chain.input_names.append(inp.name)
                # Update active_chains key
                active_chains[node.outputs[0].name] = chain

            # Check if this chain should be finalized:
            # A chain is finalized when its output is consumed by a non-elementwise op
            # or has multiple consumers. We defer finalization to handle greedy extension.
            # Finalization happens when we encounter a non-EW consumer or at the end.
            continue

    # Finalize all remaining chains
    finalized: set[int] = set()
    for chain in active_chains.values():
        chain_id = id(chain)
        if chain_id in finalized:
            continue
        finalized.add(chain_id)
        steps.append(_finalize_chain(chain, len(steps)))
        for n in chain.nodes:
            fused_nodes.add(n.name)

    # Re-sort steps into topological order based on data dependencies
    steps = _topological_sort_steps(steps, graph)

    return steps


def _try_extend_chain(
    node: OpNode,
    active_chains: dict[str, _FusionChain],
    consumer_count: dict[str, int],
    producer_map: dict[str, OpNode],
) -> _FusionChain | None:
    """Try to add a node to an existing fusion chain.

    Returns the chain if successful, None otherwise.
    """
    if not node.inputs:
        return None

    primary_input = node.inputs[0].name

    # Check if primary input was produced by a chain
    if primary_input not in active_chains:
        return None

    chain = active_chains[primary_input]

    # Single-consumer check: the producer output must only be consumed by this node
    if consumer_count.get(primary_input, 0) > 1:
        return None

    # Shape match: output shape must match chain shape
    if list(node.outputs[0].shape) != chain.shape:
        return None

    return chain


def _collect_external_inputs(node: OpNode, internal_names: set[str]) -> list[str]:
    """Collect input names that are not produced within the chain."""
    names: list[str] = []
    for inp in node.inputs:
        if inp.name not in internal_names and inp.name not in names:
            names.append(inp.name)
    return names


def _finalize_chain(chain: _FusionChain, step_idx: int) -> FusedKernelStep:
    """Convert a fusion chain into a FusedKernelStep."""
    kernel_name = f"fused_ew_{step_idx}"
    total = compute_total_elements(chain.shape)

    # Import OpNode type for generate_fused_kernel
    source = generate_fused_kernel(
        kernel_name=kernel_name,
        chain=chain.nodes,
        input_names=chain.input_names,
        output_name=chain.output_name,
    )

    return FusedKernelStep(
        kernel_name=kernel_name,
        source_code=source,
        input_buffer_names=chain.input_names,
        output_buffer_name=chain.output_name,
        total_elements=total,
    )


def _topological_sort_steps(steps: list[ExecStep], graph: IRGraph) -> list[ExecStep]:
    """Re-sort steps to respect data dependencies.

    Because fusion chains are finalized at the end, they may appear after
    steps that consume their output. This sorts them properly.
    """
    # Build output->step mapping
    step_outputs: dict[str, int] = {}
    for i, step in enumerate(steps):
        for name in _step_output_names(step):
            step_outputs[name] = i

    # Build adjacency
    n = len(steps)
    in_degree = [0] * n
    adj: dict[int, list[int]] = {i: [] for i in range(n)}

    for i, step in enumerate(steps):
        for name in _step_input_names(step):
            producer = step_outputs.get(name)
            if producer is not None and producer != i:
                adj[producer].append(i)
                in_degree[i] += 1

    # Kahn's algorithm
    queue = [i for i in range(n) if in_degree[i] == 0]
    result: list[ExecStep] = []
    while queue:
        # Stable sort: prefer original order
        queue.sort()
        idx = queue.pop(0)
        result.append(steps[idx])
        for neighbor in adj[idx]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If some steps couldn't be sorted (cycle), append them at the end
    if len(result) < n:
        sorted_indices = {id(s) for s in result}
        for s in steps:
            if id(s) not in sorted_indices:
                result.append(s)

    return result


def _step_output_names(step: ExecStep) -> list[str]:
    """Get output buffer names for a step."""
    if isinstance(step, AliasStep):
        return [step.output_buffer_name]
    if isinstance(step, CUBLASStep):
        return [step.output_buffer_name]
    if isinstance(step, FusedKernelStep):
        return [step.output_buffer_name]
    if isinstance(step, ReductionKernelStep):
        return [step.output_buffer_name]
    if isinstance(step, SpecialKernelStep):
        names = [step.output_buffer_name]
        if step.output_buffer_names:
            names.extend(n for n in step.output_buffer_names if n != step.output_buffer_name)
        return names
    return []


def _step_input_names(step: ExecStep) -> list[str]:
    """Get input buffer names for a step."""
    if isinstance(step, AliasStep):
        return [step.input_buffer_name]
    if isinstance(step, CUBLASStep):
        return step.input_buffer_names
    if isinstance(step, FusedKernelStep):
        return step.input_buffer_names
    if isinstance(step, ReductionKernelStep):
        return step.input_buffer_names
    if isinstance(step, SpecialKernelStep):
        return step.input_buffer_names
    return []
