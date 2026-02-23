"""Single-op codegen: maps individual ATen ops to KernelCall objects.

Each handler takes an OpNode and returns a KernelCall (or None for no-ops).
The main entry point is generate_single_kernel_call().
"""

from __future__ import annotations

import numpy as np

from npu_compiler.codegen_core import _DTYPE_ELEM_SIZE, _MAX_NDIM, KernelCall
from npu_compiler.ir_reader import IRGraph, OpNode
from npu_compiler.target_config import pad_channels, padded_shape_4d

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_strides(shape: list[int]) -> list[int]:
    """Compute row-major strides for a shape."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def compute_broadcast_strides(in_shape: list[int], out_shape: list[int]) -> list[int]:
    """Compute strides for a tensor broadcast to out_shape (0 for broadcast dims)."""
    ndim = len(out_shape)
    in_strides = compute_strides(in_shape)
    pad_len = ndim - len(in_shape)
    padded_in = [1] * pad_len + list(in_shape)
    padded_strides = [0] * pad_len + in_strides
    for i in range(ndim):
        if padded_in[i] == 1 and out_shape[i] != 1:
            padded_strides[i] = 0
    return padded_strides


def size_bytes(shape, dtype="float16", pad_4d=False):
    elem_size = _DTYPE_ELEM_SIZE.get(dtype, 2)
    final_shape = padded_shape_4d(shape) if pad_4d else shape
    return int(np.prod(final_shape)) * elem_size


# ---------------------------------------------------------------------------
# Binary op helpers
# ---------------------------------------------------------------------------

_BROADCAST_BINARY_KERNEL_MAP = {
    "aten.add.Tensor": "add_broadcast_kernel",
    "aten.mul.Tensor": "mul_broadcast_kernel",
    "aten.div.Tensor": "div_broadcast_kernel",
}

_ELEMENTWISE_BINARY_KERNEL_MAP = {
    "aten.add.Tensor": "add_kernel",
    "aten.mul.Tensor": "mul_kernel",
    "aten.div.Tensor": "div_kernel",
}

_ELEMENTWISE_BINARY_METAL_FILE = {
    "aten.add.Tensor": "add_relu.metal",
    "aten.mul.Tensor": "elementwise_extended.metal",
    "aten.div.Tensor": "elementwise_extended.metal",
}


def gen_binary_op(node: OpNode, op_type: str) -> KernelCall | list[KernelCall]:
    """Generate binary op kernel, using broadcast kernel when shapes differ."""
    out_shape = node.outputs[0].shape
    a_shape = node.inputs[0].shape
    b_shape = node.inputs[1].shape
    a_name = node.inputs[0].name
    b_name = node.inputs[1].name
    total = int(np.prod(out_shape))

    needs_broadcast = list(a_shape) != list(out_shape) or list(b_shape) != list(out_shape)

    if needs_broadcast:
        ndim = len(out_shape)
        a_strides = compute_broadcast_strides(a_shape, out_shape)
        b_strides = compute_broadcast_strides(b_shape, out_shape)
        return KernelCall(
            kernel_name=_BROADCAST_BINARY_KERNEL_MAP[op_type],
            kernel_source="elementwise_broadcast.metal",
            input_buffers=[a_name, b_name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["broadcast_binary_params"],
            params={
                "ndim": ndim,
                "total": total,
                "a_strides": a_strides + [0] * (_MAX_NDIM - ndim),
                "b_strides": b_strides + [0] * (_MAX_NDIM - ndim),
                "out_shape": list(out_shape) + [0] * (_MAX_NDIM - ndim),
            },
            dispatch_type="1d",
            total_threads=total,
        )
    else:
        padded = padded_shape_4d(out_shape)
        padded_total = int(np.prod(padded))
        return KernelCall(
            kernel_name=_ELEMENTWISE_BINARY_KERNEL_MAP[op_type],
            kernel_source=_ELEMENTWISE_BINARY_METAL_FILE[op_type],
            input_buffers=[a_name, b_name],
            output_buffers=[node.outputs[0].name],
            param_buffers=[],
            params={},
            dispatch_type="1d",
            total_threads=padded_total,
        )


# ---------------------------------------------------------------------------
# Individual op generators
# ---------------------------------------------------------------------------


def gen_conv_kernel(node: OpNode, graph: IRGraph, has_relu: bool, output_name: str | None = None) -> KernelCall:
    inp = node.inputs[0]
    weight = node.inputs[1]
    has_bias = len(node.inputs) > 2

    N, C_in, H, W = inp.shape
    C_out = weight.shape[0]
    KH, KW = weight.shape[2], weight.shape[3]
    stride = node.attrs.get("stride", [1, 1])
    padding = node.attrs.get("padding", [0, 0])
    groups = node.attrs.get("groups", 1)

    out_h = (H + 2 * padding[0] - KH) // stride[0] + 1
    out_w = (W + 2 * padding[1] - KW) // stride[1] + 1
    total = N * C_out * out_h * out_w

    out_name = output_name or node.outputs[0].name

    input_bufs = [inp.name, weight.name]
    if has_bias:
        input_bufs.append(node.inputs[2].name)

    return KernelCall(
        kernel_name="conv2d_kernel",
        kernel_source="conv_bn_relu.metal",
        input_buffers=input_bufs,
        output_buffers=[out_name],
        param_buffers=["conv_params"],
        params={
            "batch": N,
            "in_channels": C_in,
            "in_h": H,
            "in_w": W,
            "out_channels": C_out,
            "out_h": out_h,
            "out_w": out_w,
            "kernel_h": KH,
            "kernel_w": KW,
            "stride_h": stride[0],
            "stride_w": stride[1],
            "pad_h": padding[0],
            "pad_w": padding[1],
            "has_bias": 1 if has_bias else 0,
            "has_bn": 0,
            "has_relu": 1 if has_relu else 0,
            "groups": groups,
            "in_channels_aligned": pad_channels(C_in),
            "out_channels_aligned": pad_channels(C_out),
        },
        dispatch_type="1d",
        total_threads=total,
    )


def _gen_linear_kernel(node: OpNode, graph: IRGraph, output_name: str | None = None) -> KernelCall:
    inp = node.inputs[0]
    weight = node.inputs[1]
    has_bias = len(node.inputs) > 2

    M = inp.shape[0] if len(inp.shape) == 2 else int(np.prod(inp.shape[:-1]))
    K = inp.shape[-1]
    N = weight.shape[0]

    out_name = output_name or node.outputs[0].name
    input_bufs = [inp.name, weight.name]
    if has_bias:
        input_bufs.append(node.inputs[2].name)

    if M == 1:
        return KernelCall(
            kernel_name="matmul_vec_kernel",
            kernel_source="matmul.metal",
            input_buffers=input_bufs,
            output_buffers=[out_name],
            param_buffers=["matmul_params"],
            params={"M": M, "N": N, "K": K, "has_bias": 1 if has_bias else 0},
            dispatch_type="1d",
            total_threads=N,
        )

    return KernelCall(
        kernel_name="matmul_kernel",
        kernel_source="matmul.metal",
        input_buffers=input_bufs,
        output_buffers=[out_name],
        param_buffers=["matmul_params"],
        params={"M": M, "N": N, "K": K, "has_bias": 1 if has_bias else 0},
        dispatch_type="2d",
        grid_width=N,
        grid_height=M,
    )


def _gen_max_pool_kernel(node: OpNode) -> KernelCall:
    inp = node.inputs[0]
    N, C, H, W = inp.shape
    out_shape = node.outputs[0].shape
    out_h, out_w = out_shape[2], out_shape[3]

    kernel_size = node.attrs.get("kernel_size", [2, 2])
    stride = node.attrs.get("stride", kernel_size)
    padding = node.attrs.get("padding", [0, 0])

    total = int(np.prod(out_shape))

    return KernelCall(
        kernel_name="max_pool2d_kernel",
        kernel_source="pool.metal",
        input_buffers=[inp.name],
        output_buffers=[node.outputs[0].name],
        param_buffers=["pool_params"],
        params={
            "batch": N,
            "channels": C,
            "in_h": H,
            "in_w": W,
            "out_h": out_h,
            "out_w": out_w,
            "kernel_h": kernel_size[0],
            "kernel_w": kernel_size[1],
            "stride_h": stride[0],
            "stride_w": stride[1],
            "pad_h": padding[0] if isinstance(padding, list) else padding,
            "pad_w": padding[1] if isinstance(padding, list) else padding,
            "channels_aligned": pad_channels(C),
        },
        dispatch_type="1d",
        total_threads=total,
    )


def _gen_adaptive_avg_pool_kernel(node: OpNode) -> KernelCall:
    inp = node.inputs[0]
    N, C, H, W = inp.shape
    out_shape = node.outputs[0].shape
    out_h, out_w = out_shape[2], out_shape[3]

    total = int(np.prod(out_shape))

    return KernelCall(
        kernel_name="adaptive_avg_pool2d_kernel",
        kernel_source="pool.metal",
        input_buffers=[inp.name],
        output_buffers=[node.outputs[0].name],
        param_buffers=["pool_params"],
        params={
            "batch": N,
            "channels": C,
            "in_h": H,
            "in_w": W,
            "out_h": out_h,
            "out_w": out_w,
            "kernel_h": 0,
            "kernel_w": 0,
            "stride_h": 0,
            "stride_w": 0,
            "pad_h": 0,
            "pad_w": 0,
            "channels_aligned": pad_channels(C),
        },
        dispatch_type="1d",
        total_threads=total,
    )


def _gen_matmul_kernel(node: OpNode, graph: IRGraph) -> KernelCall:
    """Generate matmul kernel: handles 2D and batched (3D+) matmul."""
    a_shape = node.inputs[0].shape
    b_shape = node.inputs[1].shape

    if len(a_shape) >= 3:
        batch = int(np.prod(a_shape[:-2]))
        M = a_shape[-2]
        K = a_shape[-1]
        N = b_shape[-1]
        return KernelCall(
            kernel_name="batched_matmul_kernel",
            kernel_source="matmul.metal",
            input_buffers=[node.inputs[0].name, node.inputs[1].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["batched_matmul_params"],
            params={"batch": batch, "M": M, "N": N, "K": K},
            dispatch_type="3d",
            grid_width=N,
            grid_height=M,
            grid_depth=batch,
        )

    M = a_shape[0]
    K = a_shape[1]
    N = b_shape[1]

    if M == 1:
        return KernelCall(
            kernel_name="matmul_notrans_vec_kernel",
            kernel_source="matmul.metal",
            input_buffers=[node.inputs[0].name, node.inputs[1].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["matmul_params"],
            params={"M": M, "N": N, "K": K, "has_bias": 0},
            dispatch_type="1d",
            total_threads=N,
        )

    return KernelCall(
        kernel_name="matmul_notrans_kernel",
        kernel_source="matmul.metal",
        input_buffers=[node.inputs[0].name, node.inputs[1].name],
        output_buffers=[node.outputs[0].name],
        param_buffers=["matmul_params"],
        params={"M": M, "N": N, "K": K, "has_bias": 0},
        dispatch_type="2d",
        grid_width=N,
        grid_height=M,
    )


def _gen_addmm_kernel(node: OpNode, graph: IRGraph) -> KernelCall:
    """Generate addmm: bias + input @ weight.T."""
    bias = node.inputs[0]
    inp = node.inputs[1]
    weight = node.inputs[2]

    M = inp.shape[0]
    K = inp.shape[1]
    N = weight.shape[0]

    return KernelCall(
        kernel_name="matmul_kernel",
        kernel_source="matmul.metal",
        input_buffers=[inp.name, weight.name, bias.name],
        output_buffers=[node.outputs[0].name],
        param_buffers=["matmul_params"],
        params={"M": M, "N": N, "K": K, "has_bias": 1},
        dispatch_type="2d",
        grid_width=N,
        grid_height=M,
    )


# ---------------------------------------------------------------------------
# Main dispatch: single op -> KernelCall
# ---------------------------------------------------------------------------


def generate_single_kernel_call(node: OpNode, graph: IRGraph) -> KernelCall | None:  # noqa: C901
    if node.op_type == "aten.conv2d.default":
        return gen_conv_kernel(node, graph, has_relu=False)

    if node.op_type == "aten.relu.default":
        out_shape = node.outputs[0].shape
        padded = padded_shape_4d(out_shape)
        total = int(np.prod(padded))
        return KernelCall(
            kernel_name="elementwise_relu",
            kernel_source="elementwise.metal",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=[],
            params={},
            dispatch_type="1d",
            total_threads=total,
        )

    if node.op_type == "aten.add.Tensor":
        out_shape = node.outputs[0].shape
        if len(node.inputs) == 1:
            scalar = float(node.attrs.get("other", 0.0))
            total = int(np.prod(out_shape))
            return KernelCall(
                kernel_name="eltwise_add_scalar_kernel",
                kernel_source="elementwise_extended.metal",
                input_buffers=[node.inputs[0].name],
                output_buffers=[node.outputs[0].name],
                param_buffers=["scalar_params"],
                params={"scalar": scalar, "total": total},
                dispatch_type="1d",
                total_threads=total,
            )
        return gen_binary_op(node, "aten.add.Tensor")

    if node.op_type == "aten.linear.default":
        return _gen_linear_kernel(node, graph)

    if node.op_type == "aten.matmul.default":
        return _gen_matmul_kernel(node, graph)

    if node.op_type == "aten.addmm.default":
        return _gen_addmm_kernel(node, graph)

    if node.op_type == "aten.t.default":
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
        ndim = len(in_shape)
        if ndim == 2:
            total = int(np.prod(out_shape))
            strides_in = compute_strides(in_shape)
            strides_out = compute_strides(out_shape)
            return KernelCall(
                kernel_name="transpose_kernel",
                kernel_source="tensor_ops.metal",
                input_buffers=[node.inputs[0].name],
                output_buffers=[node.outputs[0].name],
                param_buffers=["transpose_params"],
                params={
                    "ndim": ndim,
                    "dim0": 0,
                    "dim1": 1,
                    "total": total,
                    "shape": list(in_shape) + [0] * (_MAX_NDIM - ndim),
                    "strides_in": strides_in + [0] * (_MAX_NDIM - ndim),
                    "strides_out": strides_out + [0] * (_MAX_NDIM - ndim),
                },
                dispatch_type="1d",
                total_threads=total,
            )
        return None

    if node.op_type == "aten.max_pool2d.default":
        return _gen_max_pool_kernel(node)

    if node.op_type == "aten.adaptive_avg_pool2d.default":
        return _gen_adaptive_avg_pool_kernel(node)

    if node.op_type in ("aten.flatten.using_ints", "aten.view.default", "aten.reshape.default"):
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape

        if len(in_shape) == 4 and len(out_shape) != 4:
            N, C, H, W = in_shape
            C_aligned = pad_channels(C)
            total = N * C * H * W
            return KernelCall(
                kernel_name="depad_4d_kernel",
                kernel_source="elementwise.metal",
                input_buffers=[node.inputs[0].name],
                output_buffers=[node.outputs[0].name],
                param_buffers=["depad_4d_params"],
                params={
                    "batch": N,
                    "channels": C,
                    "channels_aligned": C_aligned,
                    "height": H,
                    "width": W,
                },
                dispatch_type="1d",
                total_threads=total,
            )

        return KernelCall(
            kernel_name="_reshape",
            kernel_source="",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=[],
            params={"input_shape": in_shape, "output_shape": out_shape},
            dispatch_type="none",
        )

    # -- Element-wise ops (1D dispatch, total_elements) --

    _UNARY_OPS = {
        "aten.silu.default": "silu_kernel",
        "aten.neg.default": "neg_kernel",
        "aten.rsqrt.default": "rsqrt_kernel",
        "aten.cos.default": "cos_kernel",
        "aten.sin.default": "sin_kernel",
    }
    _BINARY_OPS = {
        "aten.mul.Tensor": "mul_kernel",
        "aten.div.Tensor": "div_kernel",
    }

    if node.op_type in _UNARY_OPS:
        total = int(np.prod(node.outputs[0].shape))
        return KernelCall(
            kernel_name=_UNARY_OPS[node.op_type],
            kernel_source="elementwise_extended.metal",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=[],
            params={},
            dispatch_type="1d",
            total_threads=total,
        )

    if node.op_type in _BINARY_OPS:
        if len(node.inputs) == 1:
            scalar = float(node.attrs.get("other", 1.0))
            total = int(np.prod(node.outputs[0].shape))
            return KernelCall(
                kernel_name="eltwise_mul_scalar_kernel",
                kernel_source="elementwise_extended.metal",
                input_buffers=[node.inputs[0].name],
                output_buffers=[node.outputs[0].name],
                param_buffers=["scalar_params"],
                params={"scalar": scalar, "total": total},
                dispatch_type="1d",
                total_threads=total,
            )
        return gen_binary_op(node, node.op_type)

    # -- Parameterized element-wise --

    if node.op_type == "aten.pow.Tensor_Scalar":
        total = int(np.prod(node.outputs[0].shape))
        exponent = float(node.attrs.get("exponent", 2.0))
        return KernelCall(
            kernel_name="pow_scalar_kernel",
            kernel_source="elementwise_extended.metal",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["pow_params"],
            params={"exponent": exponent},
            dispatch_type="1d",
            total_threads=total,
        )

    # -- Embedding --

    if node.op_type == "aten.embedding.default":
        weight = node.inputs[0]
        indices = node.inputs[1]
        vocab_size, embed_dim = weight.shape[0], weight.shape[1]
        seq_len = int(np.prod(indices.shape))
        return KernelCall(
            kernel_name="embedding_kernel",
            kernel_source="embedding.metal",
            input_buffers=[indices.name, weight.name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["embedding_params"],
            params={"seq_len": seq_len, "embed_dim": embed_dim, "vocab_size": vocab_size},
            dispatch_type="2d",
            grid_width=embed_dim,
            grid_height=seq_len,
        )

    # -- Zero-cost shape ops --

    if node.op_type in (
        "aten.contiguous.default",
        "aten.unsqueeze.default",
        "aten.alias.default",
        "aten.detach_.default",
    ):
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
        return KernelCall(
            kernel_name="_reshape",
            kernel_source="",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=[],
            params={"input_shape": in_shape, "output_shape": out_shape},
            dispatch_type="none",
        )

    # -- Softmax --

    if node.op_type == "aten.softmax.int":
        shape = node.inputs[0].shape
        dim = node.attrs.get("dim", -1)
        if dim < 0:
            dim += len(shape)
        cols = shape[dim]
        rows = int(np.prod(shape)) // cols
        return KernelCall(
            kernel_name="softmax_kernel",
            kernel_source="softmax.metal",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["softmax_params"],
            params={"rows": rows, "cols": cols},
            dispatch_type="1d",
            total_threads=rows,
        )

    # -- Mean (last dim reduction) --

    if node.op_type == "aten.mean.dim":
        in_shape = node.inputs[0].shape
        dims = node.attrs.get("dim", [-1])
        if dims == [-1] or dims == [len(in_shape) - 1]:
            cols = in_shape[-1]
            rows = int(np.prod(in_shape[:-1]))
            return KernelCall(
                kernel_name="mean_last_dim_kernel",
                kernel_source="elementwise_extended.metal",
                input_buffers=[node.inputs[0].name],
                output_buffers=[node.outputs[0].name],
                param_buffers=["reduce_params"],
                params={"rows": rows, "cols": cols},
                dispatch_type="1d",
                total_threads=rows,
            )
        if len(in_shape) == 4:
            return _gen_adaptive_avg_pool_kernel(node)
        return None

    # -- Transpose (dim swap) --

    if node.op_type == "aten.transpose.int":
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
        dim0 = node.attrs.get("dim0", 0)
        dim1 = node.attrs.get("dim1", 1)

        if in_shape[dim0] == 1 or in_shape[dim1] == 1:
            return KernelCall(
                kernel_name="_reshape",
                kernel_source="",
                input_buffers=[node.inputs[0].name],
                output_buffers=[node.outputs[0].name],
                param_buffers=[],
                params={"input_shape": list(in_shape), "output_shape": list(out_shape)},
                dispatch_type="none",
            )

        ndim = len(in_shape)
        total = int(np.prod(out_shape))
        strides_in = compute_strides(in_shape)
        strides_out = compute_strides(out_shape)
        return KernelCall(
            kernel_name="transpose_kernel",
            kernel_source="tensor_ops.metal",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["transpose_params"],
            params={
                "ndim": ndim,
                "dim0": dim0,
                "dim1": dim1,
                "total": total,
                "shape": list(in_shape) + [0] * (_MAX_NDIM - ndim),
                "strides_in": strides_in + [0] * (_MAX_NDIM - ndim),
                "strides_out": strides_out + [0] * (_MAX_NDIM - ndim),
            },
            dispatch_type="1d",
            total_threads=total,
        )

    # -- Cat (concatenation) --

    if node.op_type == "aten.cat.default":
        if node.inputs[0].shape == [0]:
            return KernelCall(
                kernel_name="_reshape",
                kernel_source="",
                input_buffers=[node.inputs[1].name],
                output_buffers=[node.outputs[0].name],
                param_buffers=[],
                params={"input_shape": node.inputs[1].shape, "output_shape": node.outputs[0].shape},
                dispatch_type="none",
            )

        out_shape = node.outputs[0].shape
        ndim = len(out_shape)
        axis = node.attrs.get("dim", 0)
        if axis < 0:
            axis += ndim
        total = int(np.prod(out_shape))
        in1_axis_size = node.inputs[0].shape[axis]
        strides = compute_strides(out_shape)
        return KernelCall(
            kernel_name="cat_2_kernel",
            kernel_source="tensor_ops.metal",
            input_buffers=[node.inputs[0].name, node.inputs[1].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["cat_params"],
            params={
                "axis": axis,
                "ndim": ndim,
                "total": total,
                "in1_axis_size": in1_axis_size,
                "out_shape": list(out_shape) + [0] * (_MAX_NDIM - ndim),
                "strides": strides + [0] * (_MAX_NDIM - ndim),
            },
            dispatch_type="1d",
            total_threads=total,
        )

    # -- Slice --

    if node.op_type == "aten.slice.Tensor":
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
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
        in_strides = compute_strides(in_shape)
        return KernelCall(
            kernel_name="slice_kernel",
            kernel_source="tensor_ops.metal",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["slice_params"],
            params={
                "dim": dim,
                "start": start,
                "end": end,
                "step": step,
                "ndim": ndim,
                "total": total,
                "in_shape": list(in_shape) + [0] * (_MAX_NDIM - ndim),
                "in_strides": in_strides + [0] * (_MAX_NDIM - ndim),
            },
            dispatch_type="1d",
            total_threads=total,
        )

    # -- Expand (broadcast copy) --

    if node.op_type == "aten.expand.default":
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
        ndim = len(out_shape)
        total = int(np.prod(out_shape))
        in_strides = compute_strides(in_shape)
        pad_len = ndim - len(in_shape)
        padded_in = [1] * pad_len + list(in_shape)
        padded_strides = [0] * pad_len + in_strides
        for i in range(ndim):
            if padded_in[i] == 1 and out_shape[i] != 1:
                padded_strides[i] = 0
        return KernelCall(
            kernel_name="expand_kernel",
            kernel_source="tensor_ops.metal",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["expand_params"],
            params={
                "ndim": ndim,
                "total": total,
                "in_shape": padded_in + [0] * (_MAX_NDIM - ndim),
                "out_shape": list(out_shape) + [0] * (_MAX_NDIM - ndim),
                "in_strides": padded_strides + [0] * (_MAX_NDIM - ndim),
            },
            dispatch_type="1d",
            total_threads=total,
        )

    # -- Full (constant tensor) --

    if node.op_type == "aten.full.default":
        out_shape = node.outputs[0].shape
        fill_value = float(node.attrs.get("fill_value", 0.0))
        total = int(np.prod(out_shape))
        return KernelCall(
            kernel_name="add_scalar_kernel",
            kernel_source="elementwise_extended.metal",
            input_buffers=[],
            output_buffers=[node.outputs[0].name],
            param_buffers=["scalar_params"],
            params={"scalar": fill_value, "total": total},
            dispatch_type="1d",
            total_threads=total,
        )

    # -- No-op: assertions and dropout (eval mode) --

    if node.op_type == "aten._assert_tensor_metadata.default":
        return None

    if node.op_type == "aten.dropout.default":
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
        return KernelCall(
            kernel_name="_reshape",
            kernel_source="",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=[],
            params={"input_shape": in_shape, "output_shape": out_shape},
            dispatch_type="none",
        )

    # -- Type cast (zero-cost alias in FP16 pipeline) --

    if node.op_type == "aten.to.dtype":
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
        return KernelCall(
            kernel_name="_reshape",
            kernel_source="",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=[],
            params={"input_shape": in_shape, "output_shape": out_shape},
            dispatch_type="none",
        )

    # -- Multi-output indexing (zero-cost alias) --

    if node.op_type == "<built-in function getitem>":
        idx = node.attrs.get("index", 0)
        in_name = node.inputs[idx].name if idx < len(node.inputs) else node.inputs[0].name
        out_shape = node.outputs[0].shape
        return KernelCall(
            kernel_name="_reshape",
            kernel_source="",
            input_buffers=[in_name],
            output_buffers=[node.outputs[0].name],
            param_buffers=[],
            params={"input_shape": out_shape, "output_shape": out_shape},
            dispatch_type="none",
        )

    # -- Index Copy (KV cache update) --

    if node.op_type == "aten.index_copy.default":
        in_shape = node.inputs[0].shape
        source_shape = node.inputs[2].shape
        dim = node.attrs.get("dim", 0)
        ndim = len(in_shape)

        outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
        dim_size = in_shape[dim]
        inner = int(np.prod(in_shape[dim + 1 :])) if dim + 1 < ndim else 1
        num_indices = source_shape[dim]
        total = int(np.prod(in_shape))

        return KernelCall(
            kernel_name="index_copy_kernel",
            kernel_source="index_copy.metal",
            input_buffers=[node.inputs[0].name, node.inputs[2].name, node.inputs[1].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["index_copy_params"],
            params={"outer_size": outer, "dim_size": dim_size, "inner_size": inner, "num_indices": num_indices},
            dispatch_type="1d",
            total_threads=total,
        )

    # -- RoPE (Rotary Position Embedding) --

    if node.op_type == "wrap_with_set_grad_enabled":
        inv_freq = node.inputs[0]
        positions = node.inputs[1]

        out_shape = node.outputs[0].shape
        seq_len = out_shape[-2] if len(out_shape) >= 2 else out_shape[0]
        head_dim = out_shape[-1]

        return KernelCall(
            kernel_name="rope_kernel",
            kernel_source="rope.metal",
            input_buffers=[inv_freq.name, positions.name],
            output_buffers=[node.outputs[0].name, node.outputs[1].name],
            param_buffers=["rope_params"],
            params={"seq_len": seq_len, "head_dim": head_dim},
            dispatch_type="2d",
            grid_width=head_dim,
            grid_height=seq_len,
        )

    return None


# ---------------------------------------------------------------------------
# HANDLED_OPS: single source of truth for supported ops
# ---------------------------------------------------------------------------

HANDLED_OPS: set[str] = {
    # CNN ops
    "aten.conv2d.default",
    "aten.relu.default",
    "aten.relu_.default",
    "aten.batch_norm.default",
    "aten.add.Tensor",
    "aten.add_.Tensor",
    "aten.max_pool2d.default",
    "aten.adaptive_avg_pool2d.default",
    "aten.linear.default",
    "aten.flatten.using_ints",
    "aten.view.default",
    "aten.reshape.default",
    "aten.addmm.default",
    "aten.matmul.default",
    "aten.mean.dim",
    "aten.t.default",
    # element-wise
    "aten.mul.Tensor",
    "aten.div.Tensor",
    "aten.neg.default",
    "aten.pow.Tensor_Scalar",
    "aten.rsqrt.default",
    "aten.silu.default",
    "aten.cos.default",
    "aten.sin.default",
    # tensor manipulation
    "aten.embedding.default",
    "aten.transpose.int",
    "aten.contiguous.default",
    "aten.unsqueeze.default",
    "aten.cat.default",
    "aten.slice.Tensor",
    "aten.expand.default",
    # reduction / normalization
    "aten.softmax.int",
    # misc
    "aten.full.default",
    # type casting (zero-cost alias)
    "aten.to.dtype",
    # multi-output indexing (zero-cost alias)
    "<built-in function getitem>",
    # RoPE
    "wrap_with_set_grad_enabled",
    # no-op passthrough ops
    "aten._assert_tensor_metadata.default",
    "aten.alias.default",
    "aten.detach_.default",
    "aten.dropout.default",
    # index operations
    "aten.index_copy.default",
}
