"""Code generator: maps IR graph to Metal shader execution plan.

Architecture:
    CodegenTarget (ABC) defines the interface for backend-specific code generation.
    MetalCodegenTarget is the default implementation for Apple Metal GPU.
    generate_execution_plan() uses the target to produce KernelCall objects
    that the runtime executor can dispatch.

Design trade-offs:
    - Transpose folding REMOVED: An earlier optimization folded aten.transpose.int
      into matmul by swapping N/K and setting transpose_b=True. Removed because
      (1) our custom Metal matmul kernel has no transpose_b parameter — adding one
      would double the kernel variants, and (2) MPS MPSMatrixMultiplication's
      transposeRight flag caused N/K dimension swap errors on certain shapes.
      The explicit transpose_kernel is correct at ~2% overhead per attention layer.

    - Vec matmul for M=1: Decode (M=1) uses a parallel K-reduction kernel instead
      of the tiled TILE×TILE matmul. Tiled matmul wastes (TILE-1)/TILE threads
      when M=1; vec matmul spawns N threadgroups × max_threadgroup_1d threads
      (see TargetConfig) for full occupancy.

    - 4D channel padding: Metal requires channel_alignment_bytes-aligned access
      for coalesced SIMD reads. FP16 elements are 2 bytes, so channels must be
      multiples of channel_tile (see TargetConfig). Non-4D tensors (matmul,
      embedding) skip padding because they use row-major layout where the inner
      dimension is already contiguous.

    - In-place op normalization: PyTorch's FX graph preserves in-place ops
      (relu_, add_) but our NPU pipeline requires separate input/output buffers.
      Normalizing to out-of-place at codegen is simpler than handling aliasing
      at runtime (which would need reference counting and copy-on-write).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from npu_compiler.fusion_patterns import FusedGroup, find_fusion_groups
from npu_compiler.ir_reader import IRGraph, OpNode, TensorSpec
from npu_compiler.layout import LayoutProfile, resolve_layouts
from npu_compiler.target_config import pad_channels, padded_shape_4d

# ---------------------------------------------------------------------------
# CodegenTarget: abstract interface for backend-specific kernel emission
# ---------------------------------------------------------------------------


class CodegenTarget(ABC):
    """Abstract interface for backend-specific code generation.

    Each target maps ATen ops to backend kernel specifications.
    The Metal implementation emits .metal shader file references;
    a hypothetical CUDA target would emit .cu files or PTX, etc.
    """

    @abstractmethod
    def elementwise_kernel(self, kernel_name: str) -> tuple[str, str]:
        """Return (kernel_function_name, shader_source) for an elementwise kernel."""

    @abstractmethod
    def matmul_kernel(self, is_vec: bool, transpose_b: bool) -> tuple[str, str]:
        """Return (kernel_function_name, shader_source) for matmul."""

    @abstractmethod
    def fused_kernel(self, kernel_type: str) -> tuple[str, str]:
        """Return (kernel_function_name, shader_source) for a fused kernel."""

    @abstractmethod
    def shader_source(self, filename: str) -> str:
        """Map logical shader name to backend-specific source path."""


class MetalCodegenTarget(CodegenTarget):
    """Metal GPU codegen target — emits .metal shader references."""

    _EW_MAP = {
        "elementwise_relu": "elementwise.metal",
        "silu_kernel": "elementwise_extended.metal",
        "neg_kernel": "elementwise_extended.metal",
        "rsqrt_kernel": "elementwise_extended.metal",
        "cos_kernel": "elementwise_extended.metal",
        "sin_kernel": "elementwise_extended.metal",
        "pow_scalar_kernel": "elementwise_extended.metal",
        "mul_kernel": "elementwise_extended.metal",
        "div_kernel": "elementwise_extended.metal",
        "add_kernel": "add_relu.metal",
    }

    def elementwise_kernel(self, kernel_name: str) -> tuple[str, str]:
        source = self._EW_MAP.get(kernel_name, "elementwise_extended.metal")
        return (kernel_name, source)

    def matmul_kernel(self, is_vec: bool, transpose_b: bool) -> tuple[str, str]:
        if is_vec:
            name = "matmul_vec_kernel" if transpose_b else "matmul_notrans_vec_kernel"
        else:
            name = "matmul_kernel" if transpose_b else "matmul_notrans_kernel"
        return (name, "matmul.metal")

    def fused_kernel(self, kernel_type: str) -> tuple[str, str]:
        _FUSED_MAP = {
            "conv_bn_relu": ("conv2d_kernel", "conv_bn_relu.metal"),
            "add_relu": ("add_relu_kernel", "add_relu.metal"),
            "silu_mul": ("silu_mul_kernel", "elementwise_extended.metal"),
            "rmsnorm": ("rmsnorm_kernel", "rmsnorm.metal"),
            "masked_softmax": ("masked_softmax_broadcast_kernel", "softmax.metal"),
            "decode_attention": ("fused_decode_attention_kernel", "fused_decode_attention.metal"),
        }
        return _FUSED_MAP.get(kernel_type, (kernel_type, f"{kernel_type}.metal"))

    def shader_source(self, filename: str) -> str:
        return filename


# Default target singleton
METAL_TARGET = MetalCodegenTarget()

# Metal doesn't support variable-length struct fields, so param buffers use
# fixed 6-element arrays for shape/stride. Covers all cases from scalar (ndim=0)
# to 6D batched attention tensors. Tensors with ndim < 6 are zero-padded.
_MAX_NDIM = 6

# NPU pipeline prohibits in-place ops (separate input/output buffers required),
# so we normalize in-place variants to out-of-place before codegen.
_OP_NORMALIZE = {
    "aten.relu_.default": "aten.relu.default",
    "aten.add_.Tensor": "aten.add.Tensor",
}


@dataclass
class KernelCall:
    """A single Metal kernel invocation."""

    kernel_name: str  # Metal function name
    kernel_source: str  # shader source file path (e.g. .metal)
    input_buffers: list[str]  # buffer names for inputs
    output_buffers: list[str]  # buffer names for outputs
    param_buffers: list[str]  # named param buffers
    params: dict  # kernel parameters (for struct packing)
    dispatch_type: str = "1d"  # "1d", "2d", or "3d"
    total_threads: int = 0  # for 1d dispatch
    grid_width: int = 0  # for 2d dispatch
    grid_height: int = 0  # for 2d dispatch
    grid_depth: int = 0  # for 3d dispatch


@dataclass
class BufferAllocation:
    """Memory allocation for an intermediate buffer."""

    name: str
    shape: list[int]
    alloc_shape: list[int] | None = None  # physical shape (None = same as shape)
    dtype: str = "float16"
    size_bytes: int = 0


@dataclass
class ExecutionPlan:
    """Complete execution plan for a compiled program."""

    kernel_calls: list[KernelCall]
    buffer_allocations: list[BufferAllocation]
    input_specs: list[TensorSpec]
    output_specs: list[TensorSpec]
    weight_specs: list[TensorSpec]
    weight_name_mapping: dict[str, str]
    compute_dtype: str = "bfloat16"


# Bytes per element for supported compute dtypes (FP16/BF16 = 2, FP32/INT32 = 4)
_DTYPE_ELEM_SIZE = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "int64": 8}


def _size_bytes(shape, dtype="float16", pad_4d=False):
    elem_size = _DTYPE_ELEM_SIZE.get(dtype, 2)
    final_shape = padded_shape_4d(shape) if pad_4d else shape
    return int(np.prod(final_shape)) * elem_size


def _infer_compute_dtype(graph: IRGraph) -> str:
    """Infer compute dtype from graph: bfloat16 if any tensor is bf16, else float16."""
    for spec in graph.weights:
        if spec.dtype == "bfloat16":
            return "bfloat16"
    for spec in graph.graph_inputs:
        if spec.dtype == "bfloat16":
            return "bfloat16"
    return "float16"


def _compute_strides(shape: list[int]) -> list[int]:
    """Compute row-major strides for a shape."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _compute_broadcast_strides(in_shape: list[int], out_shape: list[int]) -> list[int]:
    """Compute strides for a tensor broadcast to out_shape (0 for broadcast dims)."""
    ndim = len(out_shape)
    in_strides = _compute_strides(in_shape)
    pad_len = ndim - len(in_shape)
    padded_in = [1] * pad_len + list(in_shape)
    padded_strides = [0] * pad_len + in_strides
    for i in range(ndim):
        if padded_in[i] == 1 and out_shape[i] != 1:
            padded_strides[i] = 0
    return padded_strides


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


def _gen_binary_op(node: OpNode, op_type: str) -> KernelCall | list[KernelCall]:
    """Generate binary op kernel, using broadcast kernel when shapes differ."""
    out_shape = node.outputs[0].shape
    a_shape = node.inputs[0].shape
    b_shape = node.inputs[1].shape
    a_name = node.inputs[0].name
    b_name = node.inputs[1].name
    total = int(np.prod(out_shape))

    needs_broadcast = list(a_shape) != list(out_shape) or list(b_shape) != list(out_shape)

    if needs_broadcast:
        # Use stride-based broadcast kernel to avoid materializing the expanded tensor,
        # saving ~310 expand dispatches per Qwen decode step.
        ndim = len(out_shape)
        a_strides = _compute_broadcast_strides(a_shape, out_shape)
        b_strides = _compute_broadcast_strides(b_shape, out_shape)
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
        # Same-shape: use simple elementwise kernel
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


def _normalize_ops(graph: IRGraph):
    """Normalize in-place ops to out-of-place equivalents."""
    for node in graph.nodes:
        if node.op_type in _OP_NORMALIZE:
            node.op_type = _OP_NORMALIZE[node.op_type]


def _build_io_transforms(spec: TensorSpec, compute_dtype: str = "bfloat16") -> list[dict]:
    """Build host→NPU transform steps for an I/O tensor."""
    if spec.dtype in ("int32", "int64"):
        steps: list[dict] = [{"type": "cast", "to": "int32"}]
    else:
        steps = [{"type": "cast", "to": compute_dtype}]
    if spec.alloc_shape is not None:
        steps.append({"type": "pad", "alloc_shape": spec.alloc_shape})
    return steps


def generate_execution_plan(graph: IRGraph, profile: LayoutProfile | None = None) -> ExecutionPlan:
    """Generate a Metal execution plan from an IR graph.

    Args:
        graph: The IR graph to compile.
        profile: Optional layout profile for backend-specific layout overrides.
                 Defaults to METAL_PROFILE (PADDED_NCHW for conv/pool ops).
    """
    _normalize_ops(graph)
    fusion_result = find_fusion_groups(graph)
    compute_dtype = _infer_compute_dtype(graph)

    # Resolve layouts for all tensors using the layout system
    tensor_layouts = resolve_layouts(
        nodes=fusion_result,
        graph_inputs=graph.graph_inputs,
        weights=graph.weights,
        weight_name_mapping=graph.weight_name_mapping,
        profile=profile,
    )

    kernel_calls = []
    buffer_allocs = []
    allocated_buffers: set[str] = set()

    # Track which tensor names are graph inputs or weights (don't allocate)
    external_names = {inp.name for inp in graph.graph_inputs}
    for w in graph.weights:
        external_names.add(w.name)
    # Also add placeholder names from weight_name_mapping
    for placeholder_name in graph.weight_name_mapping:
        external_names.add(placeholder_name)

    def ensure_buffer(name: str, shape: list[int], dtype: str = compute_dtype):
        if name not in external_names and name not in allocated_buffers:
            # Use resolved layout for physical shape instead of hardcoded _PADDED_KERNELS
            resolved = tensor_layouts.get(name)
            if resolved and resolved.needs_padding:
                alloc_shape = resolved.physical_shape
            else:
                alloc_shape = list(shape)
            buffer_allocs.append(
                BufferAllocation(
                    name=name,
                    shape=shape,
                    alloc_shape=alloc_shape if alloc_shape != list(shape) else None,
                    dtype=dtype,
                    size_bytes=int(np.prod(alloc_shape)) * _DTYPE_ELEM_SIZE.get(dtype, 2),
                )
            )
            allocated_buffers.add(name)

    for item in fusion_result:
        if isinstance(item, FusedGroup):
            result = _generate_fused_kernel_call(item, graph)
        else:
            result = _generate_single_kernel_call(item, graph)

        if result is None:
            continue

        # Handle single call or list of calls (broadcasting inserts expand steps)
        call_list = result if isinstance(result, list) else [result]

        for call in call_list:
            kernel_calls.append(call)

            # Allocate output buffers
            for buf_name in call.output_buffers:
                # Find the shape from the node outputs
                shape = _find_buffer_shape(buf_name, fusion_result, graph)
                if shape:
                    ensure_buffer(buf_name, shape)
                elif buf_name.startswith("_broadcast_"):
                    # Broadcast temp buffer: infer shape from expand params
                    bcast_shape = [s for s in call.params.get("out_shape", []) if s != 0]
                    if bcast_shape:
                        ensure_buffer(buf_name, bcast_shape)

    # Set I/O alloc_shape based on per-tensor resolved layout (not model-level heuristic)
    for spec in graph.graph_inputs:
        resolved = tensor_layouts.get(spec.name)
        if resolved and resolved.needs_padding:
            spec.alloc_shape = resolved.physical_shape
        else:
            spec.alloc_shape = None
        spec.transform_steps = _build_io_transforms(spec, compute_dtype)
    for spec in graph.graph_outputs:
        resolved = tensor_layouts.get(spec.name)
        if resolved and resolved.needs_padding:
            spec.alloc_shape = resolved.physical_shape
        else:
            spec.alloc_shape = None
        spec.transform_steps = _build_io_transforms(spec, compute_dtype)

    return ExecutionPlan(
        kernel_calls=kernel_calls,
        buffer_allocations=buffer_allocs,
        input_specs=graph.graph_inputs,
        output_specs=graph.graph_outputs,
        weight_specs=graph.weights,
        weight_name_mapping=graph.weight_name_mapping,
        compute_dtype=compute_dtype,
    )


def _find_buffer_shape(name: str, fusion_result, graph: IRGraph) -> list[int] | None:
    for item in fusion_result:
        if isinstance(item, FusedGroup):
            for node in item.nodes:
                for out in node.outputs:
                    if out.name == name:
                        return out.shape
        elif isinstance(item, OpNode):
            for out in item.outputs:
                if out.name == name:
                    return out.shape
    return None


# ---------------------------------------------------------------------------
# Fused kernel codegen registry — enables adding new fusion patterns without
# modifying _generate_fused_kernel_call. Register a handler with:
#   register_fused_codegen("my_pattern", my_gen_function)
# ---------------------------------------------------------------------------

_FUSED_CODEGEN_REGISTRY: dict[str, callable] = {}


def register_fused_codegen(kernel_type: str, handler: callable):
    """Register a codegen handler for a fused kernel type.

    Handler signature: (group: FusedGroup, graph: IRGraph) -> KernelCall | None
    """
    _FUSED_CODEGEN_REGISTRY[kernel_type] = handler


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
    first_node = group.nodes[0]
    last_node = group.nodes[-1]
    return _gen_conv_kernel(
        first_node, graph, has_relu="relu" in group.kernel_type, output_name=last_node.outputs[0].name
    )


def _generate_fused_kernel_call(group: FusedGroup, graph: IRGraph) -> KernelCall | None:
    handler = _FUSED_CODEGEN_REGISTRY.get(group.kernel_type)
    if handler is not None:
        return handler(group, graph)
    return None


def _generate_single_kernel_call(node: OpNode, graph: IRGraph) -> KernelCall | None:
    if node.op_type == "aten.conv2d.default":
        return _gen_conv_kernel(node, graph, has_relu=False)

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
            # Scalar add: x + scalar
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
        return _gen_binary_op(node, "aten.add.Tensor")

    if node.op_type == "aten.linear.default":
        return _gen_linear_kernel(node, graph)

    if node.op_type == "aten.matmul.default":
        return _gen_matmul_kernel(node, graph)

    if node.op_type == "aten.addmm.default":
        return _gen_addmm_kernel(node, graph)

    if node.op_type == "aten.t.default":
        # Transpose 2D: zero-cost, just swap shape
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
        ndim = len(in_shape)
        if ndim == 2:
            total = int(np.prod(out_shape))
            strides_in = _compute_strides(in_shape)
            strides_out = _compute_strides(out_shape)
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

        # 4D tensors have channels padded to channel_tile (see TargetConfig).
        # When flattening to 2D (e.g., before linear), we must depad first
        # to produce a dense layout matching the weight matrix dimensions.
        # 4D → non-4D: need depad_4d_kernel to strip channel padding
        if len(in_shape) == 4 and len(out_shape) != 4:
            N, C, H, W = in_shape
            C_aligned = pad_channels(C)
            total = N * C * H * W  # dense output size
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

        # Non-padded reshape: zero-cost alias
        return KernelCall(
            kernel_name="_reshape",
            kernel_source="",
            input_buffers=[node.inputs[0].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=[],
            params={"input_shape": in_shape, "output_shape": out_shape},
            dispatch_type="none",
        )

    # ── Element-wise ops (1D dispatch, total_elements) ──

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
            # Scalar variant: x * scalar
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
        return _gen_binary_op(node, node.op_type)

    # ── Parameterized element-wise ──

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

    # ── Embedding ──

    if node.op_type == "aten.embedding.default":
        weight = node.inputs[0]  # (vocab_size, embed_dim)
        indices = node.inputs[1]  # (seq_len,) or (batch, seq_len)
        out_shape = node.outputs[0].shape
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

    # ── Zero-cost shape ops ──
    # These ops only change tensor metadata (shape/strides), not data.
    # Emitting them as _reshape avoids GPU dispatch entirely — the executor
    # just creates a new NPUBuffer pointing to the same Metal buffer.

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

    # ── Softmax ──
    # Dispatches one thread per row; each thread computes max-subtraction + exp + sum
    # sequentially over columns. Row-parallel is efficient because softmax reduction
    # is along the last dim and rows are independent.

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

    # ── Mean (last dim reduction) ──
    # Two paths: last-dim reduction uses dedicated kernel (1 thread/row);
    # spatial-dim reduction (ResNet global avg pool) uses adaptive_avg_pool.

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
        # Fallback: 4D adaptive avg pool style (for ResNet mean over spatial dims)
        if len(in_shape) == 4:
            return _gen_adaptive_avg_pool_kernel(node)
        return None

    # ── Transpose (dim swap) ──
    # Explicit data movement kernel (not folded into matmul) because:
    # 1. Metal custom kernel has no transpose_b parameter
    # 2. MPS matmul transpose was broken (N/K swap incorrect)
    # See: "Transpose folding: REMOVED" in MEMORY.md

    if node.op_type == "aten.transpose.int":
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
        dim0 = node.attrs.get("dim0", 0)
        dim1 = node.attrs.get("dim1", 1)

        # Optimization: when either transposed dimension has size 1,
        # the data layout is unchanged — emit a zero-cost alias instead.
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
        strides_in = _compute_strides(in_shape)
        strides_out = _compute_strides(out_shape)
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

    # ── Cat (concatenation) ──

    if node.op_type == "aten.cat.default":
        # KV cache initialization produces [0]-shaped empty tensors; cat([0], data) = data.
        # Alias avoids dispatching a kernel for a no-op concatenation.
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
        strides = _compute_strides(out_shape)
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

    # ── Slice ──

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
        in_strides = _compute_strides(in_shape)
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

    # ── Expand (broadcast copy) ──

    if node.op_type == "aten.expand.default":
        in_shape = node.inputs[0].shape
        out_shape = node.outputs[0].shape
        ndim = len(out_shape)
        total = int(np.prod(out_shape))
        # Compute input strides with broadcasting (0 for broadcast dims)
        in_strides = _compute_strides(in_shape)
        # Pad in_shape to match out ndim (left-pad with 1s)
        pad_len = ndim - len(in_shape)
        padded_in = [1] * pad_len + list(in_shape)
        padded_strides = [0] * pad_len + in_strides
        # Set stride to 0 for broadcast dimensions
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

    # ── Full (constant tensor) ──

    if node.op_type == "aten.full.default":
        # Zero-cost: allocate buffer and fill (handled via _reshape alias to a filled buffer)
        out_shape = node.outputs[0].shape
        fill_value = float(node.attrs.get("fill_value", 0.0))
        total = int(np.prod(out_shape))
        return KernelCall(
            kernel_name="add_scalar_kernel",
            kernel_source="elementwise_extended.metal",
            input_buffers=[],  # no input needed, we'll create a zero buffer
            output_buffers=[node.outputs[0].name],
            param_buffers=["scalar_params"],
            params={"scalar": fill_value, "total": total},
            dispatch_type="1d",
            total_threads=total,
        )

    # ── No-op: assertions and dropout (eval mode) ──
    # These ops exist in the IR because torch.export preserves them, but they
    # have no effect at inference time. Returning None / _reshape skips them
    # completely, avoiding unnecessary GPU dispatch overhead.

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

    # ── Type cast (zero-cost alias in FP16 pipeline) ──
    # All tensors are already in compute_dtype (fp16 or bf16), so to.dtype
    # is a no-op. The original PyTorch graph inserts these for fp32→fp16 casts
    # which our pipeline handles at weight loading / input preparation time.

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

    # ── Multi-output indexing (zero-cost alias) ──

    if node.op_type == "<built-in function getitem>":
        # Selects one output from a multi-output node by index.
        # input_buffers[0] is the producer output at the given index.
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

    # ── Index Copy (KV cache update) ──

    if node.op_type == "aten.index_copy.default":
        # index_copy(self, dim, index, source) → output
        # inputs[0]: self (input tensor)
        # inputs[1]: index tensor (int64/int32 positions)
        # inputs[2]: source tensor (values to copy)
        # attrs["dim"]: dimension along which to copy
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
            # buffer order: input, source, index → matches Metal kernel signature
            input_buffers=[node.inputs[0].name, node.inputs[2].name, node.inputs[1].name],
            output_buffers=[node.outputs[0].name],
            param_buffers=["index_copy_params"],
            params={"outer_size": outer, "dim_size": dim_size, "inner_size": inner, "num_indices": num_indices},
            dispatch_type="1d",
            total_threads=total,
        )

    # ── RoPE (Rotary Position Embedding) ──

    if node.op_type == "wrap_with_set_grad_enabled":
        # RoPE subgraph: computes cos/sin from inv_freq and positions.
        # inputs[0]: inv_freq (head_dim/2,) or (1, head_dim/2)
        # inputs[1]: positions (1, seq_len) or (seq_len,) int32
        # outputs[0]: cos (1, seq_len, head_dim)
        # outputs[1]: sin (1, seq_len, head_dim)
        inv_freq = node.inputs[0]
        positions = node.inputs[1]

        # Derive head_dim and seq_len from output shape
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


def _gen_rmsnorm_kernel(group: FusedGroup, graph: IRGraph | None = None) -> KernelCall:
    """Generate fused RMSNorm kernel from pow→mean→add(eps)→rsqrt→mul(x)→mul(weight) chain."""
    pow_node = group.nodes[0]
    last_node = group.nodes[-1]  # final mul(weight)

    # Input x is pow_node's input
    x_name = pow_node.inputs[0].name
    x_shape = pow_node.inputs[0].shape

    # eps from the add(scalar) node — it's the 3rd real node (pow, mean, add)
    add_node = None
    for n in group.nodes:
        if n.op_type == "aten.add.Tensor" and len(n.inputs) == 1:
            add_node = n
            break
    eps = float(add_node.attrs.get("other", 1e-6)) if add_node else 1e-6

    # Weight is the other input of the last mul (not the intermediate result)
    # The last mul has two inputs: one is the output of mul(x*rsqrt), the other is weight
    weight_name = None
    for inp in last_node.inputs:
        # The weight input is not produced by nodes in this group
        group_output_names = set()
        for n in group.nodes:
            for out in n.outputs:
                group_output_names.add(out.name)
        if inp.name not in group_output_names:
            weight_name = inp.name
            break
    if weight_name is None:
        # Fallback: second input of last mul
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
    """Generate fused SiLU * gate kernel: silu(gate) * up → 1 dispatch."""
    silu_node = group.nodes[0]
    mul_node = group.nodes[1]

    gate_name = silu_node.inputs[0].name  # input to silu
    # The other input to mul (not silu's output)
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
    """Generate fused masked softmax: add(scores, mask) + softmax → 1 dispatch."""
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

    # Check if mask needs broadcasting
    mask_shape = add_node.inputs[1].shape
    needs_broadcast = list(mask_shape) != list(out_shape)

    if needs_broadcast:
        ndim = len(out_shape)
        mask_strides = _compute_broadcast_strides(mask_shape, out_shape)
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
    """Generate fused decode attention kernel from transpose→matmul→scale→add→softmax→matmul chain.

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
    shape = meta["shape"]  # (B, H, S, D)
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


# Register all built-in fused codegen handlers (after function definitions).
# All handlers share uniform signature: (group: FusedGroup, graph: IRGraph) -> KernelCall
register_fused_codegen("conv_bn_relu", _gen_conv_fused_kernel)
register_fused_codegen("conv_bn", _gen_conv_fused_kernel)
register_fused_codegen("conv_relu", _gen_conv_fused_kernel)
register_fused_codegen("add_relu", _gen_add_relu_kernel)
register_fused_codegen("rmsnorm", _gen_rmsnorm_kernel)
register_fused_codegen("silu_mul", _gen_silu_mul_kernel)
register_fused_codegen("masked_softmax", _gen_masked_softmax_kernel)
register_fused_codegen("rope_rotate", _gen_rope_rotate_kernel)
register_fused_codegen("decode_attention", _gen_fused_decode_attention_kernel)


def _gen_conv_kernel(node: OpNode, graph: IRGraph, has_relu: bool, output_name: str | None = None) -> KernelCall:
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

    # Vec kernel for M=1 (decode): parallel K-reduction, better than tiled for M=1
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
        params={
            "M": M,
            "N": N,
            "K": K,
            "has_bias": 1 if has_bias else 0,
        },
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
    """Generate matmul kernel: handles 2D and batched (3D+) matmul.

    2D: A(M,K) @ B(K,N) → C(M,N) — uses matmul_kernel (non-transposed B)
    3D+: A(B,M,K) @ B(B,K,N) → C(B,M,N) — uses batched_matmul_kernel
    """
    a_shape = node.inputs[0].shape
    b_shape = node.inputs[1].shape

    if len(a_shape) >= 3:
        # Batched matmul
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

    # 2D matmul: A(M,K) @ B(K,N) → C(M,N) — B is NOT transposed
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


# All ops that the compiler pipeline can process. This is the single source of
# truth for SUPPORTED_OPS in constraint_checker.py (auto-imported).
# Includes ops handled by codegen, graph_optimizer (BN folding), and normalization.
HANDLED_OPS: set[str] = {
    # CNN ops
    "aten.conv2d.default",
    "aten.relu.default",
    "aten.relu_.default",
    "aten.batch_norm.default",  # consumed by graph_optimizer (BN folding), not codegen
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
