"""Code generator: maps IR graph to Metal shader execution plan.

Architecture:
    CodegenTarget (ABC) defines the interface for backend-specific code generation.
    MetalCodegenTarget is the default implementation for Apple Metal GPU.
    generate_execution_plan() uses the target to produce KernelCall objects
    that the runtime executor can dispatch.

Module layout:
    codegen_core.py  — data classes (KernelCall, BufferAllocation, ExecutionPlan),
                       CodegenTarget ABC, MetalCodegenTarget, shared constants.
    codegen_ops.py   — single-op codegen (_generate_single_kernel_call, HANDLED_OPS).
    codegen_fused.py — fused kernel codegen (registry + handlers).
    codegen.py       — orchestration (generate_execution_plan) + re-exports.

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

import numpy as np

# Re-export from codegen_core — these are the public API used by tests, executor, etc.
from npu_compiler.codegen_core import (  # noqa: F401
    _DTYPE_ELEM_SIZE,
    _MAX_NDIM,
    _OP_NORMALIZE,
    METAL_TARGET,
    BufferAllocation,
    CodegenTarget,
    ExecutionPlan,
    KernelCall,
    MetalCodegenTarget,
)

# Re-export from codegen_fused
from npu_compiler.codegen_fused import (  # noqa: F401
    _FUSED_CODEGEN_REGISTRY,
    register_fused_codegen,
)
from npu_compiler.codegen_fused import (
    generate_fused_kernel_call as _generate_fused_kernel_call,
)

# Re-export from codegen_ops
from npu_compiler.codegen_ops import (  # noqa: F401
    HANDLED_OPS,
)
from npu_compiler.codegen_ops import (
    generate_single_kernel_call as _generate_single_kernel_call,
)
from npu_compiler.fusion_patterns import FusedGroup
from npu_compiler.ir_reader import IRGraph, OpNode, TensorSpec
from npu_compiler.layout import LayoutProfile, resolve_layouts


def _infer_compute_dtype(graph: IRGraph) -> str:
    """Infer compute dtype from graph: bfloat16 if any tensor is bf16, else float16."""
    for spec in graph.weights:
        if spec.dtype == "bfloat16":
            return "bfloat16"
    for spec in graph.graph_inputs:
        if spec.dtype == "bfloat16":
            return "bfloat16"
    return "float16"


def _normalize_ops(graph: IRGraph):
    """Normalize in-place ops to out-of-place equivalents."""
    for node in graph.nodes:
        if node.op_type in _OP_NORMALIZE:
            node.op_type = _OP_NORMALIZE[node.op_type]


def _build_io_transforms(spec: TensorSpec, compute_dtype: str = "bfloat16") -> list[dict]:
    """Build host->NPU transform steps for an I/O tensor."""
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
    from npu_compiler.fusion_patterns import find_fusion_groups

    _normalize_ops(graph)
    fusion_result = find_fusion_groups(graph)
    compute_dtype = _infer_compute_dtype(graph)

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

    external_names = {inp.name for inp in graph.graph_inputs}
    for w in graph.weights:
        external_names.add(w.name)
    for placeholder_name in graph.weight_name_mapping:
        external_names.add(placeholder_name)

    def ensure_buffer(name: str, shape: list[int], dtype: str = compute_dtype):
        if name not in external_names and name not in allocated_buffers:
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

        call_list = result if isinstance(result, list) else [result]

        for call in call_list:
            kernel_calls.append(call)

            for buf_name in call.output_buffers:
                shape = _find_buffer_shape(buf_name, fusion_result, graph)
                if shape:
                    ensure_buffer(buf_name, shape)
                elif buf_name.startswith("_broadcast_"):
                    bcast_shape = [s for s in call.params.get("out_shape", []) if s != 0]
                    if bcast_shape:
                        ensure_buffer(buf_name, bcast_shape)

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
