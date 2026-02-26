"""Buffer allocation planner for CUDA programs.

Analyzes execution steps to determine which intermediate buffers need
to be allocated and their sizes/dtypes.
"""

from __future__ import annotations

import numpy as np

from cuda_compiler.cuda_program import (
    AliasStep,
    BufferAllocation,
    CUBLASStep,
    ExecStep,
    FusedKernelStep,
    ReductionKernelStep,
    SpecialKernelStep,
)
from npu_compiler.ir_reader import TensorSpec

_DTYPE_ELEM_SIZE = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "int64": 8}


def plan_buffers(
    steps: list[ExecStep],
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    weight_specs: list[TensorSpec],
    compute_dtype: str = "float16",
) -> list[BufferAllocation]:
    """Plan intermediate buffer allocations for a CUDA program.

    Returns a list of BufferAllocation for buffers that are neither
    graph inputs/outputs nor weights — i.e., intermediate tensors.
    """
    # Collect all known external buffer names
    external_names: set[str] = set()
    for spec in input_specs:
        external_names.add(spec.name)
    for spec in output_specs:
        external_names.add(spec.name)
    for spec in weight_specs:
        external_names.add(spec.name)

    # Collect all buffer names produced by steps, with their shapes
    produced: dict[str, list[int]] = {}
    for step in steps:
        for name in _step_output_names(step):
            shape = _step_output_shape(step, name)
            if shape is not None:
                produced[name] = shape

    # Build allocations for intermediate buffers (produced but not external)
    # Output specs also get allocations (they need pre-allocated space)
    allocations: list[BufferAllocation] = []
    allocated: set[str] = set()

    for name, shape in produced.items():
        if name in allocated:
            continue
        if name in external_names:
            # Output specs need allocation too
            is_output = any(s.name == name for s in output_specs)
            if not is_output:
                continue
        elem_size = _DTYPE_ELEM_SIZE.get(compute_dtype, 2)
        size_bytes = int(np.prod(shape)) * elem_size
        allocations.append(BufferAllocation(
            name=name,
            shape=list(shape),
            dtype=compute_dtype,
            size_bytes=size_bytes,
        ))
        allocated.add(name)

    return allocations


def _step_output_names(step: ExecStep) -> list[str]:
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


def _step_output_shape(step: ExecStep, name: str) -> list[int] | None:
    """Infer the output shape for a given buffer name from a step."""
    if isinstance(step, AliasStep):
        return step.output_shape
    if isinstance(step, CUBLASStep):
        return _blas_output_shape(step)
    if isinstance(step, FusedKernelStep):
        # Shape is total_elements as 1D
        return [step.total_elements]
    if isinstance(step, ReductionKernelStep):
        return _reduction_output_shape(step)
    if isinstance(step, SpecialKernelStep):
        return _special_output_shape(step)
    return None


def _blas_output_shape(step: CUBLASStep) -> list[int]:
    p = step.params
    if step.blas_type == "gemm":
        return [p["M"], p["N"]]
    if step.blas_type == "gemm_batched":
        return [p["batch"], p["M"], p["N"]]
    if step.blas_type == "gemm_gqa":
        return list(p["out_shape"])
    if step.blas_type == "conv2d":
        return [p["batch"], p["out_channels"], p["out_h"], p["out_w"]]
    return [1]


def _reduction_output_shape(step: ReductionKernelStep) -> list[int]:
    p = step.params
    if step.kernel_name in ("softmax_kernel", "masked_softmax_kernel"):
        return [p["rows"], p["cols"]]
    if step.kernel_name == "rmsnorm_kernel":
        return [p["rows"], p["cols"]]
    if step.kernel_name == "silu_mul_kernel":
        return [p.get("total", 1)]
    if step.kernel_name == "mean_last_dim_kernel":
        return [p["rows"]]
    if step.kernel_name in ("max_pool2d_kernel", "adaptive_avg_pool2d_kernel"):
        return [p["batch"], p["channels"], p["out_h"], p["out_w"]]
    if step.kernel_name == "batch_norm_kernel":
        return [p["batch"], p["channels"], p["spatial"]]
    return [1]


def _special_output_shape(step: SpecialKernelStep) -> list[int]:
    p = step.params
    if step.kernel_name == "embedding_kernel":
        return [p["seq_len"], p["embed_dim"]]
    if step.kernel_name == "rope_kernel":
        return [p["seq_len"], p["head_dim"]]
    if step.kernel_name == "full_kernel":
        return [p["total"]]
    if step.kernel_name == "index_copy_kernel":
        total = p["outer_size"] * p["dim_size"] * p["inner_size"]
        return [total]
    if "total" in p:
        return [p["total"]]
    return [1]
