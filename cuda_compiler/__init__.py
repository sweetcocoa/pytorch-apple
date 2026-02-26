"""CUDA subgraph compiler: compiles IR subgraphs into CUDAProgram.

Entry point: compile_subgraph(ir_dict) -> CUDAProgram

Unlike the NPU compiler (op-level codegen), the CUDA compiler operates at
subgraph level: it fuses chains of elementwise ops into single CUDA kernels
and delegates heavy compute (matmul, conv) to cuBLAS.
"""

from __future__ import annotations

from cuda_compiler.buffer_planner import plan_buffers
from cuda_compiler.cuda_program import CUDAKernelSource, CUDAProgram
from cuda_compiler.op_support import get_cuda_supported_ops as get_cuda_supported_ops
from cuda_compiler.op_support import is_cuda_op_supported as is_cuda_op_supported
from cuda_compiler.subgraph_analyzer import analyze_subgraph
from npu_compiler.ir_reader import IRGraph, load_ir_from_dict


def compile_subgraph(ir_dict: dict, compute_dtype: str = "float16") -> CUDAProgram:
    """Compile an IR dict (subgraph) into a CUDAProgram.

    Args:
        ir_dict: IR dict (from JSON or _build_sub_ir_dict).
        compute_dtype: "float16" or "bfloat16".

    Returns:
        CUDAProgram ready for CUDAExecutor.
    """
    graph = load_ir_from_dict(ir_dict)
    return compile_graph(graph, compute_dtype=compute_dtype)


def compile_graph(graph: IRGraph, compute_dtype: str = "float16") -> CUDAProgram:
    """Compile an IRGraph into a CUDAProgram.

    Pipeline:
    1. Analyze subgraph → list of ExecStep (with fusion)
    2. Collect kernel sources for NVRTC batch compilation
    3. Plan intermediate buffer allocations
    4. Package into CUDAProgram
    """
    # 1. Analyze and fuse
    steps = analyze_subgraph(graph)

    # 2. Collect input/output/weight specs
    input_specs = list(graph.graph_inputs)
    output_specs = list(graph.graph_outputs)
    weight_specs = list(graph.weights)
    weight_name_mapping = dict(graph.weight_name_mapping)

    # 3. Collect unique kernel sources for batch NVRTC compilation
    kernel_sources: list[CUDAKernelSource] = []
    seen_kernels: set[str] = set()
    from cuda_compiler.cuda_program import FusedKernelStep, ReductionKernelStep, SpecialKernelStep

    for step in steps:
        if isinstance(step, (FusedKernelStep, ReductionKernelStep, SpecialKernelStep)):
            if step.kernel_name not in seen_kernels:
                kernel_sources.append(CUDAKernelSource(
                    kernel_name=step.kernel_name,
                    source_code=step.source_code,
                ))
                seen_kernels.add(step.kernel_name)

    # 4. Plan buffer allocations
    buffer_allocations = plan_buffers(
        steps, input_specs, output_specs, weight_specs, compute_dtype,
    )

    return CUDAProgram(
        steps=steps,
        buffer_allocations=buffer_allocations,
        input_specs=input_specs,
        output_specs=output_specs,
        weight_specs=weight_specs,
        weight_name_mapping=weight_name_mapping,
        kernel_sources=kernel_sources,
        compute_dtype=compute_dtype,
    )
