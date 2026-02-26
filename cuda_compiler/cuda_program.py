"""CUDA program data model: execution steps + buffer allocations.

A CUDAProgram is the CUDA analogue of npu_compiler's CompiledProgram.
It contains a list of execution steps (cuBLAS calls, fused kernels, etc.)
and buffer allocation metadata for the executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from npu_compiler.ir_reader import TensorSpec


@dataclass
class BufferAllocation:
    """Memory allocation for an intermediate or output buffer."""

    name: str
    shape: list[int]
    dtype: str = "float16"
    size_bytes: int = 0


@dataclass
class CUDAKernelSource:
    """CUDA C source code for NVRTC JIT compilation."""

    kernel_name: str
    source_code: str


@dataclass
class CUBLASStep:
    """A cuBLAS library call (GEMM, batched GEMM, conv2d via cuDNN-like)."""

    blas_type: str  # "gemm", "gemm_batched", "conv2d"
    input_buffer_names: list[str]
    output_buffer_name: str
    params: dict  # M, N, K, transpose flags, batch, stride, padding, etc.


@dataclass
class FusedKernelStep:
    """A fused elementwise kernel (NVRTC JIT compiled)."""

    kernel_name: str
    source_code: str
    input_buffer_names: list[str]
    output_buffer_name: str
    total_elements: int
    block_size: int = 256


@dataclass
class ReductionKernelStep:
    """A reduction kernel (softmax, mean, pool, batchnorm)."""

    kernel_name: str
    source_code: str
    input_buffer_names: list[str]
    output_buffer_name: str
    params: dict  # rows, cols, etc.
    block_size: int = 256


@dataclass
class AliasStep:
    """Zero-cost alias (reshape, view, contiguous, etc.)."""

    input_buffer_name: str
    output_buffer_name: str
    input_shape: list[int]
    output_shape: list[int]


@dataclass
class SpecialKernelStep:
    """A hand-written special kernel (embedding, rope, index_copy, etc.)."""

    kernel_name: str
    source_code: str
    input_buffer_names: list[str]
    output_buffer_name: str
    output_buffer_names: list[str] = field(default_factory=list)  # for multi-output (rope)
    params: dict = field(default_factory=dict)
    block_size: int = 256
    grid_dim: tuple[int, ...] = (1,)


# Union type for all step types
ExecStep = CUBLASStep | FusedKernelStep | ReductionKernelStep | AliasStep | SpecialKernelStep


@dataclass
class CUDAProgram:
    """Complete CUDA execution program (analogous to CompiledProgram)."""

    steps: list[ExecStep]
    buffer_allocations: list[BufferAllocation]
    input_specs: list[TensorSpec]
    output_specs: list[TensorSpec]
    weight_specs: list[TensorSpec]
    weight_name_mapping: dict[str, str]
    kernel_sources: list[CUDAKernelSource] = field(default_factory=list)
    compute_dtype: str = "float16"

    @property
    def kernel_calls(self):
        """Compatibility shim — returns steps list."""
        return self.steps
