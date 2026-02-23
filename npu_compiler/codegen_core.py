"""Core codegen data structures and constants.

Shared by codegen_ops.py and codegen_fused.py to avoid circular imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from npu_compiler.ir_reader import TensorSpec

# Metal doesn't support variable-length struct fields, so param buffers use
# fixed 6-element arrays for shape/stride. Covers all cases from scalar (ndim=0)
# to 6D batched attention tensors. Tensors with ndim < 6 are zero-padded.
_MAX_NDIM = 6

# Bytes per element for supported compute dtypes (FP16/BF16 = 2, FP32/INT32 = 4)
_DTYPE_ELEM_SIZE = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "int64": 8}

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
