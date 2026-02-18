"""Executor: runs CompiledProgram on Metal GPU with batched command buffers.

Architecture:
    DispatchStrategy (ABC) defines the interface for computing grid/threadgroup
    sizes per kernel type. MetalDispatchStrategy is the default implementation
    using TargetConfig hardware parameters. This enables future backends to
    plug in their own dispatch logic without modifying the executor core.

Design trade-offs:
    - MPS for float16 only: MPSMatrixMultiplication asserts on bfloat16 input
      (Apple Metal does not support bf16 in MPS). BFloat16 models fall back
      to our custom Metal tiled/vec matmul kernels, which are ~1.5× slower
      than MPS but correct. Guarded by `self._use_mps = compute_dtype == "float16"`.

    - Single command buffer: All kernels are batched into one Metal command buffer
      to minimize GPU submission overhead (~50μs per commit). MPS matmul requires
      its own encoder (incompatible with compute encoder), so we end/reopen
      encoders when transitioning between compute and MPS dispatches.

    - Pre-packed params: Kernel parameters (shapes, strides) are constant for a
      compiled program, so struct.pack into Metal buffers happens once at init,
      not per-run. For Qwen2.5 (~1700 kernels), this saves ~85ms per forward pass.

    - Removed features: Transpose folding into matmul was removed due to incorrect
      N/K swap behavior in both MPS and custom Metal kernels.
"""

from __future__ import annotations

import os
import struct
from abc import ABC, abstractmethod

import ml_dtypes  # noqa: F401
import numpy as np

from npu_compiler.codegen import KernelCall
from npu_compiler.compiled_program import CompiledProgram
from npu_compiler.target_config import METAL_GPU, TargetConfig
from npu_runtime._mps_accel import (
    MPS_DTYPE_MAP,
    MPS_MATMUL_KERNELS,
    create_mps_matmul,
    encode_mps_matmul,
)
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device


# ---------------------------------------------------------------------------
# DispatchStrategy: abstract interface for grid/threadgroup computation
# ---------------------------------------------------------------------------

class DispatchStrategy(ABC):
    """Abstract interface for computing dispatch parameters per kernel.

    Each backend provides its own strategy. Metal uses fixed threadgroup sizes
    from TargetConfig; a CUDA backend would compute grid/block dims differently.
    """

    @abstractmethod
    def compute_dispatch(
        self, call: KernelCall, pipeline: object | None,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        """Return ((groups_x, groups_y, groups_z), (tpg_x, tpg_y, tpg_z)) or None."""


class MetalDispatchStrategy(DispatchStrategy):
    """Metal GPU dispatch strategy using TargetConfig hardware parameters."""

    _TILED_MATMUL_KERNELS = {"matmul_kernel", "matmul_notrans_kernel"}
    _VEC_MATMUL_KERNELS = {"matmul_vec_kernel", "matmul_notrans_vec_kernel"}

    def __init__(self, config: TargetConfig):
        self._config = config

    def compute_dispatch(self, call: KernelCall, pipeline: object | None):
        if call.kernel_name == "_reshape" or call.dispatch_type == "none":
            return None
        if pipeline is None:
            return None

        # Tiled matmul: fixed TILE x TILE threadgroups.
        # All threads must participate in shared memory loading + barriers.
        if call.kernel_name in self._TILED_MATMUL_KERNELS:
            T = self._config.matmul_tile
            M, N = call.params["M"], call.params["N"]
            return (((N + T - 1) // T, (M + T - 1) // T, 1), (T, T, 1))

        # Fused decode attention: BH threadgroups × D threads per group
        if call.kernel_name == "fused_decode_attention_kernel":
            bh = call.params["batch_heads"]
            D = call.params["head_dim"]
            return ((bh, 1, 1), (D, 1, 1))

        # Vec matmul (M=1): N threadgroups of max_threadgroup_1d threads (parallel K-reduction)
        if call.kernel_name in self._VEC_MATMUL_KERNELS:
            N = call.params["N"]
            return ((N, 1, 1), (self._config.max_threadgroup_1d, 1, 1))

        if call.dispatch_type == "1d":
            tpg = min(self._config.max_threadgroup_1d, pipeline.maxTotalThreadsPerThreadgroup())
            groups = (call.total_threads + tpg - 1) // tpg
            return ((groups, 1, 1), (tpg, 1, 1))
        elif call.dispatch_type == "2d":
            tpg_x = min(self._config.max_threadgroup_2d, call.grid_width)
            tpg_y = min(self._config.max_threadgroup_2d, call.grid_height)
            groups_x = (call.grid_width + tpg_x - 1) // tpg_x
            groups_y = (call.grid_height + tpg_y - 1) // tpg_y
            return ((groups_x, groups_y, 1), (tpg_x, tpg_y, 1))
        elif call.dispatch_type == "3d":
            tpg_x = min(self._config.max_threadgroup_3d_xy, call.grid_width)
            tpg_y = min(self._config.max_threadgroup_3d_xy, call.grid_height)
            tpg_z = min(self._config.max_threadgroup_3d_z, call.grid_depth)
            groups_x = (call.grid_width + tpg_x - 1) // tpg_x
            groups_y = (call.grid_height + tpg_y - 1) // tpg_y
            groups_z = (call.grid_depth + tpg_z - 1) // tpg_z
            return ((groups_x, groups_y, groups_z), (tpg_x, tpg_y, tpg_z))
        return None

# Path to metal_kernels directory
_KERNELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metal_kernels")


class Executor:
    """Executes a CompiledProgram on Metal GPU."""

    def __init__(self, program: CompiledProgram, device: Device,
                 config: TargetConfig | None = None,
                 dispatch_strategy: DispatchStrategy | None = None):
        self._program = program
        self._device = device
        self._config = config or METAL_GPU
        self._dispatch_strategy = dispatch_strategy or MetalDispatchStrategy(self._config)
        self._compute_dtype = program.compute_dtype
        self._macros = {"USE_BFLOAT": 1} if self._compute_dtype == "bfloat16" else None
        self._libraries: dict[tuple, object] = {}
        self._pipelines: dict[str, object] = {}
        # MPS matmul only supports float16; MPSMatrixMultiplication asserts on bf16 input.
        # BFloat16 models fall back to custom Metal tiled/vec matmul kernels.
        self._use_mps = self._compute_dtype == "float16"
        self._mps_dtype = MPS_DTYPE_MAP.get(self._compute_dtype, MPS_DTYPE_MAP["float16"])

        # Pre-compile all Metal shaders at init time rather than lazily during run().
        # This avoids jitter on the first forward pass (shader compilation is expensive:
        # ~10ms per file) and ensures any compilation errors surface immediately.
        for call in program.kernel_calls:
            if call.kernel_source and call.kernel_name != "_reshape":
                if self._use_mps and call.kernel_name in MPS_MATMUL_KERNELS:
                    continue
                self._ensure_pipeline(call)

        # Pre-pack parameter buffers once at init. Kernel params (shapes, strides, etc.)
        # are constant for a compiled program, so packing into Metal buffers once avoids
        # struct.pack overhead on every run() call (~1700 kernels × ~50μs = ~85ms saved).
        self._param_buffers = [self._pack_params(call) for call in program.kernel_calls]

        # Pre-compute threadgroup/grid sizes. These depend on tensor shapes which are
        # static, so caching avoids repeated division/min in the dispatch hot loop.
        self._dispatch_params = [self._precompute_dispatch(call) for call in program.kernel_calls]

        # Pre-create MPS matmul objects (only for float16 models)
        if self._use_mps:
            self._mps_matmuls = [
                create_mps_matmul(self._device, call, self._compute_dtype)
                for call in program.kernel_calls
            ]
        else:
            self._mps_matmuls = [None] * len(program.kernel_calls)

        # Pre-allocate intermediate buffer pool (reused across run() calls)
        self._buffer_pool = self._allocate_buffer_pool()

        # Conv and matmul Metal kernels always declare a bias buffer slot in their
        # function signature (simplifies shader code), even when has_bias=0. Rather
        # than branching in the shader or maintaining two kernel variants, we bind
        # a 1-element dummy buffer — the kernel reads it but multiplies by has_bias=0.
        self._dummy_bias = NPUBuffer.zeros((1,), device)

    def _ensure_pipeline(self, call: KernelCall):
        if call.kernel_name in self._pipelines:
            return

        metal_path = os.path.join(_KERNELS_DIR, call.kernel_source)
        cache_key = (metal_path, frozenset(self._macros.items()) if self._macros else frozenset())
        if cache_key not in self._libraries:
            self._libraries[cache_key] = self._device.compile_metal_file(metal_path, macros=self._macros)

        lib = self._libraries[cache_key]
        self._pipelines[call.kernel_name] = self._device.get_pipeline(lib, call.kernel_name)

    def _allocate_buffer_pool(self) -> dict[str, NPUBuffer]:
        """Pre-allocate all intermediate and output buffers once."""
        pool: dict[str, NPUBuffer] = {}
        for alloc in self._program.buffer_allocations:
            dtype = np.dtype(np.float16)
            if alloc.dtype == "bfloat16":
                dtype = np.dtype(ml_dtypes.bfloat16)
            elif alloc.dtype == "int32":
                dtype = np.dtype(np.int32)
            elif alloc.dtype == "float32":
                dtype = np.dtype(np.float32)
            pool[alloc.name] = NPUBuffer.zeros(
                tuple(alloc.shape), self._device, dtype=dtype,
                alloc_shape=tuple(alloc.alloc_shape) if alloc.alloc_shape else None,
            )
        for spec in self._program.output_specs:
            if spec.name not in pool:
                pool[spec.name] = NPUBuffer.zeros(
                    tuple(spec.shape), self._device,
                    alloc_shape=tuple(spec.alloc_shape) if spec.alloc_shape else None,
                )
        return pool

    def _precompute_dispatch(self, call: KernelCall):
        """Pre-compute threadgroup/grid sizes via DispatchStrategy."""
        if self._use_mps and call.kernel_name in MPS_MATMUL_KERNELS:
            return None
        pipeline = self._pipelines.get(call.kernel_name)
        return self._dispatch_strategy.compute_dispatch(call, pipeline)

    def run(
        self,
        inputs: dict[str, NPUBuffer],
        weights: dict[str, NPUBuffer],
    ) -> dict[str, NPUBuffer]:
        """Execute the program with given inputs and weights.

        Uses MPS for matmul operations and custom Metal compute kernels for
        everything else. Dispatches are batched into command buffers.

        Args:
            inputs: Maps input name -> NPUBuffer (with alloc_shape from compiler spec).
            weights: Maps placeholder name -> NPUBuffer (dense, no padding).

        Returns:
            Dict mapping output name -> NPUBuffer.
        """
        pool: dict[str, NPUBuffer] = {}
        pool.update(inputs)
        pool.update(weights)
        pool.update(self._buffer_pool)

        # Batch all kernels into a single Metal command buffer to minimize
        # GPU submission overhead. MPS matmul requires its own encoder (cannot
        # share with compute encoder), so we end/reopen encoders as needed.
        cmd_buf = self._device.new_command_buffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder_open = True
        batch_count = 0

        for i, call in enumerate(self._program.kernel_calls):
            if call.kernel_name == "_reshape":
                in_name = call.input_buffers[0]
                out_name = call.output_buffers[0]
                if in_name in pool:
                    out_shape = tuple(call.params["output_shape"])
                    pool[out_name] = NPUBuffer(
                        pool[in_name].native_handle, out_shape,
                        pool[in_name].dtype, self._device,
                        alloc_shape=pool[in_name].alloc_shape,
                    )
                continue

            if call.dispatch_type == "none":
                continue

            # MPS matmul: end compute encoder, encode MPS, reopen encoder
            mps_info = self._mps_matmuls[i]
            if mps_info is not None:
                if encoder_open:
                    encoder.endEncoding()
                    encoder_open = False
                encode_mps_matmul(cmd_buf, call, mps_info, pool, self._mps_dtype)
                batch_count += 1
            else:
                if not encoder_open:
                    encoder = cmd_buf.computeCommandEncoder()
                    encoder_open = True
                self._encode_kernel(encoder, call, pool, i)
                batch_count += 1

            # Metal command buffers have a finite capacity for encoded commands.
            # Exceeding it causes silent failures. max_dispatches_per_batch (10000)
            # is set conservatively below Metal's limit to trigger explicit flush.
            if batch_count >= self._config.max_dispatches_per_batch:
                if encoder_open:
                    encoder.endEncoding()
                    encoder_open = False
                cmd_buf.commit()
                cmd_buf.waitUntilCompleted()
                cmd_buf = self._device.new_command_buffer()
                batch_count = 0

        if encoder_open:
            encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Collect outputs
        outputs = {}
        for spec in self._program.output_specs:
            if spec.name in pool:
                outputs[spec.name] = pool[spec.name]

        return outputs

    def _encode_kernel(self, encoder, call: KernelCall, pool: dict[str, NPUBuffer],
                       call_idx: int):
        """Encode a single kernel dispatch into the current encoder."""
        pipeline = self._pipelines[call.kernel_name]
        encoder.setComputePipelineState_(pipeline)

        buf_idx = 0

        # Input buffers
        for name in call.input_buffers:
            if name not in pool:
                raise RuntimeError(f"Buffer '{name}' not found in pool. "
                                   f"Available: {sorted(pool.keys())}")
            encoder.setBuffer_offset_atIndex_(pool[name].native_handle, 0, buf_idx)
            buf_idx += 1

        # For kernels that need bias but input doesn't have it, provide dummy
        # (kernel signature always has a bias buffer slot even when has_bias=0)
        if call.kernel_name in ("conv2d_kernel", "matmul_kernel", "matmul_vec_kernel") and len(call.input_buffers) < 3:
            encoder.setBuffer_offset_atIndex_(self._dummy_bias.native_handle, 0, buf_idx)
            buf_idx += 1

        # Output buffers
        for name in call.output_buffers:
            encoder.setBuffer_offset_atIndex_(pool[name].native_handle, 0, buf_idx)
            buf_idx += 1

        # Parameter buffer (pre-packed at init time)
        params_buf = self._param_buffers[call_idx]
        if params_buf is not None:
            encoder.setBuffer_offset_atIndex_(params_buf, 0, buf_idx)

        # Dispatch (pre-computed grid/threadgroup sizes)
        dispatch = self._dispatch_params[call_idx]
        if dispatch is not None:
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(dispatch[0], dispatch[1])

    # Param packing specs: kernel_name(s) -> (struct_format, param_keys)
    # List-valued params are auto-unpacked.
    _PARAM_SPECS: dict[tuple[str, ...], tuple[str, list[str]]] = {
        ("conv2d_kernel",): ("19I", [
            "batch", "in_channels", "in_h", "in_w", "out_channels", "out_h", "out_w",
            "kernel_h", "kernel_w", "stride_h", "stride_w", "pad_h", "pad_w",
            "has_bias", "has_bn", "has_relu", "groups", "in_channels_aligned", "out_channels_aligned",
        ]),
        ("matmul_kernel", "matmul_notrans_kernel", "matmul_vec_kernel",
         "matmul_notrans_vec_kernel"): ("4I", ["M", "N", "K", "has_bias"]),
        ("batched_matmul_kernel",): ("4I", ["batch", "M", "N", "K"]),
        ("max_pool2d_kernel", "adaptive_avg_pool2d_kernel"): ("13I", [
            "batch", "channels", "in_h", "in_w", "out_h", "out_w",
            "kernel_h", "kernel_w", "stride_h", "stride_w", "pad_h", "pad_w", "channels_aligned",
        ]),
        ("depad_4d_kernel",): ("5I", ["batch", "channels", "channels_aligned", "height", "width"]),
        ("embedding_kernel",): ("3I", ["seq_len", "embed_dim", "vocab_size"]),
        ("pow_scalar_kernel",): ("f", ["exponent"]),
        ("add_scalar_kernel", "eltwise_add_scalar_kernel", "eltwise_mul_scalar_kernel"): ("fI", ["scalar", "total"]),
        ("softmax_kernel", "mean_last_dim_kernel"): ("2I", ["rows", "cols"]),
        ("transpose_kernel",): ("4I6I6I6I", ["ndim", "dim0", "dim1", "total", "shape", "strides_in", "strides_out"]),
        ("cat_2_kernel",): ("4I6I6I", ["axis", "ndim", "total", "in1_axis_size", "out_shape", "strides"]),
        ("slice_kernel",): ("6I6I6I", ["dim", "start", "end", "step", "ndim", "total", "in_shape", "in_strides"]),
        ("expand_kernel",): ("2I6I6I6I", ["ndim", "total", "in_shape", "out_shape", "in_strides"]),
        ("mul_broadcast_kernel", "add_broadcast_kernel", "div_broadcast_kernel"): (
            "2I6I6I6I", ["ndim", "total", "a_strides", "b_strides", "out_shape"]),
        ("rmsnorm_kernel",): ("2If", ["rows", "cols", "eps"]),
        ("masked_softmax_kernel",): ("2I", ["rows", "cols"]),
        ("masked_softmax_broadcast_kernel",): ("2II6I6I", ["rows", "cols", "ndim", "mask_strides", "out_shape"]),
        ("rope_kernel",): ("2I", ["seq_len", "head_dim"]),
        ("index_copy_kernel",): ("4I", ["outer_size", "dim_size", "inner_size", "num_indices"]),
        ("fused_decode_attention_kernel",): ("3If", ["batch_heads", "head_dim", "max_seq_len", "scale"]),
    }

    # Build flat lookup: kernel_name -> (format, keys)
    _PARAM_LOOKUP: dict[str, tuple[str, list[str]]] = {}
    for _names, _spec in _PARAM_SPECS.items():
        for _name in _names:
            _PARAM_LOOKUP[_name] = _spec

    def _pack_params(self, call: KernelCall):
        """Pack kernel parameters into a Metal buffer."""
        spec = self._PARAM_LOOKUP.get(call.kernel_name)
        if spec is None:
            return None

        fmt, keys = spec
        values = []
        for k in keys:
            v = call.params[k]
            if isinstance(v, (list, tuple)):
                values.extend(v)
            else:
                values.append(v)

        data = struct.pack(fmt, *values)
        return self._device.mtl_device.newBufferWithBytes_length_options_(data, len(data), 0)
