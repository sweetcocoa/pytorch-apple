"""CUDA executor: runs CUDAProgram on NVIDIA GPU via CuPy.

Architecture:
    - NVRTC JIT compilation at __init__ time (not per-run)
    - Module-level kernel cache (source hash → RawKernel) avoids re-NVRTC
    - cuBLAS GEMM via cupy.cublas (no extra dependency)
    - Pre-allocated intermediate buffer pool
    - run() hot path: dispatch only (no allocations)
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from cuda_compiler.cuda_program import (
    AliasStep,
    CUBLASStep,
    CUDAProgram,
    FusedKernelStep,
    ReductionKernelStep,
    SpecialKernelStep,
)

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

if TYPE_CHECKING:
    from cuda_runtime.cuda_backend import CUDABuffer

# Module-level NVRTC compilation cache: source_hash → RawKernel
# Survives across CUDAExecutor instances, avoiding redundant NVRTC calls
_KERNEL_CACHE: dict[str, "cp.RawKernel"] = {}


def _get_or_compile_kernel(source_code: str, kernel_name: str) -> "cp.RawKernel":
    """Get a compiled kernel from cache or compile via NVRTC."""
    key = hashlib.md5(source_code.encode()).hexdigest() + ":" + kernel_name
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = cp.RawKernel(source_code, kernel_name)
    _KERNEL_CACHE[key] = kernel
    return kernel


class CUDAExecutor:
    """Executes a CUDAProgram on CUDA GPU.

    Init-time work:
        - NVRTC compile all kernel sources (cached across instances)
        - Allocate intermediate buffer pool
        - Pre-build cuBLAS descriptors

    Run-time work (hot path):
        - Step dispatch loop only
    """

    def __init__(self, program: CUDAProgram):
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not installed")

        self._program = program
        self._compute_dtype = program.compute_dtype

        # Identify view-op output buffers (expand/transpose/slice use CuPy views, no kernel needed)
        _VIEW_OPS = {"expand_kernel", "transpose_kernel", "slice_kernel"}
        self._view_output_names: set[str] = set()
        for step in program.steps:
            if isinstance(step, SpecialKernelStep) and step.kernel_name in _VIEW_OPS:
                self._view_output_names.add(step.output_buffer_name)

        # 1. NVRTC compile all kernel sources (skip view-op kernels)
        self._compiled_kernels: dict[str, cp.RawKernel] = {}
        for ksrc in program.kernel_sources:
            if ksrc.kernel_name in _VIEW_OPS:
                continue  # These use CuPy native views now
            self._compiled_kernels[ksrc.kernel_name] = _get_or_compile_kernel(
                ksrc.source_code, ksrc.kernel_name,
            )

        # 2. Pre-allocate intermediate buffer pool (skip view-op outputs)
        self._buffer_pool: dict[str, cp.ndarray] = {}
        for alloc in program.buffer_allocations:
            if alloc.name in self._view_output_names:
                continue  # Views are created at dispatch time, no pre-allocation
            dtype = self._resolve_dtype(alloc.dtype)
            self._buffer_pool[alloc.name] = cp.zeros(alloc.shape, dtype=dtype)

        # 3. Pre-cache device int32 arrays for tensor op params (avoid per-call cp.array())
        self._device_arrays: dict[str, cp.ndarray] = {}

        # 4. Build dispatch table: list of (dispatch_fn, step) tuples
        # Eliminates isinstance checks in the hot loop
        self._dispatch_table: list[tuple] = []
        for step in program.steps:
            if isinstance(step, CUBLASStep):
                self._dispatch_table.append((self._dispatch_blas, step))
            elif isinstance(step, FusedKernelStep):
                self._dispatch_table.append((self._dispatch_fused, step))
            elif isinstance(step, ReductionKernelStep):
                self._dispatch_table.append((self._dispatch_reduction, step))
            elif isinstance(step, SpecialKernelStep):
                self._dispatch_table.append((self._dispatch_special, step))
            elif isinstance(step, AliasStep):
                self._dispatch_table.append((self._dispatch_alias, step))

    def _get_device_array(self, kernel_name: str, param_name: str, values: list[int]) -> cp.ndarray:
        """Get or create a cached device int32 array for kernel params."""
        key = f"{kernel_name}:{param_name}:{values}"
        cached = self._device_arrays.get(key)
        if cached is not None:
            return cached
        arr = cp.array(values, dtype=cp.int32)
        self._device_arrays[key] = arr
        return arr

    def _resolve_dtype(self, dtype_str: str) -> np.dtype:
        if dtype_str in ("float16", "bfloat16"):
            return np.dtype(np.float16)
        if dtype_str == "float32":
            return np.dtype(np.float32)
        if dtype_str == "int32":
            return np.dtype(np.int32)
        if dtype_str == "int64":
            return np.dtype(np.int64)
        return np.dtype(np.float16)

    def run(
        self,
        inputs: dict[str, CUDABuffer],
        weights: dict[str, CUDABuffer],
    ) -> dict[str, CUDABuffer]:
        """Execute the program.

        Args:
            inputs: Graph input name -> CUDABuffer.
            weights: Weight name -> CUDABuffer.

        Returns:
            Dict mapping output name -> CUDABuffer.
        """
        from cuda_runtime.cuda_backend import CUDABuffer

        # Build buffer pool: name -> cupy.ndarray
        pool: dict[str, cp.ndarray] = {}

        # Add inputs
        for name, buf in inputs.items():
            pool[name] = buf.native_handle

        # Add weights
        for name, buf in weights.items():
            pool[name] = buf.native_handle

        # Add pre-allocated intermediates
        pool.update(self._buffer_pool)

        # Dispatch loop (pre-built table, no isinstance checks)
        for dispatch_fn, step in self._dispatch_table:
            dispatch_fn(step, pool)

        # Collect outputs
        outputs: dict[str, CUDABuffer] = {}
        for spec in self._program.output_specs:
            if spec.name in pool:
                outputs[spec.name] = CUDABuffer(pool[spec.name], logical_shape=tuple(spec.shape))

        return outputs

    def _dispatch_blas(self, step: CUBLASStep, pool: dict[str, cp.ndarray]) -> None:
        """Dispatch cuBLAS operation."""
        p = step.params

        if step.blas_type == "gemm":
            A = pool[step.input_buffer_names[0]]
            B = pool[step.input_buffer_names[1]]

            M, N, K = p["M"], p["N"], p["K"]
            transpose_b = p.get("transpose_b", False)
            has_bias = p.get("has_bias", False)

            # Reshape to 2D only if needed (avoid copy)
            A_2d = A.reshape(M, K) if A.shape != (M, K) else A
            if transpose_b:
                B_2d = B.reshape(N, K) if B.shape != (N, K) else B
                # Use .T which is a zero-copy view for C-contiguous arrays
                if B_2d.flags.c_contiguous:
                    out = cp.matmul(A_2d, B_2d.T)
                else:
                    # Fallback: ascontiguousarray + .T
                    out = cp.matmul(A_2d, cp.ascontiguousarray(B_2d).T)
            else:
                B_2d = B.reshape(K, N) if B.shape != (K, N) else B
                out = cp.matmul(A_2d, B_2d)

            if has_bias and len(step.input_buffer_names) > 2:
                bias = pool[step.input_buffer_names[2]]
                out = out + bias.ravel()[:N]

            pool[step.output_buffer_name] = out

        elif step.blas_type == "gemm_batched":
            A = pool[step.input_buffer_names[0]]
            B = pool[step.input_buffer_names[1]]
            batch, M, N, K = p["batch"], p["M"], p["N"], p["K"]

            A_3d = A.reshape(batch, M, K) if A.shape != (batch, M, K) else A
            B_3d = B.reshape(batch, K, N) if B.shape != (batch, K, N) else B
            out = cp.matmul(A_3d, B_3d)
            pool[step.output_buffer_name] = out

        elif step.blas_type == "gemm_gqa":
            self._dispatch_gemm_gqa(step, pool)

        elif step.blas_type == "conv2d":
            self._dispatch_conv2d(step, pool)

    def _dispatch_gemm_gqa(self, step: CUBLASStep, pool: dict[str, cp.ndarray]) -> None:
        """Dispatch GQA-aware attention matmul.

        Instead of expanding KV from [1,kv,1,S,D] to [1,H,S,D] and doing batch=H BMM,
        we do kv_heads separate GEMMs of [q_per_kv, ...] @ [...] which is much faster
        because each GEMM has higher M (better GPU utilization) and avoids the 100MB expand copy.

        For QK^T (transpose_kv=True):
            Q: [1,H,1,D], KV: [1,kv,1,S,D]
            Per KV head: Q_group=[q_per_kv,D] @ K^T=[D,S] -> [q_per_kv,S]
            Output: [1,H,1,S]

        For Score×V (transpose_kv=False):
            Scores: [1,H,1,S], V: [1,kv,1,S,D]
            Per KV head: scores_group=[q_per_kv,S] @ V=[S,D] -> [q_per_kv,D]
            Output: [1,H,1,D]
        """
        p = step.params
        kv_heads = p["kv_heads"]
        q_per_kv = p["q_per_kv"]
        transpose_kv = p["transpose_kv"]
        out_shape = p["out_shape"]

        Q = pool[step.input_buffer_names[0]]   # Q or Scores: [1,H,1,X]
        KV = pool[step.input_buffer_names[1]]   # unexpanded KV: [1,kv,1,S,D]

        if transpose_kv:
            # QK^T: Q=[1,H,1,D], K=[1,kv,1,S,D]
            D = p["K"]
            S = p["N"]
            # Q: [1,H,1,D] -> [kv, q_per_kv, D]
            q_3d = Q.reshape(kv_heads, q_per_kv, D)
            # K: [1,kv,1,S,D] -> [kv, S, D] -> transpose last two -> [kv, D, S]
            k_3d = KV.reshape(kv_heads, S, D).transpose(0, 2, 1)  # [kv, D, S]
            # Batched GEMM: [kv, q_per_kv, D] @ [kv, D, S] -> [kv, q_per_kv, S]
            out = cp.matmul(q_3d, k_3d)
            pool[step.output_buffer_name] = out.reshape(out_shape)
        else:
            # Score×V: Scores=[1,H,1,S], V=[1,kv,1,S,D]
            S = p["K"]
            D = p["N"]
            # Scores: [1,H,1,S] -> [kv, q_per_kv, S]
            s_3d = Q.reshape(kv_heads, q_per_kv, S)
            # V: [1,kv,1,S,D] -> [kv, S, D]
            v_3d = KV.reshape(kv_heads, S, D)
            # Batched GEMM: [kv, q_per_kv, S] @ [kv, S, D] -> [kv, q_per_kv, D]
            out = cp.matmul(s_3d, v_3d)
            pool[step.output_buffer_name] = out.reshape(out_shape)

    def _dispatch_conv2d(self, step: CUBLASStep, pool: dict[str, cp.ndarray]) -> None:
        """Dispatch conv2d via cupy/custom kernel."""
        p = step.params
        inp = pool[step.input_buffer_names[0]]
        weight = pool[step.input_buffer_names[1]]

        batch = p["batch"]
        in_channels = p["in_channels"]
        in_h, in_w = p["in_h"], p["in_w"]
        out_channels = p["out_channels"]
        out_h, out_w = p["out_h"], p["out_w"]
        kh, kw = p["kernel_h"], p["kernel_w"]
        sh, sw = p["stride_h"], p["stride_w"]
        ph, pw = p["pad_h"], p["pad_w"]
        groups = p.get("groups", 1)
        has_bias = p.get("has_bias", False)

        # Use the CUDA template kernel for conv2d
        kernel = self._compiled_kernels.get("conv2d_kernel")
        if kernel is not None:
            inp_4d = inp.reshape(batch, in_channels, in_h, in_w)
            out_buf = cp.zeros((batch, out_channels, out_h, out_w), dtype=inp.dtype)
            bias_buf = pool[step.input_buffer_names[2]] if has_bias else cp.zeros((1,), dtype=inp.dtype)

            total = batch * out_channels * out_h * out_w
            block = 256
            grid = ((total + block - 1) // block,)

            kernel(
                grid, (block,),
                (inp_4d, weight, bias_buf, out_buf,
                 np.int32(batch), np.int32(in_channels), np.int32(in_h), np.int32(in_w),
                 np.int32(out_channels), np.int32(out_h), np.int32(out_w),
                 np.int32(kh), np.int32(kw), np.int32(sh), np.int32(sw),
                 np.int32(ph), np.int32(pw), np.int32(1 if has_bias else 0), np.int32(groups)),
            )
            pool[step.output_buffer_name] = out_buf
        else:
            # Fallback: im2col + cuBLAS GEMM approach
            self._conv2d_im2col(step, pool)

    def _conv2d_im2col(self, step: CUBLASStep, pool: dict[str, cp.ndarray]) -> None:
        """Conv2d via im2col + matmul (cuBLAS)."""
        p = step.params
        inp = pool[step.input_buffer_names[0]].reshape(
            p["batch"], p["in_channels"], p["in_h"], p["in_w"],
        )
        weight = pool[step.input_buffer_names[1]].reshape(
            p["out_channels"], p["in_channels"] // p.get("groups", 1),
            p["kernel_h"], p["kernel_w"],
        )

        # Pad input
        if p["pad_h"] > 0 or p["pad_w"] > 0:
            inp = cp.pad(inp, ((0, 0), (0, 0), (p["pad_h"], p["pad_h"]), (p["pad_w"], p["pad_w"])))

        batch = p["batch"]
        out_h, out_w = p["out_h"], p["out_w"]
        kh, kw = p["kernel_h"], p["kernel_w"]
        sh, sw = p["stride_h"], p["stride_w"]
        groups = p.get("groups", 1)
        ic_per_group = p["in_channels"] // groups
        oc_per_group = p["out_channels"] // groups

        # im2col
        col = cp.zeros((batch, ic_per_group * kh * kw, out_h * out_w), dtype=inp.dtype)
        for b in range(batch):
            for i in range(out_h):
                for j in range(out_w):
                    patch = inp[b, :ic_per_group, i * sh:i * sh + kh, j * sw:j * sw + kw]
                    col[b, :, i * out_w + j] = patch.ravel()

        # GEMM: weight @ col
        w_2d = weight[:oc_per_group].reshape(oc_per_group, -1)
        out = cp.matmul(w_2d, col)
        out = out.reshape(batch, p["out_channels"], out_h, out_w)

        if p.get("has_bias", False) and len(step.input_buffer_names) > 2:
            bias = pool[step.input_buffer_names[2]]
            out = out + bias.reshape(1, -1, 1, 1)

        pool[step.output_buffer_name] = out

    def _dispatch_fused(self, step: FusedKernelStep, pool: dict[str, cp.ndarray]) -> None:
        """Dispatch fused elementwise kernel."""
        kernel = self._compiled_kernels[step.kernel_name]

        args = []
        for name in step.input_buffer_names:
            buf = pool[name]
            args.append(buf.ravel() if buf.ndim > 1 else buf)
        # Output buffer
        out_buf = pool.get(step.output_buffer_name)
        if out_buf is None:
            out_buf = cp.zeros(step.total_elements, dtype=np.float16)
            pool[step.output_buffer_name] = out_buf
        args.append(out_buf.ravel() if out_buf.ndim > 1 else out_buf)
        args.append(np.int32(step.total_elements))

        block = step.block_size
        grid = ((step.total_elements + block - 1) // block,)
        kernel(grid, (block,), tuple(args))

    def _dispatch_reduction(self, step: ReductionKernelStep, pool: dict[str, cp.ndarray]) -> None:
        """Dispatch reduction kernel."""
        kernel = self._compiled_kernels[step.kernel_name]
        p = step.params

        if step.kernel_name in ("softmax_kernel", "masked_softmax_kernel"):
            inp = pool[step.input_buffer_names[0]]
            rows, cols = p["rows"], p["cols"]
            inp_flat = inp.ravel()[:rows * cols]

            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros(rows * cols, dtype=inp.dtype)
                pool[step.output_buffer_name] = out

            block = min(256, max(32, cols))  # At least a warp
            grid = (rows,)  # 1 block per row

            if step.kernel_name == "masked_softmax_kernel":
                mask = pool[step.input_buffer_names[1]]
                mask_flat = mask.ravel()[:rows * cols]
                kernel(grid, (block,), (inp_flat, mask_flat, out.ravel(), np.int32(rows), np.int32(cols)))
            else:
                kernel(grid, (block,), (inp_flat, out.ravel(), np.int32(rows), np.int32(cols)))

        elif step.kernel_name == "mean_last_dim_kernel":
            inp = pool[step.input_buffer_names[0]]
            rows, cols = p["rows"], p["cols"]
            inp_flat = inp.ravel()[:rows * cols].reshape(rows, cols)

            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros((rows,), dtype=inp.dtype)
                pool[step.output_buffer_name] = out

            block = min(256, rows)
            grid = ((rows + block - 1) // block,)
            kernel(grid, (block,), (inp_flat, out.ravel(), np.int32(rows), np.int32(cols)))

        elif step.kernel_name == "rmsnorm_kernel":
            inp = pool[step.input_buffer_names[0]]
            weight = pool[step.input_buffer_names[1]]
            rows, cols = p["rows"], p["cols"]
            eps = p.get("eps", 1e-6)

            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros(rows * cols, dtype=inp.dtype)
                pool[step.output_buffer_name] = out

            block = min(256, max(32, cols))
            grid = (rows,)
            kernel(grid, (block,), (
                inp.ravel(), weight.ravel(), out.ravel(),
                np.int32(rows), np.int32(cols), np.float32(eps),
            ))

        elif step.kernel_name == "silu_mul_kernel":
            gate = pool[step.input_buffer_names[0]]
            up = pool[step.input_buffer_names[1]]
            total = int(np.prod(gate.shape))

            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros(total, dtype=gate.dtype)
                pool[step.output_buffer_name] = out

            block = 256
            grid = ((total + block - 1) // block,)
            kernel(grid, (block,), (gate.ravel(), up.ravel(), out.ravel(), np.int32(total)))

        elif step.kernel_name == "max_pool2d_kernel":
            inp = pool[step.input_buffer_names[0]]
            batch, channels = p["batch"], p["channels"]
            total = batch * channels * p["out_h"] * p["out_w"]
            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros((batch, channels, p["out_h"], p["out_w"]), dtype=inp.dtype)
                pool[step.output_buffer_name] = out

            block = 256
            grid = ((total + block - 1) // block,)
            kernel(grid, (block,), (
                inp, out,
                np.int32(batch), np.int32(channels),
                np.int32(p["in_h"]), np.int32(p["in_w"]),
                np.int32(p["out_h"]), np.int32(p["out_w"]),
                np.int32(p["kernel_h"]), np.int32(p["kernel_w"]),
                np.int32(p["stride_h"]), np.int32(p["stride_w"]),
                np.int32(p["pad_h"]), np.int32(p["pad_w"]),
            ))

        elif step.kernel_name == "adaptive_avg_pool2d_kernel":
            inp = pool[step.input_buffer_names[0]]
            batch, channels = p["batch"], p["channels"]
            total = batch * channels * p["out_h"] * p["out_w"]
            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros((batch, channels, p["out_h"], p["out_w"]), dtype=inp.dtype)
                pool[step.output_buffer_name] = out

            block = 256
            grid = ((total + block - 1) // block,)
            kernel(grid, (block,), (
                inp, out,
                np.int32(batch), np.int32(channels),
                np.int32(p["in_h"]), np.int32(p["in_w"]),
                np.int32(p["out_h"]), np.int32(p["out_w"]),
            ))

        elif step.kernel_name == "batch_norm_kernel":
            inp = pool[step.input_buffer_names[0]]
            gamma = pool[step.input_buffer_names[1]]
            beta = pool[step.input_buffer_names[2]]
            running_mean = pool[step.input_buffer_names[3]]
            running_var = pool[step.input_buffer_names[4]]

            batch, channels, spatial = p["batch"], p["channels"], p["spatial"]
            total = batch * channels * spatial
            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros(total, dtype=inp.dtype)
                pool[step.output_buffer_name] = out

            block = 256
            grid = ((total + block - 1) // block,)
            kernel(grid, (block,), (
                inp.ravel(), gamma.ravel(), beta.ravel(),
                running_mean.ravel(), running_var.ravel(), out.ravel(),
                np.int32(batch), np.int32(channels), np.int32(spatial),
                np.float32(p["eps"]),
            ))

    def _dispatch_special(self, step: SpecialKernelStep, pool: dict[str, cp.ndarray]) -> None:
        """Dispatch special kernel (embedding, rope, etc.)."""
        p = step.params

        # View ops use CuPy native views (no compiled kernel needed)
        if step.kernel_name in ("transpose_kernel", "cat_2_kernel", "slice_kernel", "expand_kernel"):
            kernel = self._compiled_kernels.get(step.kernel_name)
            self._dispatch_tensor_op(step, kernel, pool)
            return

        kernel = self._compiled_kernels[step.kernel_name]

        if step.kernel_name == "embedding_kernel":
            indices = pool[step.input_buffer_names[0]]
            weight = pool[step.input_buffer_names[1]]
            seq_len, embed_dim = p["seq_len"], p["embed_dim"]
            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros((seq_len, embed_dim), dtype=weight.dtype)
                pool[step.output_buffer_name] = out

            block = min(256, embed_dim)
            grid = (max(1, (embed_dim + block - 1) // block), seq_len)
            kernel(grid, (block,), (
                indices.ravel().astype(cp.int32), weight.ravel(), out.ravel(),
                np.int32(seq_len), np.int32(embed_dim),
            ))

        elif step.kernel_name == "rope_kernel":
            inv_freq = pool[step.input_buffer_names[0]]
            positions = pool[step.input_buffer_names[1]]
            seq_len, head_dim = p["seq_len"], p["head_dim"]

            cos_out = pool.get(step.output_buffer_name)
            if cos_out is None:
                cos_out = cp.zeros((seq_len, head_dim), dtype=inv_freq.dtype)
                pool[step.output_buffer_name] = cos_out

            sin_out_name = step.output_buffer_names[1] if len(step.output_buffer_names) > 1 else None
            sin_out = pool.get(sin_out_name) if sin_out_name else None
            if sin_out is None:
                sin_out = cp.zeros((seq_len, head_dim), dtype=inv_freq.dtype)
                if sin_out_name:
                    pool[sin_out_name] = sin_out

            block = min(256, head_dim)
            grid = (max(1, (head_dim + block - 1) // block), seq_len)
            kernel(grid, (block,), (
                inv_freq.ravel(), positions.ravel(),
                cos_out.ravel(), sin_out.ravel(),
                np.int32(seq_len), np.int32(head_dim),
            ))

        elif step.kernel_name == "index_copy_kernel":
            src = pool[step.input_buffer_names[0]]
            values = pool[step.input_buffer_names[1]]
            indices = pool[step.input_buffer_names[2]]
            total = p["outer_size"] * p["dim_size"] * p["inner_size"]
            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros(total, dtype=src.dtype)
                pool[step.output_buffer_name] = out

            block = 256
            grid = ((total + block - 1) // block,)
            kernel(grid, (block,), (
                src.ravel(), out.ravel(), indices.ravel().astype(cp.int32), values.ravel(),
                np.int32(p["outer_size"]), np.int32(p["dim_size"]),
                np.int32(p["inner_size"]), np.int32(p["num_indices"]),
            ))

        elif step.kernel_name == "full_kernel":
            total = p["total"]
            fill_value = p["fill_value"]
            out = pool.get(step.output_buffer_name)
            if out is None:
                out = cp.zeros(total, dtype=np.float16)
                pool[step.output_buffer_name] = out

            block = 256
            grid = ((total + block - 1) // block,)
            kernel(grid, (block,), (
                out.ravel(), np.float16(fill_value), np.int32(total),
            ))

    def _dispatch_tensor_op(self, step: SpecialKernelStep, kernel, pool: dict[str, cp.ndarray]) -> None:
        """Dispatch tensor manipulation using CuPy zero-copy views where possible.

        expand, transpose, and slice use CuPy native view operations (no data copy).
        cat still uses a CUDA kernel since concatenation requires a real copy.
        """
        p = step.params

        if step.kernel_name == "transpose_kernel":
            inp = pool[step.input_buffer_names[0]]
            in_shape = p["in_shape"]
            dim0, dim1 = p["dim0"], p["dim1"]
            # Reshape to N-D, transpose dims, store view (zero-copy)
            nd = inp.reshape(in_shape)
            perm = list(range(len(in_shape)))
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
            pool[step.output_buffer_name] = nd.transpose(perm)

        elif step.kernel_name == "cat_2_kernel":
            # cat requires a real copy — use CuPy native concatenate
            in0 = pool[step.input_buffer_names[0]]
            in1 = pool[step.input_buffer_names[1]]
            out_shape = p["out_shape"]
            axis = p["axis"]
            # Reshape inputs to match output ndim for correct axis concat
            in0_shape = list(out_shape)
            in0_shape[axis] = p["in0_axis_size"]
            in1_shape = list(out_shape)
            in1_shape[axis] = out_shape[axis] - p["in0_axis_size"]
            in0_nd = in0.reshape(in0_shape)
            in1_nd = in1.reshape(in1_shape)
            pool[step.output_buffer_name] = cp.concatenate([in0_nd, in1_nd], axis=axis)

        elif step.kernel_name == "slice_kernel":
            inp = pool[step.input_buffer_names[0]]
            in_shape = p["in_shape"]
            dim = p["dim"]
            start = p["start"]
            step_val = p["step"]
            out_shape = p["out_shape"]
            # Reshape to N-D, slice, store view (zero-copy for step=1)
            nd = inp.reshape(in_shape)
            slices = [slice(None)] * len(in_shape)
            end = start + out_shape[dim] * step_val
            slices[dim] = slice(start, end, step_val)
            pool[step.output_buffer_name] = nd[tuple(slices)]

        elif step.kernel_name == "expand_kernel":
            inp = pool[step.input_buffer_names[0]]
            in_shape = p["in_shape"]
            out_shape = p["out_shape"]
            # Reshape to padded input shape, broadcast (zero-copy view)
            nd = inp.reshape(in_shape)
            pool[step.output_buffer_name] = cp.broadcast_to(nd, out_shape)

    def _dispatch_alias(self, step: AliasStep, pool: dict[str, cp.ndarray]) -> None:
        """Dispatch zero-cost alias (reshape/view)."""
        inp = pool.get(step.input_buffer_name)
        if inp is not None:
            if list(inp.shape) != step.output_shape:
                pool[step.output_buffer_name] = inp.reshape(step.output_shape)
            else:
                pool[step.output_buffer_name] = inp
