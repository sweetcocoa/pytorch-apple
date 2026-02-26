"""CUDA backend: CuPy-based Backend and DeviceBuffer implementations.

Implements the Backend ABC from npu_runtime.backend using CuPy for
CUDA GPU memory management and numpy interop.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from npu_runtime.backend import Backend, DeviceBuffer

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


class CUDABuffer(DeviceBuffer):
    """CUDA GPU buffer backed by cupy.ndarray."""

    def __init__(self, data: cp.ndarray, logical_shape: tuple[int, ...] | None = None):
        self._data = data
        self._logical_shape = logical_shape or tuple(data.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._logical_shape

    @property
    def alloc_shape(self) -> tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def size_bytes(self) -> int:
        return self._data.nbytes

    @property
    def native_handle(self) -> Any:
        """Return the underlying cupy.ndarray."""
        return self._data

    def to_numpy(self, spec=None, dtype=None) -> np.ndarray:
        """Download to CPU as numpy array, with optional shape/dtype conversion."""
        arr = cp.asnumpy(self._data)
        # If logical shape differs from physical, slice/reshape
        if self._logical_shape != tuple(arr.shape):
            # Flatten and take first N elements
            total = 1
            for s in self._logical_shape:
                total *= s
            arr = arr.ravel()[:total].reshape(self._logical_shape)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_numpy(cls, data: np.ndarray, dtype: np.dtype | None = None) -> CUDABuffer:
        """Upload numpy array to CUDA GPU."""
        if dtype is not None and data.dtype != dtype:
            data = data.astype(dtype)
        return cls(cp.asarray(data))

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.float16)) -> CUDABuffer:
        """Allocate zero-filled CUDA buffer."""
        return cls(cp.zeros(shape, dtype=dtype))

    def write_from_numpy(self, data: np.ndarray, spec=None) -> None:
        """Write numpy data into existing CUDA buffer (in-place update)."""
        if spec is not None and spec.dtype in ("float16", "bfloat16"):
            if data.dtype != np.float16:
                data = data.astype(np.float16)
        src = cp.asarray(data)
        if src.shape == self._data.shape:
            self._data[:] = src
        else:
            flat_src = src.ravel()
            flat_dst = self._data.ravel()
            n = min(len(flat_src), len(flat_dst))
            flat_dst[:n] = flat_src[:n]
        self._logical_shape = tuple(data.shape)


class CUDABackend(Backend):
    """CUDA GPU execution backend using CuPy."""

    def __init__(self, device_id: int = 0):
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not installed. Install with: uv sync --extra cuda")
        self._device_id = device_id
        self._cp_device = cp.cuda.Device(device_id)

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def device(self) -> Any:
        """Return CuPy device object."""
        return self._cp_device

    def allocate_buffer(self, data: np.ndarray, alloc_shape=None, spec=None) -> CUDABuffer:
        """Upload numpy data to CUDA GPU.

        By default converts to float16 (compute dtype for CUDA kernels).
        Integer dtypes are preserved if specified in spec.
        """
        dtype = np.float16
        if spec is not None and hasattr(spec, "dtype"):
            if spec.dtype in ("int32",):
                dtype = np.int32
            elif spec.dtype in ("int64",):
                dtype = np.int64
            # float32/float16/bfloat16 → all become float16 (compute dtype)

        if data.dtype != dtype:
            data = data.astype(dtype)

        return CUDABuffer.from_numpy(data, dtype=np.dtype(dtype))

    def allocate_zeros(self, shape, dtype=np.dtype(np.float16), alloc_shape=None) -> CUDABuffer:
        """Allocate zero-filled CUDA buffer."""
        return CUDABuffer.zeros(tuple(shape), dtype=dtype)

    def execute(self, program, inputs, weights) -> dict[str, CUDABuffer]:
        """Execute a CUDAProgram with given inputs and weights."""
        from cuda_runtime.cuda_executor import CUDAExecutor

        executor = CUDAExecutor(program)
        return executor.run(inputs, weights)

    def create_executor(self, program) -> Any:
        """Create a reusable CUDAExecutor."""
        from cuda_runtime.cuda_executor import CUDAExecutor

        return CUDAExecutor(program)

    def synchronize(self):
        """Synchronize CUDA device."""
        cp.cuda.Device(self._device_id).synchronize()
