"""NPU buffer: Metal buffer wrapper with numpy interop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ml_dtypes  # noqa: F401 — registers bfloat16 with numpy
import numpy as np

from npu_runtime.backend import DeviceBuffer
from npu_runtime.device import Device

if TYPE_CHECKING:
    from npu_compiler.ir_reader import TensorSpec

# MTLResourceStorageModeShared: CPU and GPU can both access the buffer without
# explicit copies. This is slower than Private mode for GPU-only data, but required
# for weight loading (CPU writes) and output readback (CPU reads). A future
# optimization could use Private mode for intermediate buffers.
_STORAGE_MODE_SHARED = 0  # MTLResourceStorageModeShared


def _pad_to_alloc_shape(data: np.ndarray, alloc_shape: tuple[int, ...]) -> np.ndarray:
    """Zero-pad data to the given alloc_shape. Supports arbitrary dimensionality."""
    if tuple(data.shape) == alloc_shape:
        return data
    padded = np.zeros(alloc_shape, dtype=data.dtype)
    slices = tuple(slice(0, s) for s in data.shape)
    padded[slices] = data
    return padded


class NPUBuffer(DeviceBuffer):
    """Metal-backed buffer with FP16 storage and numpy conversion.

    All data is stored as FP16 on the Metal device.
    from_numpy() converts FP32->FP16 on the CPU side before upload.
    to_numpy() reads FP16 and converts back to FP32.

    When alloc_shape is provided, data is zero-padded to the physical shape.
    The logical shape is preserved in _shape; the physical allocation shape
    is in _alloc_shape. The compiler decides alloc_shape; the runtime applies it.
    """

    def __init__(
        self,
        mtl_buffer,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
        alloc_shape: tuple[int, ...] | None = None,
    ):
        self._buffer = mtl_buffer
        self._shape = shape
        self._alloc_shape = alloc_shape or shape
        self._dtype = dtype  # storage dtype (float16)
        self._device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def alloc_shape(self) -> tuple[int, ...]:
        return self._alloc_shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def mtl_buffer(self):
        return self._buffer

    @property
    def native_handle(self):
        return self._buffer

    @property
    def size_bytes(self) -> int:
        return int(np.prod(self._alloc_shape)) * self._dtype.itemsize

    @staticmethod
    def from_numpy(
        data: np.ndarray, device: Device, alloc_shape: tuple[int, ...] | None = None, spec: TensorSpec | None = None
    ) -> NPUBuffer:
        """Create NPU buffer from numpy array.

        Args:
            data: Input numpy array.
            device: Metal device.
            alloc_shape: Physical allocation shape (compiler-decided).
                         If provided, data is zero-padded to this shape.
            spec: TensorSpec with transform_steps. If provided, transform_steps
                  are applied (cast, pad, etc.) instead of default logic.
        """
        if spec is not None:
            return _from_numpy_with_spec(data, device, spec)

        # Integer types are kept as-is (e.g., int32 for embedding indices)
        if np.issubdtype(data.dtype, np.integer):
            converted = data.astype(np.int32) if data.dtype != np.int32 else data
        elif data.dtype == ml_dtypes.bfloat16:
            converted = data
        else:
            converted = data.astype(np.float16) if data.dtype != np.float16 else data
        logical_shape = tuple(converted.shape)

        if alloc_shape is not None:
            converted = _pad_to_alloc_shape(converted, alloc_shape)

        contiguous = np.ascontiguousarray(converted)
        final_alloc_shape = tuple(contiguous.shape)

        raw_bytes = contiguous.tobytes()
        mtl_buffer = device.mtl_device.newBufferWithBytes_length_options_(
            raw_bytes,
            len(raw_bytes),
            _STORAGE_MODE_SHARED,
        )
        if mtl_buffer is None:
            raise RuntimeError(f"Failed to allocate Metal buffer ({len(raw_bytes)} bytes)")

        return NPUBuffer(mtl_buffer, logical_shape, contiguous.dtype, device, final_alloc_shape)

    @staticmethod
    def zeros(
        shape: tuple[int, ...],
        device: Device,
        dtype: np.dtype = np.dtype(np.float16),
        alloc_shape: tuple[int, ...] | None = None,
    ) -> NPUBuffer:
        """Create a zero-initialized NPU buffer.

        Args:
            shape: Logical shape.
            device: Metal device.
            dtype: Storage dtype.
            alloc_shape: Physical allocation shape (compiler-decided).
                         If provided, buffer is allocated with this shape.
        """
        logical_shape = tuple(shape)
        final_alloc_shape = alloc_shape if alloc_shape is not None else logical_shape

        size = int(np.prod(final_alloc_shape))
        size_bytes = size * dtype.itemsize

        # Metal cannot allocate 0-byte buffers; use 1-byte placeholder
        alloc_bytes = max(size_bytes, 1)
        zero_data = b"\x00" * alloc_bytes
        mtl_buffer = device.mtl_device.newBufferWithBytes_length_options_(zero_data, alloc_bytes, _STORAGE_MODE_SHARED)
        if mtl_buffer is None:
            raise RuntimeError(f"Failed to allocate Metal buffer ({alloc_bytes} bytes)")

        return NPUBuffer(mtl_buffer, logical_shape, dtype, device, final_alloc_shape)

    def write_from_numpy(self, data: np.ndarray, spec: TensorSpec | None = None) -> None:
        """Write numpy data into existing Metal buffer in-place (no allocation).

        Args:
            data: Input numpy array.
            spec: TensorSpec with transform_steps. If provided, transforms are applied.
        """
        if spec is not None and spec.transform_steps:
            result = data
            for step in spec.transform_steps:
                if step["type"] == "cast":
                    target = _DTYPE_MAP[step["to"]]
                    if result.dtype != target:
                        result = result.astype(target)
                elif step["type"] == "pad":
                    alloc_shape = tuple(step["alloc_shape"])
                    result = _pad_to_alloc_shape(result, alloc_shape)
            converted = result
        else:
            if np.issubdtype(data.dtype, np.integer):
                converted = data.astype(np.int32) if data.dtype != np.int32 else data
            elif data.dtype == ml_dtypes.bfloat16:
                converted = data
            else:
                converted = data.astype(np.float16) if data.dtype != np.float16 else data

            if self._alloc_shape != tuple(converted.shape):
                converted = _pad_to_alloc_shape(converted, self._alloc_shape)

        contiguous = np.ascontiguousarray(converted)
        raw = contiguous.tobytes()
        buf = self._buffer.contents().as_buffer(len(raw))
        buf[:] = raw
        self._shape = tuple(data.shape)
        self._dtype = contiguous.dtype

    def to_numpy(self, dtype: np.dtype = np.dtype(np.float32), spec: TensorSpec | None = None) -> np.ndarray:
        """Read buffer contents back as numpy array. Default output is FP32.

        Args:
            dtype: Output dtype (used when spec is not provided).
            spec: TensorSpec with transform_steps. If provided, inverse transforms
                  are applied automatically.
        """
        if spec is not None:
            return _to_numpy_with_spec(self, spec)

        nbytes = self.size_bytes
        if nbytes == 0:
            return np.zeros(self._shape, dtype=dtype)

        mv = self._buffer.contents().as_buffer(nbytes)
        flat = np.frombuffer(mv, dtype=self._dtype).copy()

        alloc_size = int(np.prod(self._alloc_shape))
        logical_size = int(np.prod(self._shape))

        if alloc_size == logical_size:
            # No padding — reshape directly to logical shape
            result = flat[:logical_size].reshape(self._shape)
        elif len(self._alloc_shape) == len(self._shape):
            # Same rank, different sizes — slice padding per dimension
            result = flat.reshape(self._alloc_shape)
            slices = tuple(slice(0, s) for s in self._shape)
            result = result[slices]
        else:
            # Different ranks (e.g. compiler flattened dims) — take logical_size elements
            result = flat[:logical_size].reshape(self._shape)

        if dtype != self._dtype:
            result = result.astype(dtype)
        return result


# ── spec-based transform helpers ──

_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "bfloat16": ml_dtypes.bfloat16,
    "int32": np.int32,
    "int64": np.int64,
}


def _from_numpy_with_spec(data: np.ndarray, device: Device, spec: TensorSpec) -> NPUBuffer:
    """Apply forward transform_steps (host→NPU) from spec."""
    result = data
    logical_shape = tuple(data.shape)

    for step in spec.transform_steps or []:
        if step["type"] == "cast":
            target = _DTYPE_MAP[step["to"]]
            if result.dtype != target:
                result = result.astype(target)
        elif step["type"] == "pad":
            alloc_shape = tuple(step["alloc_shape"])
            result = _pad_to_alloc_shape(result, alloc_shape)

    contiguous = np.ascontiguousarray(result)
    final_alloc_shape = tuple(contiguous.shape)

    raw_bytes = contiguous.tobytes()
    mtl_buffer = device.mtl_device.newBufferWithBytes_length_options_(
        raw_bytes,
        len(raw_bytes),
        _STORAGE_MODE_SHARED,
    )
    if mtl_buffer is None:
        raise RuntimeError(f"Failed to allocate Metal buffer ({len(raw_bytes)} bytes)")

    return NPUBuffer(mtl_buffer, logical_shape, contiguous.dtype, device, final_alloc_shape)


def _to_numpy_with_spec(buf: NPUBuffer, spec: TensorSpec) -> np.ndarray:
    """Apply inverse transform_steps (NPU→host) from spec."""
    nbytes = buf.size_bytes
    mv = buf._buffer.contents().as_buffer(nbytes)
    result = np.frombuffer(mv, dtype=buf._dtype).reshape(buf._alloc_shape).copy()

    # Walk steps in reverse, applying the inverse of each
    for step in reversed(spec.transform_steps or []):
        if step["type"] == "pad":
            # Inverse of pad: crop back to logical shape
            slices = tuple(slice(0, s) for s in spec.shape)
            result = result[slices]
        elif step["type"] == "cast":
            # Inverse of cast: restore to spec.dtype (the original host dtype)
            original_dtype = _DTYPE_MAP[spec.dtype]
            if result.dtype != original_dtype:
                result = result.astype(original_dtype)

    return result.reshape(spec.shape)
