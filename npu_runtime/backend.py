"""Abstract backend interfaces for NPU runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DeviceBuffer(ABC):
    """Abstract GPU/NPU buffer with numpy interop."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    @abstractmethod
    def alloc_shape(self) -> tuple[int, ...]:
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        ...

    @property
    @abstractmethod
    def size_bytes(self) -> int:
        ...

    @property
    @abstractmethod
    def native_handle(self) -> Any:
        """Backend-native buffer object (e.g. MTLBuffer for Metal)."""
        ...

    @abstractmethod
    def to_numpy(self, spec=None) -> np.ndarray:
        ...


class Backend(ABC):
    """Abstract NPU execution backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def allocate_buffer(self, data: np.ndarray, alloc_shape=None, spec=None) -> DeviceBuffer:
        ...

    @abstractmethod
    def allocate_zeros(self, shape, dtype=np.dtype(np.float16), alloc_shape=None) -> DeviceBuffer:
        ...

    @abstractmethod
    def execute(self, program, inputs, weights) -> dict[str, DeviceBuffer]:
        ...

    @abstractmethod
    def create_executor(self, program) -> Any:
        """Create a reusable executor for repeated runs (e.g. decode loop)."""
        ...

    @abstractmethod
    def synchronize(self):
        ...
