"""Metal GPU backend implementation."""

from __future__ import annotations

import numpy as np

from npu_compiler.target_config import METAL_GPU, TargetConfig
from npu_runtime.backend import Backend
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device
from npu_runtime.executor import Executor


class MetalBackend(Backend):
    """Backend implementation using Apple Metal GPU."""

    def __init__(self, config: TargetConfig | None = None):
        self._config = config or METAL_GPU
        self._device = Device()

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def device(self) -> Device:
        return self._device

    def allocate_buffer(self, data: np.ndarray, alloc_shape=None, spec=None) -> NPUBuffer:
        return NPUBuffer.from_numpy(data, self._device, alloc_shape=alloc_shape, spec=spec)

    def allocate_zeros(self, shape, dtype=np.dtype(np.float16), alloc_shape=None) -> NPUBuffer:
        return NPUBuffer.zeros(shape, self._device, dtype=dtype, alloc_shape=alloc_shape)

    def create_executor(self, program) -> Executor:
        return Executor(program, self._device, config=self._config)

    def execute(self, program, inputs, weights) -> dict[str, NPUBuffer]:
        executor = self.create_executor(program)
        return executor.run(inputs, weights)

    def synchronize(self):
        pass  # Metal command buffers are synchronous via waitUntilCompleted
