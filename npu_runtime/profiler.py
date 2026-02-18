"""Profiler: measure NPU execution time."""

from __future__ import annotations

import time
from dataclasses import dataclass

from npu_runtime.buffer import NPUBuffer
from npu_runtime.executor import Executor


@dataclass
class ProfileResult:
    """Profiling result with timing and iteration count.

    Dataclass (not tuple) for named access and forward-compatible extension
    (e.g., adding per-kernel breakdown without breaking callers).
    """
    total_ms: float
    iterations: int


def profile(
    executor: Executor,
    inputs: dict[str, NPUBuffer],
    weights: dict[str, NPUBuffer],
    warmup: int = 3,
    iterations: int = 10,
) -> ProfileResult:
    """Profile NPU execution.

    Runs warmup iterations then measures average execution time.
    """
    # Warmup
    for _ in range(warmup):
        executor.run(inputs, weights)

    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        executor.run(inputs, weights)
    end = time.perf_counter()

    total_ms = (end - start) / iterations * 1000

    return ProfileResult(
        total_ms=total_ms,
        iterations=iterations,
    )
