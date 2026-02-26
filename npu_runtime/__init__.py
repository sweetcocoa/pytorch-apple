from npu_runtime.backend import Backend, DeviceBuffer
from npu_runtime.dag_executor import DAGExecutor

__all__ = [
    "Backend",
    "DeviceBuffer",
    "DAGExecutor",
]

try:
    from npu_runtime.buffer import NPUBuffer
    from npu_runtime.cpu_fallback import execute_cpu_partition
    from npu_runtime.device import Device
    from npu_runtime.executor import Executor
    from npu_runtime.metal_backend import MetalBackend
    from npu_runtime.profiler import profile
    from npu_runtime.weight_loader import load_weights, load_weights_from_safetensors

    __all__ += [
        "MetalBackend",
        "Device",
        "NPUBuffer",
        "Executor",
        "execute_cpu_partition",
        "load_weights",
        "load_weights_from_safetensors",
        "profile",
    ]
except ImportError:
    pass
