"""Tests for DAG executor: mixed NPU + CPU fallback execution.

Tests use the real Metal backend for NPU partitions and torch_ir
for CPU fallback. The key scenarios:
  1. All ops supported → single NPU partition (matches existing E2E)
  2. Some ops unsupported → mixed execution → same result as CPU-only
"""

import os
import tempfile

import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn
from torch_ir import extract_ir

from npu_compiler.op_support import is_op_supported
from npu_compiler.partitioner import Partition, partition
from npu_runtime.dag_executor import DAGExecutor
from npu_runtime.metal_backend import MetalBackend


def _extract_ir_dict(model_cls, example_input_shape, model_name="test") -> dict:
    """Extract IR as an in-memory dict."""
    with torch.device("meta"):
        meta_model = model_cls()
    meta_model.eval()

    meta_input = (torch.randn(*example_input_shape, device="meta"),)
    ir = extract_ir(meta_model, meta_input, model_name=model_name)

    # Serialize to dict via JSON round-trip
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    ir.save(tmp.name)
    tmp.close()

    import json

    with open(tmp.name) as f:
        ir_dict = json.load(f)
    os.unlink(tmp.name)
    return ir_dict


def _model_weights_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def _extract_resnet_ir_dict(model_fn, model_name, input_shape=(1, 3, 224, 224)):
    """Extract IR dict for a torchvision model."""
    import json

    with torch.device("meta"):
        meta_model = model_fn(weights=None)
    meta_model.eval()
    meta_input = (torch.randn(*input_shape, device="meta"),)
    ir = extract_ir(meta_model, meta_input, model_name=model_name)
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    ir.save(tmp.name)
    tmp.close()
    with open(tmp.name) as f:
        ir_dict = json.load(f)
    os.unlink(tmp.name)
    return ir_dict


# ─── Test Model ───


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TestDAGExecutorAllSupported:
    """All ops supported → single NPU partition, result matches existing Executor."""

    def test_simple_convnet_all_npu(self, device):
        torch.manual_seed(42)
        model = SimpleConvNet()
        model.eval()

        ir_dict = _extract_ir_dict(SimpleConvNet, (1, 3, 32, 32), "SimpleConvNet")
        weights_np = _model_weights_numpy(model)

        # Partition (all supported → single NPU partition)
        plan = partition(ir_dict, is_op_supported)
        partitions = [s for s in plan.steps if isinstance(s, Partition)]
        assert len(partitions) == 1
        assert partitions[0].target == "npu"

        # Execute via DAGExecutor
        backend = MetalBackend()
        dag = DAGExecutor(plan, backend)

        input_tensor = torch.randn(1, 3, 32, 32)
        input_name = ir_dict["graph_inputs"][0]["name"]
        result = dag.execute(
            inputs={input_name: input_tensor.numpy()},
            weights=weights_np,
        )

        # Compare with CPU
        with torch.no_grad():
            cpu_output = model(input_tensor).numpy()

        npu_output = list(result.values())[0]
        npt.assert_allclose(npu_output, cpu_output, rtol=5e-2, atol=5e-2)
        assert np.argmax(npu_output) == np.argmax(cpu_output)


class TestDAGExecutorMixed:
    """Some ops marked as unsupported → mixed NPU + CPU execution."""

    def test_partial_support(self, device):
        """Mark max_pool2d as unsupported → CPU fallback for pool, rest on NPU."""
        torch.manual_seed(42)
        model = SimpleConvNet()
        model.eval()

        ir_dict = _extract_ir_dict(SimpleConvNet, (1, 3, 32, 32), "SimpleConvNet")
        weights_np = _model_weights_numpy(model)

        # Custom support: all supported except max_pool2d
        def custom_support(op_type, attrs=None):
            if op_type == "aten.max_pool2d.default":
                return False
            return is_op_supported(op_type, attrs)

        plan = partition(ir_dict, custom_support)

        # Should have multiple partitions
        partitions = [s for s in plan.steps if isinstance(s, Partition)]
        cpu_parts = [p for p in partitions if p.target == "cpu"]
        assert len(cpu_parts) >= 1, "Should have at least one CPU partition for max_pool2d"

        # Execute
        backend = MetalBackend()
        dag = DAGExecutor(plan, backend)

        input_tensor = torch.randn(1, 3, 32, 32)
        input_name = ir_dict["graph_inputs"][0]["name"]
        result = dag.execute(
            inputs={input_name: input_tensor.numpy()},
            weights=weights_np,
        )

        # Compare with CPU-only execution
        with torch.no_grad():
            cpu_output = model(input_tensor).numpy()

        npu_output = list(result.values())[0]
        npt.assert_allclose(npu_output, cpu_output, rtol=5e-2, atol=5e-2)
        assert np.argmax(npu_output) == np.argmax(cpu_output)


class TestDAGExecutorResNet:
    """E2E: ResNet-18 through DAG executor."""

    def test_resnet18_all_supported(self, device):
        import torchvision.models as models

        torch.manual_seed(0)
        model = models.resnet18(weights=None)
        model.eval()

        ir_dict = _extract_resnet_ir_dict(models.resnet18, "ResNet18")
        weights_np = _model_weights_numpy(model)

        plan = partition(ir_dict, is_op_supported)
        backend = MetalBackend()
        dag = DAGExecutor(plan, backend)

        input_tensor = torch.randn(1, 3, 224, 224)
        input_name = ir_dict["graph_inputs"][0]["name"]
        result = dag.execute(
            inputs={input_name: input_tensor.numpy()},
            weights=weights_np,
        )

        with torch.no_grad():
            cpu_output = model(input_tensor).numpy()

        npu_output = list(result.values())[0]
        npt.assert_allclose(npu_output, cpu_output, rtol=1e-1, atol=1e-1)
        assert np.argmax(npu_output) == np.argmax(cpu_output)

    def test_resnet18_npu_faster_than_cpu(self, device):
        """Benchmark: NPU (via DAG executor) should be faster than CPU at batch=16.

        At small batch sizes (1-4), PyTorch's optimized CPU kernels (MKL/oneDNN)
        beat the Metal GPU simulation due to Python-level dispatch overhead.
        At batch=16+, GPU parallelism dominates and NPU is significantly faster.
        """
        import time

        import torchvision.models as models

        batch_size = 16

        torch.manual_seed(0)
        model = models.resnet18(weights=None)
        model.eval()

        ir_dict = _extract_resnet_ir_dict(
            lambda **kw: models.resnet18(**kw),
            f"ResNet18_bs{batch_size}",
            input_shape=(batch_size, 3, 224, 224),
        )
        weights_np = _model_weights_numpy(model)

        plan = partition(ir_dict, is_op_supported)
        backend = MetalBackend()
        dag = DAGExecutor(plan, backend)

        input_tensor = torch.randn(batch_size, 3, 224, 224)
        input_name = ir_dict["graph_inputs"][0]["name"]
        input_np = input_tensor.numpy()

        # Pre-load weights once (not part of timing)
        dag.load_weights(weights_np)

        # Warmup (3 runs each)
        for _ in range(3):
            dag.execute(inputs={input_name: input_np})
        with torch.no_grad():
            for _ in range(3):
                model(input_tensor)

        n_runs = 10

        # CPU timing
        cpu_start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                model(input_tensor)
        cpu_time = time.perf_counter() - cpu_start

        # NPU timing (via DAG executor, weights already cached)
        npu_start = time.perf_counter()
        for _ in range(n_runs):
            dag.execute(inputs={input_name: input_np})
        npu_time = time.perf_counter() - npu_start

        speedup = cpu_time / npu_time
        print(f"\n  ResNet-18 benchmark (batch={batch_size}, {n_runs} runs):")
        print(f"    CPU: {cpu_time:.3f}s ({cpu_time / n_runs * 1000:.1f}ms/run)")
        print(f"    NPU: {npu_time:.3f}s ({npu_time / n_runs * 1000:.1f}ms/run)")
        print(f"    Speedup: {speedup:.2f}x")

        # NPU should be faster than CPU at batch=16
        assert speedup >= 1.0, f"NPU not faster: {speedup:.2f}x (expected >= 1.0x)"

    def test_resnet18_partial_support(self, device):
        """ResNet-18 with adaptive_avg_pool2d on CPU fallback."""
        import torchvision.models as models

        torch.manual_seed(0)
        model = models.resnet18(weights=None)
        model.eval()

        ir_dict = _extract_resnet_ir_dict(models.resnet18, "ResNet18")
        weights_np = _model_weights_numpy(model)

        def custom_support(op_type, attrs=None):
            if op_type == "aten.adaptive_avg_pool2d.default":
                return False
            return is_op_supported(op_type, attrs)

        plan = partition(ir_dict, custom_support)
        cpu_parts = [s for s in plan.steps if isinstance(s, Partition) and s.target == "cpu"]
        assert len(cpu_parts) >= 1

        backend = MetalBackend()
        dag = DAGExecutor(plan, backend)

        input_tensor = torch.randn(1, 3, 224, 224)
        input_name = ir_dict["graph_inputs"][0]["name"]
        result = dag.execute(
            inputs={input_name: input_tensor.numpy()},
            weights=weights_np,
        )

        with torch.no_grad():
            cpu_output = model(input_tensor).numpy()

        npu_output = list(result.values())[0]
        npt.assert_allclose(npu_output, cpu_output, rtol=1e-1, atol=1e-1)
        assert np.argmax(npu_output) == np.argmax(cpu_output)
