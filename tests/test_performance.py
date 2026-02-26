"""Performance benchmarks for NPU simulation.

Measures kernel execution speed and reports latency per operation.
These tests always pass (benchmarks only) but print timing data for tracking.
"""

import os
import time

import ml_dtypes
import numpy as np
import pytest

try:
    from npu_runtime.buffer import NPUBuffer
    from npu_runtime.device import Device
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

from tests.conftest import dispatch_3d, kernels_dir, make_params

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


@pytest.fixture(scope="module")
def device():
    return Device()


@pytest.fixture(scope="module")
def matmul_lib(device):
    return device.compile_metal_file(
        os.path.join(kernels_dir(), "matmul.metal"), macros={"USE_BFLOAT": 1}
    )


def _bench(fn, warmup=3, iters=10):
    """Run fn for warmup+iters, return average time in ms."""
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1000


class TestMatmulPerformance:
    """Benchmark matmul kernels at various sizes."""

    @pytest.mark.parametrize("M,N,K", [
        (128, 1536, 1536),   # Qwen prefill linear
        (128, 8960, 1536),   # Qwen prefill gate/up
        (1, 1536, 1536),     # Qwen decode linear
        (1, 8960, 1536),     # Qwen decode gate/up
    ])
    def test_matmul_transposed(self, device, matmul_lib, M, N, K):
        """Benchmark matmul_kernel (C = A @ B^T)."""
        dtype = np.dtype(ml_dtypes.bfloat16)
        A = NPUBuffer.zeros((M, K), device, dtype=dtype)
        B = NPUBuffer.zeros((N, K), device, dtype=dtype)
        bias = NPUBuffer.zeros((N,), device, dtype=dtype)
        C = NPUBuffer.zeros((M, N), device, dtype=dtype)
        params = make_params(device, "4I", M, N, K, 0)

        if M == 1:
            pipeline = device.get_pipeline(matmul_lib, "matmul_vec_kernel")

            def run():
                cmd = device.new_command_buffer()
                enc = cmd.computeCommandEncoder()
                enc.setComputePipelineState_(pipeline)
                for idx, buf in enumerate([A, B, bias, C, params]):
                    mtl = buf.mtl_buffer if isinstance(buf, NPUBuffer) else buf
                    enc.setBuffer_offset_atIndex_(mtl, 0, idx)
                enc.dispatchThreadgroups_threadsPerThreadgroup_((N, 1, 1), (256, 1, 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
        else:
            pipeline = device.get_pipeline(matmul_lib, "matmul_kernel")
            TILE = 16
            gx = (N + TILE - 1) // TILE
            gy = (M + TILE - 1) // TILE

            def run():
                cmd = device.new_command_buffer()
                enc = cmd.computeCommandEncoder()
                enc.setComputePipelineState_(pipeline)
                for idx, buf in enumerate([A, B, bias, C, params]):
                    mtl = buf.mtl_buffer if isinstance(buf, NPUBuffer) else buf
                    enc.setBuffer_offset_atIndex_(mtl, 0, idx)
                enc.dispatchThreadgroups_threadsPerThreadgroup_((gx, gy, 1), (TILE, TILE, 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()

        elapsed = _bench(run)
        gflops = 2 * M * N * K / elapsed / 1e6  # 2 ops per FMA, time in ms
        print(f"\n  matmul M={M} N={N} K={K}: {elapsed:.2f}ms ({gflops:.1f} GFLOPS)")

    def test_batched_matmul(self, device, matmul_lib):
        """Benchmark batched_matmul_kernel (attention scores)."""
        batch, M, N, K = 12, 1, 128, 128  # Qwen decode attention
        dtype = np.dtype(ml_dtypes.bfloat16)
        A = NPUBuffer.zeros((batch, M, K), device, dtype=dtype)
        B = NPUBuffer.zeros((batch, K, N), device, dtype=dtype)
        C = NPUBuffer.zeros((batch, M, N), device, dtype=dtype)
        params = make_params(device, "4I", batch, M, N, K)
        pipeline = device.get_pipeline(matmul_lib, "batched_matmul_kernel")

        def run():
            dispatch_3d(device, pipeline, [A, B, C, params], N, M, batch)

        elapsed = _bench(run)
        print(f"\n  batched_matmul batch={batch} M={M} N={N} K={K}: {elapsed:.2f}ms")


class TestProfileAPI:
    """Test the public profile() function."""

    def test_profile_returns_result(self, device):
        """profile() returns ProfileResult with timing data."""
        import npu_compiler
        from npu_runtime.executor import Executor
        from npu_runtime.profiler import profile, ProfileResult

        ir = npu_compiler.load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [4], "dtype": "float32"}],
            "graph_outputs": [{"name": "y", "shape": [4], "dtype": "float32"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [{
                "name": "relu", "op_type": "aten.relu.default",
                "inputs": [{"name": "x", "shape": [4], "dtype": "float32"}],
                "outputs": [{"name": "y", "shape": [4], "dtype": "float32"}],
                "attrs": {},
            }],
        })
        program = npu_compiler.compile_graph(ir)
        executor = Executor(program, device)
        inputs = {"x": NPUBuffer.from_numpy(np.zeros(4, dtype=np.float32), device)}

        result = profile(executor, inputs, weights={}, warmup=1, iterations=3)
        assert isinstance(result, ProfileResult)
        assert result.total_ms > 0
        assert result.iterations == 3


class TestExecutorPerformance:
    """Benchmark full executor run with compiled programs."""

    def test_executor_overhead(self, device):
        """Measure command buffer commit/wait overhead."""
        times = []
        for _ in range(50):
            t0 = time.time()
            cmd = device.new_command_buffer()
            enc = cmd.computeCommandEncoder()
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            times.append(time.time() - t0)

        avg_ms = sum(times) / len(times) * 1000
        print(f"\n  Empty commit+wait: {avg_ms:.3f}ms")
        assert avg_ms < 1.0, "Command buffer overhead should be under 1ms"

    def test_resnet_e2e_speed(self, device):
        """Benchmark ResNet-18 end-to-end (compile + run)."""
        import npu_compiler
        from npu_runtime.executor import Executor

        ir_path = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures",
                               "resnet18_ir.json")
        if not os.path.exists(ir_path):
            # Try alternate path
            ir_path = os.path.join(os.path.dirname(__file__), "fixtures", "resnet18_ir.json")
            if not os.path.exists(ir_path):
                pytest.skip("resnet18_ir.json not found")

        program = npu_compiler.compile(ir_path)
        executor = Executor(program, device)

        # Create dummy inputs
        inputs = {}
        for spec in program.input_specs:
            inputs[spec.name] = NPUBuffer.zeros(tuple(spec.shape), device)

        # Create dummy weights
        weights = {}
        for w in program.weight_specs:
            weights[w.name] = NPUBuffer.zeros(tuple(w.shape), device)

        # Benchmark
        elapsed = _bench(lambda: executor.run(inputs=inputs, weights=weights), warmup=3, iters=10)
        print(f"\n  ResNet-18 inference: {elapsed:.1f}ms ({len(program.kernel_calls)} kernels)")
