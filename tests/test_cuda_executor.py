"""Integration tests for CUDAExecutor: NVRTC compile + GPU dispatch.

These tests REQUIRE CuPy/CUDA — they test the online execution pipeline.
Skipped automatically if CuPy is not installed.
"""

import numpy as np
import numpy.testing as npt
import pytest

try:
    import cupy  # noqa: F401

    from cuda_compiler import compile_subgraph
    from cuda_compiler.cuda_program import (
        CUDAKernelSource,
        CUDAProgram,
        FusedKernelStep,
    )
    from cuda_runtime.cuda_backend import CUDABackend, CUDABuffer
    from cuda_runtime.cuda_executor import CUDAExecutor
    from npu_compiler.ir_reader import TensorSpec

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy/CUDA not available")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_relu_program() -> CUDAProgram:
    """Create a minimal program: single fused relu kernel."""
    source = """
#include <cuda_fp16.h>
extern "C" {
__global__ void fused_relu(const __half* in0, __half* out0, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    __half v0 = in0[idx];
    __half v1 = __hmax(v0, (__half)0.0);
    out0[idx] = v1;
}
}
"""
    return CUDAProgram(
        steps=[
            FusedKernelStep(
                kernel_name="fused_relu",
                source_code=source,
                input_buffer_names=["x"],
                output_buffer_name="y",
                total_elements=256,
            ),
        ],
        buffer_allocations=[],
        input_specs=[TensorSpec(name="x", shape=[256], dtype="float16")],
        output_specs=[TensorSpec(name="y", shape=[256], dtype="float16")],
        weight_specs=[],
        weight_name_mapping={},
        kernel_sources=[CUDAKernelSource(kernel_name="fused_relu", source_code=source)],
    )


# ---------------------------------------------------------------------------
# 1. CUDABuffer
# ---------------------------------------------------------------------------


class TestCUDABuffer:
    def test_from_numpy_roundtrip(self):
        data = np.random.randn(128).astype(np.float16)
        buf = CUDABuffer.from_numpy(data)
        result = buf.to_numpy()
        npt.assert_array_equal(result, data)

    def test_zeros(self):
        buf = CUDABuffer.zeros((4, 8))
        result = buf.to_numpy()
        npt.assert_array_equal(result, np.zeros((4, 8), dtype=np.float16))

    def test_shape_properties(self):
        data = np.ones((2, 3), dtype=np.float16)
        buf = CUDABuffer.from_numpy(data)
        assert buf.shape == (2, 3)
        assert buf.alloc_shape == (2, 3)
        assert buf.dtype == np.dtype(np.float16)
        assert buf.size_bytes == 2 * 3 * 2  # float16 = 2 bytes

    def test_write_from_numpy(self):
        buf = CUDABuffer.zeros((4,))
        new_data = np.array([1, 2, 3, 4], dtype=np.float16)
        buf.write_from_numpy(new_data)
        result = buf.to_numpy()
        npt.assert_array_equal(result, new_data)


# ---------------------------------------------------------------------------
# 2. CUDABackend
# ---------------------------------------------------------------------------


class TestCUDABackend:
    def test_backend_name(self):
        backend = CUDABackend()
        assert backend.name == "cuda"

    def test_allocate_buffer(self):
        backend = CUDABackend()
        data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        buf = backend.allocate_buffer(data)
        assert isinstance(buf, CUDABuffer)
        result = buf.to_numpy()
        npt.assert_array_equal(result, data)

    def test_allocate_zeros(self):
        backend = CUDABackend()
        buf = backend.allocate_zeros((8,))
        assert isinstance(buf, CUDABuffer)
        result = buf.to_numpy()
        npt.assert_array_equal(result, np.zeros(8, dtype=np.float16))

    def test_synchronize(self):
        backend = CUDABackend()
        backend.synchronize()  # should not raise


# ---------------------------------------------------------------------------
# 3. CUDAExecutor — fused kernel
# ---------------------------------------------------------------------------


class TestCUDAExecutorFused:
    def test_relu_kernel(self):
        """Fused relu kernel: negative values clamped to 0."""
        program = _make_simple_relu_program()
        executor = CUDAExecutor(program)

        data = np.random.randn(256).astype(np.float16)
        expected = np.maximum(data, 0).astype(np.float16)

        inputs = {"x": CUDABuffer.from_numpy(data)}
        outputs = executor.run(inputs, weights={})

        assert "y" in outputs
        result = outputs["y"].to_numpy()
        npt.assert_allclose(result.ravel(), expected, rtol=1e-2, atol=1e-3)

    def test_silu_mul_fused(self):
        """Compile and run a fused silu+mul kernel via compile_subgraph."""
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [
                {"name": "gate", "shape": [1, 128], "dtype": "float16"},
                {"name": "up", "shape": [1, 128], "dtype": "float16"},
            ],
            "graph_outputs": [{"name": "y", "shape": [1, 128], "dtype": "float16"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "silu",
                    "op_type": "aten.silu.default",
                    "inputs": [{"name": "gate", "shape": [1, 128], "dtype": "float16"}],
                    "outputs": [{"name": "silu_out", "shape": [1, 128], "dtype": "float16"}],
                    "attrs": {},
                },
                {
                    "name": "mul",
                    "op_type": "aten.mul.Tensor",
                    "inputs": [
                        {"name": "silu_out", "shape": [1, 128], "dtype": "float16"},
                        {"name": "up", "shape": [1, 128], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "y", "shape": [1, 128], "dtype": "float16"}],
                    "attrs": {},
                },
            ],
        }
        program = compile_subgraph(ir_dict)
        executor = CUDAExecutor(program)

        gate = np.random.randn(1, 128).astype(np.float16)
        up = np.random.randn(1, 128).astype(np.float16)

        # CPU reference: silu(x) = x / (1 + exp(-x))
        gate_f32 = gate.astype(np.float32)
        silu_ref = gate_f32 / (1 + np.exp(-gate_f32))
        expected = (silu_ref * up.astype(np.float32)).astype(np.float16)

        inputs = {
            "gate": CUDABuffer.from_numpy(gate),
            "up": CUDABuffer.from_numpy(up),
        }
        outputs = executor.run(inputs, weights={})

        result = outputs["y"].to_numpy()
        npt.assert_allclose(result, expected, rtol=5e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 4. CUDAExecutor — BLAS (cuBLAS via CuPy)
# ---------------------------------------------------------------------------


class TestCUDAExecutorBLAS:
    def test_gemm(self):
        """Linear layer (GEMM): y = x @ w.T."""
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 64], "dtype": "float16"}],
            "graph_outputs": [{"name": "y", "shape": [1, 32], "dtype": "float16"}],
            "weights": [{"name": "w", "shape": [32, 64], "dtype": "float16"}],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "linear",
                    "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 64], "dtype": "float16"},
                        {"name": "w", "shape": [32, 64], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "y", "shape": [1, 32], "dtype": "float16"}],
                    "attrs": {},
                },
            ],
        }
        program = compile_subgraph(ir_dict)
        executor = CUDAExecutor(program)

        x = np.random.randn(1, 64).astype(np.float16)
        w = np.random.randn(32, 64).astype(np.float16)
        expected = (x.astype(np.float32) @ w.astype(np.float32).T).astype(np.float16)

        inputs = {"x": CUDABuffer.from_numpy(x)}
        weights = {"w": CUDABuffer.from_numpy(w)}
        outputs = executor.run(inputs, weights)

        result = outputs["y"].to_numpy()
        npt.assert_allclose(result, expected, rtol=5e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 5. CUDAExecutor — alias (reshape)
# ---------------------------------------------------------------------------


class TestCUDAExecutorAlias:
    def test_reshape(self):
        """Reshape is zero-cost alias."""
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [2, 128], "dtype": "float16"}],
            "graph_outputs": [{"name": "y", "shape": [256], "dtype": "float16"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "reshape",
                    "op_type": "aten.reshape.default",
                    "inputs": [{"name": "x", "shape": [2, 128], "dtype": "float16"}],
                    "outputs": [{"name": "y", "shape": [256], "dtype": "float16"}],
                    "attrs": {},
                },
            ],
        }
        program = compile_subgraph(ir_dict)
        executor = CUDAExecutor(program)

        x = np.random.randn(2, 128).astype(np.float16)
        inputs = {"x": CUDABuffer.from_numpy(x)}
        outputs = executor.run(inputs, weights={})

        result = outputs["y"].to_numpy()
        npt.assert_array_equal(result.ravel(), x.ravel())


# ---------------------------------------------------------------------------
# 6. CUDAExecutor — reduction (softmax)
# ---------------------------------------------------------------------------


class TestCUDAExecutorReduction:
    def test_softmax(self):
        """Softmax on last dimension."""
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [4, 64], "dtype": "float16"}],
            "graph_outputs": [{"name": "y", "shape": [4, 64], "dtype": "float16"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "softmax",
                    "op_type": "aten.softmax.int",
                    "inputs": [{"name": "x", "shape": [4, 64], "dtype": "float16"}],
                    "outputs": [{"name": "y", "shape": [4, 64], "dtype": "float16"}],
                    "attrs": {"dim": -1},
                },
            ],
        }
        program = compile_subgraph(ir_dict)
        executor = CUDAExecutor(program)

        x = np.random.randn(4, 64).astype(np.float16)
        # CPU reference
        x_f32 = x.astype(np.float32)
        x_max = x_f32.max(axis=-1, keepdims=True)
        exp_x = np.exp(x_f32 - x_max)
        expected = (exp_x / exp_x.sum(axis=-1, keepdims=True)).astype(np.float16)

        inputs = {"x": CUDABuffer.from_numpy(x)}
        outputs = executor.run(inputs, weights={})

        result = outputs["y"].to_numpy()
        npt.assert_allclose(result, expected, rtol=5e-2, atol=1e-2)
