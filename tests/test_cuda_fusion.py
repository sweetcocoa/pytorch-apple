"""Tests for CUDA subgraph fusion correctness: fused output should match unfused.

These tests REQUIRE CuPy/CUDA — they verify that the fused CUDA kernels
produce the same results as running each op individually.
Skipped automatically if CuPy is not installed.
"""

import numpy as np
import numpy.testing as npt
import pytest

try:
    import cupy  # noqa: F401

    from cuda_compiler import compile_subgraph
    from cuda_compiler.cuda_program import CUBLASStep, FusedKernelStep
    from cuda_runtime.cuda_backend import CUDABuffer
    from cuda_runtime.cuda_executor import CUDAExecutor

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy/CUDA not available")

# ---------------------------------------------------------------------------
# Helpers: CPU reference implementations
# ---------------------------------------------------------------------------


def _cpu_relu(x):
    return np.maximum(x, 0)


def _cpu_silu(x):
    return x / (1 + np.exp(-x.astype(np.float64))).astype(x.dtype)


def _cpu_neg(x):
    return -x


def _cpu_rsqrt(x):
    return 1.0 / np.sqrt(x.astype(np.float64) + 1e-12)


def _cpu_add(a, b):
    return a + b


def _cpu_mul(a, b):
    return a * b


def _cpu_softmax(x, axis=-1):
    x_f64 = x.astype(np.float64)
    x_max = x_f64.max(axis=axis, keepdims=True)
    exp_x = np.exp(x_f64 - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def _run_program(ir_dict, inputs_np, weights_np=None):
    """Compile and run an IR dict, return output numpy dict."""
    program = compile_subgraph(ir_dict)
    executor = CUDAExecutor(program)

    inputs = {k: CUDABuffer.from_numpy(v.astype(np.float16)) for k, v in inputs_np.items()}
    weights = {}
    if weights_np:
        weights = {k: CUDABuffer.from_numpy(v.astype(np.float16)) for k, v in weights_np.items()}

    outputs = executor.run(inputs, weights)
    return {k: v.to_numpy().astype(np.float32) for k, v in outputs.items()}


# ---------------------------------------------------------------------------
# 1. Single elementwise ops (baseline — no fusion)
# ---------------------------------------------------------------------------


class TestSingleOps:
    def test_relu(self):
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 256], "dtype": "float16"}],
            "graph_outputs": [{"name": "y", "shape": [1, 256], "dtype": "float16"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [{
                "name": "relu", "op_type": "aten.relu.default",
                "inputs": [{"name": "x", "shape": [1, 256], "dtype": "float16"}],
                "outputs": [{"name": "y", "shape": [1, 256], "dtype": "float16"}],
                "attrs": {},
            }],
        }
        x = np.random.randn(1, 256).astype(np.float32)
        result = _run_program(ir_dict, {"x": x})
        expected = _cpu_relu(x)
        npt.assert_allclose(result["y"], expected, rtol=5e-2, atol=1e-2)

    def test_neg(self):
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 256], "dtype": "float16"}],
            "graph_outputs": [{"name": "y", "shape": [1, 256], "dtype": "float16"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [{
                "name": "neg", "op_type": "aten.neg.default",
                "inputs": [{"name": "x", "shape": [1, 256], "dtype": "float16"}],
                "outputs": [{"name": "y", "shape": [1, 256], "dtype": "float16"}],
                "attrs": {},
            }],
        }
        x = np.random.randn(1, 256).astype(np.float32)
        result = _run_program(ir_dict, {"x": x})
        expected = _cpu_neg(x)
        npt.assert_allclose(result["y"], expected, rtol=5e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 2. Fused elementwise chains — compare with sequential CPU execution
# ---------------------------------------------------------------------------


class TestFusedChains:
    def test_relu_add_fused(self):
        """relu(x) + y should fuse if from same chain."""
        # Note: add needs two inputs, relu_out and y. This tests binary fusion.
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [
                {"name": "x", "shape": [1, 128], "dtype": "float16"},
                {"name": "y_in", "shape": [1, 128], "dtype": "float16"},
            ],
            "graph_outputs": [{"name": "z", "shape": [1, 128], "dtype": "float16"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "relu", "op_type": "aten.relu.default",
                    "inputs": [{"name": "x", "shape": [1, 128], "dtype": "float16"}],
                    "outputs": [{"name": "relu_out", "shape": [1, 128], "dtype": "float16"}],
                    "attrs": {},
                },
                {
                    "name": "add", "op_type": "aten.add.Tensor",
                    "inputs": [
                        {"name": "relu_out", "shape": [1, 128], "dtype": "float16"},
                        {"name": "y_in", "shape": [1, 128], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "z", "shape": [1, 128], "dtype": "float16"}],
                    "attrs": {},
                },
            ],
        }
        x = np.random.randn(1, 128).astype(np.float32)
        y_in = np.random.randn(1, 128).astype(np.float32)

        result = _run_program(ir_dict, {"x": x, "y_in": y_in})
        expected = _cpu_relu(x) + y_in
        npt.assert_allclose(result["z"], expected, rtol=5e-2, atol=1e-2)

        # Verify fusion actually happened (should be 1 fused kernel)
        program = compile_subgraph(ir_dict)
        fused = [s for s in program.steps if isinstance(s, FusedKernelStep)]
        assert len(fused) == 1, f"Expected 1 fused kernel, got {len(fused)}"

    def test_silu_mul_fused(self):
        """silu(gate) * up should fuse into single kernel."""
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [
                {"name": "gate", "shape": [1, 256], "dtype": "float16"},
                {"name": "up", "shape": [1, 256], "dtype": "float16"},
            ],
            "graph_outputs": [{"name": "y", "shape": [1, 256], "dtype": "float16"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "silu", "op_type": "aten.silu.default",
                    "inputs": [{"name": "gate", "shape": [1, 256], "dtype": "float16"}],
                    "outputs": [{"name": "silu_out", "shape": [1, 256], "dtype": "float16"}],
                    "attrs": {},
                },
                {
                    "name": "mul", "op_type": "aten.mul.Tensor",
                    "inputs": [
                        {"name": "silu_out", "shape": [1, 256], "dtype": "float16"},
                        {"name": "up", "shape": [1, 256], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "y", "shape": [1, 256], "dtype": "float16"}],
                    "attrs": {},
                },
            ],
        }
        gate = np.random.randn(1, 256).astype(np.float32)
        up = np.random.randn(1, 256).astype(np.float32)

        result = _run_program(ir_dict, {"gate": gate, "up": up})
        expected = _cpu_silu(gate) * up
        npt.assert_allclose(result["y"], expected, rtol=5e-2, atol=5e-2)

        # Verify fusion
        program = compile_subgraph(ir_dict)
        fused = [s for s in program.steps if isinstance(s, FusedKernelStep)]
        assert len(fused) == 1

    def test_add_relu_fused(self):
        """add(x, y) → relu should fuse into single kernel (ResNet skip connection)."""
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [
                {"name": "a", "shape": [1, 64], "dtype": "float16"},
                {"name": "b", "shape": [1, 64], "dtype": "float16"},
            ],
            "graph_outputs": [{"name": "y", "shape": [1, 64], "dtype": "float16"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "add", "op_type": "aten.add.Tensor",
                    "inputs": [
                        {"name": "a", "shape": [1, 64], "dtype": "float16"},
                        {"name": "b", "shape": [1, 64], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "add_out", "shape": [1, 64], "dtype": "float16"}],
                    "attrs": {},
                },
                {
                    "name": "relu", "op_type": "aten.relu.default",
                    "inputs": [{"name": "add_out", "shape": [1, 64], "dtype": "float16"}],
                    "outputs": [{"name": "y", "shape": [1, 64], "dtype": "float16"}],
                    "attrs": {},
                },
            ],
        }
        a = np.random.randn(1, 64).astype(np.float32)
        b = np.random.randn(1, 64).astype(np.float32)

        result = _run_program(ir_dict, {"a": a, "b": b})
        expected = _cpu_relu(a + b)
        npt.assert_allclose(result["y"], expected, rtol=5e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 3. Mixed graph — BLAS + fused elementwise
# ---------------------------------------------------------------------------


class TestMixedGraph:
    def test_linear_relu(self):
        """linear → relu: BLAS breaks fusion, relu is separate fused kernel."""
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 64], "dtype": "float16"}],
            "graph_outputs": [{"name": "y", "shape": [1, 32], "dtype": "float16"}],
            "weights": [{"name": "w", "shape": [32, 64], "dtype": "float16"}],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "linear", "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 64], "dtype": "float16"},
                        {"name": "w", "shape": [32, 64], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "lin_out", "shape": [1, 32], "dtype": "float16"}],
                    "attrs": {},
                },
                {
                    "name": "relu", "op_type": "aten.relu.default",
                    "inputs": [{"name": "lin_out", "shape": [1, 32], "dtype": "float16"}],
                    "outputs": [{"name": "y", "shape": [1, 32], "dtype": "float16"}],
                    "attrs": {},
                },
            ],
        }
        x = np.random.randn(1, 64).astype(np.float32)
        w = np.random.randn(32, 64).astype(np.float32)

        result = _run_program(ir_dict, {"x": x}, {"w": w})
        expected = _cpu_relu(x @ w.T)
        npt.assert_allclose(result["y"], expected, rtol=5e-2, atol=5e-2)

        # Verify structure: 1 BLAS + 1 fused
        program = compile_subgraph(ir_dict)
        blas = [s for s in program.steps if isinstance(s, CUBLASStep)]
        fused = [s for s in program.steps if isinstance(s, FusedKernelStep)]
        assert len(blas) == 1
        assert len(fused) == 1

    def test_qwen_mlp_pattern(self):
        """gate_proj → silu → mul(up_proj) → down_proj (Qwen GatedMLP)."""
        ir_dict = {
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 32], "dtype": "float16"}],
            "graph_outputs": [{"name": "y", "shape": [1, 32], "dtype": "float16"}],
            "weights": [
                {"name": "gate_w", "shape": [64, 32], "dtype": "float16"},
                {"name": "up_w", "shape": [64, 32], "dtype": "float16"},
                {"name": "down_w", "shape": [32, 64], "dtype": "float16"},
            ],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "gate_proj", "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 32], "dtype": "float16"},
                        {"name": "gate_w", "shape": [64, 32], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "gate_out", "shape": [1, 64], "dtype": "float16"}],
                    "attrs": {},
                },
                {
                    "name": "up_proj", "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 32], "dtype": "float16"},
                        {"name": "up_w", "shape": [64, 32], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "up_out", "shape": [1, 64], "dtype": "float16"}],
                    "attrs": {},
                },
                {
                    "name": "silu", "op_type": "aten.silu.default",
                    "inputs": [{"name": "gate_out", "shape": [1, 64], "dtype": "float16"}],
                    "outputs": [{"name": "silu_out", "shape": [1, 64], "dtype": "float16"}],
                    "attrs": {},
                },
                {
                    "name": "mul", "op_type": "aten.mul.Tensor",
                    "inputs": [
                        {"name": "silu_out", "shape": [1, 64], "dtype": "float16"},
                        {"name": "up_out", "shape": [1, 64], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "mul_out", "shape": [1, 64], "dtype": "float16"}],
                    "attrs": {},
                },
                {
                    "name": "down_proj", "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "mul_out", "shape": [1, 64], "dtype": "float16"},
                        {"name": "down_w", "shape": [32, 64], "dtype": "float16"},
                    ],
                    "outputs": [{"name": "y", "shape": [1, 32], "dtype": "float16"}],
                    "attrs": {},
                },
            ],
        }
        np.random.seed(42)
        x = np.random.randn(1, 32).astype(np.float32)
        gate_w = np.random.randn(64, 32).astype(np.float32) * 0.1
        up_w = np.random.randn(64, 32).astype(np.float32) * 0.1
        down_w = np.random.randn(32, 64).astype(np.float32) * 0.1

        result = _run_program(
            ir_dict, {"x": x}, {"gate_w": gate_w, "up_w": up_w, "down_w": down_w},
        )

        # CPU reference
        gate_out = x @ gate_w.T
        up_out = x @ up_w.T
        silu_out = _cpu_silu(gate_out)
        mul_out = silu_out * up_out
        expected = mul_out @ down_w.T

        npt.assert_allclose(result["y"], expected, rtol=1e-1, atol=1e-1)

        # Verify structure: 3 BLAS + 1 fused (silu+mul)
        program = compile_subgraph(ir_dict)
        blas = [s for s in program.steps if isinstance(s, CUBLASStep)]
        fused = [s for s in program.steps if isinstance(s, FusedKernelStep)]
        assert len(blas) == 3
        assert len(fused) >= 1
