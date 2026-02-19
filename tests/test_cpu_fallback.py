"""Tests for CPU fallback execution."""

import numpy as np
import numpy.testing as npt
import torch

from npu_runtime.cpu_fallback import (
    _numpy_to_torch,
    _torch_to_numpy,
    execute_cpu_partition,
)


class TestNumpyTorchConversion:
    def test_float32_roundtrip(self):
        arr = np.random.randn(2, 3).astype(np.float32)
        t = _numpy_to_torch(arr)
        assert t.dtype == torch.float32
        result = _torch_to_numpy(t)
        npt.assert_array_equal(arr, result)

    def test_bfloat16_roundtrip(self):
        t_orig = torch.randn(2, 3, dtype=torch.bfloat16)
        # bfloat16 → numpy (ml_dtypes.bfloat16)
        arr = _torch_to_numpy(t_orig)
        assert arr.dtype.name == "bfloat16"
        # numpy (bfloat16) → bfloat16 tensor
        t_back = _numpy_to_torch(arr)
        assert t_back.dtype == torch.bfloat16
        npt.assert_array_equal(
            t_orig.view(torch.uint16).numpy(),
            t_back.view(torch.uint16).numpy(),
        )


class TestCPUPartitionExecution:
    def test_single_add_op(self):
        """Execute a single add op on CPU."""
        nodes = [
            {
                "name": "add_node",
                "op_type": "aten.add.Tensor",
                "inputs": [
                    {"name": "a", "shape": [2, 3], "dtype": "float32", "producer_node": None},
                    {"name": "b", "shape": [2, 3], "dtype": "float32", "producer_node": None},
                ],
                "outputs": [
                    {
                        "name": "c",
                        "shape": [2, 3],
                        "dtype": "float32",
                        "producer_node": "add_node",
                        "producer_output_idx": 0,
                    },
                ],
                "attrs": {},
            }
        ]
        a = np.ones((2, 3), dtype=np.float32) * 2.0
        b = np.ones((2, 3), dtype=np.float32) * 3.0
        pool = {"a": a, "b": b}

        result = execute_cpu_partition(nodes, pool, weights={})
        assert "c" in result
        npt.assert_allclose(result["c"], a + b)

    def test_single_relu_op(self):
        """Execute a single relu op on CPU."""
        nodes = [
            {
                "name": "relu_node",
                "op_type": "aten.relu.default",
                "inputs": [
                    {"name": "x", "shape": [2, 4], "dtype": "float32", "producer_node": None},
                ],
                "outputs": [
                    {
                        "name": "y",
                        "shape": [2, 4],
                        "dtype": "float32",
                        "producer_node": "relu_node",
                        "producer_output_idx": 0,
                    },
                ],
                "attrs": {},
            }
        ]
        x = np.array([[-1, 2, -3, 4], [5, -6, 7, -8]], dtype=np.float32)
        pool = {"x": x}

        result = execute_cpu_partition(nodes, pool, weights={})
        expected = np.maximum(x, 0)
        npt.assert_allclose(result["y"], expected)

    def test_two_node_chain(self):
        """Execute a chain: neg → relu."""
        nodes = [
            {
                "name": "neg_node",
                "op_type": "aten.neg.default",
                "inputs": [
                    {"name": "x", "shape": [2, 3], "dtype": "float32", "producer_node": None},
                ],
                "outputs": [
                    {
                        "name": "neg_out",
                        "shape": [2, 3],
                        "dtype": "float32",
                        "producer_node": "neg_node",
                        "producer_output_idx": 0,
                    },
                ],
                "attrs": {},
            },
            {
                "name": "relu_node",
                "op_type": "aten.relu.default",
                "inputs": [
                    {
                        "name": "neg_out",
                        "shape": [2, 3],
                        "dtype": "float32",
                        "producer_node": "neg_node",
                        "producer_output_idx": 0,
                    },
                ],
                "outputs": [
                    {
                        "name": "y",
                        "shape": [2, 3],
                        "dtype": "float32",
                        "producer_node": "relu_node",
                        "producer_output_idx": 0,
                    },
                ],
                "attrs": {},
            },
        ]
        x = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        pool = {"x": x}

        result = execute_cpu_partition(nodes, pool, weights={})
        expected = np.maximum(-x, 0)
        npt.assert_allclose(result["y"], expected)

    def test_mul_with_scalar_attr(self):
        """Execute mul with a scalar from attrs."""
        nodes = [
            {
                "name": "mul_node",
                "op_type": "aten.mul.Tensor",
                "inputs": [
                    {"name": "x", "shape": [2, 3], "dtype": "float32", "producer_node": None},
                    {"name": "scale", "shape": [1], "dtype": "float32", "producer_node": None},
                ],
                "outputs": [
                    {
                        "name": "y",
                        "shape": [2, 3],
                        "dtype": "float32",
                        "producer_node": "mul_node",
                        "producer_output_idx": 0,
                    },
                ],
                "attrs": {},
            }
        ]
        x = np.ones((2, 3), dtype=np.float32) * 4.0
        scale = np.array([2.0], dtype=np.float32)
        pool = {"x": x, "scale": scale}

        result = execute_cpu_partition(nodes, pool, weights={})
        npt.assert_allclose(result["y"], x * 2.0)
