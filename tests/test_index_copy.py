"""Tests for index_copy Metal kernel and codegen."""

import os

import numpy as np
import numpy.testing as npt

from npu_runtime.buffer import NPUBuffer
from tests.conftest import dispatch_1d, kernels_dir, make_params


class TestIndexCopyKernel:
    """Direct Metal kernel tests for index_copy_kernel."""

    def test_single_index_1d(self, device):
        """Copy single value into a 1D tensor."""
        # input: [10, 20, 30, 40, 50], index: [2], source: [99]
        # expected: [10, 20, 99, 40, 50]
        inp = np.array([10, 20, 30, 40, 50], dtype=np.float16)
        source = np.array([99], dtype=np.float16)
        indices = np.array([2], dtype=np.int32)

        buf_inp = NPUBuffer.from_numpy(inp, device)
        buf_src = NPUBuffer.from_numpy(source, device)
        buf_idx = NPUBuffer.from_numpy(indices, device)
        buf_out = NPUBuffer.zeros(inp.shape, device)

        # params: outer_size=1, dim_size=5, inner_size=1, num_indices=1
        params = make_params(device, "4I", 1, 5, 1, 1)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "index_copy.metal"))
        pipeline = device.get_pipeline(lib, "index_copy_kernel")
        dispatch_1d(device, pipeline, [buf_inp, buf_src, buf_idx, buf_out, params], 5)

        expected = np.array([10, 20, 99, 40, 50], dtype=np.float16)
        npt.assert_array_equal(buf_out.to_numpy(dtype=np.float16), expected)

    def test_multiple_indices(self, device):
        """Copy multiple values into a 1D tensor."""
        inp = np.arange(8, dtype=np.float16)
        source = np.array([100, 200, 300], dtype=np.float16)
        indices = np.array([1, 3, 5], dtype=np.int32)

        buf_inp = NPUBuffer.from_numpy(inp, device)
        buf_src = NPUBuffer.from_numpy(source, device)
        buf_idx = NPUBuffer.from_numpy(indices, device)
        buf_out = NPUBuffer.zeros(inp.shape, device)

        params = make_params(device, "4I", 1, 8, 1, 3)
        lib = device.compile_metal_file(os.path.join(kernels_dir(), "index_copy.metal"))
        pipeline = device.get_pipeline(lib, "index_copy_kernel")
        dispatch_1d(device, pipeline, [buf_inp, buf_src, buf_idx, buf_out, params], 8)

        expected = inp.copy()
        expected[1] = 100
        expected[3] = 200
        expected[5] = 300
        npt.assert_array_equal(buf_out.to_numpy(dtype=np.float16), expected)

    def test_kv_cache_shape(self, device):
        """Simulate KV cache update: (1, 2, 8, 4) with dim=2, single index."""
        B, H, S, D = 1, 2, 8, 4
        inp = np.random.randn(B, H, S, D).astype(np.float16)
        source = np.random.randn(B, H, 1, D).astype(np.float16)  # L=1
        indices = np.array([5], dtype=np.int32)  # write at position 5

        buf_inp = NPUBuffer.from_numpy(inp.reshape(-1), device)
        buf_src = NPUBuffer.from_numpy(source.reshape(-1), device)
        buf_idx = NPUBuffer.from_numpy(indices, device)
        buf_out = NPUBuffer.zeros((B * H * S * D,), device)

        # outer_size = B*H = 2, dim_size = S = 8, inner_size = D = 4, num_indices = 1
        params = make_params(device, "4I", B * H, S, D, 1)
        total = B * H * S * D

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "index_copy.metal"))
        pipeline = device.get_pipeline(lib, "index_copy_kernel")
        dispatch_1d(device, pipeline, [buf_inp, buf_src, buf_idx, buf_out, params], total)

        # Build expected
        expected = inp.copy()
        expected[:, :, 5:6, :] = source
        out = buf_out.to_numpy(dtype=np.float16).reshape(B, H, S, D)
        npt.assert_array_equal(out, expected)

    def test_dim0(self, device):
        """index_copy along dim=0 in a 2D tensor."""
        # (4, 3) tensor, replace row 1 and row 3
        inp = np.arange(12, dtype=np.float16).reshape(4, 3)
        source = np.array([[100, 101, 102], [300, 301, 302]], dtype=np.float16)
        indices = np.array([1, 3], dtype=np.int32)

        buf_inp = NPUBuffer.from_numpy(inp.reshape(-1), device)
        buf_src = NPUBuffer.from_numpy(source.reshape(-1), device)
        buf_idx = NPUBuffer.from_numpy(indices, device)
        buf_out = NPUBuffer.zeros((12,), device)

        # outer_size=1 (no dims before 0), dim_size=4, inner_size=3, num_indices=2
        params = make_params(device, "4I", 1, 4, 3, 2)
        lib = device.compile_metal_file(os.path.join(kernels_dir(), "index_copy.metal"))
        pipeline = device.get_pipeline(lib, "index_copy_kernel")
        dispatch_1d(device, pipeline, [buf_inp, buf_src, buf_idx, buf_out, params], 12)

        expected = inp.copy()
        expected[1] = [100, 101, 102]
        expected[3] = [300, 301, 302]
        out = buf_out.to_numpy(dtype=np.float16).reshape(4, 3)
        npt.assert_array_equal(out, expected)


class TestIndexCopyCodegen:
    """Test index_copy codegen integration."""

    def test_constraint_checker_accepts(self):
        from npu_compiler.constraint_checker import SUPPORTED_OPS
        assert "aten.index_copy.default" in SUPPORTED_OPS

    def test_codegen_produces_kernel_call(self):
        from npu_compiler.codegen import KernelCall, _generate_single_kernel_call
        from npu_compiler.ir_reader import IRGraph, OpNode, TensorSpec

        node = OpNode(
            name="index_copy_1",
            op_type="aten.index_copy.default",
            inputs=[
                TensorSpec(name="kv_cache", shape=[1, 2, 128, 64], dtype="float16"),
                TensorSpec(name="cache_pos", shape=[1], dtype="int32"),
                TensorSpec(name="new_kv", shape=[1, 2, 1, 64], dtype="float16"),
            ],
            outputs=[
                TensorSpec(name="updated_kv", shape=[1, 2, 128, 64], dtype="float16"),
            ],
            attrs={"dim": 2},
        )
        graph = IRGraph(
            graph_inputs=[], graph_outputs=[], weights=[], weight_name_mapping={},
            nodes=[node], model_name="test",
        )

        call = _generate_single_kernel_call(node, graph)
        assert call is not None
        assert isinstance(call, KernelCall)
        assert call.kernel_name == "index_copy_kernel"
        assert call.kernel_source == "index_copy.metal"
        # Buffer order: input, source, index
        assert call.input_buffers == ["kv_cache", "new_kv", "cache_pos"]
        assert call.output_buffers == ["updated_kv"]
        assert call.params["outer_size"] == 2  # B*H = 1*2
        assert call.params["dim_size"] == 128
        assert call.params["inner_size"] == 64
        assert call.params["num_indices"] == 1
        assert call.total_threads == 1 * 2 * 128 * 64
