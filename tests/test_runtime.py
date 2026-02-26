"""Tests for NPU runtime: device, buffer, elementwise kernel, weight loading."""

import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

from npu_compiler.ir_reader import TensorSpec

try:
    from npu_runtime.buffer import NPUBuffer
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


class TestDevice:
    def test_device_creation(self, device):
        assert device.name is not None
        assert len(device.name) > 0

    def test_command_queue(self, device):
        cb = device.new_command_buffer()
        assert cb is not None


class TestBuffer:
    def test_from_numpy_fp32(self, device):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf = NPUBuffer.from_numpy(data, device)
        assert buf.shape == (3,)
        assert buf.dtype == np.dtype(np.float16)
        result = buf.to_numpy()
        npt.assert_allclose(result, data, rtol=1e-3)

    def test_from_numpy_fp16(self, device):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        buf = NPUBuffer.from_numpy(data, device)
        result = buf.to_numpy(dtype=np.dtype(np.float16))
        npt.assert_array_equal(result, data)

    def test_zeros(self, device):
        buf = NPUBuffer.zeros((2, 3), device)
        result = buf.to_numpy()
        npt.assert_array_equal(result, np.zeros((2, 3), dtype=np.float32))

    def test_4d_tensor(self, device):
        data = np.random.randn(1, 3, 8, 8).astype(np.float32)
        buf = NPUBuffer.from_numpy(data, device)
        result = buf.to_numpy()
        npt.assert_allclose(result, data, rtol=1e-3, atol=1e-3)

    def test_roundtrip_preserves_shape(self, device):
        data = np.random.randn(2, 64, 16, 16).astype(np.float32)
        buf = NPUBuffer.from_numpy(data, device)
        result = buf.to_numpy()
        assert result.shape == (2, 64, 16, 16)

    def test_4d_tensor_padded(self, device):
        """Padding roundtrip: from_numpy(alloc_shape=...) -> to_numpy strips padding."""
        data = np.random.randn(1, 3, 8, 8).astype(np.float32)
        buf = NPUBuffer.from_numpy(data, device, alloc_shape=(1, 32, 8, 8))

        # Check alloc_shape is padded
        assert buf.alloc_shape == (1, 32, 8, 8), f"Expected (1,32,8,8), got {buf.alloc_shape}"
        assert buf.shape == (1, 3, 8, 8)

        # Roundtrip should strip padding
        result = buf.to_numpy()
        assert result.shape == (1, 3, 8, 8)
        npt.assert_allclose(result, data, rtol=1e-3, atol=1e-3)

    def test_zeros_padded(self, device):
        """NPUBuffer.zeros with alloc_shape should have padded alloc_shape."""
        buf = NPUBuffer.zeros((1, 10, 4, 4), device, alloc_shape=(1, 32, 4, 4))
        assert buf.shape == (1, 10, 4, 4)
        assert buf.alloc_shape == (1, 32, 4, 4)
        result = buf.to_numpy()
        assert result.shape == (1, 10, 4, 4)
        npt.assert_array_equal(result, np.zeros((1, 10, 4, 4), dtype=np.float32))


class TestBufferWithSpec:
    """Tests for spec-based from_numpy/to_numpy (transform_steps)."""

    def test_cast_only(self, device):
        """Spec with cast step only (no padding)."""
        spec = TensorSpec(
            name="x", shape=[4], dtype="float32",
            transform_steps=[{"type": "cast", "to": "float16"}],
        )
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = NPUBuffer.from_numpy(data, device, spec=spec)
        assert buf.dtype == np.dtype(np.float16)
        result = buf.to_numpy(spec=spec)
        assert result.dtype == np.float32
        npt.assert_allclose(result, data, rtol=1e-3)

    def test_cast_and_pad(self, device):
        """Spec with cast + pad steps."""
        spec = TensorSpec(
            name="x", shape=[1, 3, 8, 8], dtype="float32",
            alloc_shape=[1, 32, 8, 8],
            transform_steps=[
                {"type": "cast", "to": "float16"},
                {"type": "pad", "alloc_shape": [1, 32, 8, 8]},
            ],
        )
        data = np.random.randn(1, 3, 8, 8).astype(np.float32)
        buf = NPUBuffer.from_numpy(data, device, spec=spec)
        assert buf.alloc_shape == (1, 32, 8, 8)
        assert buf.shape == (1, 3, 8, 8)

        result = buf.to_numpy(spec=spec)
        assert result.shape == (1, 3, 8, 8)
        assert result.dtype == np.float32
        npt.assert_allclose(result, data, rtol=1e-3, atol=1e-3)

    def test_no_pad_needed(self, device):
        """Spec where shape is already aligned (no pad step)."""
        spec = TensorSpec(
            name="x", shape=[1, 64, 4, 4], dtype="float32",
            transform_steps=[{"type": "cast", "to": "float16"}],
        )
        data = np.random.randn(1, 64, 4, 4).astype(np.float32)
        buf = NPUBuffer.from_numpy(data, device, spec=spec)
        assert buf.alloc_shape == (1, 64, 4, 4)

        result = buf.to_numpy(spec=spec)
        assert result.shape == (1, 64, 4, 4)
        npt.assert_allclose(result, data, rtol=1e-3, atol=1e-3)

    def test_fp16_input(self, device):
        """Spec with FP16 input — cast is a no-op."""
        spec = TensorSpec(
            name="x", shape=[4], dtype="float32",
            transform_steps=[{"type": "cast", "to": "float16"}],
        )
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
        buf = NPUBuffer.from_numpy(data, device, spec=spec)
        result = buf.to_numpy(spec=spec)
        assert result.dtype == np.float32
        npt.assert_allclose(result, data.astype(np.float32), rtol=1e-3)


class TestElementwiseKernel:
    def test_add_kernel(self, device, metal_kernels_dir):
        import os

        a = np.random.randn(256).astype(np.float32)
        b = np.random.randn(256).astype(np.float32)
        expected = a + b

        buf_a = NPUBuffer.from_numpy(a, device)
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_out = NPUBuffer.zeros((256,), device)

        library = device.compile_metal_file(os.path.join(metal_kernels_dir, "elementwise.metal"))
        pipeline = device.get_pipeline(library, "elementwise_add")

        cb = device.new_command_buffer()
        encoder = cb.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(buf_a.mtl_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_b.mtl_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_out.mtl_buffer, 0, 2)

        # 1D dispatch
        threads_per_group = min(256, pipeline.maxTotalThreadsPerThreadgroup())
        num_groups = (256 + threads_per_group - 1) // threads_per_group
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            (num_groups, 1, 1), (threads_per_group, 1, 1)
        )
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        result = buf_out.to_numpy()
        npt.assert_allclose(result, expected, rtol=1e-2, atol=1e-3)

    def test_relu_kernel(self, device, metal_kernels_dir):
        import os

        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected = np.maximum(data, 0.0)

        buf_in = NPUBuffer.from_numpy(data, device)
        buf_out = NPUBuffer.zeros((5,), device)

        library = device.compile_metal_file(os.path.join(metal_kernels_dir, "elementwise.metal"))
        pipeline = device.get_pipeline(library, "elementwise_relu")

        cb = device.new_command_buffer()
        encoder = cb.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(buf_in.mtl_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_out.mtl_buffer, 0, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            (1, 1, 1), (5, 1, 1)
        )
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        result = buf_out.to_numpy()
        npt.assert_allclose(result, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# Public API: load_weights_from_safetensors()
# ---------------------------------------------------------------------------

class TestLoadWeightsFromSafetensors:
    """Tests for load_weights_from_safetensors() — thin wrapper over load_weights."""

    def test_roundtrip_via_safetensors(self, device):
        """Save weights to safetensors, load via load_weights_from_safetensors."""
        from npu_compiler.ir_reader import load_ir_from_dict
        from npu_compiler.codegen import generate_execution_plan
        from npu_compiler.compiled_program import CompiledProgram
        from npu_runtime.weight_loader import load_weights_from_safetensors

        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [4], "dtype": "float32"}],
            "graph_outputs": [{"name": "y", "shape": [4], "dtype": "float32"}],
            "weights": [{"name": "w", "shape": [4], "dtype": "float32"}],
            "weight_name_mapping": {"p_w": "w"},
            "nodes": [{
                "name": "relu", "op_type": "aten.relu.default",
                "inputs": [{"name": "x", "shape": [4], "dtype": "float32"}],
                "outputs": [{"name": "y", "shape": [4], "dtype": "float32"}],
                "attrs": {},
            }],
        })
        plan = generate_execution_plan(ir)
        program = CompiledProgram(
            model_name="test", execution_plan=plan, weight_recipes=[],
        )

        # Create a temporary safetensors file
        from safetensors.numpy import save_file
        weight_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({"w": weight_data}, f.name)
            path = f.name

        try:
            buffers = load_weights_from_safetensors(path, program, device)
            assert "p_w" in buffers
            result = buffers["p_w"].to_numpy()
            npt.assert_allclose(result, weight_data, rtol=1e-3)
        finally:
            os.unlink(path)

    def test_missing_weight_raises(self, device):
        """Missing weight in safetensors should raise KeyError."""
        import pytest
        from npu_compiler.ir_reader import load_ir_from_dict
        from npu_compiler.codegen import generate_execution_plan
        from npu_compiler.compiled_program import CompiledProgram
        from npu_runtime.weight_loader import load_weights_from_safetensors

        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [4], "dtype": "float32"}],
            "graph_outputs": [{"name": "y", "shape": [4], "dtype": "float32"}],
            "weights": [{"name": "w", "shape": [4], "dtype": "float32"}],
            "weight_name_mapping": {"p_w": "w"},
            "nodes": [{
                "name": "relu", "op_type": "aten.relu.default",
                "inputs": [{"name": "x", "shape": [4], "dtype": "float32"}],
                "outputs": [{"name": "y", "shape": [4], "dtype": "float32"}],
                "attrs": {},
            }],
        })
        plan = generate_execution_plan(ir)
        program = CompiledProgram(
            model_name="test", execution_plan=plan, weight_recipes=[],
        )

        from safetensors.numpy import save_file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({"wrong_name": np.zeros(4, dtype=np.float32)}, f.name)
            path = f.name

        try:
            with pytest.raises(KeyError, match="w"):
                load_weights_from_safetensors(path, program, device)
        finally:
            os.unlink(path)
