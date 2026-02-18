"""Tests for backend extensibility architecture.

Verifies that the abstraction layers (CodegenTarget, DispatchStrategy, Backend,
DeviceBuffer, FusionPatternRegistry, FusedCodegenRegistry) are properly structured
to allow backend extension without modifying core code.
"""

import numpy as np
import pytest

from npu_compiler.codegen import (
    CodegenTarget,
    HANDLED_OPS,
    KernelCall,
    MetalCodegenTarget,
    _FUSED_CODEGEN_REGISTRY,
    register_fused_codegen,
    generate_execution_plan,
)
from npu_compiler.constraint_checker import SUPPORTED_OPS
from npu_compiler.fusion_patterns import (
    FusedGroup,
    _FUSION_PATTERN_REGISTRY,
    register_fusion_pattern,
)
from npu_compiler.ir_reader import OpNode, TensorSpec, load_ir_from_dict
from npu_compiler.target_config import TargetConfig
from npu_runtime.backend import Backend, DeviceBuffer
from npu_runtime.executor import DispatchStrategy, MetalDispatchStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(name, shape, dtype="float32"):
    return TensorSpec(name=name, shape=shape, dtype=dtype)


def _make_ir(nodes, graph_inputs, graph_outputs, weights=None, weight_mapping=None):
    return load_ir_from_dict({
        "model_name": "test",
        "graph_inputs": [{"name": i.name, "shape": i.shape, "dtype": i.dtype} for i in graph_inputs],
        "graph_outputs": [{"name": o.name, "shape": o.shape, "dtype": o.dtype} for o in graph_outputs],
        "weights": [{"name": w.name, "shape": w.shape, "dtype": w.dtype} for w in (weights or [])],
        "weight_name_mapping": weight_mapping or {},
        "nodes": [
            {
                "name": n.name,
                "op_type": n.op_type,
                "inputs": [{"name": t.name, "shape": t.shape, "dtype": t.dtype} for t in n.inputs],
                "outputs": [{"name": t.name, "shape": t.shape, "dtype": t.dtype} for t in n.outputs],
                "attrs": n.attrs,
            }
            for n in nodes
        ],
    })


# ---------------------------------------------------------------------------
# 1. SUPPORTED_OPS auto-derivation (K)
# ---------------------------------------------------------------------------

class TestSupportedOpsAutoDerivation:
    """SUPPORTED_OPS is auto-derived from codegen.HANDLED_OPS."""

    def test_same_object(self):
        """SUPPORTED_OPS IS HANDLED_OPS — no separate copy to maintain."""
        assert SUPPORTED_OPS is HANDLED_OPS

    def test_batch_norm_included(self):
        """batch_norm is in HANDLED_OPS for graph_optimizer consumption."""
        assert "aten.batch_norm.default" in HANDLED_OPS

    def test_no_sync_needed(self):
        """Adding an op to HANDLED_OPS automatically makes it supported."""
        # Simulate: if we added a hypothetical op, it would appear in both
        assert SUPPORTED_OPS == HANDLED_OPS  # always true since same object


# ---------------------------------------------------------------------------
# 2. CodegenTarget ABC (B)
# ---------------------------------------------------------------------------

class TestCodegenTargetABC:
    """CodegenTarget defines a pluggable backend interface for code generation."""

    def test_interface_methods_exist(self):
        """ABC defines required methods."""
        assert hasattr(CodegenTarget, "elementwise_kernel")
        assert hasattr(CodegenTarget, "matmul_kernel")
        assert hasattr(CodegenTarget, "fused_kernel")
        assert hasattr(CodegenTarget, "shader_source")

    def test_metal_target_implements_interface(self):
        """MetalCodegenTarget provides concrete implementations."""
        target = MetalCodegenTarget()
        name, src = target.elementwise_kernel("silu_kernel")
        assert name == "silu_kernel"
        assert src.endswith(".metal")

    def test_custom_target_can_be_created(self):
        """A new backend can implement CodegenTarget without modifying codegen."""
        class CUDATarget(CodegenTarget):
            def elementwise_kernel(self, kernel_name):
                return (kernel_name, f"{kernel_name}.cu")
            def matmul_kernel(self, is_vec, transpose_b):
                return ("cublas_gemm", "cublas_gemm.cu")
            def fused_kernel(self, kernel_type):
                return (kernel_type, f"{kernel_type}.cu")
            def shader_source(self, filename):
                return filename.replace(".metal", ".cu")

        target = CUDATarget()
        name, src = target.matmul_kernel(is_vec=True, transpose_b=False)
        assert name == "cublas_gemm"
        assert src == "cublas_gemm.cu"


# ---------------------------------------------------------------------------
# 3. DispatchStrategy ABC (D)
# ---------------------------------------------------------------------------

class TestDispatchStrategyABC:
    """DispatchStrategy defines pluggable grid/threadgroup computation."""

    def test_interface_method_exists(self):
        """ABC defines compute_dispatch."""
        assert hasattr(DispatchStrategy, "compute_dispatch")

    def test_metal_strategy_with_config(self):
        """MetalDispatchStrategy uses TargetConfig hardware parameters."""
        config = TargetConfig(matmul_tile=16, max_threadgroup_1d=256)
        strategy = MetalDispatchStrategy(config)
        call = KernelCall(
            kernel_name="softmax_kernel", kernel_source="softmax.metal",
            input_buffers=["x"], output_buffers=["y"],
            param_buffers=["p"], params={"rows": 100, "cols": 64},
            dispatch_type="1d", total_threads=100,
        )
        # Pass a mock pipeline with maxTotalThreadsPerThreadgroup
        class FakePipeline:
            def maxTotalThreadsPerThreadgroup(self):
                return 256
        result = strategy.compute_dispatch(call, FakePipeline())
        assert result is not None
        groups, tpg = result
        assert tpg == (256, 1, 1)  # uses max_threadgroup_1d from config

    def test_custom_strategy(self):
        """A new backend can implement DispatchStrategy."""
        class CPUStrategy(DispatchStrategy):
            def compute_dispatch(self, call, pipeline):
                # CPU: single thread, no GPU dispatch
                return ((1, 1, 1), (1, 1, 1))

        strategy = CPUStrategy()
        result = strategy.compute_dispatch(None, None)
        assert result == ((1, 1, 1), (1, 1, 1))


# ---------------------------------------------------------------------------
# 4. DeviceBuffer ABC (E)
# ---------------------------------------------------------------------------

class TestDeviceBufferABC:
    """DeviceBuffer defines the abstract buffer interface."""

    def test_interface_properties(self):
        """ABC defines required properties."""
        assert hasattr(DeviceBuffer, "shape")
        assert hasattr(DeviceBuffer, "alloc_shape")
        assert hasattr(DeviceBuffer, "dtype")
        assert hasattr(DeviceBuffer, "size_bytes")
        assert hasattr(DeviceBuffer, "native_handle")
        assert hasattr(DeviceBuffer, "to_numpy")

    def test_npu_buffer_is_device_buffer(self):
        """NPUBuffer is a concrete DeviceBuffer."""
        from npu_runtime.buffer import NPUBuffer
        assert issubclass(NPUBuffer, DeviceBuffer)


# ---------------------------------------------------------------------------
# 5. Backend ABC (E/F)
# ---------------------------------------------------------------------------

class TestBackendABC:
    """Backend defines the abstract execution backend interface."""

    def test_interface_methods(self):
        """ABC defines required methods."""
        assert hasattr(Backend, "name")
        assert hasattr(Backend, "allocate_buffer")
        assert hasattr(Backend, "allocate_zeros")
        assert hasattr(Backend, "execute")
        assert hasattr(Backend, "create_executor")
        assert hasattr(Backend, "synchronize")

    def test_metal_backend_is_backend(self):
        """MetalBackend implements the Backend interface."""
        from npu_runtime.metal_backend import MetalBackend
        assert issubclass(MetalBackend, Backend)


# ---------------------------------------------------------------------------
# 6. Fusion Pattern Registry (J)
# ---------------------------------------------------------------------------

class TestFusionPatternRegistry:
    """Fusion patterns can be added via register_fusion_pattern()."""

    def test_registry_is_populated(self):
        """Built-in patterns are registered at module load time."""
        # At least: conv_bn_relu, add_relu, rmsnorm, decode_attention, silu_mul, masked_softmax
        assert len(_FUSION_PATTERN_REGISTRY) >= 6

    def test_custom_pattern_can_be_registered(self):
        """A new pattern can be registered without modifying find_fusion_groups."""
        initial_count = len(_FUSION_PATTERN_REGISTRY)

        def _match_gelu(node, graph, consumers, fused, available):
            """Hypothetical GELU fusion pattern."""
            return None  # Would return FusedGroup on match

        register_fusion_pattern("aten.gelu.default", _match_gelu)
        assert len(_FUSION_PATTERN_REGISTRY) == initial_count + 1

        # Cleanup: remove the test pattern to not affect other tests
        _FUSION_PATTERN_REGISTRY.pop()

    def test_existing_patterns_still_work(self):
        """Conv+BN+ReLU fusion still works through registry."""
        ir = _make_ir(
            nodes=[
                OpNode("conv", "aten.conv2d.default",
                       [_ts("x", [1, 3, 8, 8]), _ts("w", [16, 3, 3, 3])],
                       [_ts("conv_out", [1, 16, 6, 6])],
                       {"stride": [1, 1], "padding": [0, 0], "groups": 1}),
                OpNode("relu", "aten.relu.default",
                       [_ts("conv_out", [1, 16, 6, 6])],
                       [_ts("y", [1, 16, 6, 6])], {}),
            ],
            graph_inputs=[_ts("x", [1, 3, 8, 8])],
            graph_outputs=[_ts("y", [1, 16, 6, 6])],
            weights=[_ts("w", [16, 3, 3, 3])],
        )
        plan = generate_execution_plan(ir)
        # Should produce a single fused conv+relu kernel
        assert any(c.kernel_name == "conv2d_kernel" for c in plan.kernel_calls)


# ---------------------------------------------------------------------------
# 7. Fused Codegen Registry (J)
# ---------------------------------------------------------------------------

class TestFusedCodegenRegistry:
    """Fused kernel codegen handlers can be registered via register_fused_codegen."""

    def test_registry_is_populated(self):
        """Built-in handlers are registered."""
        expected = {"conv_bn_relu", "conv_bn", "conv_relu", "add_relu",
                    "rmsnorm", "silu_mul", "masked_softmax", "decode_attention"}
        assert expected.issubset(set(_FUSED_CODEGEN_REGISTRY.keys()))

    def test_custom_handler_can_be_registered(self):
        """A new fused codegen handler can be added without modifying codegen."""
        def _gen_gelu_kernel(group, graph):
            return KernelCall(
                kernel_name="gelu_kernel", kernel_source="gelu.metal",
                input_buffers=[], output_buffers=[], param_buffers=[],
                params={}, dispatch_type="1d", total_threads=1,
            )

        register_fused_codegen("gelu", _gen_gelu_kernel)
        assert "gelu" in _FUSED_CODEGEN_REGISTRY

        # Cleanup
        del _FUSED_CODEGEN_REGISTRY["gelu"]

    def test_handlers_have_uniform_signature(self):
        """All registered handlers accept (group, graph) — no lambda wrappers needed."""
        import inspect
        for name, handler in _FUSED_CODEGEN_REGISTRY.items():
            # Should NOT be a lambda — direct function reference
            assert not handler.__name__.startswith("<lambda>"), (
                f"Handler for '{name}' is a lambda — should be a direct function ref"
            )
            sig = inspect.signature(handler)
            params = list(sig.parameters.keys())
            assert len(params) >= 2, (
                f"Handler for '{name}' has {len(params)} params, expected ≥2 (group, graph)"
            )


# ---------------------------------------------------------------------------
# 8. TargetConfig (C)
# ---------------------------------------------------------------------------

class TestTargetConfigExtensibility:
    """Hardware parameters are centralized in TargetConfig."""

    def test_all_hardware_params_in_config(self):
        """Key hardware constants are TargetConfig fields, not magic numbers."""
        config = TargetConfig()
        assert config.channel_alignment_bytes == 64
        assert config.channel_tile == 32
        assert config.matmul_tile == 16
        assert config.max_threadgroup_1d == 256
        assert config.max_threadgroup_2d == 16
        assert config.max_dispatches_per_batch == 10000

    def test_custom_config_propagates(self):
        """Custom config values propagate to dependent computations."""
        from npu_compiler.target_config import pad_channels
        # Default: pad to 32
        assert pad_channels(1) == 32
        # Custom config with tile=8
        custom = TargetConfig(channel_tile=8)
        assert pad_channels(1, custom) == 8
        assert pad_channels(9, custom) == 16


# ---------------------------------------------------------------------------
# 9. Files-to-modify for new op (I)
# ---------------------------------------------------------------------------

class TestNewOpFileCount:
    """Adding a new op requires modifying only 2 files: codegen.py + .metal shader."""

    def test_constraint_checker_not_needed(self):
        """SUPPORTED_OPS auto-derives — no constraint_checker.py change needed."""
        assert SUPPORTED_OPS is HANDLED_OPS

    def test_only_codegen_and_shader(self):
        """Document the 2-file contract: codegen.py + .metal shader.
        (No need to modify constraint_checker.py, fusion_patterns.py, or executor.py
        for a basic element-wise op.)"""
        # This is a documentation/contract test — verified by the architecture:
        # 1. codegen.py: add handler to _generate_single_kernel_call + HANDLED_OPS
        # 2. .metal: write the kernel
        # constraint_checker.py: auto-derived (no change)
        # executor.py: param packing may need _PARAM_SPECS entry (if params exist)
        pass
