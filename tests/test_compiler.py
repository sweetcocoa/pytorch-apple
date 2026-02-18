"""Tests for NPU compiler pipeline."""

import os
import tempfile

import pytest

import npu_compiler
from npu_compiler.codegen import HANDLED_OPS, generate_execution_plan
from npu_compiler.compiled_program import CompiledProgram
from npu_compiler.constraint_checker import SUPPORTED_OPS, check_constraints
from npu_compiler.fusion_patterns import FusedGroup, find_fusion_groups
from npu_compiler.ir_reader import load_ir, load_ir_from_dict


@pytest.fixture
def simple_convnet_ir_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "torch_to_ir", "simple_convnet_ir.json"
    )


@pytest.fixture
def simple_convnet_ir(simple_convnet_ir_path):
    return load_ir(simple_convnet_ir_path)


class TestIRReader:
    def test_load_simple_convnet(self, simple_convnet_ir):
        ir = simple_convnet_ir
        assert ir.model_name == "SimpleConvNet"
        assert len(ir.nodes) == 10
        assert len(ir.weights) == 8
        assert len(ir.graph_inputs) == 1
        assert ir.graph_inputs[0].shape == [1, 3, 32, 32]

    def test_node_structure(self, simple_convnet_ir):
        conv = simple_convnet_ir.nodes[0]
        assert conv.op_type == "aten.conv2d.default"
        assert conv.inputs[0].name == "x"
        assert conv.outputs[0].shape == [1, 32, 32, 32]

    def test_weight_name_mapping(self, simple_convnet_ir):
        mapping = simple_convnet_ir.weight_name_mapping
        assert mapping["p_features_0_weight"] == "features.0.weight"


class TestConstraintChecker:
    def test_simple_convnet_passes(self, simple_convnet_ir):
        violations = check_constraints(simple_convnet_ir)
        assert len(violations) == 0

    def test_unsupported_op(self):
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 3, 8, 8], "dtype": "float32"}],
            "graph_outputs": [{"name": "y", "shape": [1, 3, 8, 8], "dtype": "float32"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [{
                "name": "unsupported",
                "op_type": "aten.unsupported_op.default",
                "inputs": [{"name": "x", "shape": [1, 3, 8, 8], "dtype": "float32"}],
                "outputs": [{"name": "y", "shape": [1, 3, 8, 8], "dtype": "float32"}],
                "attrs": {},
            }],
        })
        violations = check_constraints(ir)
        assert len(violations) == 1
        assert "Unsupported op" in violations[0].message

    def test_dynamic_shape_rejected(self):
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 3, 8, 8], "dtype": "float32"}],
            "graph_outputs": [{"name": "y", "shape": [-1, 3, 8, 8], "dtype": "float32"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [{
                "name": "bad",
                "op_type": "aten.relu.default",
                "inputs": [{"name": "x", "shape": [1, 3, 8, 8], "dtype": "float32"}],
                "outputs": [{"name": "y", "shape": [-1, 3, 8, 8], "dtype": "float32"}],
                "attrs": {},
            }],
        })
        violations = check_constraints(ir)
        assert any("Dynamic" in v.message for v in violations)


class TestFusionPatterns:
    def test_conv_relu_fusion(self):
        """Conv → ReLU should be fused."""
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 3, 8, 8], "dtype": "float32"}],
            "graph_outputs": [{"name": "relu", "shape": [1, 16, 8, 8], "dtype": "float32"}],
            "weights": [
                {"name": "w", "shape": [16, 3, 3, 3], "dtype": "float32"},
                {"name": "b", "shape": [16], "dtype": "float32"},
            ],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "conv",
                    "op_type": "aten.conv2d.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 3, 8, 8], "dtype": "float32"},
                        {"name": "w", "shape": [16, 3, 3, 3], "dtype": "float32"},
                        {"name": "b", "shape": [16], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "conv", "shape": [1, 16, 8, 8], "dtype": "float32"}],
                    "attrs": {"stride": [1, 1], "padding": [1, 1]},
                },
                {
                    "name": "relu",
                    "op_type": "aten.relu.default",
                    "inputs": [{"name": "conv", "shape": [1, 16, 8, 8], "dtype": "float32",
                                "producer_node": "conv", "producer_output_idx": 0}],
                    "outputs": [{"name": "relu", "shape": [1, 16, 8, 8], "dtype": "float32"}],
                    "attrs": {},
                },
            ],
        })
        groups = find_fusion_groups(ir)
        assert len(groups) == 1
        assert isinstance(groups[0], FusedGroup)
        assert groups[0].kernel_type == "conv_relu"

    def test_add_relu_fusion(self):
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [
                {"name": "a", "shape": [1, 64, 8, 8], "dtype": "float32"},
                {"name": "b", "shape": [1, 64, 8, 8], "dtype": "float32"},
            ],
            "graph_outputs": [{"name": "relu", "shape": [1, 64, 8, 8], "dtype": "float32"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "add",
                    "op_type": "aten.add.Tensor",
                    "inputs": [
                        {"name": "a", "shape": [1, 64, 8, 8], "dtype": "float32"},
                        {"name": "b", "shape": [1, 64, 8, 8], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "add", "shape": [1, 64, 8, 8], "dtype": "float32"}],
                    "attrs": {},
                },
                {
                    "name": "relu",
                    "op_type": "aten.relu.default",
                    "inputs": [{"name": "add", "shape": [1, 64, 8, 8], "dtype": "float32",
                                "producer_node": "add"}],
                    "outputs": [{"name": "relu", "shape": [1, 64, 8, 8], "dtype": "float32"}],
                    "attrs": {},
                },
            ],
        })
        groups = find_fusion_groups(ir)
        assert len(groups) == 1
        assert isinstance(groups[0], FusedGroup)
        assert groups[0].kernel_type == "add_relu"


class TestCodegen:
    def test_simple_convnet_plan(self, simple_convnet_ir):
        plan = generate_execution_plan(simple_convnet_ir)
        assert len(plan.kernel_calls) > 0
        assert len(plan.input_specs) == 1
        assert plan.input_specs[0].shape == [1, 3, 32, 32]

        # Check kernel types generated
        kernel_names = [k.kernel_name for k in plan.kernel_calls]
        assert any("conv" in k for k in kernel_names)
        assert any("matmul" in k for k in kernel_names)


class TestCompiledProgram:
    def test_roundtrip_serialization(self, simple_convnet_ir):
        program = npu_compiler.compile_graph(simple_convnet_ir)

        with tempfile.NamedTemporaryFile(suffix=".npubin", delete=False) as f:
            path = f.name

        try:
            program.save(path)
            loaded = CompiledProgram.load(path)

            assert loaded.model_name == program.model_name
            assert len(loaded.kernel_calls) == len(program.kernel_calls)
            assert len(loaded.input_specs) == len(program.input_specs)
            assert loaded.input_specs[0].shape == program.input_specs[0].shape

            # Check alloc_shape field survives serialization
            for orig, loaded_alloc in zip(program.buffer_allocations, loaded.buffer_allocations):
                assert orig.alloc_shape == loaded_alloc.alloc_shape
        finally:
            os.unlink(path)

    def test_full_compile_pipeline(self, simple_convnet_ir_path):
        program = npu_compiler.compile(simple_convnet_ir_path)
        assert program.model_name == "SimpleConvNet"
        assert len(program.kernel_calls) > 0


class TestSupportedOpsSync:
    def test_supported_ops_handled_by_codegen(self):
        """SUPPORTED_OPS의 모든 op이 codegen에서 처리되는지 검증."""
        # batch_norm is removed by the graph optimizer before codegen
        OPTIMIZER_OPS = {"aten.batch_norm.default"}
        unhandled = SUPPORTED_OPS - HANDLED_OPS - OPTIMIZER_OPS
        assert unhandled == set(), f"SUPPORTED_OPS에 있지만 codegen에서 미처리: {unhandled}"

    def test_handled_ops_in_supported_ops(self):
        """HANDLED_OPS의 모든 op이 SUPPORTED_OPS에도 등록되어 있는지 검증."""
        # Zero-cost ops that codegen handles but don't need constraint checking
        ZERO_COST_OPS = {
            "aten.view.default", "aten.reshape.default", "aten._unsafe_view.default",
            "aten.alias.default", "aten.detach_.default", "aten.clone.default",
            "aten.to.dtype", "aten.native_dropout.default",
            "aten._assert_tensor_metadata.default",
            "wrap_with_set_grad_enabled",
        }
        unregistered = HANDLED_OPS - SUPPORTED_OPS - ZERO_COST_OPS
        assert unregistered == set(), f"HANDLED_OPS에 있지만 SUPPORTED_OPS에 미등록: {unregistered}"
