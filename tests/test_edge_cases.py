"""Edge case tests and public API coverage.

Tests for boundary conditions (minimal shapes, empty inputs, dtype edges)
and direct tests for public API functions not covered elsewhere.
"""

import numpy as np
import pytest

from npu_compiler.codegen import generate_execution_plan
from npu_compiler.constraint_checker import check_constraints, pad_channels
from npu_compiler.graph_optimizer import eliminate_noop_ops, fold_batch_norms
from npu_compiler.ir_reader import IRGraph, OpNode, TensorSpec, load_ir_from_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _ts(name, shape, dtype="float32"):
    return TensorSpec(name=name, shape=shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Edge case: minimal shapes
# ---------------------------------------------------------------------------

class TestMinimalShapes:
    """Test boundary shapes like (1,1,1,1), (1,), scalar-like tensors."""

    def test_relu_1d_scalar(self):
        """Shape (1,) — single element tensor."""
        ir = _make_ir(
            nodes=[OpNode("relu", "aten.relu.default",
                          [_ts("x", [1])], [_ts("y", [1])], {})],
            graph_inputs=[_ts("x", [1])],
            graph_outputs=[_ts("y", [1])],
        )
        plan = generate_execution_plan(ir)
        assert len(plan.kernel_calls) == 1
        assert plan.kernel_calls[0].total_threads == 1

    def test_relu_4d_ones(self):
        """Shape (1,1,1,1) — minimal 4D tensor, channels padded to 32."""
        ir = _make_ir(
            nodes=[OpNode("relu", "aten.relu.default",
                          [_ts("x", [1, 1, 1, 1])], [_ts("y", [1, 1, 1, 1])], {})],
            graph_inputs=[_ts("x", [1, 1, 1, 1])],
            graph_outputs=[_ts("y", [1, 1, 1, 1])],
        )
        plan = generate_execution_plan(ir)
        assert len(plan.kernel_calls) == 1
        # 4D dispatch uses padded channels: pad_channels(1) = 32
        assert plan.kernel_calls[0].total_threads == 1 * 32 * 1 * 1

    def test_matmul_1x1(self):
        """Minimal matmul: (1,1) × (1,1) → (1,1)."""
        ir = _make_ir(
            nodes=[OpNode("mm", "aten.linear.default",
                          [_ts("x", [1, 1]), _ts("w", [1, 1])],
                          [_ts("y", [1, 1])], {})],
            graph_inputs=[_ts("x", [1, 1])],
            graph_outputs=[_ts("y", [1, 1])],
            weights=[_ts("w", [1, 1])],
        )
        plan = generate_execution_plan(ir)
        assert len(plan.kernel_calls) == 1
        assert plan.kernel_calls[0].params["M"] == 1
        assert plan.kernel_calls[0].params["N"] == 1
        assert plan.kernel_calls[0].params["K"] == 1

    def test_softmax_single_col(self):
        """Softmax with cols=1 — degenerate case, output always 1.0."""
        ir = _make_ir(
            nodes=[OpNode("sm", "aten.softmax.int",
                          [_ts("x", [2, 1])], [_ts("y", [2, 1])],
                          {"dim": -1})],
            graph_inputs=[_ts("x", [2, 1])],
            graph_outputs=[_ts("y", [2, 1])],
        )
        plan = generate_execution_plan(ir)
        assert len(plan.kernel_calls) == 1
        assert plan.kernel_calls[0].params["cols"] == 1


# ---------------------------------------------------------------------------
# Edge case: large shape
# ---------------------------------------------------------------------------

class TestLargeShape:
    """Test with larger tensor dimensions to ensure no overflow in codegen."""

    def test_large_embedding(self):
        """Large vocab embedding: vocab=151936, embed=1536 (Qwen2.5 scale)."""
        ir = _make_ir(
            nodes=[OpNode("emb", "aten.embedding.default",
                          [_ts("weight", [151936, 1536]), _ts("ids", [1, 1024], dtype="int64")],
                          [_ts("out", [1, 1024, 1536])], {})],
            graph_inputs=[_ts("ids", [1, 1024], dtype="int64")],
            graph_outputs=[_ts("out", [1, 1024, 1536])],
            weights=[_ts("weight", [151936, 1536])],
        )
        plan = generate_execution_plan(ir)
        assert len(plan.kernel_calls) == 1
        assert plan.kernel_calls[0].params["vocab_size"] == 151936


# ---------------------------------------------------------------------------
# Edge case: dtype boundaries
# ---------------------------------------------------------------------------

class TestDtypeEdgeCases:
    """Test dtype handling at boundaries."""

    def test_int64_input_cast(self):
        """int64 inputs (token IDs) should produce int32 IO transform."""
        ir = _make_ir(
            nodes=[OpNode("emb", "aten.embedding.default",
                          [_ts("weight", [1000, 64]), _ts("ids", [4], dtype="int64")],
                          [_ts("out", [4, 64])], {})],
            graph_inputs=[_ts("ids", [4], dtype="int64")],
            graph_outputs=[_ts("out", [4, 64])],
            weights=[_ts("weight", [1000, 64])],
        )
        plan = generate_execution_plan(ir)
        # Verify int64 input gets int32 cast in IO spec
        id_spec = next(s for s in plan.input_specs if s.name == "ids")
        assert id_spec.dtype == "int64"
        assert any(t["type"] == "cast" and t["to"] == "int32" for t in id_spec.transform_steps)

    def test_bfloat16_compute_dtype_inference(self):
        """IR with bfloat16 weights should infer bfloat16 compute dtype."""
        ir = _make_ir(
            nodes=[OpNode("relu", "aten.relu.default",
                          [_ts("x", [2, 4], dtype="bfloat16")],
                          [_ts("y", [2, 4], dtype="bfloat16")], {})],
            graph_inputs=[_ts("x", [2, 4], dtype="bfloat16")],
            graph_outputs=[_ts("y", [2, 4], dtype="bfloat16")],
        )
        plan = generate_execution_plan(ir)
        assert plan.compute_dtype == "bfloat16"


# ---------------------------------------------------------------------------
# Public API: eliminate_noop_ops()
# ---------------------------------------------------------------------------

class TestEliminateNoopOps:
    """Direct tests for eliminate_noop_ops() — covers removal and rewiring."""

    def test_removes_assert(self):
        ir = _make_ir(
            nodes=[
                OpNode("relu", "aten.relu.default",
                       [_ts("x", [4])], [_ts("r", [4])], {}),
                OpNode("assert", "aten._assert_tensor_metadata.default",
                       [_ts("r", [4])], [], {}),
            ],
            graph_inputs=[_ts("x", [4])],
            graph_outputs=[_ts("r", [4])],
        )
        removed = eliminate_noop_ops(ir)
        assert removed == 1
        assert len(ir.nodes) == 1
        assert ir.nodes[0].name == "relu"

    def test_rewires_alias(self):
        """alias(x) → y should rewire downstream consumers to x."""
        ir = _make_ir(
            nodes=[
                OpNode("a", "aten.alias.default",
                       [_ts("x", [4])], [_ts("aliased", [4])], {}),
                OpNode("relu", "aten.relu.default",
                       [_ts("aliased", [4])], [_ts("y", [4])], {}),
            ],
            graph_inputs=[_ts("x", [4])],
            graph_outputs=[_ts("y", [4])],
        )
        removed = eliminate_noop_ops(ir)
        assert removed == 1
        assert len(ir.nodes) == 1
        # relu's input should be rewired from "aliased" to "x"
        assert ir.nodes[0].inputs[0].name == "x"

    def test_rewires_detach(self):
        ir = _make_ir(
            nodes=[
                OpNode("d", "aten.detach_.default",
                       [_ts("x", [4])], [_ts("detached", [4])], {}),
                OpNode("relu", "aten.relu.default",
                       [_ts("detached", [4])], [_ts("y", [4])], {}),
            ],
            graph_inputs=[_ts("x", [4])],
            graph_outputs=[_ts("y", [4])],
        )
        removed = eliminate_noop_ops(ir)
        assert removed == 1
        assert ir.nodes[0].inputs[0].name == "x"

    def test_rewires_graph_output(self):
        """If graph output refers to removed node's output, it should be rewired."""
        ir = _make_ir(
            nodes=[
                OpNode("a", "aten.alias.default",
                       [_ts("x", [4])], [_ts("y", [4])], {}),
            ],
            graph_inputs=[_ts("x", [4])],
            graph_outputs=[_ts("y", [4])],
        )
        removed = eliminate_noop_ops(ir)
        assert removed == 1
        assert ir.graph_outputs[0].name == "x"


# ---------------------------------------------------------------------------
# Public API: fold_batch_norms()
# ---------------------------------------------------------------------------

class TestFoldBatchNorms:
    """Direct tests for fold_batch_norms()."""

    def test_conv_bn_fold(self):
        """Conv2d → BN should be folded into single conv with recipe."""
        conv_inputs = [
            _ts("x", [1, 3, 8, 8]),
            _ts("conv_w", [16, 3, 3, 3]),
            _ts("conv_b", [16]),
        ]
        conv_outputs = [_ts("conv_out", [1, 16, 6, 6])]
        bn_inputs = [
            _ts("conv_out", [1, 16, 6, 6]),
            _ts("bn_w", [16]),    # gamma
            _ts("bn_b", [16]),    # beta
            _ts("bn_mean", [16]),
            _ts("bn_var", [16]),
        ]
        bn_outputs = [_ts("bn_out", [1, 16, 6, 6])]

        ir = _make_ir(
            nodes=[
                OpNode("conv", "aten.conv2d.default", conv_inputs, conv_outputs,
                       {"stride": [1, 1], "padding": [0, 0], "groups": 1}),
                OpNode("bn", "aten.batch_norm.default", bn_inputs, bn_outputs,
                       {"eps": 1e-5, "training": False}),
            ],
            graph_inputs=[_ts("x", [1, 3, 8, 8])],
            graph_outputs=[_ts("bn_out", [1, 16, 6, 6])],
            weights=[
                _ts("conv_w", [16, 3, 3, 3]), _ts("conv_b", [16]),
                _ts("bn_w", [16]), _ts("bn_b", [16]),
                _ts("bn_mean", [16]), _ts("bn_var", [16]),
            ],
        )
        result = fold_batch_norms(ir)
        # BN should be removed, only conv remains
        assert len(result.graph.nodes) == 1
        assert result.graph.nodes[0].op_type == "aten.conv2d.default"
        # Conv output should be renamed to BN output
        assert result.graph.nodes[0].outputs[0].name == "bn_out"
        # Should produce a weight recipe
        assert len(result.weight_recipes) == 1
        assert result.weight_recipes[0].transform == "bn_fold"

    def test_standalone_conv_no_fold(self):
        """Conv2d without following BN should not be modified."""
        ir = _make_ir(
            nodes=[
                OpNode("conv", "aten.conv2d.default",
                       [_ts("x", [1, 3, 8, 8]), _ts("w", [16, 3, 3, 3])],
                       [_ts("y", [1, 16, 6, 6])],
                       {"stride": [1, 1], "padding": [0, 0], "groups": 1}),
            ],
            graph_inputs=[_ts("x", [1, 3, 8, 8])],
            graph_outputs=[_ts("y", [1, 16, 6, 6])],
            weights=[_ts("w", [16, 3, 3, 3])],
        )
        result = fold_batch_norms(ir)
        assert len(result.graph.nodes) == 1
        assert len(result.weight_recipes) == 0


# ---------------------------------------------------------------------------
# Public API: compile()
# ---------------------------------------------------------------------------

class TestCompileAPI:
    """Test the top-level compile() function."""

    def test_compile_graph_roundtrip(self):
        """compile_graph → CompiledProgram has correct fields."""
        import npu_compiler

        ir = _make_ir(
            nodes=[OpNode("relu", "aten.relu.default",
                          [_ts("x", [2, 64])], [_ts("y", [2, 64])], {})],
            graph_inputs=[_ts("x", [2, 64])],
            graph_outputs=[_ts("y", [2, 64])],
        )
        program = npu_compiler.compile_graph(ir)
        assert program.model_name == "test"
        assert len(program.kernel_calls) == 1
        assert len(program.input_specs) == 1
        assert len(program.output_specs) == 1


# ---------------------------------------------------------------------------
# Constraint checker edge cases
# ---------------------------------------------------------------------------

class TestConstraintEdgeCases:
    """Edge cases for constraint checking."""

    def test_pad_channels_already_aligned(self):
        """32 channels → 32 (no padding needed)."""
        assert pad_channels(32) == 32

    def test_pad_channels_1(self):
        """1 channel → 32."""
        assert pad_channels(1) == 32

    def test_pad_channels_33(self):
        """33 channels → 64."""
        assert pad_channels(33) == 64

    def test_zero_shape_allowed_for_cat(self):
        """Zero-size tensors are allowed for cat ops (KV cache init)."""
        ir = _make_ir(
            nodes=[OpNode("cat", "aten.cat.default",
                          [_ts("empty", [0]), _ts("data", [4, 64])],
                          [_ts("out", [4, 64])],
                          {"dim": 0})],
            graph_inputs=[_ts("empty", [0]), _ts("data", [4, 64])],
            graph_outputs=[_ts("out", [4, 64])],
        )
        violations = check_constraints(ir)
        assert len(violations) == 0
