"""Regression tests for known past bugs.

Each test targets a specific bug that was discovered during development,
preventing re-introduction of the same issue.
"""

import numpy as np

from npu_compiler.codegen import generate_execution_plan
from npu_compiler.fusion_patterns import FusedGroup, find_fusion_groups
from npu_compiler.ir_reader import load_ir_from_dict


class TestInPlaceOpNormalization:
    """Regression: ResNet uses in-place relu_ and add_ which must be normalized."""

    def test_relu_inplace_normalized(self):
        """aten.relu_.default must be treated as aten.relu.default."""
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 64], "dtype": "float32"}],
            "graph_outputs": [{"name": "y", "shape": [1, 64], "dtype": "float32"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [{
                "name": "relu",
                "op_type": "aten.relu_.default",
                "inputs": [{"name": "x", "shape": [1, 64], "dtype": "float32"}],
                "outputs": [{"name": "y", "shape": [1, 64], "dtype": "float32"}],
                "attrs": {},
            }],
        })
        plan = generate_execution_plan(ir)
        # Should generate a relu kernel (not crash on unhandled relu_)
        assert len(plan.kernel_calls) == 1
        assert plan.kernel_calls[0].kernel_name == "elementwise_relu"

    def test_add_inplace_normalized(self):
        """aten.add_.Tensor must be treated as aten.add.Tensor."""
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [
                {"name": "a", "shape": [1, 64], "dtype": "float32"},
                {"name": "b", "shape": [1, 64], "dtype": "float32"},
            ],
            "graph_outputs": [{"name": "y", "shape": [1, 64], "dtype": "float32"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [{
                "name": "add",
                "op_type": "aten.add_.Tensor",
                "inputs": [
                    {"name": "a", "shape": [1, 64], "dtype": "float32"},
                    {"name": "b", "shape": [1, 64], "dtype": "float32"},
                ],
                "outputs": [{"name": "y", "shape": [1, 64], "dtype": "float32"}],
                "attrs": {},
            }],
        })
        plan = generate_execution_plan(ir)
        assert len(plan.kernel_calls) == 1
        assert "add" in plan.kernel_calls[0].kernel_name


class TestSiLUGateOrderBug:
    """Regression: SiLU+Gate fusion must check data availability.

    In Qwen2.5, the IR node order may place silu BEFORE the up_proj linear
    that produces mul's other input. Fusing at silu position would read
    uninitialized buffer data. Fix: only fuse when mul's other input is
    in the 'available' set (produced before silu in graph order).
    """

    def test_silu_gate_not_fused_when_input_unavailable(self):
        """silu(gate) * up should NOT fuse when up is produced AFTER silu."""
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [
                {"name": "x", "shape": [1, 64], "dtype": "float32"},
            ],
            "graph_outputs": [{"name": "mul_out", "shape": [1, 64], "dtype": "float32"}],
            "weights": [
                {"name": "gate_w", "shape": [64, 64], "dtype": "float32"},
                {"name": "up_w", "shape": [64, 64], "dtype": "float32"},
            ],
            "weight_name_mapping": {},
            "nodes": [
                # gate_proj = x @ gate_w.T
                {
                    "name": "gate_proj",
                    "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 64], "dtype": "float32"},
                        {"name": "gate_w", "shape": [64, 64], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "gate_out", "shape": [1, 64], "dtype": "float32"}],
                    "attrs": {},
                },
                # silu(gate_out) — happens BEFORE up_proj
                {
                    "name": "silu",
                    "op_type": "aten.silu.default",
                    "inputs": [{"name": "gate_out", "shape": [1, 64], "dtype": "float32",
                                "producer_node": "gate_proj"}],
                    "outputs": [{"name": "silu_out", "shape": [1, 64], "dtype": "float32"}],
                    "attrs": {},
                },
                # up_proj = x @ up_w.T — happens AFTER silu
                {
                    "name": "up_proj",
                    "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 64], "dtype": "float32"},
                        {"name": "up_w", "shape": [64, 64], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "up_out", "shape": [1, 64], "dtype": "float32"}],
                    "attrs": {},
                },
                # silu_out * up_out — up_out is NOT available at silu position
                {
                    "name": "mul",
                    "op_type": "aten.mul.Tensor",
                    "inputs": [
                        {"name": "silu_out", "shape": [1, 64], "dtype": "float32",
                         "producer_node": "silu"},
                        {"name": "up_out", "shape": [1, 64], "dtype": "float32",
                         "producer_node": "up_proj"},
                    ],
                    "outputs": [{"name": "mul_out", "shape": [1, 64], "dtype": "float32"}],
                    "attrs": {},
                },
            ],
        })
        groups = find_fusion_groups(ir)
        # silu+mul should NOT be fused (up_out not available when silu runs)
        fused_types = [g.kernel_type for g in groups if isinstance(g, FusedGroup)]
        assert "silu_mul" not in fused_types, (
            "silu+mul was fused despite up_out not being available at silu position"
        )

    def test_silu_gate_fused_when_input_available(self):
        """silu(gate) * up SHOULD fuse when up is produced BEFORE silu."""
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [
                {"name": "x", "shape": [1, 64], "dtype": "float32"},
            ],
            "graph_outputs": [{"name": "mul_out", "shape": [1, 64], "dtype": "float32"}],
            "weights": [
                {"name": "gate_w", "shape": [64, 64], "dtype": "float32"},
                {"name": "up_w", "shape": [64, 64], "dtype": "float32"},
            ],
            "weight_name_mapping": {},
            "nodes": [
                # up_proj = x @ up_w.T — happens BEFORE silu
                {
                    "name": "up_proj",
                    "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 64], "dtype": "float32"},
                        {"name": "up_w", "shape": [64, 64], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "up_out", "shape": [1, 64], "dtype": "float32"}],
                    "attrs": {},
                },
                # gate_proj = x @ gate_w.T
                {
                    "name": "gate_proj",
                    "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 64], "dtype": "float32"},
                        {"name": "gate_w", "shape": [64, 64], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "gate_out", "shape": [1, 64], "dtype": "float32"}],
                    "attrs": {},
                },
                # silu(gate_out) — happens AFTER up_proj
                {
                    "name": "silu",
                    "op_type": "aten.silu.default",
                    "inputs": [{"name": "gate_out", "shape": [1, 64], "dtype": "float32",
                                "producer_node": "gate_proj"}],
                    "outputs": [{"name": "silu_out", "shape": [1, 64], "dtype": "float32"}],
                    "attrs": {},
                },
                # silu_out * up_out — up_out IS available at silu position
                {
                    "name": "mul",
                    "op_type": "aten.mul.Tensor",
                    "inputs": [
                        {"name": "silu_out", "shape": [1, 64], "dtype": "float32",
                         "producer_node": "silu"},
                        {"name": "up_out", "shape": [1, 64], "dtype": "float32",
                         "producer_node": "up_proj"},
                    ],
                    "outputs": [{"name": "mul_out", "shape": [1, 64], "dtype": "float32"}],
                    "attrs": {},
                },
            ],
        })
        groups = find_fusion_groups(ir)
        fused_types = [g.kernel_type for g in groups if isinstance(g, FusedGroup)]
        assert "silu_mul" in fused_types, (
            "silu+mul should be fused when up_out is available at silu position"
        )


class TestCatEmptyInputAlias:
    """Regression: cat([0]-shaped tensor, data) should alias to data, not crash."""

    def test_cat_empty_first_input(self):
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [
                {"name": "empty", "shape": [0], "dtype": "float32"},
                {"name": "data", "shape": [4, 64], "dtype": "float32"},
            ],
            "graph_outputs": [{"name": "out", "shape": [4, 64], "dtype": "float32"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [{
                "name": "cat",
                "op_type": "aten.cat.default",
                "inputs": [
                    {"name": "empty", "shape": [0], "dtype": "float32"},
                    {"name": "data", "shape": [4, 64], "dtype": "float32"},
                ],
                "outputs": [{"name": "out", "shape": [4, 64], "dtype": "float32"}],
                "attrs": {"dim": 0},
            }],
        })
        plan = generate_execution_plan(ir)
        # Should generate a reshape (alias), not a cat kernel
        assert len(plan.kernel_calls) == 1
        assert plan.kernel_calls[0].kernel_name == "_reshape"
        # Input should be "data", not "empty"
        assert plan.kernel_calls[0].input_buffers == ["data"]


class TestTransposeFoldingRemoved:
    """Regression: transpose folding was removed because N/K swap was incorrect.

    Verify that transpose operations still generate explicit transpose kernels,
    not silently folded into matmul (which would produce wrong results).
    """

    def test_transpose_generates_explicit_kernel(self):
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [2, 3, 4, 8], "dtype": "float32"}],
            "graph_outputs": [{"name": "out", "shape": [2, 3, 8, 4], "dtype": "float32"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [{
                "name": "transpose",
                "op_type": "aten.transpose.int",
                "inputs": [{"name": "x", "shape": [2, 3, 4, 8], "dtype": "float32"}],
                "outputs": [{"name": "out", "shape": [2, 3, 8, 4], "dtype": "float32"}],
                "attrs": {"dim0": 2, "dim1": 3},
            }],
        })
        plan = generate_execution_plan(ir)
        assert len(plan.kernel_calls) == 1
        assert plan.kernel_calls[0].kernel_name == "transpose_kernel"
        # Should NOT be folded (no _reshape or no-op)


class TestMPSBFloat16Fallback:
    """Regression: MPS matmul doesn't support bfloat16.

    MPSMatrixMultiplication asserts on bf16 input. BFloat16 models must
    use custom Metal tiled/vec matmul kernels instead of MPS acceleration.
    If this guard is removed, bf16 models will crash at runtime.
    """

    def test_bf16_disables_mps(self):
        """BFloat16 compute_dtype must disable MPS acceleration."""
        from npu_compiler.compiled_program import CompiledProgram
        from npu_compiler.codegen import ExecutionPlan, KernelCall
        from npu_compiler.ir_reader import TensorSpec
        from npu_runtime.device import Device
        from npu_runtime.executor import Executor

        # Create a minimal bf16 program with a matmul
        plan = ExecutionPlan(
            kernel_calls=[KernelCall(
                kernel_name="matmul_vec_kernel", kernel_source="matmul.metal",
                input_buffers=["x", "w"], output_buffers=["y"],
                param_buffers=["matmul_params"],
                params={"M": 1, "N": 64, "K": 64, "has_bias": 0},
                dispatch_type="1d", total_threads=64,
            )],
            buffer_allocations=[],
            input_specs=[TensorSpec(name="x", shape=[1, 64], dtype="bfloat16")],
            output_specs=[TensorSpec(name="y", shape=[1, 64], dtype="bfloat16")],
            weight_specs=[TensorSpec(name="w", shape=[64, 64], dtype="bfloat16")],
            weight_name_mapping={},
            compute_dtype="bfloat16",
        )
        program = CompiledProgram(
            model_name="test_bf16", execution_plan=plan, weight_recipes=[],
        )
        device = Device()
        executor = Executor(program, device)

        # MPS must be disabled for bf16
        assert not executor._use_mps, (
            "MPS should be disabled for bfloat16 models — "
            "MPSMatrixMultiplication asserts on bf16 input"
        )
        # MPS matmul objects should all be None
        assert all(m is None for m in executor._mps_matmuls), (
            "MPS matmul objects should not be created for bf16 models"
        )

    def test_fp16_enables_mps(self):
        """Float16 compute_dtype should enable MPS acceleration."""
        from npu_compiler.compiled_program import CompiledProgram
        from npu_compiler.codegen import ExecutionPlan, KernelCall
        from npu_compiler.ir_reader import TensorSpec
        from npu_runtime.device import Device
        from npu_runtime.executor import Executor

        plan = ExecutionPlan(
            kernel_calls=[KernelCall(
                kernel_name="matmul_vec_kernel", kernel_source="matmul.metal",
                input_buffers=["x", "w"], output_buffers=["y"],
                param_buffers=["matmul_params"],
                params={"M": 1, "N": 64, "K": 64, "has_bias": 0},
                dispatch_type="1d", total_threads=64,
            )],
            buffer_allocations=[],
            input_specs=[TensorSpec(name="x", shape=[1, 64], dtype="float16")],
            output_specs=[TensorSpec(name="y", shape=[1, 64], dtype="float16")],
            weight_specs=[TensorSpec(name="w", shape=[64, 64], dtype="float16")],
            weight_name_mapping={},
            compute_dtype="float16",
        )
        program = CompiledProgram(
            model_name="test_fp16", execution_plan=plan, weight_recipes=[],
        )
        device = Device()
        executor = Executor(program, device)

        # MPS should be enabled for fp16
        assert executor._use_mps, (
            "MPS should be enabled for float16 models for performance"
        )


class TestTransposeFoldingStaysRemoved:
    """Regression: transpose folding must stay removed.

    The original optimization tried to fold transpose into matmul by swapping
    N/K dimensions, but: (1) Metal custom kernel has no transpose_b parameter,
    (2) MPS matmul transpose flag caused N/K swap errors. The folding was
    removed and transpose always generates an explicit kernel.
    """

    def test_transpose_before_matmul_not_folded(self):
        """transpose → matmul should produce TWO kernels, not one."""
        ir = load_ir_from_dict({
            "model_name": "test",
            "graph_inputs": [
                {"name": "a", "shape": [4, 8], "dtype": "float32"},
                {"name": "b", "shape": [8, 4], "dtype": "float32"},
            ],
            "graph_outputs": [{"name": "out", "shape": [4, 4], "dtype": "float32"}],
            "weights": [],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "t",
                    "op_type": "aten.t.default",
                    "inputs": [{"name": "b", "shape": [8, 4], "dtype": "float32"}],
                    "outputs": [{"name": "bt", "shape": [4, 8], "dtype": "float32"}],
                    "attrs": {},
                },
                {
                    "name": "mm",
                    "op_type": "aten.matmul.default",
                    "inputs": [
                        {"name": "a", "shape": [4, 8], "dtype": "float32"},
                        {"name": "bt", "shape": [4, 8], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "out", "shape": [4, 4], "dtype": "float32"}],
                    "attrs": {},
                },
            ],
        })
        plan = generate_execution_plan(ir)
        kernel_names = [c.kernel_name for c in plan.kernel_calls]
        # Must have explicit transpose, NOT folded into matmul
        assert "transpose_kernel" in kernel_names, (
            "Transpose should produce explicit kernel, not be folded into matmul"
        )


class TestSupportedOpsSync:
    """Regression: SUPPORTED_OPS must be auto-derived from HANDLED_OPS."""

    def test_ops_sync(self):
        """Verify SUPPORTED_OPS is the exact same object as HANDLED_OPS
        (auto-derived, no manual duplication)."""
        from npu_compiler.codegen import HANDLED_OPS
        from npu_compiler.constraint_checker import SUPPORTED_OPS

        # After the auto-derivation change, these are literally the same set.
        assert SUPPORTED_OPS is HANDLED_OPS, (
            "SUPPORTED_OPS should be auto-derived from HANDLED_OPS, not a separate copy"
        )

    def test_batch_norm_in_handled_ops(self):
        """batch_norm must be in HANDLED_OPS even though codegen doesn't emit a
        kernel for it — the graph_optimizer consumes it during BN folding."""
        from npu_compiler.codegen import HANDLED_OPS

        assert "aten.batch_norm.default" in HANDLED_OPS
