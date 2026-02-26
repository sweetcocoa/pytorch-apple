"""Tests for CUDA subgraph compiler: op classification, subgraph analysis, codegen, buffer planning.

These tests do NOT require CuPy/CUDA — they test the offline compilation pipeline only.
"""


from cuda_compiler.cuda_codegen import generate_fused_kernel, is_fusible_elementwise
from cuda_compiler.cuda_program import (
    AliasStep,
    CUBLASStep,
    CUDAProgram,
    FusedKernelStep,
    ReductionKernelStep,
    SpecialKernelStep,
)
from cuda_compiler.op_classify import OpCategory, classify_op
from cuda_compiler.op_support import get_cuda_supported_ops, is_cuda_op_supported
from cuda_compiler.subgraph_analyzer import analyze_subgraph
from npu_compiler.ir_reader import OpNode, TensorSpec, load_ir_from_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(name, shape, dtype="float32"):
    return TensorSpec(name=name, shape=shape, dtype=dtype)


def _make_graph(nodes, graph_inputs, graph_outputs, weights=None, weight_mapping=None):
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
# 1. Op classification
# ---------------------------------------------------------------------------

class TestOpClassify:
    def test_matmul_is_blas(self):
        assert classify_op("aten.matmul.default") == OpCategory.ANCHOR_BLAS

    def test_linear_is_blas(self):
        assert classify_op("aten.linear.default") == OpCategory.ANCHOR_BLAS

    def test_conv2d_is_blas(self):
        assert classify_op("aten.conv2d.default") == OpCategory.ANCHOR_BLAS

    def test_relu_is_elementwise(self):
        assert classify_op("aten.relu.default") == OpCategory.ELEMENTWISE

    def test_silu_is_elementwise(self):
        assert classify_op("aten.silu.default") == OpCategory.ELEMENTWISE

    def test_add_is_elementwise(self):
        assert classify_op("aten.add.Tensor") == OpCategory.ELEMENTWISE

    def test_mul_is_elementwise(self):
        assert classify_op("aten.mul.Tensor") == OpCategory.ELEMENTWISE

    def test_softmax_is_reduction(self):
        assert classify_op("aten.softmax.int") == OpCategory.REDUCTION

    def test_reshape_is_alias(self):
        assert classify_op("aten.reshape.default") == OpCategory.SHAPE_ALIAS

    def test_view_is_alias(self):
        assert classify_op("aten.view.default") == OpCategory.SHAPE_ALIAS

    def test_embedding_is_special(self):
        assert classify_op("aten.embedding.default") == OpCategory.SPECIAL

    def test_noop(self):
        assert classify_op("aten._assert_tensor_metadata.default") == OpCategory.NOOP


# ---------------------------------------------------------------------------
# 2. Op support
# ---------------------------------------------------------------------------

class TestOpSupport:
    def test_cuda_supports_conv(self):
        assert is_cuda_op_supported("aten.conv2d.default")

    def test_cuda_supports_matmul(self):
        assert is_cuda_op_supported("aten.matmul.default")

    def test_cuda_supports_relu(self):
        assert is_cuda_op_supported("aten.relu.default")

    def test_cuda_supports_silu(self):
        assert is_cuda_op_supported("aten.silu.default")

    def test_unsupported_op(self):
        assert not is_cuda_op_supported("aten.some_weird_op.default")

    def test_supported_ops_set(self):
        ops = get_cuda_supported_ops()
        assert "aten.linear.default" in ops
        assert len(ops) >= 30


# ---------------------------------------------------------------------------
# 3. Subgraph analysis — fusion
# ---------------------------------------------------------------------------

class TestSubgraphAnalyzer:
    def test_single_relu(self):
        """Single elementwise op → single fused kernel."""
        graph = _make_graph(
            nodes=[
                OpNode("relu", "aten.relu.default",
                       [_ts("x", [1, 256])], [_ts("y", [1, 256])]),
            ],
            graph_inputs=[_ts("x", [1, 256])],
            graph_outputs=[_ts("y", [1, 256])],
        )
        steps = analyze_subgraph(graph)
        fused = [s for s in steps if isinstance(s, FusedKernelStep)]
        assert len(fused) == 1
        assert fused[0].total_elements == 256

    def test_silu_mul_fusion(self):
        """silu → mul should fuse into single kernel."""
        graph = _make_graph(
            nodes=[
                OpNode("silu", "aten.silu.default",
                       [_ts("gate", [1, 256])], [_ts("silu_out", [1, 256])]),
                OpNode("mul", "aten.mul.Tensor",
                       [_ts("silu_out", [1, 256]), _ts("up", [1, 256])],
                       [_ts("y", [1, 256])]),
            ],
            graph_inputs=[_ts("gate", [1, 256]), _ts("up", [1, 256])],
            graph_outputs=[_ts("y", [1, 256])],
        )
        steps = analyze_subgraph(graph)
        fused = [s for s in steps if isinstance(s, FusedKernelStep)]
        assert len(fused) == 1
        assert "silu" in fused[0].source_code.lower() or "hexp" in fused[0].source_code

    def test_blas_breaks_fusion(self):
        """matmul between elementwise ops creates separate fused groups."""
        graph = _make_graph(
            nodes=[
                OpNode("relu1", "aten.relu.default",
                       [_ts("x", [1, 256])], [_ts("relu_out", [1, 256])]),
                OpNode("linear", "aten.linear.default",
                       [_ts("relu_out", [1, 256]), _ts("w", [128, 256])],
                       [_ts("lin_out", [1, 128])]),
                OpNode("relu2", "aten.relu.default",
                       [_ts("lin_out", [1, 128])], [_ts("y", [1, 128])]),
            ],
            graph_inputs=[_ts("x", [1, 256])],
            graph_outputs=[_ts("y", [1, 128])],
            weights=[_ts("w", [128, 256])],
        )
        steps = analyze_subgraph(graph)
        blas = [s for s in steps if isinstance(s, CUBLASStep)]
        fused = [s for s in steps if isinstance(s, FusedKernelStep)]
        assert len(blas) == 1
        assert len(fused) == 2  # relu1 and relu2 are separate chains

    def test_multi_consumer_breaks_fusion(self):
        """When an op output has multiple consumers, it becomes a fusion barrier."""
        graph = _make_graph(
            nodes=[
                OpNode("relu", "aten.relu.default",
                       [_ts("x", [1, 256])], [_ts("relu_out", [1, 256])]),
                OpNode("add", "aten.add.Tensor",
                       [_ts("relu_out", [1, 256]), _ts("relu_out", [1, 256])],
                       [_ts("y", [1, 256])]),
            ],
            graph_inputs=[_ts("x", [1, 256])],
            graph_outputs=[_ts("y", [1, 256])],
        )
        steps = analyze_subgraph(graph)
        fused = [s for s in steps if isinstance(s, FusedKernelStep)]
        # relu_out consumed by both inputs of add → cannot fuse
        assert len(fused) == 2

    def test_reshape_is_alias(self):
        """Reshape produces an AliasStep."""
        graph = _make_graph(
            nodes=[
                OpNode("reshape", "aten.reshape.default",
                       [_ts("x", [2, 128])], [_ts("y", [256])]),
            ],
            graph_inputs=[_ts("x", [2, 128])],
            graph_outputs=[_ts("y", [256])],
        )
        steps = analyze_subgraph(graph)
        aliases = [s for s in steps if isinstance(s, AliasStep)]
        assert len(aliases) == 1

    def test_softmax_produces_reduction(self):
        """Softmax produces a ReductionKernelStep."""
        graph = _make_graph(
            nodes=[
                OpNode("softmax", "aten.softmax.int",
                       [_ts("x", [4, 128])], [_ts("y", [4, 128])],
                       attrs={"dim": -1}),
            ],
            graph_inputs=[_ts("x", [4, 128])],
            graph_outputs=[_ts("y", [4, 128])],
        )
        steps = analyze_subgraph(graph)
        reductions = [s for s in steps if isinstance(s, ReductionKernelStep)]
        assert len(reductions) == 1
        assert reductions[0].kernel_name == "softmax_kernel"

    def test_embedding_produces_special(self):
        """Embedding produces a SpecialKernelStep."""
        graph = _make_graph(
            nodes=[
                OpNode("embed", "aten.embedding.default",
                       [_ts("weight", [1000, 256]), _ts("indices", [4])],
                       [_ts("y", [4, 256])]),
            ],
            graph_inputs=[_ts("indices", [4])],
            graph_outputs=[_ts("y", [4, 256])],
            weights=[_ts("weight", [1000, 256])],
        )
        steps = analyze_subgraph(graph)
        specials = [s for s in steps if isinstance(s, SpecialKernelStep)]
        assert len(specials) == 1
        assert specials[0].kernel_name == "embedding_kernel"

    def test_conv_produces_blas(self):
        """Conv2d produces a CUBLASStep."""
        graph = _make_graph(
            nodes=[
                OpNode("conv", "aten.conv2d.default",
                       [_ts("x", [1, 3, 8, 8]), _ts("w", [16, 3, 3, 3])],
                       [_ts("y", [1, 16, 6, 6])],
                       attrs={"stride": [1, 1], "padding": [0, 0], "groups": 1}),
            ],
            graph_inputs=[_ts("x", [1, 3, 8, 8])],
            graph_outputs=[_ts("y", [1, 16, 6, 6])],
            weights=[_ts("w", [16, 3, 3, 3])],
        )
        steps = analyze_subgraph(graph)
        blas = [s for s in steps if isinstance(s, CUBLASStep)]
        assert len(blas) == 1
        assert blas[0].blas_type == "conv2d"

    def test_rmsnorm_fusion(self):
        """pow → mean → add(eps) → rsqrt → mul(x) → mul(weight) → single RMSNorm kernel."""
        graph = _make_graph(
            nodes=[
                OpNode("pow", "aten.pow.Tensor_Scalar",
                       [_ts("x", [1, 256])], [_ts("pow_out", [1, 256])],
                       attrs={"exponent": 2}),
                OpNode("mean", "aten.mean.dim",
                       [_ts("pow_out", [1, 256])], [_ts("mean_out", [1, 1])],
                       attrs={"dim": [-1]}),
                OpNode("add_eps", "aten.add.Tensor",
                       [_ts("mean_out", [1, 1]), _ts("eps", [1])],
                       [_ts("add_out", [1, 1])],
                       attrs={"other": 1e-6}),
                OpNode("rsqrt", "aten.rsqrt.default",
                       [_ts("add_out", [1, 1])], [_ts("rsqrt_out", [1, 1])]),
                OpNode("mul1", "aten.mul.Tensor",
                       [_ts("rsqrt_out", [1, 1]), _ts("x", [1, 256])],
                       [_ts("mul1_out", [1, 256])]),
                OpNode("mul2", "aten.mul.Tensor",
                       [_ts("mul1_out", [1, 256]), _ts("w", [256])],
                       [_ts("y", [1, 256])]),
            ],
            graph_inputs=[_ts("x", [1, 256])],
            graph_outputs=[_ts("y", [1, 256])],
            weights=[_ts("w", [256])],
        )
        steps = analyze_subgraph(graph)
        reductions = [s for s in steps if isinstance(s, ReductionKernelStep)]
        rmsnorm = [s for s in reductions if s.kernel_name == "rmsnorm_kernel"]
        assert len(rmsnorm) == 1
        assert rmsnorm[0].params["cols"] == 256
        # No leftover elementwise steps from the fused chain
        fused = [s for s in steps if isinstance(s, FusedKernelStep)]
        assert len(fused) == 0

    def test_masked_softmax_fusion(self):
        """add(scores, mask) → softmax → single masked_softmax kernel."""
        graph = _make_graph(
            nodes=[
                OpNode("add", "aten.add.Tensor",
                       [_ts("scores", [4, 128]), _ts("mask", [4, 128])],
                       [_ts("masked", [4, 128])]),
                OpNode("softmax", "aten.softmax.int",
                       [_ts("masked", [4, 128])], [_ts("y", [4, 128])],
                       attrs={"dim": -1}),
            ],
            graph_inputs=[_ts("scores", [4, 128]), _ts("mask", [4, 128])],
            graph_outputs=[_ts("y", [4, 128])],
        )
        steps = analyze_subgraph(graph)
        reductions = [s for s in steps if isinstance(s, ReductionKernelStep)]
        masked = [s for s in reductions if s.kernel_name == "masked_softmax_kernel"]
        assert len(masked) == 1
        assert masked[0].params["rows"] == 4
        assert masked[0].params["cols"] == 128


# ---------------------------------------------------------------------------
# 4. Codegen
# ---------------------------------------------------------------------------

class TestCUDACodegen:
    def test_fusible_check(self):
        assert is_fusible_elementwise("aten.relu.default")
        assert is_fusible_elementwise("aten.silu.default")
        assert is_fusible_elementwise("aten.add.Tensor")
        assert not is_fusible_elementwise("aten.matmul.default")

    def test_generate_relu_kernel(self):
        node = OpNode("relu", "aten.relu.default",
                       [_ts("x", [256])], [_ts("y", [256])])
        source = generate_fused_kernel("fused_relu", [node], ["x"], "y")
        assert "fused_relu" in source
        assert "__global__" in source
        assert "__hmax" in source
        assert "cuda_fp16.h" in source

    def test_generate_silu_mul_kernel(self):
        nodes = [
            OpNode("silu", "aten.silu.default",
                   [_ts("gate", [256])], [_ts("silu_out", [256])]),
            OpNode("mul", "aten.mul.Tensor",
                   [_ts("silu_out", [256]), _ts("up", [256])],
                   [_ts("y", [256])]),
        ]
        source = generate_fused_kernel("fused_silu_mul", nodes, ["gate", "up"], "y")
        assert "fused_silu_mul" in source
        assert "hexp" in source  # silu expression
        assert "*" in source  # mul


# ---------------------------------------------------------------------------
# 5. Buffer planner
# ---------------------------------------------------------------------------

class TestBufferPlanner:
    def test_intermediate_buffer_allocated(self):
        """Intermediate buffer (not input/output/weight) should be allocated."""
        from cuda_compiler import compile_subgraph

        ir_dict = {
            "model_name": "test",
            "graph_inputs": [{"name": "x", "shape": [1, 256], "dtype": "float32"}],
            "graph_outputs": [{"name": "y", "shape": [1, 128], "dtype": "float32"}],
            "weights": [{"name": "w", "shape": [128, 256], "dtype": "float32"}],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "relu", "op_type": "aten.relu.default",
                    "inputs": [{"name": "x", "shape": [1, 256], "dtype": "float32"}],
                    "outputs": [{"name": "relu_out", "shape": [1, 256], "dtype": "float32"}],
                    "attrs": {},
                },
                {
                    "name": "linear", "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "relu_out", "shape": [1, 256], "dtype": "float32"},
                        {"name": "w", "shape": [128, 256], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "y", "shape": [1, 128], "dtype": "float32"}],
                    "attrs": {},
                },
            ],
        }
        program = compile_subgraph(ir_dict)
        assert isinstance(program, CUDAProgram)
        assert len(program.steps) >= 2
        # Should have buffer allocations for intermediate results
        alloc_names = {a.name for a in program.buffer_allocations}
        assert len(alloc_names) >= 1


# ---------------------------------------------------------------------------
# 6. compile_subgraph() integration
# ---------------------------------------------------------------------------

class TestCompileSubgraph:
    def test_resnet_basic_block(self):
        """conv → relu → conv → add → relu (ResNet basic block)."""
        from cuda_compiler import compile_subgraph

        ir_dict = {
            "model_name": "basic_block",
            "graph_inputs": [{"name": "x", "shape": [1, 64, 56, 56], "dtype": "float32"}],
            "graph_outputs": [{"name": "y", "shape": [1, 64, 56, 56], "dtype": "float32"}],
            "weights": [
                {"name": "w1", "shape": [64, 64, 3, 3], "dtype": "float32"},
                {"name": "w2", "shape": [64, 64, 3, 3], "dtype": "float32"},
            ],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "conv1", "op_type": "aten.conv2d.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 64, 56, 56], "dtype": "float32"},
                        {"name": "w1", "shape": [64, 64, 3, 3], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "conv1_out", "shape": [1, 64, 56, 56], "dtype": "float32"}],
                    "attrs": {"stride": [1, 1], "padding": [1, 1], "groups": 1},
                },
                {
                    "name": "relu1", "op_type": "aten.relu.default",
                    "inputs": [{"name": "conv1_out", "shape": [1, 64, 56, 56], "dtype": "float32"}],
                    "outputs": [{"name": "relu1_out", "shape": [1, 64, 56, 56], "dtype": "float32"}],
                    "attrs": {},
                },
                {
                    "name": "conv2", "op_type": "aten.conv2d.default",
                    "inputs": [
                        {"name": "relu1_out", "shape": [1, 64, 56, 56], "dtype": "float32"},
                        {"name": "w2", "shape": [64, 64, 3, 3], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "conv2_out", "shape": [1, 64, 56, 56], "dtype": "float32"}],
                    "attrs": {"stride": [1, 1], "padding": [1, 1], "groups": 1},
                },
                {
                    "name": "add", "op_type": "aten.add.Tensor",
                    "inputs": [
                        {"name": "conv2_out", "shape": [1, 64, 56, 56], "dtype": "float32"},
                        {"name": "x", "shape": [1, 64, 56, 56], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "add_out", "shape": [1, 64, 56, 56], "dtype": "float32"}],
                    "attrs": {},
                },
                {
                    "name": "relu2", "op_type": "aten.relu.default",
                    "inputs": [{"name": "add_out", "shape": [1, 64, 56, 56], "dtype": "float32"}],
                    "outputs": [{"name": "y", "shape": [1, 64, 56, 56], "dtype": "float32"}],
                    "attrs": {},
                },
            ],
        }
        program = compile_subgraph(ir_dict)
        assert isinstance(program, CUDAProgram)

        blas = [s for s in program.steps if isinstance(s, CUBLASStep)]
        fused = [s for s in program.steps if isinstance(s, FusedKernelStep)]
        assert len(blas) == 2  # two conv2d
        # relu1 is a single fused kernel, add+relu2 fuse together
        assert len(fused) >= 1

    def test_qwen_mlp_pattern(self):
        """linear → silu → mul → linear (Qwen GatedMLP)."""
        from cuda_compiler import compile_subgraph

        ir_dict = {
            "model_name": "qwen_mlp",
            "graph_inputs": [{"name": "x", "shape": [1, 1536], "dtype": "float32"}],
            "graph_outputs": [{"name": "y", "shape": [1, 1536], "dtype": "float32"}],
            "weights": [
                {"name": "gate_w", "shape": [8960, 1536], "dtype": "float32"},
                {"name": "up_w", "shape": [8960, 1536], "dtype": "float32"},
                {"name": "down_w", "shape": [1536, 8960], "dtype": "float32"},
            ],
            "weight_name_mapping": {},
            "nodes": [
                {
                    "name": "gate_proj", "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 1536], "dtype": "float32"},
                        {"name": "gate_w", "shape": [8960, 1536], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "gate_out", "shape": [1, 8960], "dtype": "float32"}],
                    "attrs": {},
                },
                {
                    "name": "up_proj", "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "x", "shape": [1, 1536], "dtype": "float32"},
                        {"name": "up_w", "shape": [8960, 1536], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "up_out", "shape": [1, 8960], "dtype": "float32"}],
                    "attrs": {},
                },
                {
                    "name": "silu", "op_type": "aten.silu.default",
                    "inputs": [{"name": "gate_out", "shape": [1, 8960], "dtype": "float32"}],
                    "outputs": [{"name": "silu_out", "shape": [1, 8960], "dtype": "float32"}],
                    "attrs": {},
                },
                {
                    "name": "mul", "op_type": "aten.mul.Tensor",
                    "inputs": [
                        {"name": "silu_out", "shape": [1, 8960], "dtype": "float32"},
                        {"name": "up_out", "shape": [1, 8960], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "mul_out", "shape": [1, 8960], "dtype": "float32"}],
                    "attrs": {},
                },
                {
                    "name": "down_proj", "op_type": "aten.linear.default",
                    "inputs": [
                        {"name": "mul_out", "shape": [1, 8960], "dtype": "float32"},
                        {"name": "down_w", "shape": [1536, 8960], "dtype": "float32"},
                    ],
                    "outputs": [{"name": "y", "shape": [1, 1536], "dtype": "float32"}],
                    "attrs": {},
                },
            ],
        }
        program = compile_subgraph(ir_dict)

        blas = [s for s in program.steps if isinstance(s, CUBLASStep)]
        fused = [s for s in program.steps if isinstance(s, FusedKernelStep)]
        assert len(blas) == 3  # gate_proj, up_proj, down_proj
        # silu + mul should fuse into one kernel
        assert len(fused) >= 1
        # Check the fused kernel contains silu
        silu_fused = [f for f in fused if "hexp" in f.source_code]
        assert len(silu_fused) == 1
