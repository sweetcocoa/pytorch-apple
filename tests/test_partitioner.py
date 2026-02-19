"""Tests for graph partitioner."""

from npu_compiler.partitioner import Partition, TransferOp, partition


def _make_node(name, op_type, input_names, output_names, attrs=None):
    return {
        "name": name,
        "op_type": op_type,
        "inputs": [{"name": n, "shape": [1, 64], "dtype": "float32"} for n in input_names],
        "outputs": [
            {"name": n, "shape": [1, 64], "dtype": "float32", "producer_node": name, "producer_output_idx": i}
            for i, n in enumerate(output_names)
        ],
        "attrs": attrs or {},
    }


def _make_ir(nodes, graph_input_names=None, graph_output_names=None):
    if graph_input_names is None:
        graph_input_names = ["x"]
    if graph_output_names is None:
        graph_output_names = [o["name"] for o in nodes[-1]["outputs"]]
    return {
        "model_name": "test",
        "graph_inputs": [{"name": n, "shape": [1, 64], "dtype": "float32"} for n in graph_input_names],
        "graph_outputs": [{"name": n, "shape": [1, 64], "dtype": "float32"} for n in graph_output_names],
        "weights": [],
        "weight_name_mapping": {},
        "nodes": nodes,
    }


class TestPartitionAllSupported:
    def test_single_npu_partition(self):
        """All ops supported -> single NPU partition, no transfer ops."""
        nodes = [
            _make_node("conv", "aten.conv2d.default", ["x"], ["conv_out"]),
            _make_node("relu", "aten.relu.default", ["conv_out"], ["relu_out"]),
        ]
        ir = _make_ir(nodes)
        plan = partition(ir, lambda op, _attrs=None: True)

        assert len(plan.steps) == 1
        assert isinstance(plan.steps[0], Partition)
        assert plan.steps[0].target == "npu"
        assert len(plan.steps[0].nodes) == 2

    def test_empty_graph(self):
        ir = _make_ir([], graph_output_names=[])
        ir["nodes"] = []
        plan = partition(ir, lambda op, _attrs=None: True)
        assert len(plan.steps) == 0


class TestPartitionAllUnsupported:
    def test_single_cpu_partition(self):
        """No ops supported -> single CPU partition."""
        nodes = [
            _make_node("gelu", "aten.gelu.default", ["x"], ["gelu_out"]),
            _make_node("tanh", "aten.tanh.default", ["gelu_out"], ["tanh_out"]),
        ]
        ir = _make_ir(nodes)
        plan = partition(ir, lambda op, _attrs=None: False)

        assert len(plan.steps) == 1
        assert isinstance(plan.steps[0], Partition)
        assert plan.steps[0].target == "cpu"
        assert len(plan.steps[0].nodes) == 2


class TestPartitionMixed:
    def test_npu_cpu_npu_split(self):
        """NPU -> CPU -> NPU split with transfer ops."""
        nodes = [
            _make_node("conv", "aten.conv2d.default", ["x"], ["conv_out"]),
            _make_node("relu", "aten.relu.default", ["conv_out"], ["relu_out"]),
            _make_node("gelu", "aten.gelu.default", ["relu_out"], ["gelu_out"]),
            _make_node("linear", "aten.linear.default", ["gelu_out"], ["linear_out"]),
        ]
        ir = _make_ir(nodes)
        supported = {"aten.conv2d.default", "aten.relu.default", "aten.linear.default"}
        plan = partition(ir, lambda op, _attrs=None: op in supported)

        partitions = [s for s in plan.steps if isinstance(s, Partition)]
        transfers = [s for s in plan.steps if isinstance(s, TransferOp)]

        assert len(partitions) == 3
        assert len(transfers) == 2

        assert partitions[0].target == "npu"
        assert len(partitions[0].nodes) == 2

        assert partitions[1].target == "cpu"
        assert len(partitions[1].nodes) == 1

        assert partitions[2].target == "npu"
        assert len(partitions[2].nodes) == 1

        assert transfers[0].direction == "to_cpu"
        assert transfers[1].direction == "to_npu"

    def test_transfer_op_tensor_names(self):
        """Transfer ops carry the correct boundary tensor names."""
        nodes = [
            _make_node("conv", "aten.conv2d.default", ["x"], ["conv_out"]),
            _make_node("gelu", "aten.gelu.default", ["conv_out"], ["gelu_out"]),
            _make_node("relu", "aten.relu.default", ["gelu_out"], ["relu_out"]),
        ]
        ir = _make_ir(nodes)
        supported = {"aten.conv2d.default", "aten.relu.default"}
        plan = partition(ir, lambda op, _attrs=None: op in supported)

        transfers = [s for s in plan.steps if isinstance(s, TransferOp)]
        assert len(transfers) == 2
        assert "conv_out" in transfers[0].tensor_names
        assert transfers[0].direction == "to_cpu"
        assert "gelu_out" in transfers[1].tensor_names
        assert transfers[1].direction == "to_npu"

    def test_consecutive_unsupported_grouped(self):
        """Multiple consecutive unsupported ops form one CPU partition."""
        nodes = [
            _make_node("conv", "aten.conv2d.default", ["x"], ["conv_out"]),
            _make_node("gelu", "aten.gelu.default", ["conv_out"], ["gelu_out"]),
            _make_node("tanh", "aten.tanh.default", ["gelu_out"], ["tanh_out"]),
            _make_node("relu", "aten.relu.default", ["tanh_out"], ["relu_out"]),
        ]
        ir = _make_ir(nodes)
        supported = {"aten.conv2d.default", "aten.relu.default"}
        plan = partition(ir, lambda op, _attrs=None: op in supported)

        partitions = [s for s in plan.steps if isinstance(s, Partition)]
        assert len(partitions) == 3
        assert partitions[1].target == "cpu"
        assert len(partitions[1].nodes) == 2


class TestBoundaryIO:
    def test_partition_input_names(self):
        """Partition inputs include tensors consumed from outside."""
        nodes = [
            _make_node("conv", "aten.conv2d.default", ["x"], ["conv_out"]),
            _make_node("gelu", "aten.gelu.default", ["conv_out"], ["gelu_out"]),
        ]
        ir = _make_ir(nodes)
        supported = {"aten.conv2d.default"}
        plan = partition(ir, lambda op, _attrs=None: op in supported)

        partitions = [s for s in plan.steps if isinstance(s, Partition)]
        cpu_part = [p for p in partitions if p.target == "cpu"][0]
        assert "conv_out" in cpu_part.input_names

    def test_partition_output_names(self):
        """Partition outputs include tensors consumed by later partitions."""
        nodes = [
            _make_node("conv", "aten.conv2d.default", ["x"], ["conv_out"]),
            _make_node("gelu", "aten.gelu.default", ["conv_out"], ["gelu_out"]),
            _make_node("relu", "aten.relu.default", ["gelu_out"], ["relu_out"]),
        ]
        ir = _make_ir(nodes)
        supported = {"aten.conv2d.default", "aten.relu.default"}
        plan = partition(ir, lambda op, _attrs=None: op in supported)

        partitions = [s for s in plan.steps if isinstance(s, Partition)]
        npu_p0 = partitions[0]
        assert "conv_out" in npu_p0.output_names


class TestPartitionPlanStructure:
    def test_step_order_preserved(self):
        """Steps maintain topological order."""
        nodes = [
            _make_node("n0", "aten.conv2d.default", ["x"], ["t0"]),
            _make_node("n1", "aten.gelu.default", ["t0"], ["t1"]),
            _make_node("n2", "aten.relu.default", ["t1"], ["t2"]),
        ]
        ir = _make_ir(nodes)
        supported = {"aten.conv2d.default", "aten.relu.default"}
        plan = partition(ir, lambda op, _attrs=None: op in supported)

        types = [type(s).__name__ for s in plan.steps]
        assert types == ["Partition", "TransferOp", "Partition", "TransferOp", "Partition"]

    def test_partition_ids_sequential(self):
        nodes = [
            _make_node("n0", "aten.conv2d.default", ["x"], ["t0"]),
            _make_node("n1", "aten.gelu.default", ["t0"], ["t1"]),
            _make_node("n2", "aten.relu.default", ["t1"], ["t2"]),
        ]
        ir = _make_ir(nodes)
        supported = {"aten.conv2d.default", "aten.relu.default"}
        plan = partition(ir, lambda op, _attrs=None: op in supported)

        partitions = [s for s in plan.steps if isinstance(s, Partition)]
        ids = [p.partition_id for p in partitions]
        assert ids == [0, 1, 2]

    def test_original_ir_preserved(self):
        nodes = [_make_node("n0", "aten.conv2d.default", ["x"], ["t0"])]
        ir = _make_ir(nodes)
        plan = partition(ir, lambda op, _attrs=None: True)
        assert plan.original_ir is ir
