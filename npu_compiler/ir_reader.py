"""Load torch_to_ir IR JSON into internal graph representation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str = "float32"
    alloc_shape: list[int] | None = None  # physical shape (compiler-decided, None = same as shape)
    transform_steps: list[dict] | None = None  # host→NPU transform pipeline (cast, pad, etc.)
    producer_node: str | None = None
    producer_output_idx: int = 0


@dataclass
class OpNode:
    name: str
    op_type: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    attrs: dict = field(default_factory=dict)


@dataclass
class IRGraph:
    model_name: str
    graph_inputs: list[TensorSpec]
    graph_outputs: list[TensorSpec]
    weights: list[TensorSpec]
    weight_name_mapping: dict[str, str]
    nodes: list[OpNode]
    constants: dict = field(default_factory=dict)

    def get_node_by_name(self, name: str) -> OpNode | None:
        for node in self.nodes:
            if node.name == name:
                return node
        return None


def _parse_tensor_spec(d: dict) -> TensorSpec:
    return TensorSpec(
        name=d["name"],
        shape=d["shape"],
        dtype=d.get("dtype", "float32"),
        alloc_shape=d.get("alloc_shape"),
        transform_steps=d.get("transform_steps"),
        producer_node=d.get("producer_node"),
        producer_output_idx=d.get("producer_output_idx", 0),
    )


def _parse_op_node(d: dict) -> OpNode:
    return OpNode(
        name=d["name"],
        op_type=d["op_type"],
        inputs=[_parse_tensor_spec(i) for i in d["inputs"]],
        outputs=[_parse_tensor_spec(o) for o in d["outputs"]],
        attrs=d.get("attrs", {}),
    )


def load_ir(path: str) -> IRGraph:
    """Load IR JSON file into IRGraph."""
    with open(path) as f:
        data = json.load(f)

    return IRGraph(
        model_name=data.get("model_name", "unknown"),
        graph_inputs=[_parse_tensor_spec(i) for i in data["graph_inputs"]],
        graph_outputs=[_parse_tensor_spec(o) for o in data["graph_outputs"]],
        weights=[_parse_tensor_spec(w) for w in data["weights"]],
        weight_name_mapping=data.get("weight_name_mapping", {}),
        nodes=[_parse_op_node(n) for n in data["nodes"]],
        constants=data.get("constants", {}),
    )


def load_ir_from_dict(data: dict) -> IRGraph:
    """Load IR from a dict (for testing)."""
    return IRGraph(
        model_name=data.get("model_name", "unknown"),
        graph_inputs=[_parse_tensor_spec(i) for i in data["graph_inputs"]],
        graph_outputs=[_parse_tensor_spec(o) for o in data["graph_outputs"]],
        weights=[_parse_tensor_spec(w) for w in data["weights"]],
        weight_name_mapping=data.get("weight_name_mapping", {}),
        nodes=[_parse_op_node(n) for n in data["nodes"]],
        constants=data.get("constants", {}),
    )
