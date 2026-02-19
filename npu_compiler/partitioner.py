"""Graph partitioner: splits IR into NPU and CPU partitions.

The IR nodes are already topologically sorted (guaranteed by torch.export).
The partitioner assigns each node to NPU or CPU based on op support,
groups consecutive same-target nodes into partitions, and inserts
TransferOp at device boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal


@dataclass
class Partition:
    """A contiguous group of IR nodes assigned to the same device."""

    partition_id: int
    target: Literal["npu", "cpu"]
    nodes: list[dict]
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)


@dataclass
class TransferOp:
    """Represents a device-to-device data transfer at a partition boundary."""

    tensor_names: list[str]
    direction: Literal["to_cpu", "to_npu"]


@dataclass
class PartitionPlan:
    """Complete execution plan: partitions interleaved with transfer ops."""

    steps: list[Partition | TransferOp]
    original_ir: dict


def partition(ir_dict: dict, is_supported_fn: Callable[[str, dict | None], bool]) -> PartitionPlan:
    """Partition an IR graph into NPU/CPU segments with transfer ops.

    Args:
        ir_dict: Full IR dict (as loaded from JSON or constructed in-memory).
        is_supported_fn: Callable(op_type, attrs) -> bool.

    Returns:
        PartitionPlan with ordered steps (Partition and TransferOp).
    """
    nodes = ir_dict.get("nodes", [])
    if not nodes:
        return PartitionPlan(steps=[], original_ir=ir_dict)

    tagged: list[tuple[Literal["npu", "cpu"], dict]] = []
    for node in nodes:
        target: Literal["npu", "cpu"] = "npu" if is_supported_fn(node["op_type"], node.get("attrs")) else "cpu"
        tagged.append((target, node))

    groups: list[tuple[Literal["npu", "cpu"], list[dict]]] = []
    current_target, first_node = tagged[0]
    current_nodes_list = [first_node]
    for target, node in tagged[1:]:
        if target == current_target:
            current_nodes_list.append(node)
        else:
            groups.append((current_target, current_nodes_list))
            current_target = target
            current_nodes_list = [node]
    groups.append((current_target, current_nodes_list))

    partitions = [
        Partition(partition_id=idx, target=target, nodes=group_nodes)
        for idx, (target, group_nodes) in enumerate(groups)
    ]

    produced_by = _compute_boundary_io(partitions, ir_dict)

    steps: list[Partition | TransferOp] = []
    for i, p in enumerate(partitions):
        if i > 0 and partitions[i - 1].target != p.target:
            boundary = _find_boundary_tensors(partitions[:i], p, produced_by)
            if boundary:
                direction: Literal["to_cpu", "to_npu"] = "to_cpu" if p.target == "cpu" else "to_npu"
                steps.append(TransferOp(tensor_names=boundary, direction=direction))
        steps.append(p)

    return PartitionPlan(steps=steps, original_ir=ir_dict)


def _compute_boundary_io(partitions: list[Partition], ir_dict: dict) -> dict[int, set[str]]:
    """Compute input_names and output_names for each partition.

    Returns produced_by dict for reuse by _find_boundary_tensors.
    """
    produced_by: dict[int, set[str]] = {p.partition_id: set() for p in partitions}
    for p in partitions:
        for node in p.nodes:
            for out in node.get("outputs", []):
                produced_by[p.partition_id].add(out["name"])

    graph_output_names = {o["name"] for o in ir_dict.get("graph_outputs", [])}

    # Build reverse index: tensor_name -> partition_id that produced it
    tensor_producer: dict[str, int] = {}
    for pid, names in produced_by.items():
        for name in names:
            tensor_producer[name] = pid

    for p in partitions:
        local_produced = produced_by[p.partition_id]
        input_set: set[str] = set()
        for node in p.nodes:
            for inp in node.get("inputs", []):
                if inp["name"] not in local_produced:
                    input_set.add(inp["name"])
        p.input_names = sorted(input_set)

    # O(n) output computation using reverse index
    consumed_outside: dict[int, set[str]] = {p.partition_id: set() for p in partitions}
    for p in partitions:
        for name in p.input_names:
            producer_pid = tensor_producer.get(name)
            if producer_pid is not None and producer_pid != p.partition_id:
                consumed_outside[producer_pid].add(name)

    for p in partitions:
        output_set = consumed_outside[p.partition_id] | (produced_by[p.partition_id] & graph_output_names)
        p.output_names = sorted(output_set)

    return produced_by


def _find_boundary_tensors(
    preceding: list[Partition], current: Partition, produced_by: dict[int, set[str]]
) -> list[str]:
    """Find tensors that cross from preceding partitions into current."""
    all_produced: set[str] = set()
    for p in preceding:
        all_produced |= produced_by[p.partition_id]

    return sorted(name for name in current.input_names if name in all_produced)
