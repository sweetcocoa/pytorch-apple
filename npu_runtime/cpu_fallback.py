"""CPU fallback execution for unsupported ops.

Uses torch_ir's schema-based ATen executor to run ops that the NPU
cannot handle. Tensors flow as numpy arrays (the DAG executor's common
currency), converted to/from torch.Tensor at the CPU partition boundary.

bfloat16 handling:
  numpy has no native bfloat16. We keep tensors as torch.Tensor during
  CPU execution and only convert to numpy at the boundary via uint16
  reinterpret (zero-copy).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_ir import IR, IRExecutor
from torch_ir.ir import OpNode, TensorMeta


def _tensor_meta_from_dict(d: dict) -> TensorMeta:
    """Convert an IR tensor dict to a TensorMeta object."""
    return TensorMeta(
        name=d["name"],
        shape=list(d["shape"]),
        dtype=d.get("dtype", "float32"),
        producer_node=d.get("producer_node"),
        producer_output_idx=d.get("producer_output_idx", 0),
    )


def execute_cpu_partition(
    partition_nodes: list[dict],
    tensor_pool: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    weight_name_mapping: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Execute a CPU partition using torch_ir's ATen fallback executor.

    Args:
        partition_nodes: List of IR node dicts for this CPU partition.
        tensor_pool: Name->ndarray mapping of available tensors (inputs + intermediates).
        weights: Name->ndarray mapping of model weights.
        weight_name_mapping: FX placeholder name -> state_dict key mapping.

    Returns:
        Dict mapping output tensor names -> numpy arrays produced by this partition.
    """
    sub_ir = _build_sub_ir(partition_nodes, tensor_pool, weights, weight_name_mapping)

    torch_weights = {}
    for w in sub_ir.weights:
        name = w.name
        if name in weights:
            torch_weights[name] = _numpy_to_torch(weights[name])
        elif weight_name_mapping:
            sd_key = weight_name_mapping.get(name)
            if sd_key and sd_key in weights:
                torch_weights[name] = _numpy_to_torch(weights[sd_key])

    input_tensors = [_numpy_to_torch(tensor_pool[gi.name]) for gi in sub_ir.graph_inputs]

    executor = IRExecutor(sub_ir, weights=torch_weights)
    outputs = executor.execute(tuple(input_tensors))

    return {out_meta.name: _torch_to_numpy(outputs[i]) for i, out_meta in enumerate(sub_ir.graph_outputs)}


def _build_sub_ir(
    nodes: list[dict],
    tensor_pool: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    weight_name_mapping: dict[str, str] | None,
) -> IR:
    """Build a minimal IR object for the given partition nodes."""
    produced: set[str] = set()
    for node in nodes:
        for out in node.get("outputs", []):
            produced.add(out["name"])

    external_inputs: dict[str, dict] = {}
    for node in nodes:
        for inp in node.get("inputs", []):
            name = inp["name"]
            if name not in produced and name not in external_inputs:
                external_inputs[name] = inp

    weight_names = set(weights.keys())
    if weight_name_mapping:
        weight_names |= set(weight_name_mapping.keys())

    sub_graph_inputs: list[TensorMeta] = []
    sub_weights: list[TensorMeta] = []

    for name, meta in external_inputs.items():
        tm = _tensor_meta_from_dict(meta)
        if name in weight_names:
            sub_weights.append(tm)
        else:
            sub_graph_inputs.append(tm)

    sub_graph_outputs = [_tensor_meta_from_dict(out) for node in nodes for out in node.get("outputs", [])]

    ir_nodes = [
        OpNode(
            name=node["name"],
            op_type=node["op_type"],
            inputs=[_tensor_meta_from_dict(inp) for inp in node.get("inputs", [])],
            outputs=[_tensor_meta_from_dict(out) for out in node.get("outputs", [])],
            attrs=node.get("attrs", {}),
        )
        for node in nodes
    ]

    sub_wnm = {}
    if weight_name_mapping:
        sub_weight_names = {w.name for w in sub_weights}
        sub_wnm = {k: v for k, v in weight_name_mapping.items() if k in sub_weight_names}

    return IR(
        nodes=ir_nodes,
        graph_inputs=sub_graph_inputs,
        graph_outputs=sub_graph_outputs,
        weights=sub_weights,
        weight_name_mapping=sub_wnm,
        model_name="cpu_partition",
    )


def build_cpu_executor(
    partition_nodes: list[dict],
    weights: dict[str, np.ndarray],
    weight_name_mapping: dict[str, str] | None = None,
) -> tuple[IR, IRExecutor, dict[str, torch.Tensor]]:
    """Build and return a reusable CPU executor for a partition.

    Returns:
        Tuple of (sub_ir, executor, torch_weights) that can be cached and reused.
    """
    # Build a dummy tensor_pool with just weight names so _build_sub_ir can
    # classify external inputs correctly.
    sub_ir = _build_sub_ir(partition_nodes, {}, weights, weight_name_mapping)

    torch_weights = {}
    for w in sub_ir.weights:
        name = w.name
        if name in weights:
            torch_weights[name] = _numpy_to_torch(weights[name])
        elif weight_name_mapping:
            sd_key = weight_name_mapping.get(name)
            if sd_key and sd_key in weights:
                torch_weights[name] = _numpy_to_torch(weights[sd_key])

    executor = IRExecutor(sub_ir, weights=torch_weights)
    return sub_ir, executor, torch_weights


def execute_cpu_partition_cached(
    sub_ir: IR,
    executor: IRExecutor,
    tensor_pool: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Execute a cached CPU partition executor with the given tensor pool.

    Args:
        sub_ir: Pre-built IR for this partition.
        executor: Pre-built IRExecutor for this partition.
        tensor_pool: Name->ndarray mapping of available tensors.

    Returns:
        Dict mapping output tensor names -> numpy arrays produced by this partition.
    """
    input_tensors = [_numpy_to_torch(tensor_pool[gi.name]) for gi in sub_ir.graph_inputs]
    outputs = executor.execute(tuple(input_tensors))
    return {out_meta.name: _torch_to_numpy(outputs[i]) for i, out_meta in enumerate(sub_ir.graph_outputs)}


def _numpy_to_torch(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor, handling bfloat16."""
    if arr.dtype == np.uint16:
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.from_numpy(arr.copy())


def _torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array, handling bfloat16."""
    import ml_dtypes  # noqa: F811

    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16).copy()
    return t.numpy().copy()
