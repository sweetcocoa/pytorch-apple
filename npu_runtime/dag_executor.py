"""DAG executor: orchestrates mixed NPU + CPU fallback execution.

Runs a PartitionPlan by dispatching NPU partitions to the Backend,
CPU partitions to torch_ir's executor, and transfer ops via the
Backend's upload/download methods.
"""

from __future__ import annotations

import numpy as np

import npu_compiler
from npu_compiler.compiled_program import CompiledProgram
from npu_compiler.partitioner import Partition, PartitionPlan, TransferOp
from npu_runtime.backend import Backend, DeviceBuffer
from npu_runtime.cpu_fallback import execute_cpu_partition
from npu_runtime.weight_loader import load_weights


class DAGExecutor:
    """Execute a partitioned graph across NPU and CPU.

    Uses the Backend ABC for hardware-agnostic NPU execution and
    torch_ir's ATen fallback for CPU partitions.
    """

    def __init__(self, plan: PartitionPlan, backend: Backend):
        self.plan = plan
        self.backend = backend
        self.npu_programs: dict[int, CompiledProgram] = {}
        self._npu_executors: dict[int, object] = {}
        self._npu_weight_cache: dict[int, dict[str, DeviceBuffer]] = {}
        self._cpu_weights: dict[str, np.ndarray] | None = None

        for step in plan.steps:
            if isinstance(step, Partition) and step.target == "npu":
                sub_ir_dict = _build_sub_ir_dict(step, plan.original_ir)
                program = npu_compiler.compile(sub_ir_dict)
                self.npu_programs[step.partition_id] = program
                self._npu_executors[step.partition_id] = backend.create_executor(program)

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Pre-load and cache weights for all NPU partitions."""
        self._cpu_weights = weights
        self._npu_weight_cache.clear()
        for step in self.plan.steps:
            if isinstance(step, Partition) and step.target == "npu":
                program = self.npu_programs[step.partition_id]
                self._npu_weight_cache[step.partition_id] = load_weights(
                    weights,
                    program,
                    self.backend.device,
                )

    def execute(
        self,
        inputs: dict[str, np.ndarray],
        weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the full partitioned graph.

        Args:
            inputs: Graph input name -> numpy array.
            weights: If None, uses cached weights from load_weights().

        Returns:
            Graph output name -> numpy array.
        """
        if weights is not None and not self._npu_weight_cache:
            self.load_weights(weights)
        elif weights is not None:
            self._cpu_weights = weights

        cpu_weights = self._cpu_weights or weights or {}

        tensor_pool: dict[str, np.ndarray | DeviceBuffer] = {}
        tensor_pool.update(inputs)

        npu_weight_cache = self._npu_weight_cache

        for step in self.plan.steps:
            if isinstance(step, TransferOp):
                self._execute_transfer(step, tensor_pool)
            elif isinstance(step, Partition):
                if step.target == "npu":
                    self._execute_npu_partition(
                        step,
                        tensor_pool,
                        npu_weight_cache[step.partition_id],
                    )
                else:
                    self._execute_cpu_partition(step, tensor_pool, cpu_weights)

        graph_outputs = self.plan.original_ir.get("graph_outputs", [])
        result: dict[str, np.ndarray] = {}
        for out in graph_outputs:
            name = out["name"]
            val = tensor_pool[name]
            if isinstance(val, np.ndarray):
                result[name] = val
            else:
                result[name] = val.to_numpy()
        return result

    def _execute_npu_partition(
        self,
        partition: Partition,
        tensor_pool: dict,
        npu_weights: dict[str, DeviceBuffer],
    ) -> None:
        program = self.npu_programs[partition.partition_id]
        executor = self._npu_executors[partition.partition_id]

        npu_inputs: dict[str, DeviceBuffer] = {}
        for spec in program.input_specs:
            name = spec.name
            val = tensor_pool.get(name)
            if val is None:
                continue
            if isinstance(val, np.ndarray):
                npu_inputs[name] = self.backend.allocate_buffer(
                    val,
                    alloc_shape=spec.alloc_shape,
                    spec=spec,
                )
            else:
                npu_inputs[name] = val

        output_buffers = executor.run(npu_inputs, npu_weights)
        for name, buf in output_buffers.items():
            tensor_pool[name] = buf

    def _execute_cpu_partition(
        self,
        partition: Partition,
        tensor_pool: dict,
        weights: dict[str, np.ndarray],
    ) -> None:
        # Preserve storage dtype (e.g., bfloat16) instead of converting to float32
        cpu_pool: dict[str, np.ndarray] = {}
        for name, val in tensor_pool.items():
            if isinstance(val, np.ndarray):
                cpu_pool[name] = val
            else:
                cpu_pool[name] = val.to_numpy(dtype=val.dtype)

        result = execute_cpu_partition(
            partition_nodes=partition.nodes,
            tensor_pool=cpu_pool,
            weights=weights,
            weight_name_mapping=self.plan.original_ir.get("weight_name_mapping"),
        )
        tensor_pool.update(result)

    def _execute_transfer(self, op: TransferOp, pool: dict) -> None:
        for name in op.tensor_names:
            val = pool.get(name)
            if val is None:
                raise RuntimeError(
                    f"Transfer op expected tensor '{name}' but it was not found in pool. "
                    f"Available: {sorted(pool.keys())[:20]}"
                )
            if op.direction == "to_cpu":
                if not isinstance(val, np.ndarray):
                    pool[name] = val.to_numpy(dtype=val.dtype)
            else:  # to_npu
                if isinstance(val, np.ndarray):
                    pool[name] = self.backend.allocate_buffer(val)


def _build_sub_ir_dict(partition: Partition, original_ir: dict) -> dict:
    """Build an IR dict for a single NPU partition."""
    produced: set[str] = set()
    for node in partition.nodes:
        for out in node.get("outputs", []):
            produced.add(out["name"])

    external_inputs: dict[str, dict] = {}
    for node in partition.nodes:
        for inp in node.get("inputs", []):
            name = inp["name"]
            if name not in produced and name not in external_inputs:
                external_inputs[name] = inp

    orig_weight_map = original_ir.get("weight_name_mapping", {})
    orig_weight_sd_keys = {w["name"] for w in original_ir.get("weights", [])}

    sub_graph_inputs = []
    sub_weights = []
    sub_wnm: dict[str, str] = {}

    for name, meta in external_inputs.items():
        is_weight = False

        if name in orig_weight_map:
            sd_key = orig_weight_map[name]
            sub_weights.append({"name": sd_key, "shape": meta["shape"], "dtype": meta.get("dtype", "float32")})
            sub_wnm[name] = sd_key
            is_weight = True
        elif name in orig_weight_sd_keys:
            sub_weights.append({"name": name, "shape": meta["shape"], "dtype": meta.get("dtype", "float32")})
            is_weight = True

        if not is_weight:
            sub_graph_inputs.append(
                {"name": meta["name"], "shape": meta["shape"], "dtype": meta.get("dtype", "float32")}
            )

    # Include original graph inputs not directly referenced by nodes but
    # potentially needed by the compiler (kernel fusion may introduce
    # implicit buffer references like cache_position for fused attention).
    sub_gi_names = {g["name"] for g in sub_graph_inputs}
    for gi in original_ir.get("graph_inputs", []):
        if gi["name"] not in sub_gi_names and gi["name"] not in produced:
            sub_graph_inputs.append({"name": gi["name"], "shape": gi["shape"], "dtype": gi.get("dtype", "float32")})

    orig_graph_output_names = {o["name"] for o in original_ir.get("graph_outputs", [])}
    needed_output_names = orig_graph_output_names | set(partition.output_names)

    produced_outputs: dict[str, dict] = {}
    for node in partition.nodes:
        for out in node.get("outputs", []):
            produced_outputs[out["name"]] = out

    sub_graph_outputs = []
    for name in sorted(needed_output_names & set(produced_outputs.keys())):
        out = produced_outputs[name]
        sub_graph_outputs.append(
            {
                "name": out["name"],
                "shape": out["shape"],
                "dtype": out.get("dtype", "float32"),
                "producer_node": out.get("producer_node"),
                "producer_output_idx": out.get("producer_output_idx", 0),
            }
        )

    return {
        "model_name": original_ir.get("model_name", "sub_partition"),
        "graph_inputs": sub_graph_inputs,
        "graph_outputs": sub_graph_outputs,
        "weights": sub_weights,
        "weight_name_mapping": sub_wnm,
        "nodes": partition.nodes,
        "constants": original_ir.get("constants", {}),
    }
