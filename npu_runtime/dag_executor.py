"""DAG executor: orchestrates mixed NPU + CPU fallback execution.

Runs a PartitionPlan by dispatching NPU partitions to the Backend,
CPU partitions to torch_ir's executor, and transfer ops via the
Backend's upload/download methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

import npu_compiler
from npu_compiler.compiled_program import CompiledProgram
from npu_compiler.partitioner import Partition, PartitionPlan, TransferOp
from npu_runtime.backend import Backend, DeviceBuffer

from npu_runtime.cpu_fallback import build_cpu_executor, execute_cpu_partition_cached

try:
    from npu_runtime.buffer import NPUBuffer
    from npu_runtime.weight_loader import load_weights
except ImportError:
    NPUBuffer = None  # type: ignore[assignment,misc]
    load_weights = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from torch_ir import IR, IRExecutor


class DAGExecutor:
    """Execute a partitioned graph across NPU and CPU.

    Uses the Backend ABC for hardware-agnostic NPU execution and
    torch_ir's ATen fallback for CPU partitions.
    """

    def __init__(
        self,
        plan: PartitionPlan,
        backend: Backend,
        compile_fn: Callable[[dict], Any] | None = None,
    ):
        self.plan = plan
        self.backend = backend
        self._compile_fn = compile_fn or npu_compiler.compile
        self.npu_programs: dict[int, CompiledProgram] = {}
        self._npu_executors: dict[int, object] = {}
        self._npu_weight_cache: dict[int, dict[str, DeviceBuffer]] = {}
        self._cpu_weights: dict[str, np.ndarray] | None = None

        # CPU partition caches (populated in load_weights)
        self._cpu_irs: dict[int, IR] = {}
        self._cpu_executors: dict[int, IRExecutor] = {}

        # NPU input buffer caches (populated in _init_npu_input_buffers)
        self._npu_input_buffers: dict[int, dict[str, NPUBuffer]] = {}

        # Pre-cache metadata
        self._output_names: list[str] = [o["name"] for o in plan.original_ir.get("graph_outputs", [])]
        self._weight_name_mapping: dict[str, str] | None = plan.original_ir.get("weight_name_mapping")

        # Pre-compute needed input names per CPU partition
        self._cpu_needed_inputs: dict[int, set[str]] = {}

        for step in plan.steps:
            if isinstance(step, Partition) and step.target == "npu":
                sub_ir_dict = _build_sub_ir_dict(step, plan.original_ir)
                program = self._compile_fn(sub_ir_dict)
                self.npu_programs[step.partition_id] = program
                self._npu_executors[step.partition_id] = backend.create_executor(program)
            elif isinstance(step, Partition) and step.target == "cpu":
                needed = set()
                produced: set[str] = set()
                for node in step.nodes:
                    for inp in node.get("inputs", []):
                        needed.add(inp["name"])
                    for out in node.get("outputs", []):
                        produced.add(out["name"])
                # Only external inputs (not produced within the partition)
                self._cpu_needed_inputs[step.partition_id] = needed - produced

        # Build dispatch table for execute() loop
        self._dispatch_table: list[tuple[str, Partition | TransferOp, CompiledProgram | None, object | None]] = []
        for step in plan.steps:
            if isinstance(step, TransferOp):
                self._dispatch_table.append(("transfer", step, None, None))
            elif isinstance(step, Partition) and step.target == "npu":
                self._dispatch_table.append(
                    (
                        "npu",
                        step,
                        self.npu_programs[step.partition_id],
                        self._npu_executors[step.partition_id],
                    )
                )
            elif isinstance(step, Partition):
                self._dispatch_table.append(("cpu", step, None, None))

    def _init_npu_input_buffers(self) -> None:
        """Pre-allocate NPU input buffers for all NPU partitions."""
        for step in self.plan.steps:
            if isinstance(step, Partition) and step.target == "npu":
                program = self.npu_programs[step.partition_id]
                bufs: dict[str, DeviceBuffer] = {}
                for spec in program.input_specs:
                    if NPUBuffer is not None:
                        bufs[spec.name] = NPUBuffer.zeros(
                            tuple(spec.shape),
                            self.backend.device,
                            alloc_shape=tuple(spec.alloc_shape) if spec.alloc_shape else None,
                        )
                    else:
                        bufs[spec.name] = self.backend.allocate_zeros(tuple(spec.shape), dtype=np.dtype(np.float16))
                self._npu_input_buffers[step.partition_id] = bufs

    def _load_weights_for_backend(
        self, weights: dict[str, np.ndarray], program: Any
    ) -> dict[str, DeviceBuffer]:
        """Load weights using Metal weight_loader or generic backend upload."""
        if load_weights is not None:
            return load_weights(weights, program, self.backend.device)

        # Generic path: upload weights directly via backend (CUDA, etc.)
        # Convert to compute dtype (float16 for CUDA kernels)
        compute_dtype = getattr(program, "compute_dtype", "float16")
        np_dtype = np.float16 if compute_dtype in ("float16", "bfloat16") else np.float32

        reverse_map: dict[str, str] = {}
        if hasattr(program, "weight_name_mapping"):
            for placeholder, sd_key in program.weight_name_mapping.items():
                reverse_map[sd_key] = placeholder
        buffers: dict[str, DeviceBuffer] = {}
        for w_spec in program.weight_specs:
            sd_key = w_spec.name
            placeholder = reverse_map.get(sd_key, sd_key)
            if sd_key in weights:
                arr = weights[sd_key]
            elif placeholder in weights:
                arr = weights[placeholder]
            else:
                continue
            # Cast to compute dtype before upload
            if arr.dtype != np_dtype:
                arr = arr.astype(np_dtype)
            buffers[placeholder] = self.backend.allocate_buffer(arr)
        return buffers

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Pre-load and cache weights for all NPU partitions."""
        self._cpu_weights = weights
        self._npu_weight_cache.clear()
        for step in self.plan.steps:
            if isinstance(step, Partition) and step.target == "npu":
                program = self.npu_programs[step.partition_id]
                self._npu_weight_cache[step.partition_id] = self._load_weights_for_backend(
                    weights, program
                )

        # Cache CPU partition executors
        self._cpu_irs.clear()
        self._cpu_executors.clear()
        for step in self.plan.steps:
            if isinstance(step, Partition) and step.target == "cpu":
                sub_ir, executor, _ = build_cpu_executor(
                    step.nodes,
                    weights,
                    self._weight_name_mapping,
                )
                self._cpu_irs[step.partition_id] = sub_ir
                self._cpu_executors[step.partition_id] = executor

        # Pre-allocate NPU input buffers
        if not self._npu_input_buffers:
            self._init_npu_input_buffers()

    def execute(
        self,
        inputs: dict[str, np.ndarray | DeviceBuffer],
        weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray | DeviceBuffer]:
        """Run the full partitioned graph.

        Args:
            inputs: Graph input name -> numpy array or DeviceBuffer.
            weights: If None, uses cached weights from load_weights().

        Returns:
            Graph output name -> numpy array or DeviceBuffer (pass-through).
        """
        if weights is not None and not self._npu_weight_cache:
            self.load_weights(weights)
        elif weights is not None:
            self._cpu_weights = weights

        cpu_weights = self._cpu_weights or weights or {}

        tensor_pool: dict[str, np.ndarray | DeviceBuffer] = {}
        tensor_pool.update(inputs)

        npu_weight_cache = self._npu_weight_cache

        for kind, step, program, executor in self._dispatch_table:
            if kind == "transfer":
                self._execute_transfer(step, tensor_pool)
            elif kind == "npu":
                self._execute_npu_partition(
                    step,
                    tensor_pool,
                    npu_weight_cache[step.partition_id],
                    program,
                    executor,
                )
            else:
                self._execute_cpu_partition(step, tensor_pool, cpu_weights)

        # Return values as-is (DeviceBuffer or numpy) — no forced conversion
        result: dict[str, np.ndarray | DeviceBuffer] = {}
        for name in self._output_names:
            result[name] = tensor_pool[name]
        return result

    def _execute_npu_partition(
        self,
        partition: Partition,
        tensor_pool: dict,
        npu_weights: dict[str, DeviceBuffer],
        program: CompiledProgram,
        executor: object,
    ) -> None:
        pre_alloc = self._npu_input_buffers.get(partition.partition_id)

        npu_inputs: dict[str, DeviceBuffer] = {}
        for spec in program.input_specs:
            name = spec.name
            val = tensor_pool.get(name)
            if val is None:
                continue
            if isinstance(val, np.ndarray):
                if pre_alloc and name in pre_alloc:
                    pre_alloc[name].write_from_numpy(val, spec=spec)
                    npu_inputs[name] = pre_alloc[name]
                else:
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
        pid = partition.partition_id

        # Use cached executor if available
        if pid in self._cpu_executors:
            # Only convert needed inputs instead of entire tensor_pool
            needed = self._cpu_needed_inputs.get(pid)
            cpu_pool: dict[str, np.ndarray] = {}
            if needed:
                for name in needed:
                    val = tensor_pool.get(name)
                    if val is None:
                        continue
                    if isinstance(val, np.ndarray):
                        cpu_pool[name] = val
                    else:
                        cpu_pool[name] = val.to_numpy(dtype=val.dtype)

            result = execute_cpu_partition_cached(
                self._cpu_irs[pid],
                self._cpu_executors[pid],
                cpu_pool,
            )
        else:
            # Fallback: selective copy + uncached execution
            from npu_runtime.cpu_fallback import execute_cpu_partition

            needed = self._cpu_needed_inputs.get(pid)
            cpu_pool = {}
            if needed:
                for name in needed:
                    val = tensor_pool.get(name)
                    if val is None:
                        continue
                    if isinstance(val, np.ndarray):
                        cpu_pool[name] = val
                    else:
                        cpu_pool[name] = val.to_numpy(dtype=val.dtype)
            else:
                for name, val in tensor_pool.items():
                    if isinstance(val, np.ndarray):
                        cpu_pool[name] = val
                    else:
                        cpu_pool[name] = val.to_numpy(dtype=val.dtype)

            result = execute_cpu_partition(
                partition_nodes=partition.nodes,
                tensor_pool=cpu_pool,
                weights=weights,
                weight_name_mapping=self._weight_name_mapping,
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
