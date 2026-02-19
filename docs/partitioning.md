# Graph Partitioning

## Overview

Not all ops in a model can run on the NPU. When the IR contains unsupported ops, the graph is **partitioned** into NPU and CPU segments that execute alternately. Transfer ops move tensors between devices at partition boundaries.

```
IR → op_support (tag each node) → partitioner (group + transfer ops) → DAGExecutor
     │                              │                                     │
     ├─ NPU partition ─────────────→ compile(dict) → Backend.execute()    │
     ├─ TransferOp ────────────────→ allocate_buffer / to_numpy           │
     └─ CPU partition ─────────────→ torch_ir IRExecutor (ATen fallback)  │
```

## Pipeline

### 1. Op Support Check

`npu_compiler.op_support.is_op_supported(op_type)` checks whether an op can run on the NPU. The support table mirrors `codegen.HANDLED_OPS` (50+ ops including zero-cost aliases). Ops not in this table are routed to CPU fallback via the partition path.

### 2. Partitioning

`npu_compiler.partitioner.partition(ir_dict, is_supported_fn)` performs contiguous grouping:

1. **Tag** each node as `npu` or `cpu`
2. **Group** consecutive same-target nodes into `Partition` objects
3. **Compute boundary I/O** using producer_node tracking
4. **Insert `TransferOp`** at device transitions

The `is_supported_fn` callback accepts `(op_type, attrs)` and returns `bool`. You can wrap `is_op_supported` to add custom overrides (e.g., force specific ops to CPU for debugging):

```python
def support_fn(op_type, _attrs=None):
    if op_type in force_cpu_ops:
        return False
    return is_op_supported(op_type)
```

**Example** (7-node graph):
```
Node:     [conv] [relu] [matmul] [add] [relu2] [concat] [softmax]
Target:    npu    npu     npu     npu    npu      cpu      npu

→ Step 0: Partition(npu, [conv, relu, matmul, add, relu2])
  Step 1: TransferOp(to_cpu, [relu2_output])
  Step 2: Partition(cpu, [concat])
  Step 3: TransferOp(to_npu, [concat_output])
  Step 4: Partition(npu, [softmax])
```

### 3. DAG Execution

`npu_runtime.DAGExecutor` orchestrates the plan:

- **NPU partitions**: Each NPU partition is compiled via `npu_compiler.compile(sub_ir_dict)` at `__init__` time (compile-once, run-many). Executed via `Backend.create_executor()` → `executor.run()`.
- **CPU partitions**: Executed via `torch_ir.IRExecutor` (schema-based ATen fallback for any ATen op).
- **Transfer ops**: `backend.allocate_buffer(numpy_array)` for to_npu; `device_buffer.to_numpy()` for to_cpu. bfloat16 dtype is preserved through transfers.

Call `dag.load_weights(weights_dict)` before `dag.execute()` to pre-cache NPU weight buffers (uploaded once, reused across runs).

### 4. bfloat16 Handling

numpy has no native bfloat16. The pipeline uses `ml_dtypes.bfloat16`:
- `torch.bfloat16` → `.view(torch.uint16)` → `.numpy()` → `.view(ml_dtypes.bfloat16)`
- Restore: `arr.view(np.uint16)` → `torch.from_numpy().view(torch.bfloat16)`
- dtype preserved through NPU↔CPU transfers (no lossy float32 conversion)

## API

```python
import json
from npu_compiler import partition, is_op_supported
from npu_runtime import DAGExecutor, MetalBackend

# 1. Load IR
ir_dict = json.load(open("model.json"))

# 2. Partition
plan = partition(ir_dict, is_op_supported)

# 3. Compile + load weights
backend = MetalBackend()
dag = DAGExecutor(plan, backend)
dag.load_weights(weights_dict)

# 4. Execute
result = dag.execute(inputs={"x": input_array})
```

## Data Structures

| Class | Module | Description |
|-------|--------|-------------|
| `Partition` | `npu_compiler.partitioner` | Contiguous group of nodes on same device |
| `TransferOp` | `npu_compiler.partitioner` | Device-to-device data transfer |
| `PartitionPlan` | `npu_compiler.partitioner` | Ordered steps (partitions + transfers) |
| `DAGExecutor` | `npu_runtime.dag_executor` | Executes a PartitionPlan |

## NPU Framework Comparison

All major NPU/accelerator frameworks follow the same 4-step pattern:
`capability_check() → partition() → compile_subgraphs() → orchestrate_execution()`

| System | Compile Unit | Op Support | Fallback | Notes |
|--------|:------------:|:----------:|:--------:|-------|
| **TensorRT** | Subgraph → engine | Layer support matrix (dtype) | CUDA/CPU | FP16/INT8 auto-quantize. `INetworkDefinition` subgraph split |
| **XLA (TPU/GPU)** | HLO IR → plan | HLO op set | CPU fallback | JIT compile. `send`/`recv` for device transfer |
| **ExecuTorch** | `preprocess()→blob` | Partitioner tagging | `call_delegate` + CPU | Mobile/edge. `to_backend()` delegate pattern |
| **ONNX Runtime EP** | EP subgraph | Priority-ordered providers | Next EP fallback | `GetCapability()` per EP. Priority-ordered assignment |
| **Qualcomm QNN** | Graph → context binary | Op support table (per SoC) | CPU delegation | Hexagon DSP. `QnnBackend::IsNodeSupported()` |
| **AWS Neuron** | Graph → NEFF | Graph partitioner | CPU fallback | `torch_neuronx.trace()` auto-partitions. XLA-based |
| **Apple ANE** | CoreML subgraph | Op compatibility table | CPU/GPU fallback | `MLComputeUnits`. Auto ANE/GPU/CPU distribution |
| **Intel OpenVINO** | Graph → blob | Plugin op support | HETERO plugin | `HETERO:GPU,CPU` priority format |
| **pytorch-apple** | Sub-IR → npubin | `is_op_supported()` | torch_ir executor | Contiguous grouping. Backend ABC |
