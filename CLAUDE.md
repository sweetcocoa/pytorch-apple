# CLAUDE.md

## Project Overview

NPU compiler/runtime backend that compiles PyTorch IR (from torch_to_ir) to Metal compute shaders, simulating NPU execution on Mac M4 Pro GPU. AOT compilation with weight/buffer separation.

## Commands

```bash
# Install dependencies
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_kernels.py -v

# Lint
uv run ruff check npu_compiler/ npu_runtime/ tests/
```

## Architecture

```
torch_to_ir IR JSON ─┬─→ npu_compiler.compile() → CompiledProgram (.npubin) → Executor → Metal GPU
                     └─→ partition() → PartitionPlan → DAGExecutor → NPU + CPU mixed execution
```

### npu_compiler/ (offline)
- **ir_reader.py** — loads torch_to_ir IR JSON into internal graph
- **constraint_checker.py** — validates NPU constraints (static shape, channel alignment)
- **graph_optimizer.py** — BN folding, op fusion, layout optimization
- **fusion_patterns.py** — pattern matching for fused kernels
- **codegen.py** — orchestration (generate_execution_plan) + re-exports
- **codegen_core.py** — data classes (KernelCall, ExecutionPlan), CodegenTarget ABC
- **codegen_ops.py** — single-op codegen (ATen op → KernelCall), HANDLED_OPS
- **codegen_fused.py** — fused kernel codegen (registry + handlers)
- **compiled_program.py** — CompiledProgram serialization (.npubin)
- **op_support.py** — op support table (`is_op_supported()`, `get_supported_ops()`)
- **partitioner.py** — graph partitioning (`partition()`, `Partition`, `TransferOp`, `PartitionPlan`)

### npu_runtime/ (online)
- **backend.py** — Backend ABC + DeviceBuffer abstraction (hardware-agnostic)
- **metal_backend.py** — MetalBackend implementation
- **device.py** — Metal device management (pyobjc)
- **buffer.py** — NPU buffer (from_numpy, to_numpy, zeros)
- **executor.py** — command buffer batching execution (single program)
- **dag_executor.py** — DAGExecutor (mixed NPU + CPU partition execution)
- **cpu_fallback.py** — CPU fallback via torch_ir.IRExecutor
- **weight_loader.py** — safetensors → NPU buffer with recipe-based transform
- **profiler.py** — kernel/transfer/total time measurement

### metal_kernels/ — Metal Compute Shaders
- **matmul.metal** — tiled matmul (TILE=16), vec matmul (VEC_TPG=256 parallel K-reduction), batched matmul
- **conv_bn_relu.metal** — Conv2d with optional BN folding + ReLU fusion
- **add_relu.metal** — element-wise add with optional ReLU fusion
- **elementwise.metal** — basic add/relu, 4D depadding
- **elementwise_extended.metal** — silu, mul, div, neg, pow, rsqrt, cos, sin, scalar ops
- **softmax.metal** — numerically stable softmax (max subtraction)
- **embedding.metal** — token embedding lookup
- **rope.metal** — rotary position embedding (2D dispatch)
- **tensor_ops.metal** — transpose, cat, slice, expand
- **index_copy.metal** — KV cache update (in-place by position)
- **pool.metal** — max pool, adaptive average pool
- **common.metal** — NCHW indexing utilities

## Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| ResNet-18/50 | Full E2E | Top-5 agreement with PyTorch CPU |
| Qwen2.5-1.5B | Prefill + Decode | BFloat16, KV cache via index_copy |

## NPU Constraints

- Static shape only (compile-time fixed)
- Channel axis: multiple of 64 (64-byte alignment, 32 FP16 elements)
- FP16/BF16 compute types (FP32 auto-converted)
- No in-place operations (normalized to out-of-place at codegen)
- Fusion patterns: Conv+BN+ReLU, Add+ReLU

## Supported Ops (50+)

- **CNN**: conv2d, batch_norm, relu, max_pool2d, adaptive_avg_pool2d
- **Linear**: matmul (tiled + vec), linear, bmm (batched + notrans)
- **Element-wise**: add, mul, div, neg, silu, pow, rsqrt, cos, sin, relu
- **Tensor**: embedding, transpose, cat, slice, expand, index_copy
- **Reduction**: softmax (stable), mean_last_dim
- **Positional**: rope (rotary position embedding)
- **Zero-cost aliases**: reshape, view, alias, detach, to.dtype, getitem, dropout

## Data Flow

### Single program path (all ops NPU-supported)
1. **Compile**: IR JSON → constraint check → noop elimination → BN folding → codegen → CompiledProgram
2. **Weights**: safetensors → load_weights(recipe) → [BN fold + pad + FP16/BF16] → NPU buffer
3. **Inputs**: numpy → from_numpy(data, spec) → [cast + pad] → NPU buffer
4. **Execute**: Executor batches all kernels into single Metal command buffer → GPU

### Partition path (mixed NPU + CPU)
1. **Partition**: IR JSON → `partition(ir_dict, is_op_supported)` → PartitionPlan
2. **Compile**: DAGExecutor compiles each NPU partition via `npu_compiler.compile(sub_ir_dict)` at init
3. **Weights**: `dag.load_weights(weights_dict)` pre-caches NPU weight buffers
4. **Execute**: DAGExecutor runs NPU partitions on Metal, CPU partitions via torch_ir, transfers at boundaries

## Performance Notes

- Tiled matmul (TILE=16) with shared memory for M>1 (prefill)
- Vec matmul (256-thread parallel K-reduction) for M=1 (decode)
- MPS acceleration for float16 matmul operations
- Single command encoder per run (minimizes Metal overhead)
- Pre-packed parameter buffers (struct.pack at init, not per-run)

## Style

- Python 3.11+, type hints, @dataclass for data
- Line length: 120 (ruff)
- Documentation: Korean, Code: English
- Follow Karpathy Guidelines (simplicity first, surgical changes)

## Agent Workflow Principles

When using Claude Code Task tool for specialized subagents, apply these domain-specific principles:

### Compiler Agent (codegen, fusion, layout)
- **Backend abstraction**: All codegen goes through `CodegenTarget` ABC. New backends (CUDA, FPGA, NPU ASIC) implement this interface — never hardcode Metal-specific logic outside `MetalCodegenTarget`.
- **DMA awareness**: Current Metal backend uses shared memory (no explicit DMA). But `ExecutionPlan` must be extensible for backends with separate device memory — future: add `TransferCall` nodes between host↔device, inter-partition data movement costs in `PartitionPlan`.
- **Artifact format**: `.npubin` (msgpack) must be both efficient and human-inspectable. Include: kernel list, buffer layout, weight mapping, compute dtype. Use `compiled_program.py:describe()` for text dump.
- **Verify**: `uv run pytest tests/test_compiler.py tests/test_fusion_correctness.py tests/test_extensibility.py -v`

### Runtime Agent (executor, buffer, device)
- **Init vs exec separation**: ALL expensive work (shader compilation, param packing, buffer allocation, MPS object creation) happens in `Executor.__init__()`. The `run()` hot path must only do: buffer lookup → encode → dispatch → commit. No allocations, no struct.pack, no pipeline lookups in `run()`.
- **Backend abstraction**: `DispatchStrategy` ABC for grid/threadgroup computation. `Backend` ABC + `DeviceBuffer` ABC for hardware-agnostic execution. Metal-specific code lives only in `metal_backend.py`, `device.py`, `_mps_accel.py`.
- **Verify**: `uv run pytest tests/test_runtime.py tests/test_kernels.py tests/test_dag_executor.py -v`

### Metal Shader Agent (*.metal files)
- **Portability**: Use `compute_t` typedef (not raw `half`/`bfloat`). `#ifdef USE_BFLOAT` for dtype switching. Param structs must match `Executor._PARAM_SPECS` exactly.
- **Verify**: `uv run pytest tests/test_kernels.py tests/test_extended_kernels.py -v`

### E2E Validation Agent
- **Correctness first**: Pearson r > 0.99 vs CPU reference. Top-5 token agreement. Argmax match.
- **Regression guard**: Run full suite after any change: `uv run pytest tests/ -v`
- **Bisect strategy**: If output is wrong, disable all fusions → verify → re-enable one by one.
