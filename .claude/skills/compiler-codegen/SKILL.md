---
name: compiler-codegen
description: Rules for modifying the NPU compiler codegen pipeline. Use when adding ops, changing kernel mapping, or modifying fusion patterns.
---

# Compiler Codegen Rules

## Module Structure
- `codegen_core.py` — KernelCall, ExecutionPlan, CodegenTarget ABC (shared types)
- `codegen_ops.py` — single-op codegen + HANDLED_OPS
- `codegen_fused.py` — fused kernel codegen registry + handlers
- `codegen.py` — orchestration (generate_execution_plan) + re-exports for backwards compat

## Adding a New Op
1. Add op string to `HANDLED_OPS` in `codegen_ops.py`
2. Add handler in `generate_single_kernel_call()` in `codegen_ops.py` — return `KernelCall` or `None`
3. If op needs params, add struct spec to `Executor._PARAM_SPECS`
4. Add Metal kernel (or reuse existing shader file)
5. Add test in `tests/test_kernels.py` or `test_extended_kernels.py`

## KernelCall Conventions
- `dispatch_type`: `"1d"` (elementwise), `"2d"` (matmul/embedding), `"3d"` (batched matmul), `"none"` (alias)
- `_reshape` kernel name = zero-cost alias (no GPU dispatch, just buffer pointer sharing)
- `_MAX_NDIM=6`: shape/stride arrays padded to 6 elements for fixed Metal struct layout
- `total_threads`: for 1D dispatch only. `grid_width/height/depth`: for 2D/3D dispatch.

## In-place Op Normalization
`_OP_NORMALIZE` dict maps in-place ops to out-of-place: `relu_` → `relu`, `add_` → `add.Tensor`. Applied before codegen. NPU requires separate input/output buffers.

## Binary Op Pattern
- Same-shape → simple elementwise kernel (with 4D channel padding)
- Different-shape → stride-based broadcast kernel (`elementwise_broadcast.metal`), avoids materializing expanded tensor
- Scalar variant (1 input + scalar attr) → `eltwise_*_scalar_kernel`

## Fusion Pattern Rules (fusion_patterns.py)
- Register via `register_fusion_pattern(trigger_op, match_fn)`
- Match function signature: `(node, graph, consumers, fused_names, available) -> FusedGroup | None`
- Registration order = priority. Conv+BN+ReLU before Add+ReLU.
- **CRITICAL**: Check `available` set for all non-local inputs. Fused kernel runs at first node position.
- Passthrough skip: up to 4 ops (expand, to.dtype, _assert, dropout) between fusible nodes.

## Fused Codegen Registry (codegen_fused.py)
- Register via `register_fused_codegen(kernel_type, handler)`
- Handler signature: `(group: FusedGroup, graph: IRGraph) -> KernelCall | None`
- Each `FusedGroup.kernel_type` must have a registered handler.

## Layout System
- 4D tensors (conv/pool): PADDED_NCHW — channels padded to `channel_tile` alignment
- Non-4D tensors (matmul, embedding): row-major, no padding
- `resolve_layouts()` determines per-tensor layout before codegen
- Flattening 4D→2D: emit `depad_4d_kernel` to strip channel padding

## Param Packing
- `Executor._PARAM_SPECS`: maps kernel names to `(struct_format, param_keys)`
- List-valued params auto-unpacked into struct fields
- Adding new kernel params: add entry to `_PARAM_SPECS`, match Metal struct layout
