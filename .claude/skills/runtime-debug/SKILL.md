---
name: runtime-debug
description: Debugging guide for NPU runtime on Metal GPU. Use when investigating incorrect outputs, silent failures, or Metal API issues.
---

# Runtime Debug Guide

## Bisect Strategy (incorrect output)
1. Disable ALL fusions → verify unfused output is correct
2. Re-enable fusions one by one → bug is in the last-enabled fusion
3. For each fusion: compare fused vs unfused output tensor (Pearson r, argmax)

## Correctness Metrics (verify_logits.py)
- Pearson r > 0.99: correct execution
- Top-5 token overlap: should be 100% agreement
- Argmax mismatch: indicates a bug, not just numerical drift

## pyobjc Metal API
- `newBufferWithBytes_length_options_()` takes raw `bytes`, NOT ctypes pointers
- `buffer.contents()` returns `objc.varlist` → use `.as_buffer(nbytes)` for `memoryview`
- Reference pattern: `npu_runtime/buffer.py`

## Critical Constraints
- **MPS bf16 unsupported**: `MPSMatrixMultiplication` asserts on bfloat16. Guard: `self._use_mps = compute_dtype == "float16"`
- **Transpose folding REMOVED**: Don't fold transpose into matmul. Use explicit `transpose_kernel`.
- **Command buffer overflow**: Metal has finite capacity. `max_dispatches_per_batch=10000` triggers flush. Silent failures = check batch count.
- **Pre-packed params**: Packed once at `Executor.__init__`. Changing param structure requires re-creating Executor.

## Fusion Data Availability
Fused kernel runs at first node's position. If output is garbage, check that ALL inputs exist before fusion point in graph. See: `available` set in `fusion_patterns.py`.

## Metal Shader Compilation
- `max(half, half)` is ambiguous → use `max(float(...), 0.0f)`
- bf16 shaders: `#ifdef USE_BFLOAT` → `typedef bfloat compute_t`

## Buffer Dtypes
- `ml_dtypes.bfloat16` in `_DTYPE_MAP`
- `from_numpy` preserves integer types (int32 for indices, cache_position)

## Test Commands
```bash
uv run pytest tests/ -v                          # all
uv run pytest tests/test_kernels.py -v            # kernel-level
uv run pytest tests/test_fusion_correctness.py -v # fusion
uv run pytest tests/test_e2e_resnet.py -v         # ResNet E2E
uv run pytest tests/test_e2e_qwen.py -v           # Qwen E2E
```
