---
name: metal-shader
description: Rules for writing and modifying Metal compute shaders for the NPU backend. Use when creating or editing .metal files.
---

# Metal Shader Rules

## File Organization
- `metal_kernels/common.metal` — shared types, `#ifdef USE_BFLOAT` typedef
- Each shader file groups related kernels (e.g., `elementwise_extended.metal` has silu, neg, rsqrt, cos, sin, pow, mul, div)
- Fused kernels get dedicated files when complex (rmsnorm.metal, fused_decode_attention.metal)

## BFloat16 Support
```metal
#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif
```
- Compile-time macro `USE_BFLOAT=1` passed via `MTLCompileOptions` for bf16 models
- All kernels use `compute_t` for input/output buffer types

## Common Pitfalls
- `max(half, half)` is ambiguous in Metal — use `max(float(x), 0.0f)` pattern
- Metal requires `channel_alignment_bytes`-aligned access for coalesced SIMD reads
- Metal command buffers have finite command capacity — executor flushes at 10000 dispatches
- Metal structs cannot have variable-length fields — use fixed `_MAX_NDIM=6` arrays

## Dispatch Patterns
- **1D** (elementwise): `total_threads` elements, threadgroup = `max_threadgroup_1d`
- **2D** (matmul, embedding): `grid_width × grid_height`, threadgroup = `max_threadgroup_2d²`
- **3D** (batched matmul): `grid_width × grid_height × grid_depth`
- **Tiled matmul**: fixed `TILE × TILE` threadgroups (all threads participate in shared memory)
- **Vec matmul** (M=1): N threadgroups × max_threadgroup_1d (parallel K-reduction)

## Parameter Struct Convention
Kernel params passed as a single Metal buffer with packed struct. Layout must match `Executor._PARAM_SPECS` exactly.

Example (conv):
```metal
struct ConvParams {
    uint batch, in_channels, in_h, in_w;
    uint out_channels, out_h, out_w;
    uint kernel_h, kernel_w, stride_h, stride_w;
    uint pad_h, pad_w, has_bias, has_bn, has_relu;
    uint groups, in_channels_aligned, out_channels_aligned;
};
```

## Shared Memory Usage
- Tiled matmul: `TILE × TILE` shared arrays for A and B tiles
- Fused decode attention: `scores[MAX_SEQ_LEN]` + `q_local[HEAD_DIM]` per threadgroup
- Metal limit: 32KB per threadgroup — constrains fused attention to decode (M=1) only

## Buffer Slot Convention
Metal kernel function signature order: inputs → outputs → params
```metal
kernel void my_kernel(
    device const compute_t* input [[buffer(0)]],
    device compute_t* output       [[buffer(1)]],
    constant MyParams& params      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
);
```
Conv/matmul always declare bias slot even when unused (has_bias=0 multiplied away).

## Adding a New Kernel
1. Write kernel in appropriate .metal file (or new file if distinct category)
2. Use `compute_t` for all floating-point buffers
3. Match param struct to `Executor._PARAM_SPECS` entry
4. Add codegen handler in `codegen_ops.py` or `codegen_fused.py`
5. Test with `uv run pytest tests/test_kernels.py -v`
