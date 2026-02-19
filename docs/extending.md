# Extension Guide

How to add a new operator to the NPU simulation pipeline.

## Steps

### 1. Add Metal Kernel

Create or extend a `.metal` file in `metal_kernels/`:

```metal
#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// Always use compute_t for storage, float for arithmetic
kernel void my_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    // Optional: parameter struct
    constant MyParams &p           [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    float x = float(input[tid]);
    output[tid] = compute_t(/* compute */);
}
```

!!! note "Conventions"
    - Use `compute_t` typedef for all storage (handles float16/bfloat16)
    - Cast to `float` for arithmetic to avoid precision loss
    - Avoid `max(half, half)` — use `max(float(...), 0.0f)` pattern for bfloat16 compatibility
    - Guard bounds with `if (tid >= total) return;`

### 2. Add Codegen Handler

In `npu_compiler/codegen.py`, add a handler in `_generate_single_kernel_call()`:

```python
if node.op_type == "aten.my_op.default":
    total = int(np.prod(node.outputs[0].shape))
    return KernelCall(
        kernel_name="my_kernel",
        metal_file="my_metal_file.metal",
        input_buffers=[node.inputs[0].name],
        output_buffers=[node.outputs[0].name],
        param_buffers=["my_params"],
        params={"total": total, "some_attr": node.attrs.get("attr", 0)},
        dispatch_type="1d",
        total_threads=total,
    )
```

### 3. Add Param Spec to Executor

In `npu_runtime/executor.py`, add to `_PARAM_SPECS`:

```python
("my_kernel",): ("2I", ["total", "some_attr"]),
```

The format string uses `struct.pack` format codes:
- `I` = uint32
- `f` = float32
- `6I` = array of 6 uint32s (auto-unpacked from list params)

### 4. Register in Op Support Table

Add to `HANDLED_OPS` in `npu_compiler/codegen.py` (the constraint checker imports from there) and to `_SUPPORTED_OPS` in `npu_compiler/op_support.py` (used by the graph partitioner):

```python
# npu_compiler/codegen.py — HANDLED_OPS set
HANDLED_OPS.add("aten.my_op.default")

# npu_compiler/op_support.py — _SUPPORTED_OPS set
_SUPPORTED_OPS.add("aten.my_op.default")
```

Both sets must stay in sync. `HANDLED_OPS` controls the single-program compile path; `_SUPPORTED_OPS` controls the partition path (ops not in `_SUPPORTED_OPS` fall back to CPU via DAGExecutor).

### 5. Add Tests

```python
class TestMyKernel:
    def test_my_op(self, device):
        # Prepare input
        x = np.random.randn(4, 64).astype(np.float32)

        # PyTorch reference
        ref = torch.tensor(x).my_op().numpy()

        # Metal kernel
        lib = device.compile_metal_file(os.path.join(kernels_dir(), "my_file.metal"))
        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros((4, 64), device)
        params = make_params(device, "I", 256)
        pipeline = device.get_pipeline(lib, "my_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], 256)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, atol=1e-3)
```

## Adding Fusion Patterns

### 1. Define Pattern in `fusion_patterns.py`

```python
# Pattern: op_a → op_b → fused
if node.op_type == "aten.op_a.default":
    next_nodes = consumers.get(node.outputs[0].name, [])
    if (len(next_nodes) == 1
            and next_nodes[0].op_type == "aten.op_b.default"
            and next_nodes[0].name not in fused_node_names):
        b_node = next_nodes[0]
        fused_node_names.add(node.name)
        fused_node_names.add(b_node.name)
        result.append(FusedGroup(
            name=f"fused_{node.name}",
            kernel_type="my_fusion",
            nodes=[node, b_node],
        ))
        i += 1
        continue
```

### 2. Handle in Codegen

In `_generate_fused_kernel_call()`:

```python
if group.kernel_type == "my_fusion":
    return _gen_my_fused_kernel(group)
```

## Dispatch Types

| Type | Use Case | Grid |
|------|----------|------|
| `1d` | Element-wise, reductions | `total_threads` |
| `2d` | 2D ops (embedding, matmul) | `grid_width x grid_height` |
| `3d` | Batched ops (BMM) | `grid_width x grid_height x grid_depth` |
| `none` | Zero-cost aliases | No dispatch |
