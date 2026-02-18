# API Reference

## Modules

### npu_compiler

The offline compilation pipeline that transforms IR into Metal execution plans.

| Module | Description |
|--------|-------------|
| [`ir_reader`](compiler.md#ir-reader) | Load torch_to_ir IR JSON |
| [`constraint_checker`](compiler.md#constraint-checker) | Validate NPU constraints |
| [`graph_optimizer`](compiler.md#graph-optimizer) | BN folding, noop elimination |
| [`fusion_patterns`](compiler.md#fusion-patterns) | Op fusion pattern matching |
| [`codegen`](compiler.md#code-generator) | Metal kernel code generation |
| [`compiled_program`](compiler.md#compiled-program) | Serialization (.npubin) |

### npu_runtime

The online execution engine using Metal GPU.

| Module | Description |
|--------|-------------|
| [`device`](runtime.md#device) | Metal device management |
| [`buffer`](runtime.md#npubuffer) | GPU memory (NPUBuffer) |
| [`executor`](runtime.md#executor) | Command buffer batching |
| [`weight_loader`](runtime.md#weight-loader) | safetensors loading |
| [`profiler`](runtime.md#profiler) | Kernel timing |

### metal_kernels

Metal compute shaders for all supported operations.

| File | Kernels |
|------|---------|
| [`matmul.metal`](kernels.md#matmul) | Tiled, vec, batched matmul |
| [`rmsnorm.metal`](kernels.md#rmsnorm) | Fused RMSNorm |
| [`softmax.metal`](kernels.md#softmax) | Softmax, masked softmax |
| [`elementwise_extended.metal`](kernels.md#elementwise) | Unary/binary ops, SiLU+mul |
| [`elementwise_broadcast.metal`](kernels.md#broadcast) | Broadcast binary ops |
