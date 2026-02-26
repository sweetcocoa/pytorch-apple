# NPU Simulation

NPU compiler/runtime backend that compiles PyTorch IR to Metal compute shaders (macOS) or CUDA kernels (NVIDIA GPU), simulating NPU execution.

## Supported Models

| Model | Status | Compute Dtype |
|-------|--------|---------------|
| ResNet-18 | Full E2E (top-5 match) | float16 |
| ResNet-50 | Full E2E (top-5 match) | float16 |
| Qwen2.5-1.5B | Prefill + Decode | bfloat16 |

## Quick Start

```bash
# Install (Metal backend, macOS)
uv sync --extra dev

# Install (CUDA backend, Linux/Windows with NVIDIA GPU)
uv sync --extra dev --extra cuda

# Run tests
uv run pytest tests/ -v

# Run Qwen example (graph partition pipeline)
uv sync --extra examples
cd examples && uv run python run_qwen_graph.py --prompt "Hello"
```

## Architecture

```
PyTorch Model → torch_to_ir → IR JSON ─┬─→ compile()  → .npubin → Executor → Metal GPU
                                        ├─→ partition() → DAGExecutor → NPU + CPU mixed (Metal)
                                        └─→ partition() → DAGExecutor → CUDA + CPU mixed (CUDA)
```

### Metal Backend (macOS)
- Op-level codegen: 1 ATen op → 1 Metal kernel
- Op fusion: Conv+BN+ReLU, Add+ReLU, RMSNorm, SiLU+Gate, Masked Softmax
- MPS-accelerated matmul (float16 + bfloat16)

### CUDA Backend (NVIDIA GPU)
- Subgraph-level codegen: N elementwise ops → 1 fused CUDA kernel
- Greedy elementwise chain fusion (silu+mul, add+relu, etc.)
- cuBLAS for GEMM via CuPy, NVRTC JIT for custom kernels

### Common
- IR loading, constraint validation, BN folding, graph partitioning
- **DAGExecutor**: mixed GPU + CPU fallback execution via Backend ABC
- `compile_fn` parameter selects Metal or CUDA compilation

## Performance Optimizations

| Optimization | Impact |
|-------------|--------|
| MPS BFloat16 matmul | ~70% latency reduction for LLM decode |
| Broadcast binary ops | Eliminates ~310 expand dispatches |
| Fused RMSNorm | 8 dispatches → 1 (×57 layers) |
| SiLU+Gate fusion | 2 dispatches → 1 (×28 layers) |
| Masked Softmax fusion | 2 dispatches → 1 (×28 layers) |
| Transpose folding | Eliminates transpose before attention matmul |

## Documentation

```bash
uv sync --extra docs
uv run mkdocs serve  # http://localhost:8000
```

Available in English and Korean.

## 50+ Supported Ops

CNN, linear algebra, element-wise, tensor manipulation, reductions, positional encoding, and fused kernels. See [docs/operators.md](docs/operators.md) for the full list. Ops not in this list are automatically routed to CPU fallback via the [graph partition pipeline](docs/partitioning.md).
