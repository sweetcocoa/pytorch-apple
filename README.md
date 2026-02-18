# NPU Simulation

NPU compiler/runtime backend that compiles PyTorch IR to Metal compute shaders, simulating NPU execution on Mac Apple Silicon GPU.

## Supported Models

| Model | Status | Compute Dtype |
|-------|--------|---------------|
| ResNet-18 | Full E2E (top-5 match) | float16 |
| ResNet-50 | Full E2E (top-5 match) | float16 |
| Qwen2.5-1.5B | Prefill + Decode | bfloat16 |

## Quick Start

```bash
# Install
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run Qwen example
uv sync --extra examples
cd examples && uv run python run_qwen.py --prompt "Hello" --max-tokens 8
```

## Architecture

```
PyTorch Model → torch_to_ir → IR JSON → npu_compiler → .npubin → npu_runtime → Metal GPU
```

### Compiler (offline)
- IR loading, constraint validation, BN folding
- Op fusion: Conv+BN+ReLU, Add+ReLU, RMSNorm, SiLU+Gate, Masked Softmax
- Broadcast folding, transpose folding into MPS matmul

### Runtime (online)
- MPS-accelerated matmul (float16 + bfloat16)
- Pre-compiled shaders, pre-packed parameters
- Single command buffer batching

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

CNN, linear algebra, element-wise, tensor manipulation, reductions, positional encoding, and fused kernels. See [docs/operators.md](docs/operators.md) for the full list.
