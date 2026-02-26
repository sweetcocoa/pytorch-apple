# Setup

## Requirements

### Metal Backend (macOS)
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### CUDA Backend (Linux/Windows)
- NVIDIA GPU (Compute Capability 7.0+, e.g. RTX 3090, A100)
- CUDA Driver 12.x
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd npu-simulation/pytorch-apple

# Install runtime dependencies (Metal backend)
uv sync

# Install CUDA backend dependencies
uv sync --extra cuda

# Install dev dependencies (for tests)
uv sync --extra dev

# Install example dependencies (for Qwen/ResNet demos)
uv sync --extra examples

# Install docs dependencies (for building documentation)
uv sync --extra docs
```

## Project Structure

```
pytorch-apple/
├── npu_compiler/           # Offline compilation pipeline
│   ├── ir_reader.py        # Load torch_to_ir IR JSON
│   ├── constraint_checker.py # NPU constraint validation
│   ├── graph_optimizer.py  # BN folding, noop elimination
│   ├── fusion_patterns.py  # Op fusion pattern matching
│   ├── codegen.py          # Metal kernel code generation
│   ├── compiled_program.py # Serialization (.npubin)
│   ├── op_support.py       # Op support table (is_op_supported)
│   └── partitioner.py      # Graph partitioning (NPU/CPU split)
├── npu_runtime/            # Online execution on Metal GPU
│   ├── backend.py          # Backend ABC (hardware-agnostic)
│   ├── metal_backend.py    # MetalBackend implementation
│   ├── device.py           # Metal device management
│   ├── buffer.py           # NPUBuffer (GPU memory)
│   ├── executor.py         # Command buffer batching (single program)
│   ├── dag_executor.py     # DAGExecutor (mixed NPU + CPU)
│   ├── cpu_fallback.py     # CPU fallback via torch_ir
│   ├── weight_loader.py    # safetensors → NPU buffers
│   └── profiler.py         # Kernel timing measurement
├── cuda_compiler/             # CUDA offline compilation (subgraph-level)
│   ├── op_classify.py         # Op category classification
│   ├── op_support.py          # CUDA op support table
│   ├── subgraph_analyzer.py   # Fusion analysis (greedy elementwise)
│   ├── cuda_codegen.py        # Fused CUDA kernel code generation
│   ├── cuda_templates.py      # Pre-written CUDA kernel templates
│   ├── cuda_program.py        # CUDAProgram data model
│   └── buffer_planner.py      # Intermediate buffer allocation
├── cuda_runtime/              # CUDA online execution via CuPy
│   ├── cuda_backend.py        # CUDABackend + CUDABuffer
│   └── cuda_executor.py       # CUDAExecutor (NVRTC + cuBLAS)
├── metal_kernels/             # Metal compute shaders
│   ├── matmul.metal           # Tiled + vec matmul
│   ├── conv_bn_relu.metal     # Conv2d with fusion
│   ├── elementwise*.metal     # Element-wise ops
│   ├── softmax.metal          # Softmax + masked softmax
│   ├── rmsnorm.metal          # Fused RMSNorm
│   ├── tensor_ops.metal       # Transpose, cat, slice, expand
│   ├── embedding.metal        # Token embedding lookup
│   ├── rope.metal             # Rotary position embedding
│   └── ...
├── tests/                     # Test suite
├── examples/                  # Demo scripts
└── docs/                      # This documentation
```

## Dependencies

### Runtime (Metal)
| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `ml-dtypes` | BFloat16 support |
| `pyobjc-framework-Metal` | Metal API bindings |
| `pyobjc-framework-MetalPerformanceShaders` | MPS matmul |
| `pytorch-ir` | IR extraction + CPU fallback execution (torch_to_ir) |

### Runtime (CUDA)
| Package | Purpose |
|---------|---------|
| `cupy-cuda12x` | CUDA runtime, NVRTC JIT, cuBLAS |

### Dev
| Package | Purpose |
|---------|---------|
| `pytest` | Testing |
| `torch` | Reference computations |
| `ruff` | Linting |

## Verification

```bash
# Run all tests
uv run pytest tests/ -v

# Expected: 180+ passed, 1-2 skipped (model download)
```
