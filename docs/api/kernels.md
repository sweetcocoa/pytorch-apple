# Metal Kernels Reference

## Matmul

**File**: `matmul.metal`

| Kernel | Description | Dispatch |
|--------|-------------|----------|
| `matmul_kernel` | Tiled matmul C = A @ B^T (TILE=16, shared memory) | 2D: (N/16, M/16) |
| `matmul_vec_kernel` | Vector matmul for M=1, parallel K-reduction (256 threads) | 1D: N threadgroups |
| `matmul_notrans_kernel` | C = A @ B (B not transposed) | 2D |
| `matmul_notrans_vec_kernel` | Vector variant for M=1 | 1D |
| `batched_matmul_kernel` | C[b] = A[b] @ B[b] | 3D: (N, M, batch) |

**Parameters**: `MatmulParams { M, N, K, has_bias }`

## RMSNorm

**File**: `rmsnorm.metal`

| Kernel | Description | Dispatch |
|--------|-------------|----------|
| `rmsnorm_kernel` | Fused: x * rsqrt(mean(x^2) + eps) * weight | 1D: rows |

**Parameters**: `RMSNormParams { rows, cols, eps }`

Replaces 8 separate dispatches (pow, mean, add, rsqrt, expand, mul, expand, mul).

## Softmax

**File**: `softmax.metal`

| Kernel | Description | Dispatch |
|--------|-------------|----------|
| `softmax_kernel` | Numerically stable softmax (3-pass: max, exp+sum, normalize) | 1D: rows |
| `masked_softmax_kernel` | Fused add(scores, mask) + softmax | 1D: rows |
| `masked_softmax_broadcast_kernel` | Masked softmax with broadcast mask | 1D: rows |

**Parameters**: `SoftmaxParams { rows, cols }`

## Elementwise

**File**: `elementwise_extended.metal`

### Unary
| Kernel | Operation |
|--------|-----------|
| `silu_kernel` | x * sigmoid(x) |
| `neg_kernel` | -x |
| `rsqrt_kernel` | 1/sqrt(x) |
| `cos_kernel` | cos(x) |
| `sin_kernel` | sin(x) |
| `pow_scalar_kernel` | x^exp |

### Binary
| Kernel | Operation |
|--------|-----------|
| `mul_kernel` | a * b |
| `div_kernel` | a / b |

### Fused
| Kernel | Operation |
|--------|-----------|
| `silu_mul_kernel` | silu(gate) * up |

### Scalar
| Kernel | Operation |
|--------|-----------|
| `add_scalar_kernel` | fill with constant |
| `eltwise_add_scalar_kernel` | x + scalar |
| `eltwise_mul_scalar_kernel` | x * scalar |

### Reduction
| Kernel | Operation |
|--------|-----------|
| `mean_last_dim_kernel` | mean along last dimension |

## Broadcast

**File**: `elementwise_broadcast.metal`

Stride-based broadcast binary operations. Eliminates the need for separate expand+elementwise dispatches.

| Kernel | Operation |
|--------|-----------|
| `mul_broadcast_kernel` | a * b with broadcasting |
| `add_broadcast_kernel` | a + b with broadcasting |
| `div_broadcast_kernel` | a / b with broadcasting |

**Parameters**: `BroadcastBinaryParams { ndim, total, a_strides[6], b_strides[6], out_shape[6] }`

Stride of 0 means broadcast along that dimension.

## Tensor Operations

**File**: `tensor_ops.metal`

| Kernel | Description |
|--------|-------------|
| `transpose_kernel` | Swap two dimensions |
| `cat_2_kernel` | Concatenate 2 tensors along axis |
| `slice_kernel` | Extract sub-tensor |
| `expand_kernel` | Broadcast copy |

## Other Kernels

| File | Kernel | Description |
|------|--------|-------------|
| `embedding.metal` | `embedding_kernel` | Token embedding lookup |
| `rope.metal` | `rope_kernel` | Rotary position embedding |
| `index_copy.metal` | `index_copy_kernel` | KV cache position update |
| `conv_bn_relu.metal` | `conv2d_kernel` | Conv2d with optional BN+ReLU |
| `add_relu.metal` | `add_kernel`, `add_relu_kernel` | Add with optional ReLU |
| `pool.metal` | `max_pool2d_kernel`, `adaptive_avg_pool2d_kernel` | Pooling |

## CUDA Kernels

The CUDA backend generates kernels via two mechanisms:

### 1. Fused Elementwise Kernels (codegen)

Generated at compile time by `cuda_compiler.cuda_codegen.generate_fused_kernel()`. Chains of elementwise ops (relu, silu, add, mul, div, neg, pow, rsqrt, cos, sin) are fused into a single CUDA C kernel and JIT-compiled via NVRTC.

Example: `silu(gate) * up` generates:
```cuda
__global__ void fused_ew_0(const __half* in0, const __half* in1,
                           __half* out0, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    __half v0 = in0[idx];
    __half v1 = (v0 / ((__half)1.0 + hexp(-v0)));  // silu
    __half v2 = in1[idx];
    out0[idx] = (v1 * v2);                          // mul
}
```

### 2. Pre-written Template Kernels

Defined in `cuda_compiler/cuda_templates.py`:

| Template | Operations |
|----------|-----------|
| `softmax_kernel` | Numerically stable softmax (max, exp, sum, normalize) |
| `mean_last_dim_kernel` | Mean reduction on last dimension |
| `embedding_kernel` | Token embedding lookup |
| `rope_kernel` | Rotary position embedding |
| `index_copy_kernel` | KV cache position update |
| `conv2d_kernel` | Conv2d (direct implementation) |
| `batch_norm_kernel` | Batch normalization |
| `max_pool2d_kernel` | Max pooling |
| `adaptive_avg_pool2d_kernel` | Adaptive average pooling |
| `transpose_kernel` | N-dim transpose |
| `cat_2_kernel` | 2-input concatenation |
| `slice_kernel` | Tensor slicing |
| `expand_kernel` | Broadcast expansion |
| `full_kernel` | Constant fill |

### 3. cuBLAS (via CuPy)

BLAS operations (matmul, linear, bmm, conv2d) use `cupy.matmul()` which dispatches to cuBLAS internally. No custom CUDA kernels needed for GEMM.
