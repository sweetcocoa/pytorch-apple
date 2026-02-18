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
