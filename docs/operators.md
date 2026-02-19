# Operator Support

Ops listed below run on the NPU. Ops not in this list are automatically routed to CPU fallback via the [graph partition pipeline](partitioning.md).

## Supported Operations (50+)

### CNN Operations
| Op Type | Metal Kernel | Notes |
|---------|-------------|-------|
| `aten.conv2d.default` | `conv2d_kernel` | Groups, bias, padding, stride |
| `aten.batch_norm.default` | (folded) | Folded into conv weights |
| `aten.relu.default` | `elementwise_relu` | Also fused with conv/add |
| `aten.relu_.default` | `elementwise_relu` | In-place variant |
| `aten.max_pool2d.default` | `max_pool2d_kernel` | |
| `aten.adaptive_avg_pool2d.default` | `adaptive_avg_pool2d_kernel` | |

### Linear Algebra
| Op Type | Metal Kernel | Notes |
|---------|-------------|-------|
| `aten.linear.default` | `matmul_kernel` / `matmul_vec_kernel` | MPS accelerated, vec for M=1 |
| `aten.matmul.default` | `matmul_notrans_kernel` / `batched_matmul_kernel` | 2D and batched 3D+ |
| `aten.addmm.default` | `matmul_kernel` | With bias |
| `aten.t.default` | `transpose_kernel` | 2D transpose |

### Element-wise Unary
| Op Type | Metal Kernel | Notes |
|---------|-------------|-------|
| `aten.silu.default` | `silu_kernel` | x * sigmoid(x) |
| `aten.neg.default` | `neg_kernel` | |
| `aten.rsqrt.default` | `rsqrt_kernel` | |
| `aten.cos.default` | `cos_kernel` | |
| `aten.sin.default` | `sin_kernel` | |
| `aten.pow.Tensor_Scalar` | `pow_scalar_kernel` | |

### Element-wise Binary
| Op Type | Metal Kernel | Notes |
|---------|-------------|-------|
| `aten.add.Tensor` | `add_kernel` / `add_broadcast_kernel` | Broadcast-aware |
| `aten.add_.Tensor` | `add_kernel` | In-place variant |
| `aten.mul.Tensor` | `mul_kernel` / `mul_broadcast_kernel` | Broadcast-aware |
| `aten.div.Tensor` | `div_kernel` / `div_broadcast_kernel` | Broadcast-aware |

### Scalar Operations
| Op Type | Metal Kernel | Notes |
|---------|-------------|-------|
| `aten.add.Tensor` (scalar) | `eltwise_add_scalar_kernel` | Single input + attr |
| `aten.mul.Tensor` (scalar) | `eltwise_mul_scalar_kernel` | Single input + attr |
| `aten.full.default` | `add_scalar_kernel` | Constant fill |

### Tensor Operations
| Op Type | Metal Kernel | Notes |
|---------|-------------|-------|
| `aten.embedding.default` | `embedding_kernel` | Token lookup |
| `aten.transpose.int` | `transpose_kernel` | N-dim, can be folded into matmul |
| `aten.cat.default` | `cat_2_kernel` | 2-input concat |
| `aten.slice.Tensor` | `slice_kernel` | Along any dim |
| `aten.expand.default` | `expand_kernel` | Broadcast copy |
| `aten.index_copy.default` | `index_copy_kernel` | KV cache update |

### Reduction
| Op Type | Metal Kernel | Notes |
|---------|-------------|-------|
| `aten.softmax.int` | `softmax_kernel` | Numerically stable |
| `aten.mean.dim` | `mean_last_dim_kernel` | Last dim reduction |

### Positional Encoding
| Op Type | Metal Kernel | Notes |
|---------|-------------|-------|
| `wrap_with_set_grad_enabled` | `rope_kernel` | Rotary position embedding |

### Fused Kernels
| Pattern | Metal Kernel | Ops Replaced |
|---------|-------------|-------------|
| RMSNorm | `rmsnorm_kernel` | pow, mean, add, rsqrt, expand, mul, mul |
| SiLU+Gate | `silu_mul_kernel` | silu, mul |
| Masked Softmax | `masked_softmax_kernel` | add, softmax |
| Add+ReLU | `add_relu_kernel` | add, relu |

### Zero-cost Aliases (no dispatch)
| Op Type | Notes |
|---------|-------|
| `aten.reshape.default` | Buffer alias |
| `aten.view.default` | Buffer alias |
| `aten.flatten.using_ints` | Buffer alias (4D->non-4D needs depad) |
| `aten.contiguous.default` | Identity |
| `aten.unsqueeze.default` | Shape change only |
| `aten.alias.default` | Identity |
| `aten.detach_.default` | Identity |
| `aten.to.dtype` | Identity (dtype handled at boundary) |
| `aten.dropout.default` | Identity (eval mode) |
| `<built-in function getitem>` | Multi-output index |
| `aten._assert_tensor_metadata.default` | Metadata assertion (no-op) |
