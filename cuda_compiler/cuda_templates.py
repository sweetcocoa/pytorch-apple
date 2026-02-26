"""Pre-written CUDA kernel templates for non-fusible operations.

These templates are compiled via NVRTC at runtime. Each function returns
CUDA C source code as a string.
"""

from __future__ import annotations

SOFTMAX_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
// Block-per-row softmax: threads cooperate on each row via warp shuffle + shared memory.
// 3-pass: (1) parallel max, (2) parallel exp+sum, (3) parallel normalize.
__global__ void softmax_kernel(const __half* input, __half* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* in_row = input + row * cols;
    __half* out_row = output + row * cols;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int num_warps = block_size / 32;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Pass 1: find max (strided loop + warp shuffle reduction)
    float local_max = -65504.0f;
    for (int i = tid; i < cols; i += block_size) {
        float v = __half2float(in_row[i]);
        if (v > local_max) local_max = v;
    }
    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }
    // Cross-warp reduction via shared memory (sequential to avoid warp divergence)
    __shared__ float smem[8];
    if (lane_id == 0) smem[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float v = smem[0];
        for (int i = 1; i < num_warps; i++) {
            if (smem[i] > v) v = smem[i];
        }
        smem[0] = v;
    }
    __syncthreads();
    float row_max = smem[0];

    // Pass 2: compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float v = expf(__half2float(in_row[i]) - row_max);
        out_row[i] = __float2half(v);
        local_sum += v;
    }
    // Warp-level sum reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float v = smem[0];
        for (int i = 1; i < num_warps; i++) {
            v += smem[i];
        }
        smem[0] = v;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem[0];

    // Pass 3: normalize
    for (int i = tid; i < cols; i += block_size) {
        out_row[i] = __float2half(__half2float(out_row[i]) * inv_sum);
    }
}
}
"""

MEAN_LAST_DIM_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void mean_last_dim_kernel(const __half* input, __half* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const __half* in_row = input + row * cols;
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        sum += __half2float(in_row[i]);
    }
    output[row] = __float2half(sum / (float)cols);
}
}
"""

EMBEDDING_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void embedding_kernel(const int* indices, const __half* weight,
                                  __half* output, int seq_len, int embed_dim) {
    int s = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seq_len || d >= embed_dim) return;

    int idx = indices[s];
    output[s * embed_dim + d] = weight[idx * embed_dim + d];
}
}
"""

ROPE_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void rope_kernel(const __half* inv_freq, const __half* positions,
                            __half* cos_out, __half* sin_out,
                            int seq_len, int head_dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (d >= head_dim || s >= seq_len) return;

    float pos = __half2float(positions[s]);
    float freq = __half2float(inv_freq[d]);
    float angle = pos * freq;

    cos_out[s * head_dim + d] = __float2half(cosf(angle));
    sin_out[s * head_dim + d] = __float2half(sinf(angle));
}
}
"""

INDEX_COPY_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void index_copy_kernel(const __half* src, __half* dst,
                                   const int* indices, const __half* values,
                                   int outer_size, int dim_size,
                                   int inner_size, int num_indices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * dim_size * inner_size;
    if (tid >= total) return;

    // Copy src to dst first
    dst[tid] = src[tid];
    __syncthreads();

    // Now handle index_copy part
    int inner_idx = tid % inner_size;
    int outer_idx = tid / (dim_size * inner_size);
    int dim_idx = (tid / inner_size) % dim_size;

    for (int i = 0; i < num_indices; i++) {
        if (indices[i] == dim_idx) {
            int src_offset = outer_idx * num_indices * inner_size + i * inner_size + inner_idx;
            dst[tid] = values[src_offset];
        }
    }
}
}
"""

TRANSPOSE_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void transpose_kernel(const __half* input, __half* output,
                                  int ndim, int dim0, int dim1, int total,
                                  const int* in_shape, const int* in_strides,
                                  const int* out_strides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose output linear index into multi-dim index
    int remaining = idx;
    int coords[6];
    for (int d = 0; d < ndim; d++) {
        coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }

    // Swap dims for input indexing
    int tmp = coords[dim0];
    coords[dim0] = coords[dim1];
    coords[dim1] = tmp;

    // Compute input linear index
    int in_idx = 0;
    for (int d = 0; d < ndim; d++) {
        in_idx += coords[d] * in_strides[d];
    }

    output[idx] = input[in_idx];
}
}
"""

CAT_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void cat_2_kernel(const __half* in0, const __half* in1,
                              __half* output, int axis, int ndim, int total,
                              int in0_axis_size, const int* out_shape,
                              const int* out_strides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose output index
    int remaining = idx;
    int coords[6];
    for (int d = 0; d < ndim; d++) {
        coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }

    int axis_coord = coords[axis];

    if (axis_coord < in0_axis_size) {
        // Read from in0 — compute linear index in in0
        int in0_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            in0_idx += coords[d] * stride;
            stride *= (d == axis) ? in0_axis_size : out_shape[d];
        }
        output[idx] = in0[in0_idx];
    } else {
        // Read from in1 — adjust axis coord
        coords[axis] -= in0_axis_size;
        int in1_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            in1_idx += coords[d] * stride;
            stride *= (d == axis) ? (out_shape[axis] - in0_axis_size) : out_shape[d];
        }
        output[idx] = in1[in1_idx];
    }
}
}
"""

SLICE_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void slice_kernel(const __half* input, __half* output,
                              int dim, int start, int step, int ndim, int total,
                              const int* in_shape, const int* in_strides,
                              const int* out_shape) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose output index into multi-dim coords
    int remaining = idx;
    int coords[6];
    int out_stride = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = remaining % out_shape[d];
        remaining /= out_shape[d];
    }

    // Adjust the sliced dimension
    coords[dim] = start + coords[dim] * step;

    // Compute input linear index
    int in_idx = 0;
    for (int d = 0; d < ndim; d++) {
        in_idx += coords[d] * in_strides[d];
    }

    output[idx] = input[in_idx];
}
}
"""

EXPAND_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void expand_kernel(const __half* input, __half* output,
                               int ndim, int total,
                               const int* in_shape, const int* out_shape,
                               const int* in_strides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose output index
    int remaining = idx;
    int in_idx = 0;
    int out_stride = 1;

    for (int d = ndim - 1; d >= 0; d--) {
        int coord = remaining % out_shape[d];
        remaining /= out_shape[d];
        // If input dim is 1, broadcast (stride = 0)
        if (in_shape[d] != 1) {
            in_idx += coord * in_strides[d];
        }
    }

    output[idx] = input[in_idx];
}
}
"""

FULL_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void full_kernel(__half* output, __half value, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] = value;
}
}
"""

MAX_POOL2D_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void max_pool2d_kernel(const __half* input, __half* output,
                                   int batch, int channels,
                                   int in_h, int in_w, int out_h, int out_w,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int n = idx / (out_w * out_h * channels);

    float max_val = -65504.0f;  // -inf for half
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                float v = __half2float(input[in_idx]);
                if (v > max_val) max_val = v;
            }
        }
    }
    output[idx] = __float2half(max_val);
}
}
"""

ADAPTIVE_AVG_POOL2D_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void adaptive_avg_pool2d_kernel(const __half* input, __half* output,
                                            int batch, int channels,
                                            int in_h, int in_w,
                                            int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int n = idx / (out_w * out_h * channels);

    int ih_start = (oh * in_h) / out_h;
    int ih_end = ((oh + 1) * in_h) / out_h;
    int iw_start = (ow * in_w) / out_w;
    int iw_end = ((ow + 1) * in_w) / out_w;

    float sum = 0.0f;
    int count = 0;
    for (int ih = ih_start; ih < ih_end; ih++) {
        for (int iw = iw_start; iw < iw_end; iw++) {
            int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
            sum += __half2float(input[in_idx]);
            count++;
        }
    }
    output[idx] = __float2half(sum / (float)count);
}
}
"""

CONV2D_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void conv2d_kernel(const __half* input, const __half* weight,
                               const __half* bias, __half* output,
                               int batch, int in_channels, int in_h, int in_w,
                               int out_channels, int out_h, int out_w,
                               int kernel_h, int kernel_w,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int has_bias, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % out_channels;
    int n = idx / (out_w * out_h * out_channels);

    int group_size_in = in_channels / groups;
    int group_size_out = out_channels / groups;
    int g = oc / group_size_out;
    int oc_in_group = oc % group_size_out;

    float sum = 0.0f;
    for (int ic = 0; ic < group_size_in; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = ((n * in_channels + g * group_size_in + ic) * in_h + ih) * in_w + iw;
                    int w_idx = ((oc * group_size_in + ic) * kernel_h + kh) * kernel_w + kw;
                    sum += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
                }
            }
        }
    }
    if (has_bias) {
        sum += __half2float(bias[oc]);
    }
    output[idx] = __float2half(sum);
}
}
"""

BATCH_NORM_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
__global__ void batch_norm_kernel(const __half* input,
                                   const __half* gamma, const __half* beta,
                                   const __half* running_mean, const __half* running_var,
                                   __half* output,
                                   int batch, int channels, int spatial,
                                   float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * spatial;
    if (idx >= total) return;

    int c = (idx / spatial) % channels;
    float x = __half2float(input[idx]);
    float mean = __half2float(running_mean[c]);
    float var = __half2float(running_var[c]);
    float g = __half2float(gamma[c]);
    float b = __half2float(beta[c]);

    float normalized = (x - mean) / sqrtf(var + eps);
    output[idx] = __float2half(g * normalized + b);
}
}
"""

RMSNORM_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
// Fused RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
// Block-per-row, warp shuffle reduction for mean(x^2).
__global__ void rmsnorm_kernel(const __half* input, const __half* weight,
                                __half* output, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* in_row = input + row * cols;
    __half* out_row = output + row * cols;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // Compute sum of squares (strided)
    float local_ss = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float v = __half2float(in_row[i]);
        local_ss += v * v;
    }
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);
    }
    int num_warps = block_size / 32;
    __shared__ float smem[8];
    if (lane_id == 0) smem[warp_id] = local_ss;
    __syncthreads();
    if (tid == 0) {
        float v = smem[0];
        for (int i = 1; i < num_warps; i++) v += smem[i];
        smem[0] = v;
    }
    __syncthreads();

    float rms = rsqrtf(smem[0] / (float)cols + eps);

    // Normalize and scale
    for (int i = tid; i < cols; i += block_size) {
        float v = __half2float(in_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * rms * w);
    }
}
}
"""

SILU_MUL_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
// Fused SiLU(gate) * up: avoids materializing silu intermediate.
__global__ void silu_mul_kernel(const __half* gate, const __half* up,
                                 __half* output, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    output[idx] = __float2half(silu_g * u);
}
}
"""

MASKED_SOFTMAX_KERNEL = r"""
#include <cuda_fp16.h>
extern "C" {
// Fused add(input, mask) + softmax: block-per-row with warp shuffle.
__global__ void masked_softmax_kernel(const __half* input, const __half* mask,
                                       __half* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* in_row = input + row * cols;
    const __half* mask_row = mask + row * cols;
    __half* out_row = output + row * cols;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int num_warps = block_size / 32;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // Pass 1: max(input + mask)
    float local_max = -65504.0f;
    for (int i = tid; i < cols; i += block_size) {
        float v = __half2float(in_row[i]) + __half2float(mask_row[i]);
        if (v > local_max) local_max = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }
    __shared__ float smem[8];
    if (lane_id == 0) smem[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float v = smem[0];
        for (int i = 1; i < num_warps; i++) {
            if (smem[i] > v) v = smem[i];
        }
        smem[0] = v;
    }
    __syncthreads();
    float row_max = smem[0];

    // Pass 2: exp(input + mask - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float v = expf(__half2float(in_row[i]) + __half2float(mask_row[i]) - row_max);
        out_row[i] = __float2half(v);
        local_sum += v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float v = smem[0];
        for (int i = 1; i < num_warps; i++) {
            v += smem[i];
        }
        smem[0] = v;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem[0];

    // Pass 3: normalize
    for (int i = tid; i < cols; i += block_size) {
        out_row[i] = __float2half(__half2float(out_row[i]) * inv_sum);
    }
}
}
"""

# Template name -> source code mapping
TEMPLATE_MAP: dict[str, str] = {
    "softmax_kernel": SOFTMAX_KERNEL,
    "mean_last_dim_kernel": MEAN_LAST_DIM_KERNEL,
    "embedding_kernel": EMBEDDING_KERNEL,
    "rope_kernel": ROPE_KERNEL,
    "index_copy_kernel": INDEX_COPY_KERNEL,
    "transpose_kernel": TRANSPOSE_KERNEL,
    "cat_2_kernel": CAT_KERNEL,
    "slice_kernel": SLICE_KERNEL,
    "expand_kernel": EXPAND_KERNEL,
    "full_kernel": FULL_KERNEL,
    "max_pool2d_kernel": MAX_POOL2D_KERNEL,
    "adaptive_avg_pool2d_kernel": ADAPTIVE_AVG_POOL2D_KERNEL,
    "conv2d_kernel": CONV2D_KERNEL,
    "batch_norm_kernel": BATCH_NORM_KERNEL,
    "rmsnorm_kernel": RMSNORM_KERNEL,
    "silu_mul_kernel": SILU_MUL_KERNEL,
    "masked_softmax_kernel": MASKED_SOFTMAX_KERNEL,
}
