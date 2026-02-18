#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

struct SoftmaxParams {
    uint rows;
    uint cols;
};

// Numerically stable softmax along last axis.
// Each thread handles one row. FP32 accumulation.
// tid = row index, dispatch 1D with total_threads = rows.
kernel void softmax_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    constant SoftmaxParams &p      [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.rows) return;

    uint base = tid * p.cols;

    // Pass 1: find max for numerical stability
    float max_val = float(input[base]);
    for (uint i = 1; i < p.cols; i++) {
        float v = float(input[base + i]);
        max_val = max(max_val, v);
    }

    // All-masked row: output zeros (avoids -inf - (-inf) = NaN)
    if (max_val == -INFINITY) {
        for (uint i = 0; i < p.cols; i++) {
            output[base + i] = compute_t(0.0f);
        }
        return;
    }

    // Pass 2: exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < p.cols; i++) {
        float e = exp(float(input[base + i]) - max_val);
        output[base + i] = compute_t(e);  // temp store
        sum += e;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < p.cols; i++) {
        output[base + i] = compute_t(float(output[base + i]) * inv_sum);
    }
}

// ── Fused masked softmax: add(scores, mask) + softmax in 1 dispatch ──
// Same-shape mask (no broadcasting needed).

kernel void masked_softmax_kernel(
    device const compute_t *scores [[buffer(0)]],
    device const compute_t *mask   [[buffer(1)]],
    device compute_t *output       [[buffer(2)]],
    constant SoftmaxParams &p      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.rows) return;

    uint base = tid * p.cols;

    // Pass 1: add mask and find max
    float max_val = float(scores[base]) + float(mask[base]);
    for (uint i = 1; i < p.cols; i++) {
        float v = float(scores[base + i]) + float(mask[base + i]);
        max_val = max(max_val, v);
    }

    if (max_val == -INFINITY) {
        for (uint i = 0; i < p.cols; i++) {
            output[base + i] = compute_t(0.0f);
        }
        return;
    }

    // Pass 2: exp(scores + mask - max) and sum
    float sum = 0.0f;
    for (uint i = 0; i < p.cols; i++) {
        float e = exp(float(scores[base + i]) + float(mask[base + i]) - max_val);
        output[base + i] = compute_t(e);
        sum += e;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < p.cols; i++) {
        output[base + i] = compute_t(float(output[base + i]) * inv_sum);
    }
}

// ── Fused masked softmax with broadcast mask ──
// Mask may be smaller than scores (e.g. [1,1,1,S] broadcast to [B,H,1,S]).

struct MaskedSoftmaxBroadcastParams {
    uint rows;
    uint cols;
    uint ndim;
    uint mask_strides[6];
    uint out_shape[6];
};

kernel void masked_softmax_broadcast_kernel(
    device const compute_t *scores [[buffer(0)]],
    device const compute_t *mask   [[buffer(1)]],
    device compute_t *output       [[buffer(2)]],
    constant MaskedSoftmaxBroadcastParams &p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.rows) return;

    uint base = tid * p.cols;

    // Compute row's base coordinates (all dims except last)
    uint row_linear = tid;
    uint row_coords[6];
    // Last dim will be iterated over in the loop
    for (int d = int(p.ndim) - 2; d >= 0; d--) {
        row_coords[d] = row_linear % p.out_shape[d];
        row_linear /= p.out_shape[d];
    }

    // Compute mask base offset for this row (all dims except last)
    uint mask_row_offset = 0;
    for (uint d = 0; d + 1 < p.ndim; d++) {
        mask_row_offset += row_coords[d] * p.mask_strides[d];
    }

    uint mask_col_stride = p.mask_strides[p.ndim - 1];

    // Pass 1: add mask and find max
    float max_val = float(scores[base]) + float(mask[mask_row_offset]);
    for (uint i = 1; i < p.cols; i++) {
        float v = float(scores[base + i]) + float(mask[mask_row_offset + i * mask_col_stride]);
        max_val = max(max_val, v);
    }

    if (max_val == -INFINITY) {
        for (uint i = 0; i < p.cols; i++) {
            output[base + i] = compute_t(0.0f);
        }
        return;
    }

    // Pass 2: exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < p.cols; i++) {
        float e = exp(float(scores[base + i]) + float(mask[mask_row_offset + i * mask_col_stride]) - max_val);
        output[base + i] = compute_t(e);
        sum += e;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < p.cols; i++) {
        output[base + i] = compute_t(float(output[base + i]) * inv_sum);
    }
}
