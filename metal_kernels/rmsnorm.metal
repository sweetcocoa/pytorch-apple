#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// Fused RMSNorm: replaces pow→mean→add_scalar→rsqrt→expand→mul→expand→mul chain.
// Each thread processes one row: sum_sq → rsqrt(mean + eps) → scale × weight.
// Dispatch 1D: total_threads = rows.

struct RMSNormParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void rmsnorm_kernel(
    device const compute_t *input   [[buffer(0)]],
    device const compute_t *weight  [[buffer(1)]],
    device compute_t *output        [[buffer(2)]],
    constant RMSNormParams &p       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.rows) return;

    uint base = tid * p.cols;

    // Pass 1: compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = 0; i < p.cols; i++) {
        float v = float(input[base + i]);
        sum_sq += v * v;
    }

    // RMS normalization factor: rsqrt(mean(x^2) + eps)
    float scale = rsqrt(sum_sq / float(p.cols) + p.eps);

    // Pass 2: normalize and multiply by weight
    for (uint i = 0; i < p.cols; i++) {
        float v = float(input[base + i]);
        float w = float(weight[i]);
        output[base + i] = compute_t(v * scale * w);
    }
}
