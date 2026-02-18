#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

kernel void elementwise_add(
    device const compute_t *a [[buffer(0)]],
    device const compute_t *b [[buffer(1)]],
    device compute_t *out     [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = a[tid] + b[tid];
}

kernel void elementwise_relu(
    device const compute_t *input [[buffer(0)]],
    device compute_t *output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = compute_t(max(float(input[tid]), 0.0f));
}

// Parameters for depad_4d: strips channel padding from 4D → dense 2D
struct Depad4DParams {
    uint batch;
    uint channels;           // logical channels
    uint channels_aligned;   // aligned channels (64-byte aligned)
    uint height;
    uint width;
};

// Depad 4D tensor to dense 2D: input[N, C_aligned, H, W] → output[N, C*H*W]
// tid = dense output index in [0, N*C*H*W)
kernel void depad_4d_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    constant Depad4DParams &p      [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.batch * p.channels * p.height * p.width;
    if (tid >= total) return;

    uint w = tid % p.width;
    uint h = (tid / p.width) % p.height;
    uint c = (tid / (p.width * p.height)) % p.channels;
    uint n = tid / (p.width * p.height * p.channels);

    uint in_idx = ((n * p.channels_aligned + c) * p.height + h) * p.width + w;
    output[tid] = input[in_idx];
}
