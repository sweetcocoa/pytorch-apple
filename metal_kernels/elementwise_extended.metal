#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// ── Unary element-wise kernels (FP32 compute, compute_t storage) ──

kernel void silu_kernel(
    device const compute_t *input [[buffer(0)]],
    device compute_t *output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float x = float(input[tid]);
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    output[tid] = compute_t(x * sigmoid_x);
}

kernel void neg_kernel(
    device const compute_t *input [[buffer(0)]],
    device compute_t *output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = compute_t(-float(input[tid]));
}

kernel void rsqrt_kernel(
    device const compute_t *input [[buffer(0)]],
    device compute_t *output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = compute_t(rsqrt(float(input[tid])));
}

kernel void cos_kernel(
    device const compute_t *input [[buffer(0)]],
    device compute_t *output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = compute_t(cos(float(input[tid])));
}

kernel void sin_kernel(
    device const compute_t *input [[buffer(0)]],
    device compute_t *output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = compute_t(sin(float(input[tid])));
}

// ── Fused SiLU + mul kernel (GatedMLP: silu(gate) * up) ──

kernel void silu_mul_kernel(
    device const compute_t *gate [[buffer(0)]],
    device const compute_t *up   [[buffer(1)]],
    device compute_t *output     [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    float g = float(gate[tid]);
    float sigmoid_g = 1.0f / (1.0f + exp(-g));
    output[tid] = compute_t(g * sigmoid_g * float(up[tid]));
}

// ── Binary element-wise kernels ──

kernel void mul_kernel(
    device const compute_t *a [[buffer(0)]],
    device const compute_t *b [[buffer(1)]],
    device compute_t *output  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = compute_t(float(a[tid]) * float(b[tid]));
}

kernel void div_kernel(
    device const compute_t *a [[buffer(0)]],
    device const compute_t *b [[buffer(1)]],
    device compute_t *output  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = compute_t(float(a[tid]) / float(b[tid]));
}

// ── Parameterized element-wise kernels ──

struct PowParams {
    float exponent;
};

kernel void pow_scalar_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    constant PowParams &p          [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = compute_t(pow(float(input[tid]), p.exponent));
}

struct ScalarParams {
    float scalar;
    uint total;
};

kernel void add_scalar_kernel(
    device compute_t *output       [[buffer(0)]],
    constant ScalarParams &p       [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    output[tid] = compute_t(p.scalar);
}

// input + scalar → output
kernel void eltwise_add_scalar_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    constant ScalarParams &p       [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    output[tid] = compute_t(float(input[tid]) + p.scalar);
}

// input * scalar → output
kernel void eltwise_mul_scalar_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    constant ScalarParams &p       [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    output[tid] = compute_t(float(input[tid]) * p.scalar);
}

// ── Reduction kernels ──

struct ReduceParams {
    uint rows;
    uint cols;
};

kernel void mean_last_dim_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    constant ReduceParams &p       [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.rows) return;

    float sum = 0.0f;
    uint base = tid * p.cols;
    for (uint i = 0; i < p.cols; i++) {
        sum += float(input[base + i]);
    }
    output[tid] = compute_t(sum / float(p.cols));
}
