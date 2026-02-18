#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

kernel void add_relu_kernel(
    device const compute_t *a   [[buffer(0)]],
    device const compute_t *b   [[buffer(1)]],
    device compute_t *output    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    float val = float(a[tid]) + float(b[tid]);
    output[tid] = compute_t(max(val, 0.0f));
}

kernel void add_kernel(
    device const compute_t *a   [[buffer(0)]],
    device const compute_t *b   [[buffer(1)]],
    device compute_t *output    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = compute_t(float(a[tid]) + float(b[tid]));
}
