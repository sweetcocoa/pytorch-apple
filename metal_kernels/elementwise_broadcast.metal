#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// Broadcast binary ops: fuse expand + elementwise into a single dispatch.
// Stride-based indexing: stride=0 means broadcast along that dimension.

struct BroadcastBinaryParams {
    uint ndim;
    uint total;
    uint a_strides[6];
    uint b_strides[6];
    uint out_shape[6];
};

// Helper: decompose linear index into multi-index, compute input offsets via strides.
inline uint2 broadcast_indices(uint tid, constant BroadcastBinaryParams &p) {
    uint remaining = tid;
    uint a_offset = 0;
    uint b_offset = 0;
    for (int d = int(p.ndim) - 1; d >= 0; d--) {
        uint coord = remaining % p.out_shape[d];
        remaining /= p.out_shape[d];
        a_offset += coord * p.a_strides[d];
        b_offset += coord * p.b_strides[d];
    }
    return uint2(a_offset, b_offset);
}

kernel void mul_broadcast_kernel(
    device const compute_t *a [[buffer(0)]],
    device const compute_t *b [[buffer(1)]],
    device compute_t *output  [[buffer(2)]],
    constant BroadcastBinaryParams &p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    uint2 idx = broadcast_indices(tid, p);
    output[tid] = compute_t(float(a[idx.x]) * float(b[idx.y]));
}

kernel void add_broadcast_kernel(
    device const compute_t *a [[buffer(0)]],
    device const compute_t *b [[buffer(1)]],
    device compute_t *output  [[buffer(2)]],
    constant BroadcastBinaryParams &p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    uint2 idx = broadcast_indices(tid, p);
    output[tid] = compute_t(float(a[idx.x]) + float(b[idx.y]));
}

kernel void div_broadcast_kernel(
    device const compute_t *a [[buffer(0)]],
    device const compute_t *b [[buffer(1)]],
    device compute_t *output  [[buffer(2)]],
    constant BroadcastBinaryParams &p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    uint2 idx = broadcast_indices(tid, p);
    output[tid] = compute_t(float(a[idx.x]) / float(b[idx.y]));
}
