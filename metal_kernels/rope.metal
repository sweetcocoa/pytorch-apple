#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

struct RopeParams {
    uint seq_len;
    uint head_dim;
};

// Rotary Position Embedding (RoPE) kernel.
// inv_freq: (head_dim/2,) compute_t — precomputed inverse frequencies
// positions: (seq_len,) int32 — position indices
// out_cos: (1, seq_len, head_dim) compute_t
// out_sin: (1, seq_len, head_dim) compute_t
// 2D dispatch: (head_dim, seq_len)
kernel void rope_kernel(
    device const compute_t *inv_freq    [[buffer(0)]],
    device const int  *positions        [[buffer(1)]],
    device compute_t *out_cos           [[buffer(2)]],
    device compute_t *out_sin           [[buffer(3)]],
    constant RopeParams &p              [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint dim = tid.x;   // head_dim index
    uint seq = tid.y;   // seq_len index

    if (dim >= p.head_dim || seq >= p.seq_len) return;

    // inv_freq has head_dim/2 entries; cos/sin repeat for dim and dim+half
    uint half_dim = p.head_dim / 2;
    uint freq_idx = dim % half_dim;

    float freq = float(inv_freq[freq_idx]);
    float pos = float(positions[seq]);
    float angle = pos * freq;

    uint out_idx = seq * p.head_dim + dim;
    out_cos[out_idx] = compute_t(cos(angle));
    out_sin[out_idx] = compute_t(sin(angle));
}
