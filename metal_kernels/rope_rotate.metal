#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// Fused Rotary Position Embedding (RoPE) rotation kernel.
// Replaces 7 separate ops: mul(cos) + slice + slice + neg + cat + mul(sin) + add
//
// Applies: output = x * cos + rotate_half(x) * sin
// where rotate_half([x1, x2]) = [-x2, x1]  (halves along last dim)
//
// Inputs:
//   x:   (B, H, S, D) — input tensor (Q or K heads)
//   cos: (1, 1, S, D) — cosine values (broadcast over B, H)
//   sin: (1, 1, S, D) — sine values (broadcast over B, H)
// Output:
//   out: (B, H, S, D) — rotated tensor
//
// 1D dispatch: total_elements = B * H * S * D

struct RopeRotateParams {
    uint total_elements;
    uint head_dim;  // D
    uint seq_len;   // S
};

kernel void rope_rotate_kernel(
    device const compute_t *x       [[buffer(0)]],
    device const compute_t *cos_buf [[buffer(1)]],
    device const compute_t *sin_buf [[buffer(2)]],
    device compute_t       *output  [[buffer(3)]],
    constant RopeRotateParams &p    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.total_elements) return;

    uint d = gid % p.head_dim;
    uint s_idx = (gid / p.head_dim) % p.seq_len;
    uint half_dim = p.head_dim / 2;

    // cos/sin are (1, 1, S, D) — index by (s_idx, d)
    uint cs_idx = s_idx * p.head_dim + d;
    float c = float(cos_buf[cs_idx]);
    float s = float(sin_buf[cs_idx]);
    float xi = float(x[gid]);

    // rotate_half: for d < D/2, partner is x[d + D/2] (negated)
    //              for d >= D/2, partner is x[d - D/2]
    uint base = gid - d;  // start of this element's last-dim slice
    float partner;
    if (d < half_dim) {
        partner = -float(x[base + d + half_dim]);
    } else {
        partner = float(x[base + d - half_dim]);
    }

    output[gid] = compute_t(xi * c + partner * s);
}
