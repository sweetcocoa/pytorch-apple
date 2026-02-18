#include <metal_stdlib>
using namespace metal;

// Utility: convert linear index to NCHW coordinates
inline uint4 idx_to_nchw(uint idx, uint C, uint H, uint W) {
    uint w = idx % W;
    uint h = (idx / W) % H;
    uint c = (idx / (W * H)) % C;
    uint n = idx / (W * H * C);
    return uint4(n, c, h, w);
}

// Utility: convert NCHW coordinates to linear index
inline uint nchw_to_idx(uint n, uint c, uint h, uint w, uint C, uint H, uint W) {
    return ((n * C + c) * H + h) * W + w;
}
