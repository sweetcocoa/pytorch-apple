#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

struct EmbeddingParams {
    uint seq_len;
    uint embed_dim;
    uint vocab_size;
};

// input_ids: (seq_len,) uint32 indices
// weight: (vocab_size, embed_dim) compute_t
// output: (seq_len, embed_dim) compute_t
// 2D dispatch: (embed_dim, seq_len)
kernel void embedding_kernel(
    device const uint *input_ids      [[buffer(0)]],
    device const compute_t *weight    [[buffer(1)]],
    device compute_t *output          [[buffer(2)]],
    constant EmbeddingParams &p       [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;  // embed_dim
    uint row = tid.y;  // seq_len

    if (row >= p.seq_len || col >= p.embed_dim) return;

    uint token_id = input_ids[row];
    output[row * p.embed_dim + col] = weight[token_id * p.embed_dim + col];
}
