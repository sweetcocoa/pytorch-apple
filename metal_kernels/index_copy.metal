#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// index_copy(input, dim, index, source) → output
// Copies source values into input at positions specified by index along `dim`.
// output[..., index[l], ...] = source[..., l, ...] for each l in [0, num_indices)
// All other positions: output[i] = input[i]

struct IndexCopyParams {
    uint outer_size;   // product of dims before `dim` (B * H)
    uint dim_size;     // size along `dim` (max_seq)
    uint inner_size;   // product of dims after `dim` (D)
    uint num_indices;  // number of positions to update (L)
};

kernel void index_copy_kernel(
    device const compute_t *input   [[buffer(0)]],   // (outer, dim_size, inner)
    device const compute_t *source  [[buffer(1)]],   // (outer, num_indices, inner)
    device const int       *indices [[buffer(2)]],   // (num_indices,) int32
    device compute_t       *output  [[buffer(3)]],   // same shape as input
    constant IndexCopyParams &p     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.outer_size * p.dim_size * p.inner_size;
    if (tid >= total) return;

    uint inner = tid % p.inner_size;
    uint dim_pos = (tid / p.inner_size) % p.dim_size;
    uint outer = tid / (p.inner_size * p.dim_size);

    // Check if this position should be replaced by source
    for (uint l = 0; l < p.num_indices; l++) {
        if (dim_pos == (uint)indices[l]) {
            uint src_idx = outer * p.num_indices * p.inner_size + l * p.inner_size + inner;
            output[tid] = source[src_idx];
            return;
        }
    }
    output[tid] = input[tid];
}
