#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// Tile size for shared-memory tiled matmul.
// Threadgroup is always TILE x TILE threads.
#define TILE 16

struct MatmulParams {
    uint M;     // rows of A / output
    uint N;     // cols of B / output
    uint K;     // cols of A / rows of B
    uint has_bias;
};

// Tiled matmul: C = A @ B^T + bias
// A: (M, K), B: (N, K) stored row-major (weight matrix, transposed), C: (M, N)
// Dispatch: threadgroups = (ceil(N/TILE), ceil(M/TILE)), threads_per_tg = (TILE, TILE)
kernel void matmul_kernel(
    device const compute_t *A      [[buffer(0)]],
    device const compute_t *B      [[buffer(1)]],
    device const compute_t *bias   [[buffer(2)]],
    device compute_t *C            [[buffer(3)]],
    constant MatmulParams &p       [[buffer(4)]],
    uint2 tg_pos   [[threadgroup_position_in_grid]],
    uint2 local_tid [[thread_position_in_threadgroup]]
) {
    uint tx = local_tid.x;  // N direction (column)
    uint ty = local_tid.y;  // M direction (row)

    uint global_row = tg_pos.y * TILE + ty;
    uint global_col = tg_pos.x * TILE + tx;

    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    float sum = 0.0f;

    for (uint k_base = 0; k_base < p.K; k_base += TILE) {
        // Load A tile: As[m_local][k_local]
        // Thread (tx, ty): ty → row offset in tile, tx → k offset
        uint a_col = k_base + tx;
        As[ty][tx] = (global_row < p.M && a_col < p.K)
            ? float(A[global_row * p.K + a_col]) : 0.0f;

        // Load B tile: Bs[n_local][k_local] (B is N x K, transposed weight)
        // Thread (tx, ty): tx → N offset in tile, ty → k offset
        uint b_col = k_base + ty;
        Bs[tx][ty] = (global_col < p.N && b_col < p.K)
            ? float(B[global_col * p.K + b_col]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // C[row, col] = sum_k A[row, k] * B[col, k]
        for (uint kk = 0; kk < TILE; kk++) {
            sum += As[ty][kk] * Bs[tx][kk];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_row < p.M && global_col < p.N) {
        if (p.has_bias) {
            sum += float(bias[global_col]);
        }
        C[global_row * p.N + global_col] = compute_t(sum);
    }
}

// Tiled non-transposed matmul: C = A @ B (B is NOT transposed)
// A: (M, K), B: (K, N), C: (M, N)
// Dispatch: threadgroups = (ceil(N/TILE), ceil(M/TILE)), threads_per_tg = (TILE, TILE)
kernel void matmul_notrans_kernel(
    device const compute_t *A      [[buffer(0)]],
    device const compute_t *B      [[buffer(1)]],
    device compute_t *C            [[buffer(2)]],
    constant MatmulParams &p       [[buffer(3)]],
    uint2 tg_pos   [[threadgroup_position_in_grid]],
    uint2 local_tid [[thread_position_in_threadgroup]]
) {
    uint tx = local_tid.x;  // N direction
    uint ty = local_tid.y;  // M direction

    uint global_row = tg_pos.y * TILE + ty;
    uint global_col = tg_pos.x * TILE + tx;

    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    float sum = 0.0f;

    for (uint k_base = 0; k_base < p.K; k_base += TILE) {
        // Load A tile: As[m_local][k_local]
        uint a_col = k_base + tx;
        As[ty][tx] = (global_row < p.M && a_col < p.K)
            ? float(A[global_row * p.K + a_col]) : 0.0f;

        // Load B tile: Bs[k_local][n_local] (B is K x N)
        // Thread (tx, ty): ty → k offset, tx → N offset
        uint b_row = k_base + ty;
        Bs[ty][tx] = (b_row < p.K && global_col < p.N)
            ? float(B[b_row * p.N + global_col]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // C[row, col] = sum_k A[row, k] * B[k, col]
        for (uint kk = 0; kk < TILE; kk++) {
            sum += As[ty][kk] * Bs[kk][tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_row < p.M && global_col < p.N) {
        C[global_row * p.N + global_col] = compute_t(sum);
    }
}

// Vector-matrix multiply: C = A @ B^T + bias (M=1, parallel K-reduction)
// A: (1, K), B: (N, K), C: (1, N)
// Each threadgroup computes one output element; threads split the K reduction.
// Dispatch: N threadgroups of VEC_TPG threads.
#define VEC_TPG 256

kernel void matmul_vec_kernel(
    device const compute_t *A      [[buffer(0)]],
    device const compute_t *B      [[buffer(1)]],
    device const compute_t *bias   [[buffer(2)]],
    device compute_t *C            [[buffer(3)]],
    constant MatmulParams &p       [[buffer(4)]],
    uint tg_id    [[threadgroup_position_in_grid]],
    uint local_id [[thread_index_in_threadgroup]]
) {
    if (tg_id >= p.N) return;

    float sum = 0.0f;
    uint b_base = tg_id * p.K;
    for (uint k = local_id; k < p.K; k += VEC_TPG) {
        sum += float(A[k]) * float(B[b_base + k]);
    }

    threadgroup float shared[VEC_TPG];
    shared[local_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = VEC_TPG / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared[local_id] += shared[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        float val = shared[0];
        if (p.has_bias) val += float(bias[tg_id]);
        C[tg_id] = compute_t(val);
    }
}

// Vector-matrix multiply: C = A @ B (non-transposed, M=1, parallel K-reduction)
// A: (1, K), B: (K, N), C: (1, N)
kernel void matmul_notrans_vec_kernel(
    device const compute_t *A      [[buffer(0)]],
    device const compute_t *B      [[buffer(1)]],
    device compute_t *C            [[buffer(2)]],
    constant MatmulParams &p       [[buffer(3)]],
    uint tg_id    [[threadgroup_position_in_grid]],
    uint local_id [[thread_index_in_threadgroup]]
) {
    if (tg_id >= p.N) return;

    float sum = 0.0f;
    for (uint k = local_id; k < p.K; k += VEC_TPG) {
        sum += float(A[k]) * float(B[k * p.N + tg_id]);
    }

    threadgroup float shared[VEC_TPG];
    shared[local_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = VEC_TPG / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared[local_id] += shared[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        C[tg_id] = compute_t(shared[0]);
    }
}

// Batched matmul: C[b] = A[b] @ B[b]
// A: (batch, M, K), B: (batch, K, N), C: (batch, M, N)
// 3D dispatch: (N, M, batch)
struct BatchedMatmulParams {
    uint batch;
    uint M;
    uint N;
    uint K;
};

kernel void batched_matmul_kernel(
    device const compute_t *A             [[buffer(0)]],
    device const compute_t *B             [[buffer(1)]],
    device compute_t *C                   [[buffer(2)]],
    constant BatchedMatmulParams &p       [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;    // N
    uint row = tid.y;    // M
    uint batch = tid.z;  // batch

    if (batch >= p.batch || row >= p.M || col >= p.N) return;

    uint a_offset = batch * p.M * p.K;
    uint b_offset = batch * p.K * p.N;
    uint c_offset = batch * p.M * p.N;

    float sum = 0.0;
    for (uint k = 0; k < p.K; k++) {
        sum += float(A[a_offset + row * p.K + k]) * float(B[b_offset + k * p.N + col]);
    }

    C[c_offset + row * p.N + col] = compute_t(sum);
}
