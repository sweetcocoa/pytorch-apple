#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// Fused decode attention: transpose(K) + Q×K^T + scale + add(mask) + softmax + attn×V
// Replaces ~6 separate dispatches per attention head per layer.
//
// Dispatch: B*H threadgroups, head_dim threads per group.
// Each threadgroup computes one head's attention output.
//
// Inputs:
//   Q:  (B*H, 1, D) contiguous as (B*H, D) — query for decode step
//   K:  (B*H, max_seq, D) — key cache (NOT transposed)
//   V:  (B*H, max_seq, D) — value cache
//   mask: (1, 1, 1, max_seq) — causal mask (broadcast over B,H)
//   cache_position: (1,) int32 — current decode position
// Output:
//   out: (B*H, 1, D) contiguous as (B*H, D)

struct FusedDecodeAttentionParams {
    uint batch_heads;   // B * H
    uint head_dim;      // D (e.g. 128)
    uint max_seq_len;   // S (e.g. 129)
    float scale;        // 1/sqrt(D)
};

kernel void fused_decode_attention_kernel(
    device const compute_t *Q              [[buffer(0)]],
    device const compute_t *K              [[buffer(1)]],
    device const compute_t *V              [[buffer(2)]],
    device const compute_t *mask           [[buffer(3)]],
    device const int       *cache_position [[buffer(4)]],
    device compute_t       *output         [[buffer(5)]],
    constant FusedDecodeAttentionParams &p [[buffer(6)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tpg   [[threads_per_threadgroup]]
) {
    if (gid >= p.batch_heads) return;

    uint D = p.head_dim;
    uint valid_len = uint(cache_position[0]) + 1;
    if (valid_len > p.max_seq_len) valid_len = p.max_seq_len;

    // Pointers for this head
    device const compute_t *q_ptr = Q + gid * D;
    device const compute_t *k_ptr = K + gid * p.max_seq_len * D;
    device const compute_t *v_ptr = V + gid * p.max_seq_len * D;

    // Shared memory: scores[max 2048] + q_shared[D]
    // max_seq_len <= 2048, D <= 256
    threadgroup float scores[2048];
    threadgroup float q_shared[256];

    // Load Q into shared memory (each thread loads its element)
    if (tid < D) {
        q_shared[tid] = float(q_ptr[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: Q × K^T (each thread handles ceil(valid_len/tpg) positions)
    // For each position i, compute dot(Q, K[i]) * scale + mask[i]
    for (uint i = tid; i < valid_len; i += tpg) {
        float dot = 0.0f;
        device const compute_t *k_row = k_ptr + i * D;
        for (uint d = 0; d < D; d++) {
            dot += q_shared[d] * float(k_row[d]);
        }
        // Apply scale and mask
        float m = float(mask[i]);  // mask is (1,1,1,S), broadcast
        scores[i] = dot * p.scale + m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Softmax over valid_len scores (thread 0 does serial)
    if (tid == 0) {
        // Find max
        float max_val = scores[0];
        for (uint i = 1; i < valid_len; i++) {
            max_val = max(max_val, scores[i]);
        }

        // Handle all-masked case
        if (max_val == -INFINITY) {
            for (uint i = 0; i < valid_len; i++) {
                scores[i] = 0.0f;
            }
        } else {
            // Exp and sum
            float sum = 0.0f;
            for (uint i = 0; i < valid_len; i++) {
                float e = exp(scores[i] - max_val);
                scores[i] = e;
                sum += e;
            }
            // Normalize
            float inv_sum = 1.0f / sum;
            for (uint i = 0; i < valid_len; i++) {
                scores[i] *= inv_sum;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Attn × V — thread d computes output[d] = sum(attn[i] * V[i,d])
    if (tid < D) {
        float acc = 0.0f;
        for (uint i = 0; i < valid_len; i++) {
            acc += scores[i] * float(v_ptr[i * D + tid]);
        }
        output[gid * D + tid] = compute_t(acc);
    }
}
