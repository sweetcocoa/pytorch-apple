#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

struct PoolParams {
    uint batch;
    uint channels;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
    uint channels_aligned;  // aligned channel count (64-byte aligned)
};

// Max pool 2D
kernel void max_pool2d_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    constant PoolParams &p         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.batch * p.channels * p.out_h * p.out_w;
    if (tid >= total) return;

    uint ow = tid % p.out_w;
    uint oh = (tid / p.out_w) % p.out_h;
    uint c  = (tid / (p.out_w * p.out_h)) % p.channels;
    uint n  = tid / (p.out_w * p.out_h * p.channels);

    float max_val = -INFINITY;

    for (uint kh = 0; kh < p.kernel_h; kh++) {
        for (uint kw = 0; kw < p.kernel_w; kw++) {
            int ih = (int)(oh * p.stride_h + kh) - (int)p.pad_h;
            int iw = (int)(ow * p.stride_w + kw) - (int)p.pad_w;

            if (ih >= 0 && ih < (int)p.in_h && iw >= 0 && iw < (int)p.in_w) {
                uint idx = ((n * p.channels_aligned + c) * p.in_h + (uint)ih) * p.in_w + (uint)iw;
                max_val = max(max_val, float(input[idx]));
            }
        }
    }

    uint out_idx = ((n * p.channels_aligned + c) * p.out_h + oh) * p.out_w + ow;
    output[out_idx] = compute_t(max_val);
}

// Adaptive average pool 2D (global average pooling: out_h=1, out_w=1)
kernel void adaptive_avg_pool2d_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    constant PoolParams &p         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = p.batch * p.channels * p.out_h * p.out_w;
    if (tid >= total) return;

    uint ow = tid % p.out_w;
    uint oh = (tid / p.out_w) % p.out_h;
    uint c  = (tid / (p.out_w * p.out_h)) % p.channels;
    uint n  = tid / (p.out_w * p.out_h * p.channels);

    // Compute adaptive window boundaries
    uint ih_start = oh * p.in_h / p.out_h;
    uint ih_end   = (oh + 1) * p.in_h / p.out_h;
    uint iw_start = ow * p.in_w / p.out_w;
    uint iw_end   = (ow + 1) * p.in_w / p.out_w;

    float sum = 0.0;
    uint count = 0;

    for (uint ih = ih_start; ih < ih_end; ih++) {
        for (uint iw = iw_start; iw < iw_end; iw++) {
            uint idx = ((n * p.channels_aligned + c) * p.in_h + ih) * p.in_w + iw;
            sum += float(input[idx]);
            count++;
        }
    }

    uint out_idx = ((n * p.channels_aligned + c) * p.out_h + oh) * p.out_w + ow;
    output[out_idx] = compute_t(sum / float(count));
}
