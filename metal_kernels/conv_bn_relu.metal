#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// Parameters passed as a constant buffer
struct ConvParams {
    uint batch;
    uint in_channels;
    uint in_h;
    uint in_w;
    uint out_channels;
    uint out_h;
    uint out_w;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
    uint has_bias;      // 1 if bias is present
    uint has_bn;        // 1 if BN params are folded
    uint has_relu;      // 1 if ReLU activation
    uint groups;        // for grouped convolution
    uint in_channels_aligned;   // aligned input channels (64-byte aligned)
    uint out_channels_aligned;  // aligned output channels (64-byte aligned)
};

// Conv2d kernel with optional bias and ReLU.
// BN is folded into weight/bias at compile time by graph_optimizer.
kernel void conv2d_kernel(
    device const compute_t *input   [[buffer(0)]],
    device const compute_t *weight  [[buffer(1)]],
    device const compute_t *bias    [[buffer(2)]],
    device compute_t *output        [[buffer(3)]],
    constant ConvParams &p          [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread computes one output element
    uint total = p.batch * p.out_channels * p.out_h * p.out_w;
    if (tid >= total) return;

    uint ow = tid % p.out_w;
    uint oh = (tid / p.out_w) % p.out_h;
    uint oc = (tid / (p.out_w * p.out_h)) % p.out_channels;
    uint n  = tid / (p.out_w * p.out_h * p.out_channels);

    uint group_size_in = p.in_channels / p.groups;
    uint group_size_out = p.out_channels / p.groups;
    uint g = oc / group_size_out;

    float sum = 0.0;

    for (uint ic = 0; ic < group_size_in; ic++) {
        uint abs_ic = g * group_size_in + ic;
        for (uint kh = 0; kh < p.kernel_h; kh++) {
            for (uint kw = 0; kw < p.kernel_w; kw++) {
                int ih = (int)(oh * p.stride_h + kh) - (int)p.pad_h;
                int iw = (int)(ow * p.stride_w + kw) - (int)p.pad_w;

                if (ih >= 0 && ih < (int)p.in_h && iw >= 0 && iw < (int)p.in_w) {
                    uint in_idx = ((n * p.in_channels_aligned + abs_ic) * p.in_h + (uint)ih) * p.in_w + (uint)iw;
                    uint w_idx = ((oc * group_size_in + ic) * p.kernel_h + kh) * p.kernel_w + kw;
                    sum += float(input[in_idx]) * float(weight[w_idx]);
                }
            }
        }
    }

    if (p.has_bias) {
        sum += float(bias[oc]);
    }

    if (p.has_relu) {
        sum = max(sum, 0.0f);
    }

    uint out_idx = ((n * p.out_channels_aligned + oc) * p.out_h + oh) * p.out_w + ow;
    output[out_idx] = compute_t(sum);
}
