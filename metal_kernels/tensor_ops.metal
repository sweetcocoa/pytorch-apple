#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// ── Transpose: swap two dimensions ──

struct TransposeParams {
    uint ndim;
    uint dim0;
    uint dim1;
    uint total;
    uint shape[6];
    uint strides_in[6];
    uint strides_out[6];
};

kernel void transpose_kernel(
    device const compute_t *input     [[buffer(0)]],
    device compute_t *output          [[buffer(1)]],
    constant TransposeParams &p       [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;

    // Decompose output linear index into output multi-index
    uint remaining = tid;
    uint out_idx[6];
    for (int d = int(p.ndim) - 1; d >= 0; d--) {
        // Use strides_out to recover coordinates
        // Actually compute from output shape (which is input shape with dim0,dim1 swapped)
        uint dim_size;
        if (uint(d) == p.dim0) {
            dim_size = p.shape[p.dim1];
        } else if (uint(d) == p.dim1) {
            dim_size = p.shape[p.dim0];
        } else {
            dim_size = p.shape[d];
        }
        out_idx[d] = remaining % dim_size;
        remaining /= dim_size;
    }

    // Map output index to input index by swapping dim0 and dim1
    uint in_linear = 0;
    for (uint d = 0; d < p.ndim; d++) {
        uint coord;
        if (d == p.dim0) {
            coord = out_idx[p.dim1];
        } else if (d == p.dim1) {
            coord = out_idx[p.dim0];
        } else {
            coord = out_idx[d];
        }
        in_linear += coord * p.strides_in[d];
    }

    output[tid] = input[in_linear];
}

// ── Cat: concatenate two tensors along an axis ──

struct CatParams {
    uint axis;
    uint ndim;
    uint total;
    uint in1_axis_size;
    uint out_shape[6];
    uint strides[6];
};

kernel void cat_2_kernel(
    device const compute_t *input1   [[buffer(0)]],
    device const compute_t *input2   [[buffer(1)]],
    device compute_t *output         [[buffer(2)]],
    constant CatParams &p            [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;

    // Decompose output linear index into multi-index
    uint remaining = tid;
    uint idx[6];
    for (int d = int(p.ndim) - 1; d >= 0; d--) {
        idx[d] = remaining % p.out_shape[d];
        remaining /= p.out_shape[d];
    }

    uint axis_coord = idx[p.axis];

    if (axis_coord < p.in1_axis_size) {
        // From input1: compute linear index in input1
        uint in1_linear = 0;
        uint in1_stride = 1;
        for (int d = int(p.ndim) - 1; d >= 0; d--) {
            in1_linear += idx[d] * in1_stride;
            if (uint(d) == p.axis) {
                in1_stride *= p.in1_axis_size;
            } else {
                in1_stride *= p.out_shape[d];
            }
        }
        output[tid] = input1[in1_linear];
    } else {
        // From input2: adjust axis coordinate
        idx[p.axis] -= p.in1_axis_size;
        uint in2_axis_size = p.out_shape[p.axis] - p.in1_axis_size;
        uint in2_linear = 0;
        uint in2_stride = 1;
        for (int d = int(p.ndim) - 1; d >= 0; d--) {
            in2_linear += idx[d] * in2_stride;
            if (uint(d) == p.axis) {
                in2_stride *= in2_axis_size;
            } else {
                in2_stride *= p.out_shape[d];
            }
        }
        output[tid] = input2[in2_linear];
    }
}

// ── Slice: extract a sub-tensor along one dimension ──

struct SliceParams {
    uint dim;
    uint start;
    uint end;
    uint step;
    uint ndim;
    uint total;
    uint in_shape[6];
    uint in_strides[6];
};

kernel void slice_kernel(
    device const compute_t *input   [[buffer(0)]],
    device compute_t *output        [[buffer(1)]],
    constant SliceParams &p         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;

    // Compute output shape (same as in_shape except sliced dim)
    uint out_shape[6];
    for (uint d = 0; d < p.ndim; d++) {
        if (d == p.dim) {
            out_shape[d] = (p.end - p.start + p.step - 1) / p.step;
        } else {
            out_shape[d] = p.in_shape[d];
        }
    }

    // Decompose output index
    uint remaining = tid;
    uint idx[6];
    for (int d = int(p.ndim) - 1; d >= 0; d--) {
        idx[d] = remaining % out_shape[d];
        remaining /= out_shape[d];
    }

    // Map to input index
    uint in_linear = 0;
    for (uint d = 0; d < p.ndim; d++) {
        uint coord = idx[d];
        if (d == p.dim) {
            coord = p.start + coord * p.step;
        }
        in_linear += coord * p.in_strides[d];
    }

    output[tid] = input[in_linear];
}

// ── Expand: broadcast copy ──

struct ExpandParams {
    uint ndim;
    uint total;
    uint in_shape[6];
    uint out_shape[6];
    uint in_strides[6];
};

kernel void expand_kernel(
    device const compute_t *input    [[buffer(0)]],
    device compute_t *output         [[buffer(1)]],
    constant ExpandParams &p         [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;

    // Decompose output index
    uint remaining = tid;
    uint in_linear = 0;
    for (int d = int(p.ndim) - 1; d >= 0; d--) {
        uint coord = remaining % p.out_shape[d];
        remaining /= p.out_shape[d];
        // If input dim is 1 (broadcast), stride is 0 so coord doesn't matter
        in_linear += coord * p.in_strides[d];
    }

    output[tid] = input[in_linear];
}
