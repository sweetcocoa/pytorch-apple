"""Tests for Metal compute kernels against PyTorch CPU reference."""

import os

import numpy as np
import numpy.testing as npt
import torch
import torch.nn.functional as F

from npu_compiler.constraint_checker import pad_channels
from npu_runtime.buffer import NPUBuffer
from tests.conftest import dispatch_1d, dispatch_tiled_2d, kernels_dir, make_params


def _alloc_4d(shape):
    """Compute alloc_shape for a 4D tensor with channel alignment."""
    N, C, H, W = shape
    return (N, pad_channels(C), H, W)


class TestConv2dKernel:
    def test_conv2d_basic(self, device):
        """Test conv2d 3x3 kernel against PyTorch."""
        N, C_in, H, W = 1, 3, 8, 8
        C_out, KH, KW = 16, 3, 3
        stride, pad = 1, 1

        input_np = np.random.randn(N, C_in, H, W).astype(np.float32)
        weight_np = np.random.randn(C_out, C_in, KH, KW).astype(np.float32)
        bias_np = np.random.randn(C_out).astype(np.float32)

        # PyTorch reference
        ref = F.conv2d(
            torch.tensor(input_np), torch.tensor(weight_np), torch.tensor(bias_np),
            stride=stride, padding=pad
        ).numpy()

        out_h, out_w = ref.shape[2], ref.shape[3]
        total = N * C_out * out_h * out_w

        buf_in = NPUBuffer.from_numpy(input_np, device, alloc_shape=_alloc_4d((N, C_in, H, W)))
        buf_w = NPUBuffer.from_numpy(weight_np, device)
        buf_b = NPUBuffer.from_numpy(bias_np, device)
        buf_out = NPUBuffer.zeros((N, C_out, out_h, out_w), device,
                                  alloc_shape=_alloc_4d((N, C_out, out_h, out_w)))

        # ConvParams: 19 uint32 fields
        params = make_params(
            device, "19I",
            N, C_in, H, W,           # batch, in_channels, in_h, in_w
            C_out, out_h, out_w,      # out_channels, out_h, out_w
            KH, KW,                   # kernel_h, kernel_w
            stride, stride,           # stride_h, stride_w
            pad, pad,                 # pad_h, pad_w
            1, 0, 0,                  # has_bias, has_bn, has_relu
            1,                        # groups
            pad_channels(C_in), pad_channels(C_out),  # aligned channels
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "conv_bn_relu.metal"))
        pipeline = device.get_pipeline(lib, "conv2d_kernel")

        dispatch_1d(device, pipeline, [buf_in, buf_w, buf_b, buf_out, params], total)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-2, atol=1e-2)

    def test_conv2d_no_bias(self, device):
        """Test conv2d without bias."""
        N, C_in, H, W = 1, 3, 8, 8
        C_out, KH, KW = 16, 3, 3

        input_np = np.random.randn(N, C_in, H, W).astype(np.float32)
        weight_np = np.random.randn(C_out, C_in, KH, KW).astype(np.float32)

        ref = F.conv2d(
            torch.tensor(input_np), torch.tensor(weight_np),
            stride=1, padding=1
        ).numpy()

        out_h, out_w = ref.shape[2], ref.shape[3]
        total = N * C_out * out_h * out_w

        buf_in = NPUBuffer.from_numpy(input_np, device, alloc_shape=_alloc_4d((N, C_in, H, W)))
        buf_w = NPUBuffer.from_numpy(weight_np, device)
        # Dummy bias buffer (not used since has_bias=0)
        buf_b = NPUBuffer.zeros((C_out,), device)
        buf_out = NPUBuffer.zeros((N, C_out, out_h, out_w), device,
                                  alloc_shape=_alloc_4d((N, C_out, out_h, out_w)))

        params = make_params(
            device, "19I",
            N, C_in, H, W, C_out, out_h, out_w,
            KH, KW, 1, 1, 1, 1,
            0, 0, 0, 1,
            pad_channels(C_in), pad_channels(C_out),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "conv_bn_relu.metal"))
        pipeline = device.get_pipeline(lib, "conv2d_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_w, buf_b, buf_out, params], total)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-2, atol=1e-2)

    def test_conv2d_with_relu(self, device):
        """Test conv2d + relu fused."""
        N, C_in, H, W = 1, 3, 8, 8
        C_out, KH, KW = 16, 3, 3

        input_np = np.random.randn(N, C_in, H, W).astype(np.float32)
        weight_np = np.random.randn(C_out, C_in, KH, KW).astype(np.float32)
        bias_np = np.random.randn(C_out).astype(np.float32)

        ref = F.relu(F.conv2d(
            torch.tensor(input_np), torch.tensor(weight_np), torch.tensor(bias_np),
            stride=1, padding=1
        )).numpy()

        out_h, out_w = ref.shape[2], ref.shape[3]
        total = N * C_out * out_h * out_w

        buf_in = NPUBuffer.from_numpy(input_np, device, alloc_shape=_alloc_4d((N, C_in, H, W)))
        buf_w = NPUBuffer.from_numpy(weight_np, device)
        buf_b = NPUBuffer.from_numpy(bias_np, device)
        buf_out = NPUBuffer.zeros((N, C_out, out_h, out_w), device,
                                  alloc_shape=_alloc_4d((N, C_out, out_h, out_w)))

        params = make_params(
            device, "19I",
            N, C_in, H, W, C_out, out_h, out_w,
            KH, KW, 1, 1, 1, 1,
            1, 0, 1, 1,  # has_bias=1, has_bn=0, has_relu=1
            pad_channels(C_in), pad_channels(C_out),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "conv_bn_relu.metal"))
        pipeline = device.get_pipeline(lib, "conv2d_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_w, buf_b, buf_out, params], total)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-2, atol=1e-2)


class TestConvBNReluFusion:
    def test_bn_folding_matches_unfused(self, device):
        """BN folding into conv weight/bias should match conv->BN->ReLU."""
        N, C_in, H, W = 1, 3, 8, 8
        C_out = 16

        input_np = np.random.randn(N, C_in, H, W).astype(np.float32)
        conv_w = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
        conv_b = np.random.randn(C_out).astype(np.float32)
        bn_mean = np.random.randn(C_out).astype(np.float32)
        bn_var = np.abs(np.random.randn(C_out).astype(np.float32)) + 0.1
        bn_gamma = np.random.randn(C_out).astype(np.float32)
        bn_beta = np.random.randn(C_out).astype(np.float32)

        # PyTorch reference: conv -> bn -> relu (unfused)
        inp_t = torch.tensor(input_np)
        out = F.conv2d(inp_t, torch.tensor(conv_w), torch.tensor(conv_b), padding=1)
        out = F.batch_norm(out, torch.tensor(bn_mean), torch.tensor(bn_var),
                           torch.tensor(bn_gamma), torch.tensor(bn_beta), training=False)
        ref = F.relu(out).numpy()

        # Fold BN into conv weights
        eps = 1e-5
        std = np.sqrt(bn_var + eps)
        scale = bn_gamma / std
        folded_w = conv_w * scale.reshape(-1, 1, 1, 1)
        folded_b = (conv_b - bn_mean) * scale + bn_beta

        out_h, out_w = ref.shape[2], ref.shape[3]
        total = N * C_out * out_h * out_w

        buf_in = NPUBuffer.from_numpy(input_np, device, alloc_shape=_alloc_4d((N, C_in, H, W)))
        buf_w = NPUBuffer.from_numpy(folded_w, device)
        buf_b = NPUBuffer.from_numpy(folded_b, device)
        buf_out = NPUBuffer.zeros((N, C_out, out_h, out_w), device,
                                  alloc_shape=_alloc_4d((N, C_out, out_h, out_w)))

        params = make_params(
            device, "19I",
            N, C_in, H, W, C_out, out_h, out_w,
            3, 3, 1, 1, 1, 1,
            1, 1, 1, 1,  # has_bias=1, has_bn=1, has_relu=1
            pad_channels(C_in), pad_channels(C_out),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "conv_bn_relu.metal"))
        pipeline = device.get_pipeline(lib, "conv2d_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_w, buf_b, buf_out, params], total)

        result = buf_out.to_numpy()
        # FP16 accumulation in folded BN weights introduces slightly more error
        npt.assert_allclose(result, ref, rtol=2e-2, atol=2e-2)


class TestAddReluKernel:
    def test_add_relu(self, device):
        a = np.random.randn(1, 64, 8, 8).astype(np.float32)
        b = np.random.randn(1, 64, 8, 8).astype(np.float32)
        ref = np.maximum(a + b, 0.0)

        alloc = _alloc_4d(a.shape)
        buf_a = NPUBuffer.from_numpy(a, device, alloc_shape=alloc)
        buf_b = NPUBuffer.from_numpy(b, device, alloc_shape=alloc)
        padded_total = int(np.prod(buf_a.alloc_shape))
        buf_out = NPUBuffer.zeros(a.shape, device, alloc_shape=alloc)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "add_relu.metal"))
        pipeline = device.get_pipeline(lib, "add_relu_kernel")
        dispatch_1d(device, pipeline, [buf_a, buf_b, buf_out], padded_total)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-2, atol=1e-3)

    def test_add_only(self, device):
        a = np.random.randn(256).astype(np.float32)
        b = np.random.randn(256).astype(np.float32)
        ref = a + b

        buf_a = NPUBuffer.from_numpy(a, device)
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_out = NPUBuffer.zeros(a.shape, device)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "add_relu.metal"))
        pipeline = device.get_pipeline(lib, "add_kernel")
        dispatch_1d(device, pipeline, [buf_a, buf_b, buf_out], 256)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-2, atol=1e-3)


class TestMatmulKernel:
    def test_linear(self, device):
        """Test matmul as PyTorch linear: out = input @ weight.T + bias."""
        M, K, N = 1, 128, 64
        input_np = np.random.randn(M, K).astype(np.float32)
        weight_np = np.random.randn(N, K).astype(np.float32)  # (N, K) like nn.Linear
        bias_np = np.random.randn(N).astype(np.float32)

        ref = (input_np @ weight_np.T + bias_np).astype(np.float32)

        buf_a = NPUBuffer.from_numpy(input_np, device)
        buf_w = NPUBuffer.from_numpy(weight_np, device)
        buf_b = NPUBuffer.from_numpy(bias_np, device)
        buf_out = NPUBuffer.zeros((M, N), device)

        params = make_params(device, "4I", M, N, K, 1)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "matmul.metal"))
        pipeline = device.get_pipeline(lib, "matmul_kernel")
        dispatch_tiled_2d(device, pipeline, [buf_a, buf_w, buf_b, buf_out, params], N, M)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-2, atol=1e-1)

    def test_matmul_no_bias(self, device):
        M, K, N = 2, 64, 32
        input_np = np.random.randn(M, K).astype(np.float32)
        weight_np = np.random.randn(N, K).astype(np.float32)

        ref = (input_np @ weight_np.T).astype(np.float32)

        buf_a = NPUBuffer.from_numpy(input_np, device)
        buf_w = NPUBuffer.from_numpy(weight_np, device)
        buf_b = NPUBuffer.zeros((N,), device)
        buf_out = NPUBuffer.zeros((M, N), device)

        params = make_params(device, "4I", M, N, K, 0)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "matmul.metal"))
        pipeline = device.get_pipeline(lib, "matmul_kernel")
        dispatch_tiled_2d(device, pipeline, [buf_a, buf_w, buf_b, buf_out, params], N, M)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-2, atol=1e-1)


class TestPoolKernel:
    def test_max_pool2d(self, device):
        N, C, H, W = 1, 16, 8, 8
        KH, KW, SH, SW = 2, 2, 2, 2

        input_np = np.random.randn(N, C, H, W).astype(np.float32)
        ref = F.max_pool2d(torch.tensor(input_np), kernel_size=2, stride=2).numpy()

        out_h, out_w = ref.shape[2], ref.shape[3]
        total = N * C * out_h * out_w

        buf_in = NPUBuffer.from_numpy(input_np, device, alloc_shape=_alloc_4d((N, C, H, W)))
        buf_out = NPUBuffer.zeros((N, C, out_h, out_w), device,
                                  alloc_shape=_alloc_4d((N, C, out_h, out_w)))

        params = make_params(
            device, "13I",
            N, C, H, W, out_h, out_w, KH, KW, SH, SW, 0, 0,
            pad_channels(C),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "pool.metal"))
        pipeline = device.get_pipeline(lib, "max_pool2d_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], total)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-3, atol=1e-3)

    def test_adaptive_avg_pool2d(self, device):
        """Test global average pooling (output 1x1)."""
        N, C, H, W = 1, 64, 7, 7
        out_h, out_w = 1, 1

        input_np = np.random.randn(N, C, H, W).astype(np.float32)
        ref = F.adaptive_avg_pool2d(torch.tensor(input_np), (1, 1)).numpy()

        total = N * C * out_h * out_w

        buf_in = NPUBuffer.from_numpy(input_np, device, alloc_shape=_alloc_4d((N, C, H, W)))
        buf_out = NPUBuffer.zeros((N, C, out_h, out_w), device,
                                  alloc_shape=_alloc_4d((N, C, out_h, out_w)))

        params = make_params(
            device, "13I",
            N, C, H, W, out_h, out_w, 0, 0, 0, 0, 0, 0,
            pad_channels(C),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "pool.metal"))
        pipeline = device.get_pipeline(lib, "adaptive_avg_pool2d_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], total)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-2, atol=1e-2)


class TestDepad4DKernel:
    def test_depad_strips_padding(self, device):
        """4D padded -> 2D dense should strip channel padding correctly."""
        N, C, H, W = 1, 64, 7, 7
        C_aligned = pad_channels(C)  # 64 -> 64 (already aligned)

        input_np = np.random.randn(N, C, H, W).astype(np.float32)
        ref = input_np.reshape(N, C * H * W)

        buf_in = NPUBuffer.from_numpy(input_np, device, alloc_shape=_alloc_4d((N, C, H, W)))
        dense_total = N * C * H * W
        buf_out = NPUBuffer.zeros((N, dense_total), device)

        params = make_params(device, "5I", N, C, C_aligned, H, W)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise.metal"))
        pipeline = device.get_pipeline(lib, "depad_4d_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], dense_total)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-3, atol=1e-3)

    def test_depad_unaligned_channels(self, device):
        """Test depad with channels not aligned to 32 (e.g., 3 channels)."""
        N, C, H, W = 1, 3, 4, 4
        C_aligned = pad_channels(C)  # 3 -> 32

        input_np = np.random.randn(N, C, H, W).astype(np.float32)
        ref = input_np.reshape(N, C * H * W)

        buf_in = NPUBuffer.from_numpy(input_np, device, alloc_shape=_alloc_4d((N, C, H, W)))
        dense_total = N * C * H * W
        buf_out = NPUBuffer.zeros((N, dense_total), device)

        params = make_params(device, "5I", N, C, C_aligned, H, W)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise.metal"))
        pipeline = device.get_pipeline(lib, "depad_4d_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], dense_total)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=1e-3, atol=1e-3)
