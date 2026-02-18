"""Fusion ON/OFF correctness comparison tests.

Each test verifies that fused kernel output matches the unfused (individual op)
output within tolerance, catching regressions in fused kernel implementations.
"""

import os

import numpy as np
import numpy.testing as npt
import torch

from npu_runtime.buffer import NPUBuffer
from tests.conftest import dispatch_1d, dispatch_2d, kernels_dir, make_params


class TestRMSNormFusion:
    """Compare fused RMSNorm kernel vs decomposed 8-step implementation."""

    def test_rmsnorm_fused_vs_decomposed(self, device):
        seq_len, hidden = 4, 64
        eps = 1e-6

        np.random.seed(42)
        x = np.random.randn(seq_len, hidden).astype(np.float32)
        weight = np.random.randn(hidden).astype(np.float32)

        # PyTorch reference
        x_t = torch.tensor(x)
        variance = x_t.pow(2).mean(-1, keepdim=True)
        ref = (x_t * torch.rsqrt(variance + eps) * torch.tensor(weight)).numpy()

        # --- Unfused (decomposed, 8 steps) ---
        lib_ew = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        lib_add = device.compile_metal_file(os.path.join(kernels_dir(), "add_relu.metal"))
        lib_tensor = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))
        total = seq_len * hidden

        buf_x = NPUBuffer.from_numpy(x, device)
        buf_x2 = NPUBuffer.zeros((seq_len, hidden), device)
        pow_params = make_params(device, "f", 2.0)
        dispatch_1d(device, device.get_pipeline(lib_ew, "pow_scalar_kernel"),
                     [buf_x, buf_x2, pow_params], total)

        buf_mean = NPUBuffer.zeros((seq_len,), device)
        mean_params = make_params(device, "2I", seq_len, hidden)
        dispatch_1d(device, device.get_pipeline(lib_ew, "mean_last_dim_kernel"),
                     [buf_x2, buf_mean, mean_params], seq_len)

        eps_arr = np.full(seq_len, eps, dtype=np.float32)
        buf_eps = NPUBuffer.from_numpy(eps_arr, device)
        buf_var_eps = NPUBuffer.zeros((seq_len,), device)
        dispatch_1d(device, device.get_pipeline(lib_add, "add_kernel"),
                     [buf_mean, buf_eps, buf_var_eps], seq_len)

        buf_rsqrt = NPUBuffer.zeros((seq_len,), device)
        dispatch_1d(device, device.get_pipeline(lib_ew, "rsqrt_kernel"),
                     [buf_var_eps, buf_rsqrt], seq_len)

        buf_rsqrt_exp = NPUBuffer.zeros((seq_len, hidden), device)
        expand_params = make_params(
            device, "2I6I6I6I",
            2, total,
            seq_len, 1, 0, 0, 0, 0,
            seq_len, hidden, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0,
        )
        dispatch_1d(device, device.get_pipeline(lib_tensor, "expand_kernel"),
                     [buf_rsqrt, buf_rsqrt_exp, expand_params], total)

        buf_normed = NPUBuffer.zeros((seq_len, hidden), device)
        dispatch_1d(device, device.get_pipeline(lib_ew, "mul_kernel"),
                     [buf_x, buf_rsqrt_exp, buf_normed], total)

        buf_weight = NPUBuffer.from_numpy(weight, device)
        buf_weight_exp = NPUBuffer.zeros((seq_len, hidden), device)
        expand_w_params = make_params(
            device, "2I6I6I6I",
            2, total,
            1, hidden, 0, 0, 0, 0,
            seq_len, hidden, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
        )
        dispatch_1d(device, device.get_pipeline(lib_tensor, "expand_kernel"),
                     [buf_weight, buf_weight_exp, expand_w_params], total)

        buf_unfused = NPUBuffer.zeros((seq_len, hidden), device)
        dispatch_1d(device, device.get_pipeline(lib_ew, "mul_kernel"),
                     [buf_normed, buf_weight_exp, buf_unfused], total)

        unfused_result = buf_unfused.to_numpy()

        # --- Fused (single RMSNorm kernel) ---
        lib_rmsnorm = device.compile_metal_file(os.path.join(kernels_dir(), "rmsnorm.metal"))
        buf_x2 = NPUBuffer.from_numpy(x, device)
        buf_fused = NPUBuffer.zeros((seq_len, hidden), device)
        rmsnorm_params = make_params(device, "2If", seq_len, hidden, eps)
        dispatch_2d(device, device.get_pipeline(lib_rmsnorm, "rmsnorm_kernel"),
                     [buf_x2, buf_weight, buf_fused, rmsnorm_params], hidden, seq_len)

        fused_result = buf_fused.to_numpy()

        # Both should match reference
        npt.assert_allclose(unfused_result, ref, rtol=5e-2, atol=5e-2)
        npt.assert_allclose(fused_result, ref, rtol=5e-2, atol=5e-2)
        # Fused and unfused should match each other
        npt.assert_allclose(fused_result, unfused_result, rtol=5e-2, atol=5e-2)


class TestSiLUMulFusion:
    """Compare fused silu_mul kernel vs separate silu + mul."""

    def test_silu_mul_fused_vs_separate(self, device):
        total = 256
        np.random.seed(42)
        gate = np.random.randn(total).astype(np.float32)
        up = np.random.randn(total).astype(np.float32)

        # PyTorch reference
        ref = (torch.nn.functional.silu(torch.tensor(gate)) * torch.tensor(up)).numpy()

        lib_ew = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))

        # --- Unfused: silu then mul ---
        buf_gate = NPUBuffer.from_numpy(gate, device)
        buf_up = NPUBuffer.from_numpy(up, device)
        buf_silu = NPUBuffer.zeros((total,), device)
        dispatch_1d(device, device.get_pipeline(lib_ew, "silu_kernel"),
                     [buf_gate, buf_silu], total)
        buf_unfused = NPUBuffer.zeros((total,), device)
        dispatch_1d(device, device.get_pipeline(lib_ew, "mul_kernel"),
                     [buf_silu, buf_up, buf_unfused], total)
        unfused_result = buf_unfused.to_numpy()

        # --- Fused: silu_mul_kernel ---
        buf_gate2 = NPUBuffer.from_numpy(gate, device)
        buf_up2 = NPUBuffer.from_numpy(up, device)
        buf_fused = NPUBuffer.zeros((total,), device)
        dispatch_1d(device, device.get_pipeline(lib_ew, "silu_mul_kernel"),
                     [buf_gate2, buf_up2, buf_fused], total)
        fused_result = buf_fused.to_numpy()

        npt.assert_allclose(unfused_result, ref, rtol=1e-2, atol=1e-2)
        npt.assert_allclose(fused_result, ref, rtol=1e-2, atol=1e-2)
        npt.assert_allclose(fused_result, unfused_result, rtol=1e-3, atol=1e-3)


class TestAddReluFusion:
    """Compare fused add_relu kernel vs separate add + relu."""

    def test_add_relu_fused_vs_separate(self, device):
        total = 256
        np.random.seed(42)
        a = np.random.randn(total).astype(np.float32)
        b = np.random.randn(total).astype(np.float32)

        ref = np.maximum(a + b, 0)

        lib_add = device.compile_metal_file(os.path.join(kernels_dir(), "add_relu.metal"))
        lib_ew = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise.metal"))

        # --- Unfused: add then relu ---
        buf_a = NPUBuffer.from_numpy(a, device)
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_sum = NPUBuffer.zeros((total,), device)
        dispatch_1d(device, device.get_pipeline(lib_add, "add_kernel"),
                     [buf_a, buf_b, buf_sum], total)
        buf_unfused = NPUBuffer.zeros((total,), device)
        dispatch_1d(device, device.get_pipeline(lib_ew, "elementwise_relu"),
                     [buf_sum, buf_unfused], total)
        unfused_result = buf_unfused.to_numpy()

        # --- Fused: add_relu_kernel ---
        buf_a2 = NPUBuffer.from_numpy(a, device)
        buf_b2 = NPUBuffer.from_numpy(b, device)
        buf_fused = NPUBuffer.zeros((total,), device)
        dispatch_1d(device, device.get_pipeline(lib_add, "add_relu_kernel"),
                     [buf_a2, buf_b2, buf_fused], total)
        fused_result = buf_fused.to_numpy()

        npt.assert_allclose(unfused_result, ref, rtol=1e-3, atol=1e-3)
        npt.assert_allclose(fused_result, ref, rtol=1e-3, atol=1e-3)
        npt.assert_allclose(fused_result, unfused_result, rtol=1e-5, atol=1e-5)


class TestMaskedSoftmaxFusion:
    """Compare fused masked softmax vs separate add + softmax."""

    def test_masked_softmax_fused_vs_separate(self, device):
        rows, cols = 4, 16
        np.random.seed(42)
        scores = np.random.randn(rows, cols).astype(np.float32)
        mask = np.zeros((rows, cols), dtype=np.float32)
        mask[:, cols // 2:] = -1e4  # mask out second half

        ref = torch.softmax(torch.tensor(scores + mask), dim=-1).numpy()

        lib_add = device.compile_metal_file(os.path.join(kernels_dir(), "add_relu.metal"))
        lib_softmax = device.compile_metal_file(os.path.join(kernels_dir(), "softmax.metal"))

        # --- Unfused: add then softmax ---
        buf_scores = NPUBuffer.from_numpy(scores, device)
        buf_mask = NPUBuffer.from_numpy(mask, device)
        buf_masked = NPUBuffer.zeros((rows, cols), device)
        dispatch_1d(device, device.get_pipeline(lib_add, "add_kernel"),
                     [buf_scores, buf_mask, buf_masked], rows * cols)

        buf_unfused = NPUBuffer.zeros((rows, cols), device)
        softmax_params = make_params(device, "2I", rows, cols)
        dispatch_1d(device, device.get_pipeline(lib_softmax, "softmax_kernel"),
                     [buf_masked, buf_unfused, softmax_params], rows)
        unfused_result = buf_unfused.to_numpy()

        # --- Fused: masked_softmax_broadcast_kernel ---
        # Param format: "2II6I6I" → rows, cols, ndim, mask_strides[6], out_shape[6]
        buf_scores2 = NPUBuffer.from_numpy(scores, device)
        buf_mask2 = NPUBuffer.from_numpy(mask, device)
        buf_fused = NPUBuffer.zeros((rows, cols), device)
        # mask_strides: (cols, 1) for same-shape (rows, cols) row-major
        ms_params = make_params(
            device, "2II6I6I",
            rows, cols,
            2,                           # ndim
            cols, 1, 0, 0, 0, 0,        # mask_strides (row-major for same shape)
            rows, cols, 0, 0, 0, 0,     # out_shape
        )
        dispatch_1d(device, device.get_pipeline(lib_softmax, "masked_softmax_broadcast_kernel"),
                     [buf_scores2, buf_mask2, buf_fused, ms_params], rows)
        fused_result = buf_fused.to_numpy()

        npt.assert_allclose(unfused_result, ref, rtol=1e-2, atol=1e-2)
        npt.assert_allclose(fused_result, ref, rtol=1e-2, atol=1e-2)
        npt.assert_allclose(fused_result, unfused_result, rtol=1e-3, atol=1e-3)


class TestBroadcastBinaryFusion:
    """Compare broadcast binary kernel vs expand + elementwise."""

    def test_broadcast_add_vs_expand_add(self, device):
        rows, cols = 4, 64
        np.random.seed(42)
        a = np.random.randn(rows, cols).astype(np.float32)
        b = np.random.randn(1, cols).astype(np.float32)  # broadcast dim 0

        ref = a + b

        lib_bcast = device.compile_metal_file(
            os.path.join(kernels_dir(), "elementwise_broadcast.metal"))
        lib_add = device.compile_metal_file(os.path.join(kernels_dir(), "add_relu.metal"))
        lib_tensor = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))

        total = rows * cols

        # --- Unfused: expand b to (rows, cols) then element-wise add ---
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_b_exp = NPUBuffer.zeros((rows, cols), device)
        expand_params = make_params(
            device, "2I6I6I6I",
            2, total,
            1, cols, 0, 0, 0, 0,
            rows, cols, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
        )
        dispatch_1d(device, device.get_pipeline(lib_tensor, "expand_kernel"),
                     [buf_b, buf_b_exp, expand_params], total)

        buf_a = NPUBuffer.from_numpy(a, device)
        buf_unfused = NPUBuffer.zeros((rows, cols), device)
        dispatch_1d(device, device.get_pipeline(lib_add, "add_kernel"),
                     [buf_a, buf_b_exp, buf_unfused], total)
        unfused_result = buf_unfused.to_numpy()

        # --- Fused: single broadcast add kernel ---
        # Param format: "2I6I6I6I" → ndim, total, a_strides[6], b_strides[6], out_shape[6]
        buf_a2 = NPUBuffer.from_numpy(a, device)
        buf_b2 = NPUBuffer.from_numpy(b, device)
        buf_fused = NPUBuffer.zeros((rows, cols), device)
        # a_strides: (cols, 1) for (rows, cols) row-major; b_strides: (0, 1) for (1, cols) broadcast
        bcast_params = make_params(
            device, "2I6I6I6I",
            2, total,
            cols, 1, 0, 0, 0, 0,        # a_strides
            0, 1, 0, 0, 0, 0,           # b_strides (broadcast dim 0)
            rows, cols, 0, 0, 0, 0,     # out_shape
        )
        dispatch_1d(device, device.get_pipeline(lib_bcast, "add_broadcast_kernel"),
                     [buf_a2, buf_b2, buf_fused, bcast_params], total)
        fused_result = buf_fused.to_numpy()

        npt.assert_allclose(unfused_result, ref, rtol=1e-3, atol=1e-3)
        npt.assert_allclose(fused_result, ref, rtol=1e-3, atol=1e-3)
        npt.assert_allclose(fused_result, unfused_result, rtol=1e-5, atol=1e-5)
