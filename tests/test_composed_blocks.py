"""Tests for composed blocks: RMSNorm, self-attention, gated MLP.

These tests chain multiple kernels to validate end-to-end correctness
of transformer building blocks.
"""

import os

import numpy as np
import numpy.testing as npt
import torch

from npu_runtime.buffer import NPUBuffer
from tests.conftest import dispatch_1d, dispatch_3d, dispatch_tiled_2d, kernels_dir, make_params


class TestRMSNorm:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight

    Decomposed: pow(x,2) -> mean -> add_scalar(eps) -> rsqrt -> mul(x) -> mul(weight)
    """

    def test_rms_norm(self, device):
        seq_len, hidden = 4, 64
        eps = 1e-6

        x = np.random.randn(seq_len, hidden).astype(np.float32)
        weight = np.random.randn(hidden).astype(np.float32)

        # PyTorch reference
        x_t = torch.tensor(x)
        variance = x_t.pow(2).mean(-1, keepdim=True)
        ref = (x_t * torch.rsqrt(variance + eps) * torch.tensor(weight)).numpy()

        lib_ew = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        total = seq_len * hidden

        # Step 1: pow(x, 2)
        buf_x = NPUBuffer.from_numpy(x, device)
        buf_x2 = NPUBuffer.zeros((seq_len, hidden), device)
        pow_params = make_params(device, "f", 2.0)
        pipeline = device.get_pipeline(lib_ew, "pow_scalar_kernel")
        dispatch_1d(device, pipeline, [buf_x, buf_x2, pow_params], total)

        # Step 2: mean(x^2, dim=-1) -> (seq_len,)
        buf_mean = NPUBuffer.zeros((seq_len,), device)
        mean_params = make_params(device, "2I", seq_len, hidden)
        pipeline = device.get_pipeline(lib_ew, "mean_last_dim_kernel")
        dispatch_1d(device, pipeline, [buf_x2, buf_mean, mean_params], seq_len)

        # Step 3: add eps
        eps_arr = np.full(seq_len, eps, dtype=np.float32)
        buf_eps = NPUBuffer.from_numpy(eps_arr, device)
        buf_var_eps = NPUBuffer.zeros((seq_len,), device)
        lib_add = device.compile_metal_file(os.path.join(kernels_dir(), "add_relu.metal"))
        pipeline = device.get_pipeline(lib_add, "add_kernel")
        dispatch_1d(device, pipeline, [buf_mean, buf_eps, buf_var_eps], seq_len)

        # Step 4: rsqrt
        buf_rsqrt = NPUBuffer.zeros((seq_len,), device)
        pipeline = device.get_pipeline(lib_ew, "rsqrt_kernel")
        dispatch_1d(device, pipeline, [buf_var_eps, buf_rsqrt], seq_len)

        # Step 5: expand rsqrt (seq_len,) -> (seq_len, hidden)
        buf_rsqrt_exp = NPUBuffer.zeros((seq_len, hidden), device)
        lib_tensor = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))
        expand_params = make_params(
            device, "2I6I6I6I",
            2, total,
            seq_len, 1, 0, 0, 0, 0,
            seq_len, hidden, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0,
        )
        pipeline = device.get_pipeline(lib_tensor, "expand_kernel")
        dispatch_1d(device, pipeline, [buf_rsqrt, buf_rsqrt_exp, expand_params], total)

        # Step 6: x * rsqrt
        buf_normed = NPUBuffer.zeros((seq_len, hidden), device)
        pipeline = device.get_pipeline(lib_ew, "mul_kernel")
        dispatch_1d(device, pipeline, [buf_x, buf_rsqrt_exp, buf_normed], total)

        # Step 7: normed * weight (broadcast weight across rows)
        buf_weight = NPUBuffer.from_numpy(weight, device)
        buf_weight_exp = NPUBuffer.zeros((seq_len, hidden), device)
        expand_w_params = make_params(
            device, "2I6I6I6I",
            2, total,
            1, hidden, 0, 0, 0, 0,
            seq_len, hidden, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
        )
        pipeline = device.get_pipeline(lib_tensor, "expand_kernel")
        dispatch_1d(device, pipeline, [buf_weight, buf_weight_exp, expand_w_params], total)

        buf_out = NPUBuffer.zeros((seq_len, hidden), device)
        pipeline = device.get_pipeline(lib_ew, "mul_kernel")
        dispatch_1d(device, pipeline, [buf_normed, buf_weight_exp, buf_out], total)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=5e-2, atol=5e-2)


class TestSelfAttention:
    """Self-attention: Q@K^T/sqrt(d) -> softmax -> @V

    Using batched matmul kernels.
    """

    def test_self_attention(self, device):
        B, H, S, D = 1, 2, 4, 8
        BH = B * H

        np.random.seed(42)
        Q = np.random.randn(BH, S, D).astype(np.float32)
        K = np.random.randn(BH, S, D).astype(np.float32)
        V = np.random.randn(BH, S, D).astype(np.float32)

        # PyTorch reference
        Q_t = torch.tensor(Q)
        K_t = torch.tensor(K)
        V_t = torch.tensor(V)
        scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        ref = torch.matmul(attn, V_t).numpy()

        lib_matmul = device.compile_metal_file(os.path.join(kernels_dir(), "matmul.metal"))
        lib_softmax = device.compile_metal_file(os.path.join(kernels_dir(), "softmax.metal"))
        lib_ew = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        lib_tensor = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))

        def strides(shape):
            s = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                s[i] = s[i + 1] * shape[i + 1]
            return s

        # Step 1: K^T
        buf_K = NPUBuffer.from_numpy(K, device)
        buf_Kt = NPUBuffer.zeros((BH, D, S), device)
        kt_total = BH * D * S
        t_params = make_params(
            device, "4I6I6I6I",
            3, 1, 2, kt_total,
            *[BH, S, D], *([0] * 3),
            *strides([BH, S, D]), *([0] * 3),
            *strides([BH, D, S]), *([0] * 3),
        )
        pipeline = device.get_pipeline(lib_tensor, "transpose_kernel")
        dispatch_1d(device, pipeline, [buf_K, buf_Kt, t_params], kt_total)

        # Step 2: Q @ K^T
        buf_Q = NPUBuffer.from_numpy(Q, device)
        buf_scores = NPUBuffer.zeros((BH, S, S), device)
        mm_params = make_params(device, "4I", BH, S, S, D)
        pipeline = device.get_pipeline(lib_matmul, "batched_matmul_kernel")
        dispatch_3d(device, pipeline, [buf_Q, buf_Kt, buf_scores, mm_params], S, S, BH)

        # Step 3: scores / sqrt(D)
        scale = np.full(BH * S * S, D ** 0.5, dtype=np.float32)
        buf_scale = NPUBuffer.from_numpy(scale, device)
        buf_scaled = NPUBuffer.zeros((BH * S * S,), device)
        pipeline = device.get_pipeline(lib_ew, "div_kernel")
        dispatch_1d(device, pipeline, [buf_scores, buf_scale, buf_scaled], BH * S * S)

        # Step 4: softmax
        buf_attn = NPUBuffer.zeros((BH * S * S,), device)
        softmax_params = make_params(device, "2I", BH * S, S)
        pipeline = device.get_pipeline(lib_softmax, "softmax_kernel")
        dispatch_1d(device, pipeline, [buf_scaled, buf_attn, softmax_params], BH * S)

        # Step 5: attn @ V
        buf_V = NPUBuffer.from_numpy(V, device)
        buf_out = NPUBuffer.zeros((BH, S, D), device)
        buf_attn_3d = NPUBuffer(buf_attn.mtl_buffer, (BH, S, S), buf_attn.dtype, device)
        mm_params2 = make_params(device, "4I", BH, S, D, S)
        pipeline = device.get_pipeline(lib_matmul, "batched_matmul_kernel")
        dispatch_3d(device, pipeline, [buf_attn_3d, buf_V, buf_out, mm_params2], D, S, BH)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=5e-2, atol=5e-2)


class TestGatedMLP:
    """Gated MLP: SiLU(x @ gate_w.T) * (x @ up_w.T) -> @ down_w.T

    Using matmul + silu + mul kernels.
    """

    def test_gated_mlp(self, device):
        seq_len, hidden, intermediate = 4, 32, 64

        np.random.seed(42)
        x = np.random.randn(seq_len, hidden).astype(np.float32)
        gate_w = np.random.randn(intermediate, hidden).astype(np.float32)
        up_w = np.random.randn(intermediate, hidden).astype(np.float32)
        down_w = np.random.randn(hidden, intermediate).astype(np.float32)

        # PyTorch reference
        x_t = torch.tensor(x)
        gate = torch.nn.functional.silu(x_t @ torch.tensor(gate_w).T)
        up = x_t @ torch.tensor(up_w).T
        ref = (gate * up) @ torch.tensor(down_w).T
        ref = ref.numpy()

        lib_matmul = device.compile_metal_file(os.path.join(kernels_dir(), "matmul.metal"))
        lib_ew = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))

        buf_x = NPUBuffer.from_numpy(x, device)

        # Step 1: x @ gate_w.T
        buf_gate_w = NPUBuffer.from_numpy(gate_w, device)
        buf_gate_b = NPUBuffer.zeros((intermediate,), device)
        buf_gate_proj = NPUBuffer.zeros((seq_len, intermediate), device)
        mm_params = make_params(device, "4I", seq_len, intermediate, hidden, 0)
        pipeline = device.get_pipeline(lib_matmul, "matmul_kernel")
        dispatch_tiled_2d(device, pipeline, [buf_x, buf_gate_w, buf_gate_b, buf_gate_proj, mm_params],
                          intermediate, seq_len)

        # Step 2: SiLU(gate_proj)
        total_inter = seq_len * intermediate
        buf_gate_silu = NPUBuffer.zeros((seq_len, intermediate), device)
        pipeline = device.get_pipeline(lib_ew, "silu_kernel")
        dispatch_1d(device, pipeline, [buf_gate_proj, buf_gate_silu], total_inter)

        # Step 3: x @ up_w.T
        buf_up_w = NPUBuffer.from_numpy(up_w, device)
        buf_up_b = NPUBuffer.zeros((intermediate,), device)
        buf_up_proj = NPUBuffer.zeros((seq_len, intermediate), device)
        pipeline = device.get_pipeline(lib_matmul, "matmul_kernel")
        dispatch_tiled_2d(device, pipeline, [buf_x, buf_up_w, buf_up_b, buf_up_proj, mm_params],
                          intermediate, seq_len)

        # Step 4: gate_silu * up_proj
        buf_mul_out = NPUBuffer.zeros((seq_len, intermediate), device)
        pipeline = device.get_pipeline(lib_ew, "mul_kernel")
        dispatch_1d(device, pipeline, [buf_gate_silu, buf_up_proj, buf_mul_out], total_inter)

        # Step 5: result @ down_w.T
        buf_down_w = NPUBuffer.from_numpy(down_w, device)
        buf_down_b = NPUBuffer.zeros((hidden,), device)
        buf_out = NPUBuffer.zeros((seq_len, hidden), device)
        mm_params2 = make_params(device, "4I", seq_len, hidden, intermediate, 0)
        pipeline = device.get_pipeline(lib_matmul, "matmul_kernel")
        dispatch_tiled_2d(device, pipeline, [buf_mul_out, buf_down_w, buf_down_b, buf_out, mm_params2],
                          hidden, seq_len)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, rtol=5e-2, atol=5e-1)
