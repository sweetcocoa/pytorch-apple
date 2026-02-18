"""Tests for extended Metal compute kernels against PyTorch/NumPy CPU reference."""

import os

import numpy as np
import numpy.testing as npt
import torch

from npu_runtime.buffer import NPUBuffer
from tests.conftest import dispatch_1d, dispatch_2d, dispatch_3d, dispatch_tiled_2d, kernels_dir, make_params

# ── Element-wise kernels ──

class TestSiluKernel:
    def test_silu(self, device):
        x = np.random.randn(256).astype(np.float32)
        ref = x * (1.0 / (1.0 + np.exp(-x)))  # silu = x * sigmoid(x)

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "silu_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out], 256)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)


class TestMulKernel:
    def test_mul(self, device):
        a = np.random.randn(512).astype(np.float32)
        b = np.random.randn(512).astype(np.float32)
        ref = a * b

        buf_a = NPUBuffer.from_numpy(a, device)
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_out = NPUBuffer.zeros(a.shape, device)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "mul_kernel")
        dispatch_1d(device, pipeline, [buf_a, buf_b, buf_out], 512)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)


class TestDivKernel:
    def test_div(self, device):
        a = np.random.randn(256).astype(np.float32)
        b = np.random.uniform(0.5, 2.0, 256).astype(np.float32)  # avoid div by zero
        ref = a / b

        buf_a = NPUBuffer.from_numpy(a, device)
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_out = NPUBuffer.zeros(a.shape, device)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "div_kernel")
        dispatch_1d(device, pipeline, [buf_a, buf_b, buf_out], 256)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)


class TestNegKernel:
    def test_neg(self, device):
        x = np.random.randn(256).astype(np.float32)
        ref = -x

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "neg_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out], 256)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-3, atol=1e-3)


class TestPowKernel:
    def test_pow_scalar(self, device):
        x = np.random.uniform(0.1, 2.0, 256).astype(np.float32)
        exponent = 2.0
        ref = x ** exponent

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)
        params = make_params(device, "f", exponent)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "pow_scalar_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], 256)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-2)


class TestRsqrtKernel:
    def test_rsqrt(self, device):
        x = np.random.uniform(0.1, 10.0, 256).astype(np.float32)
        ref = 1.0 / np.sqrt(x)

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "rsqrt_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out], 256)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)


class TestCosKernel:
    def test_cos(self, device):
        x = np.random.randn(256).astype(np.float32)
        ref = np.cos(x)

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "cos_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out], 256)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)


class TestSinKernel:
    def test_sin(self, device):
        x = np.random.randn(256).astype(np.float32)
        ref = np.sin(x)

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "sin_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out], 256)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)


# ── Embedding ──

class TestEmbeddingKernel:
    def test_embedding(self, device):
        vocab_size, embed_dim, seq_len = 100, 64, 8
        weight = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        indices = np.random.randint(0, vocab_size, seq_len).astype(np.int32)
        ref = weight[indices]  # (seq_len, embed_dim)

        buf_ids = NPUBuffer.from_numpy(indices, device)
        buf_w = NPUBuffer.from_numpy(weight, device)
        buf_out = NPUBuffer.zeros((seq_len, embed_dim), device)
        params = make_params(device, "3I", seq_len, embed_dim, vocab_size)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "embedding.metal"))
        pipeline = device.get_pipeline(lib, "embedding_kernel")
        dispatch_2d(device, pipeline, [buf_ids, buf_w, buf_out, params], embed_dim, seq_len)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-2)


# ── Softmax ──

class TestSoftmaxKernel:
    def test_softmax_1d(self, device):
        x = np.random.randn(128).astype(np.float32)
        ref = torch.softmax(torch.tensor(x), dim=-1).numpy()

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)
        params = make_params(device, "2I", 1, 128)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "softmax.metal"))
        pipeline = device.get_pipeline(lib, "softmax_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], 1)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)

    def test_softmax_2d(self, device):
        """Softmax along last dim of (4, 64) tensor."""
        x = np.random.randn(4, 64).astype(np.float32)
        ref = torch.softmax(torch.tensor(x), dim=-1).numpy()

        rows, cols = 4, 64
        buf_in = NPUBuffer.from_numpy(x.reshape(-1), device)
        buf_out = NPUBuffer.zeros((rows * cols,), device)
        params = make_params(device, "2I", rows, cols)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "softmax.metal"))
        pipeline = device.get_pipeline(lib, "softmax_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], rows)

        result = buf_out.to_numpy().reshape(rows, cols)
        npt.assert_allclose(result, ref, rtol=1e-2, atol=1e-3)

    def test_softmax_large_values(self, device):
        """Softmax numerical stability with large values."""
        x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        ref = torch.softmax(torch.tensor(x), dim=-1).numpy()

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)
        params = make_params(device, "2I", 1, 3)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "softmax.metal"))
        pipeline = device.get_pipeline(lib, "softmax_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], 1)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)


# ── Mean last dim ──

class TestMeanLastDimKernel:
    def test_mean_last_dim(self, device):
        x = np.random.randn(8, 64).astype(np.float32)
        ref = x.mean(axis=-1)

        buf_in = NPUBuffer.from_numpy(x.reshape(-1), device)
        buf_out = NPUBuffer.zeros((8,), device)
        params = make_params(device, "2I", 8, 64)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "mean_last_dim_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], 8)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-2)


# ── Transpose ──

class TestTransposeKernel:
    def test_transpose_2d(self, device):
        x = np.random.randn(4, 8).astype(np.float32)
        ref = x.T  # (8, 4)

        in_shape = [4, 8]
        out_shape = [8, 4]
        ndim = 2
        total = 32
        strides_in = [8, 1]
        strides_out = [4, 1]

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(tuple(out_shape), device)
        params = make_params(
            device, "4I6I6I6I",
            ndim, 0, 1, total,
            *in_shape, *([0] * (6 - ndim)),
            *strides_in, *([0] * (6 - ndim)),
            *strides_out, *([0] * (6 - ndim)),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))
        pipeline = device.get_pipeline(lib, "transpose_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], total)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-3, atol=1e-3)

    def test_transpose_4d_attention(self, device):
        """Transpose for attention: (B, S, H, D) -> swap dims 1,2 -> (B, H, S, D)."""
        B, S, H, D = 1, 4, 2, 8
        x = np.random.randn(B, S, H, D).astype(np.float32)
        ref = x.transpose(0, 2, 1, 3)  # (B, H, S, D)

        in_shape = [B, S, H, D]
        out_shape = [B, H, S, D]
        ndim = 4

        def strides(shape):
            s = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                s[i] = s[i + 1] * shape[i + 1]
            return s

        total = int(np.prod(out_shape))
        strides_in = strides(in_shape)
        strides_out = strides(out_shape)

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(tuple(out_shape), device)
        params = make_params(
            device, "4I6I6I6I",
            ndim, 1, 2, total,
            *in_shape, *([0] * (6 - ndim)),
            *strides_in, *([0] * (6 - ndim)),
            *strides_out, *([0] * (6 - ndim)),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))
        pipeline = device.get_pipeline(lib, "transpose_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], total)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-3, atol=1e-3)


# ── Batched Matmul ──

class TestBatchedMatmulKernel:
    def test_batched_matmul(self, device):
        B, M, K, N = 2, 4, 8, 6
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(B, K, N).astype(np.float32)
        ref = np.matmul(a, b)  # (B, M, N)

        buf_a = NPUBuffer.from_numpy(a, device)
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_out = NPUBuffer.zeros((B, M, N), device)
        params = make_params(device, "4I", B, M, N, K)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "matmul.metal"))
        pipeline = device.get_pipeline(lib, "batched_matmul_kernel")
        dispatch_3d(device, pipeline, [buf_a, buf_b, buf_out, params], N, M, B)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-1)

    def test_batched_matmul_attention_pattern(self, device):
        """(B*H, S, D) @ (B*H, D, S) -> (B*H, S, S) -- attention scores."""
        BH, S, D = 4, 8, 16
        q = np.random.randn(BH, S, D).astype(np.float32)
        k = np.random.randn(BH, D, S).astype(np.float32)
        ref = np.matmul(q, k)

        buf_q = NPUBuffer.from_numpy(q, device)
        buf_k = NPUBuffer.from_numpy(k, device)
        buf_out = NPUBuffer.zeros((BH, S, S), device)
        params = make_params(device, "4I", BH, S, S, D)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "matmul.metal"))
        pipeline = device.get_pipeline(lib, "batched_matmul_kernel")
        dispatch_3d(device, pipeline, [buf_q, buf_k, buf_out, params], S, S, BH)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-1)


# ── Cat ──

class TestCatKernel:
    def test_cat_1d(self, device):
        a = np.random.randn(8).astype(np.float32)
        b = np.random.randn(4).astype(np.float32)
        ref = np.concatenate([a, b])

        out_shape = [12]
        ndim = 1
        total = 12
        strides = [1]

        buf_a = NPUBuffer.from_numpy(a, device)
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_out = NPUBuffer.zeros(tuple(out_shape), device)
        params = make_params(
            device, "4I6I6I",
            0, ndim, total, 8,  # axis, ndim, total, in1_axis_size
            *out_shape, *([0] * (6 - ndim)),
            *strides, *([0] * (6 - ndim)),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))
        pipeline = device.get_pipeline(lib, "cat_2_kernel")
        dispatch_1d(device, pipeline, [buf_a, buf_b, buf_out, params], total)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-3, atol=1e-3)

    def test_cat_2d_axis1(self, device):
        a = np.random.randn(2, 4).astype(np.float32)
        b = np.random.randn(2, 3).astype(np.float32)
        ref = np.concatenate([a, b], axis=1)  # (2, 7)

        out_shape = [2, 7]
        ndim = 2
        total = 14

        def strides(shape):
            s = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                s[i] = s[i + 1] * shape[i + 1]
            return s

        buf_a = NPUBuffer.from_numpy(a, device)
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_out = NPUBuffer.zeros(tuple(out_shape), device)
        params = make_params(
            device, "4I6I6I",
            1, ndim, total, 4,
            *out_shape, *([0] * (6 - ndim)),
            *strides(out_shape), *([0] * (6 - ndim)),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))
        pipeline = device.get_pipeline(lib, "cat_2_kernel")
        dispatch_1d(device, pipeline, [buf_a, buf_b, buf_out, params], total)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-3, atol=1e-3)


# ── Slice ──

class TestSliceKernel:
    def test_slice_1d(self, device):
        x = np.arange(16).astype(np.float32)
        ref = x[2:10:2]  # [2, 4, 6, 8]

        in_shape = [16]
        ndim = 1
        total = 4
        in_strides = [1]

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros((4,), device)
        params = make_params(
            device, "6I6I6I",
            0, 2, 10, 2, ndim, total,
            *in_shape, *([0] * (6 - ndim)),
            *in_strides, *([0] * (6 - ndim)),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))
        pipeline = device.get_pipeline(lib, "slice_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], total)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-3, atol=1e-3)

    def test_slice_2d(self, device):
        x = np.arange(24).reshape(4, 6).astype(np.float32)
        ref = x[1:3, :]  # rows 1 and 2

        in_shape = [4, 6]
        ndim = 2
        total = 12
        in_strides = [6, 1]

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros((2, 6), device)
        params = make_params(
            device, "6I6I6I",
            0, 1, 3, 1, ndim, total,  # dim=0, start=1, end=3, step=1
            *in_shape, *([0] * (6 - ndim)),
            *in_strides, *([0] * (6 - ndim)),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))
        pipeline = device.get_pipeline(lib, "slice_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], total)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-3, atol=1e-3)


# ── Expand ──

class TestExpandKernel:
    def test_expand_broadcast(self, device):
        x = np.random.randn(1, 4).astype(np.float32)
        ref = np.broadcast_to(x, (3, 4))

        in_shape = [1, 4]
        out_shape = [3, 4]
        ndim = 2
        total = 12
        in_strides = [0, 1]  # stride 0 for broadcast dim

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(tuple(out_shape), device)
        params = make_params(
            device, "2I6I6I6I",
            ndim, total,
            *in_shape, *([0] * (6 - ndim)),
            *out_shape, *([0] * (6 - ndim)),
            *in_strides, *([0] * (6 - ndim)),
        )

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "tensor_ops.metal"))
        pipeline = device.get_pipeline(lib, "expand_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], total)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-3, atol=1e-3)


# ── Non-transposed matmul ──

class TestMatmulNotransKernel:
    def test_matmul_notrans(self, device):
        M, K, N = 4, 8, 6
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        ref = a @ b

        buf_a = NPUBuffer.from_numpy(a, device)
        buf_b = NPUBuffer.from_numpy(b, device)
        buf_out = NPUBuffer.zeros((M, N), device)
        params = make_params(device, "4I", M, N, K, 0)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "matmul.metal"))
        pipeline = device.get_pipeline(lib, "matmul_notrans_kernel")
        dispatch_tiled_2d(device, pipeline, [buf_a, buf_b, buf_out, params], N, M)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-1)


# ── Scalar element-wise ──

class TestEltwiseAddScalarKernel:
    def test_add_scalar(self, device):
        x = np.random.randn(256).astype(np.float32)
        scalar = 1e-6
        ref = x + scalar

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)
        params = make_params(device, "fI", scalar, 256)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "eltwise_add_scalar_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], 256)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)


class TestEltwiseMulScalarKernel:
    def test_mul_scalar(self, device):
        x = np.random.randn(256).astype(np.float32)
        scalar = 0.08838834764831845  # 1/sqrt(128), attention scaling
        ref = x * scalar

        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros(x.shape, device)
        params = make_params(device, "fI", scalar, 256)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "elementwise_extended.metal"))
        pipeline = device.get_pipeline(lib, "eltwise_mul_scalar_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], 256)

        npt.assert_allclose(buf_out.to_numpy(), ref, rtol=1e-2, atol=1e-3)


# ── RoPE ──

class TestRopeKernel:
    def test_rope_basic(self, device):
        """RoPE: cos/sin from inv_freq and position indices."""
        head_dim = 128
        seq_len = 8
        half_dim = head_dim // 2

        # inv_freq: (half_dim,) — typical values from Qwen2.5
        inv_freq = (1.0 / (10000.0 ** (np.arange(half_dim, dtype=np.float32) / half_dim)))
        positions = np.arange(seq_len, dtype=np.int32)

        # NumPy reference: freq[seq, dim] = positions[seq] * inv_freq[dim % half_dim]
        freq = positions[:, None].astype(np.float32) * inv_freq[None, :]  # (seq, half_dim)
        ref_cos = np.cos(np.concatenate([freq, freq], axis=1))  # (seq, head_dim)
        ref_sin = np.sin(np.concatenate([freq, freq], axis=1))

        buf_inv = NPUBuffer.from_numpy(inv_freq.astype(np.float16), device)
        buf_pos = NPUBuffer.from_numpy(positions, device)
        buf_cos = NPUBuffer.zeros((seq_len, head_dim), device)
        buf_sin = NPUBuffer.zeros((seq_len, head_dim), device)
        params = make_params(device, "2I", seq_len, head_dim)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "rope.metal"))
        pipeline = device.get_pipeline(lib, "rope_kernel")
        dispatch_2d(device, pipeline, [buf_inv, buf_pos, buf_cos, buf_sin, params], head_dim, seq_len)

        npt.assert_allclose(buf_cos.to_numpy(), ref_cos, rtol=1e-2, atol=1e-2)
        npt.assert_allclose(buf_sin.to_numpy(), ref_sin, rtol=1e-2, atol=1e-2)

    def test_rope_nonzero_positions(self, device):
        """RoPE with non-sequential positions (decode phase)."""
        head_dim = 64
        seq_len = 1
        half_dim = head_dim // 2

        inv_freq = (1.0 / (10000.0 ** (np.arange(half_dim, dtype=np.float32) / half_dim)))
        positions = np.array([42], dtype=np.int32)  # decode at position 42

        freq = positions[:, None].astype(np.float32) * inv_freq[None, :]
        ref_cos = np.cos(np.concatenate([freq, freq], axis=1))
        ref_sin = np.sin(np.concatenate([freq, freq], axis=1))

        buf_inv = NPUBuffer.from_numpy(inv_freq.astype(np.float16), device)
        buf_pos = NPUBuffer.from_numpy(positions, device)
        buf_cos = NPUBuffer.zeros((seq_len, head_dim), device)
        buf_sin = NPUBuffer.zeros((seq_len, head_dim), device)
        params = make_params(device, "2I", seq_len, head_dim)

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "rope.metal"))
        pipeline = device.get_pipeline(lib, "rope_kernel")
        dispatch_2d(device, pipeline, [buf_inv, buf_pos, buf_cos, buf_sin, params], head_dim, seq_len)

        npt.assert_allclose(buf_cos.to_numpy(), ref_cos, rtol=1e-2, atol=1e-2)
        npt.assert_allclose(buf_sin.to_numpy(), ref_sin, rtol=1e-2, atol=1e-2)


# ── Fused Decode Attention ──

def _dispatch_fused_decode_attn(device, buffers, batch_heads, head_dim):
    """Dispatch fused_decode_attention_kernel: BH threadgroups × D threads."""
    lib = device.compile_metal_file(os.path.join(kernels_dir(), "fused_decode_attention.metal"))
    pipeline = device.get_pipeline(lib, "fused_decode_attention_kernel")
    cmd_buf = device.new_command_buffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for idx, buf in enumerate(buffers):
        mtl = buf.mtl_buffer if isinstance(buf, NPUBuffer) else buf
        encoder.setBuffer_offset_atIndex_(mtl, 0, idx)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        (batch_heads, 1, 1), (head_dim, 1, 1)
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()


def _numpy_decode_attention(Q, K, V, mask, valid_len, scale):
    """NumPy reference for fused decode attention.

    Q: (BH, D), K: (BH, S, D), V: (BH, S, D), mask: (S,)
    Returns: (BH, D)
    """
    BH, D = Q.shape
    out = np.zeros((BH, D), dtype=np.float64)
    for h in range(BH):
        # Q × K^T (only valid_len positions)
        scores = np.dot(K[h, :valid_len, :], Q[h, :]) * scale  # (valid_len,)
        scores = scores + mask[:valid_len]
        # Softmax
        max_s = np.max(scores)
        exp_s = np.exp(scores - max_s)
        attn = exp_s / np.sum(exp_s)
        # Attn × V
        out[h, :] = np.dot(attn, V[h, :valid_len, :])
    return out


class TestFusedDecodeAttention:
    def _run_test(self, device, B, H, D, max_seq, valid_len):
        BH = B * H
        scale = 1.0 / np.sqrt(D)

        np.random.seed(42)
        Q = np.random.randn(BH, D).astype(np.float32)
        K = np.random.randn(BH, max_seq, D).astype(np.float32)
        V = np.random.randn(BH, max_seq, D).astype(np.float32)

        # Causal mask: 0 for valid positions, -inf for masked
        mask = np.full(max_seq, -np.inf, dtype=np.float32)
        mask[:valid_len] = 0.0

        cache_pos = np.array([valid_len - 1], dtype=np.int32)  # 0-indexed

        ref = _numpy_decode_attention(Q, K, V, mask, valid_len, scale)

        buf_q = NPUBuffer.from_numpy(Q.astype(np.float16), device)
        buf_k = NPUBuffer.from_numpy(K.astype(np.float16), device)
        buf_v = NPUBuffer.from_numpy(V.astype(np.float16), device)
        buf_mask = NPUBuffer.from_numpy(mask.astype(np.float16), device)
        buf_pos = NPUBuffer.from_numpy(cache_pos, device)
        buf_out = NPUBuffer.zeros((BH, D), device)
        params = make_params(device, "3If", BH, D, max_seq, scale)

        _dispatch_fused_decode_attn(
            device,
            [buf_q, buf_k, buf_v, buf_mask, buf_pos, buf_out, params],
            BH, D,
        )
        result = buf_out.to_numpy().astype(np.float64)
        npt.assert_allclose(result, ref, rtol=5e-2, atol=1e-2)

    def test_small(self, device):
        """Small attention: 1 batch, 2 heads, D=64, seq=16, valid=8."""
        self._run_test(device, B=1, H=2, D=64, max_seq=16, valid_len=8)

    def test_full_seq(self, device):
        """Full sequence: all positions valid."""
        self._run_test(device, B=1, H=4, D=64, max_seq=32, valid_len=32)

    def test_single_position(self, device):
        """Single valid position (first decode step)."""
        self._run_test(device, B=1, H=4, D=128, max_seq=64, valid_len=1)

    def test_qwen_like(self, device):
        """Qwen2.5-like dimensions: 12 heads, D=128, max_seq=129."""
        self._run_test(device, B=1, H=12, D=128, max_seq=129, valid_len=10)
