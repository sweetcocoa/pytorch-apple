"""MPS (Metal Performance Shaders) acceleration for matmul operations."""

from __future__ import annotations

import MetalPerformanceShaders as MPS

from npu_compiler.codegen import KernelCall
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device

# Matmul kernel names that should use MPS acceleration
MPS_MATMUL_KERNELS = {"matmul_kernel", "matmul_notrans_kernel", "batched_matmul_kernel",
                      "matmul_vec_kernel", "matmul_notrans_vec_kernel"}

# MPS datatype mapping
MPS_DTYPE_MAP = {
    "bfloat16": MPS.MPSDataTypeBFloat16,
    "float16": MPS.MPSDataTypeFloat16,
}


def create_mps_matmul(device: Device, call: KernelCall, compute_dtype: str):
    """Pre-create an MPSMatrixMultiplication object for a matmul kernel call."""
    if call.kernel_name not in MPS_MATMUL_KERNELS:
        return None

    p = call.params
    mtl_device = device.mtl_device

    if call.kernel_name in ("matmul_kernel", "matmul_vec_kernel"):
        M, N, K = p["M"], p["N"], p["K"]
        has_bias = p.get("has_bias", 0)
        beta = 1.0 if has_bias else 0.0
        init = MPS.MPSMatrixMultiplication.alloc()
        mps_op = init.initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
            mtl_device, False, True, M, N, K, 1.0, beta)
        return {
            "op": mps_op,
            "M": M, "N": N, "K": K, "has_bias": has_bias,
            "type": "matmul_kernel",
        }
    elif call.kernel_name in ("matmul_notrans_kernel", "matmul_notrans_vec_kernel"):
        M, N, K = p["M"], p["N"], p["K"]
        init = MPS.MPSMatrixMultiplication.alloc()
        mps_op = init.initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
            mtl_device, False, False, M, N, K, 1.0, 0.0)
        return {
            "op": mps_op,
            "M": M, "N": N, "K": K, "has_bias": 0,
            "type": "matmul_notrans_kernel",
        }
    elif call.kernel_name == "batched_matmul_kernel":
        batch, M, N, K = p["batch"], p["M"], p["N"], p["K"]
        init = MPS.MPSMatrixMultiplication.alloc()
        mps_op = init.initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
            mtl_device, False, False, M, N, K, 1.0, 0.0)
        return {
            "op": mps_op,
            "batch": batch, "M": M, "N": N, "K": K,
            "type": "batched_matmul_kernel",
        }
    return None


def _broadcast_bias_to_c(cmd_buf, bias_buf, c_buf, M, N, elem_size):
    """Copy bias(N,) into each row of C(M,N) so MPS can use beta=1.0."""
    blit = cmd_buf.blitCommandEncoder()
    row_bytes = N * elem_size
    for row in range(M):
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            bias_buf, 0, c_buf, row * row_bytes, row_bytes)
    blit.endEncoding()


def encode_mps_matmul(cmd_buf, call: KernelCall, info: dict,
                      pool: dict[str, NPUBuffer], mps_dtype):
    """Encode an MPS matrix multiplication into the command buffer."""
    dt = mps_dtype
    elem_size = 2  # both float16 and bfloat16 are 2 bytes

    if info["type"] == "batched_matmul_kernel":
        batch, M, N, K = info["batch"], info["M"], info["N"], info["K"]
        a_buf = pool[call.input_buffers[0]].native_handle
        b_buf = pool[call.input_buffers[1]].native_handle
        c_buf = pool[call.output_buffers[0]].native_handle

        desc_a = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType_(
            M, K, batch, K * elem_size, M * K * elem_size, dt)
        desc_b = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType_(
            K, N, batch, N * elem_size, K * N * elem_size, dt)
        desc_c = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType_(
            M, N, batch, N * elem_size, M * N * elem_size, dt)

        mat_a = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(a_buf, desc_a)
        mat_b = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(b_buf, desc_b)
        mat_c = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(c_buf, desc_c)

        info["op"].encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
            cmd_buf, mat_a, mat_b, mat_c)
    else:
        M, N, K = info["M"], info["N"], info["K"]
        has_bias = info.get("has_bias", 0)
        a_buf = pool[call.input_buffers[0]].native_handle
        b_buf = pool[call.input_buffers[1]].native_handle
        c_buf = pool[call.output_buffers[0]].native_handle

        if info["type"] == "matmul_kernel":
            desc_a = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
                M, K, K * elem_size, dt)
            desc_b = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
                N, K, K * elem_size, dt)
        else:
            desc_a = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
                M, K, K * elem_size, dt)
            desc_b = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
                K, N, N * elem_size, dt)

        desc_c = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            M, N, N * elem_size, dt)

        mat_a = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(a_buf, desc_a)
        mat_b = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(b_buf, desc_b)

        if has_bias:
            bias_buf = pool[call.input_buffers[2]].native_handle
            _broadcast_bias_to_c(cmd_buf, bias_buf, c_buf, M, N, elem_size)

        mat_c = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(c_buf, desc_c)

        info["op"].encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
            cmd_buf, mat_a, mat_b, mat_c)
