"""Shared fixtures and helpers for NPU simulation tests."""

import os
import struct

import pytest

from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device


@pytest.fixture(scope="session")
def device():
    """Session-scoped Metal device."""
    return Device()


@pytest.fixture
def metal_kernels_dir():
    """Path to metal_kernels/ directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "metal_kernels")


def kernels_dir():
    """Return path to metal_kernels/ directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "metal_kernels")


def make_params(device, fmt, *values):
    """Create a Metal buffer from struct-packed parameters."""
    data = struct.pack(fmt, *values)
    return device.mtl_device.newBufferWithBytes_length_options_(data, len(data), 0)


def _set_buffers(encoder, buffers):
    for idx, buf in enumerate(buffers):
        mtl = buf.mtl_buffer if isinstance(buf, NPUBuffer) else buf
        encoder.setBuffer_offset_atIndex_(mtl, 0, idx)


def dispatch_1d(device, pipeline, buffers, total_threads):
    """Encode and dispatch a 1D compute kernel."""
    cmd_buf = device.new_command_buffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    _set_buffers(encoder, buffers)
    tpg = min(256, pipeline.maxTotalThreadsPerThreadgroup())
    groups = (total_threads + tpg - 1) // tpg
    encoder.dispatchThreadgroups_threadsPerThreadgroup_((groups, 1, 1), (tpg, 1, 1))
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()


def dispatch_2d(device, pipeline, buffers, width, height):
    """Encode and dispatch a 2D compute kernel."""
    cmd_buf = device.new_command_buffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    _set_buffers(encoder, buffers)
    tpg_x = min(16, width)
    tpg_y = min(16, height)
    groups_x = (width + tpg_x - 1) // tpg_x
    groups_y = (height + tpg_y - 1) // tpg_y
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        (groups_x, groups_y, 1), (tpg_x, tpg_y, 1)
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()


def dispatch_tiled_2d(device, pipeline, buffers, width, height, tile=16):
    """Dispatch a tiled 2D kernel with fixed tile-sized threadgroups."""
    cmd_buf = device.new_command_buffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    _set_buffers(encoder, buffers)
    groups_x = (width + tile - 1) // tile
    groups_y = (height + tile - 1) // tile
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        (groups_x, groups_y, 1), (tile, tile, 1)
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()


def dispatch_3d(device, pipeline, buffers, width, height, depth):
    """Encode and dispatch a 3D compute kernel."""
    cmd_buf = device.new_command_buffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    _set_buffers(encoder, buffers)
    tpg_x = min(8, width)
    tpg_y = min(8, height)
    tpg_z = min(4, depth)
    groups_x = (width + tpg_x - 1) // tpg_x
    groups_y = (height + tpg_y - 1) // tpg_y
    groups_z = (depth + tpg_z - 1) // tpg_z
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        (groups_x, groups_y, groups_z), (tpg_x, tpg_y, tpg_z)
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()
