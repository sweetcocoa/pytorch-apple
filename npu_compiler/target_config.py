"""Hardware target configuration for NPU backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetConfig:
    """Hardware-specific constants for a target backend."""
    name: str = "metal_gpu"
    channel_alignment_bytes: int = 64
    channel_tile: int = 32
    matmul_tile: int = 16
    max_threadgroup_1d: int = 256
    max_threadgroup_2d: int = 16
    max_threadgroup_3d_xy: int = 8
    max_threadgroup_3d_z: int = 4
    max_dispatches_per_batch: int = 10000


METAL_GPU = TargetConfig()


# ---------------------------------------------------------------------------
# Channel padding utilities (target-dependent, used by constraint_checker & codegen)
# ---------------------------------------------------------------------------
# Placed here (not in constraint_checker) to break the circular dependency:
# codegen → constraint_checker → codegen (via HANDLED_OPS).

def pad_channels(channels: int, config: TargetConfig = METAL_GPU) -> int:
    """Round up channel count to nearest multiple of channel_tile (64-byte alignment)."""
    tile = config.channel_tile
    return ((channels + tile - 1) // tile) * tile


def padded_shape_4d(shape: list[int], config: TargetConfig = METAL_GPU) -> list[int]:
    """Return shape with channels padded to alignment. Non-4D shapes unchanged."""
    if len(shape) == 4:
        N, C, H, W = shape
        return [N, pad_channels(C, config), H, W]
    return list(shape)
