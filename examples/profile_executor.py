"""Deep profiling of executor performance bottlenecks."""

from __future__ import annotations

import time

import numpy as np

import npu_compiler
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device
from npu_runtime.executor import Executor


def profile_decode_step():
    """Profile a single decode step to find bottlenecks."""
    # Compile decode program
    print("Compiling decode IR...")
    program = npu_compiler.compile("qwen2_decode_ir.json")
    print(f"  Kernels: {len(program.kernel_calls)}")
    print(f"  Buffer allocations: {len(program.buffer_allocations)}")
    print(f"  Compute dtype: {program.compute_dtype}")

    # Count kernel types
    kernel_counts = {}
    reshape_count = 0
    real_dispatch_count = 0
    for call in program.kernel_calls:
        kernel_counts[call.kernel_name] = kernel_counts.get(call.kernel_name, 0) + 1
        if call.kernel_name == "_reshape":
            reshape_count += 1
        elif call.dispatch_type != "none":
            real_dispatch_count += 1

    print(f"\n  Reshape (zero-cost alias): {reshape_count}")
    print(f"  Real GPU dispatches: {real_dispatch_count}")
    print(f"  No-dispatch (non-reshape): {len(program.kernel_calls) - reshape_count - real_dispatch_count}")
    print("\n  Top kernel types:")
    for name, count in sorted(kernel_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {name}: {count}")

    device = Device()
    print(f"\nDevice: {device.name}")

    # Create executor (this pre-compiles shaders)
    print("\nCreating executor (shader compilation)...")
    t0 = time.perf_counter()
    executor = Executor(program, device)
    print(f"  Executor init: {(time.perf_counter() - t0) * 1000:.1f}ms")

    # Prepare minimal decode inputs
    prefill_seq_len = 32  # matches IR extraction
    decode_inputs_np = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.zeros((1, 1, 1, prefill_seq_len + 1), dtype=np.float16),
        "position_ids": np.array([[5]], dtype=np.int64),
        "cache_position": np.array([5], dtype=np.int64),
    }

    # Create fake weight buffers (zeros)
    print("\nAllocating fake weights...")
    t0 = time.perf_counter()
    weights = {}
    for spec in program.weight_specs:
        if any(s == 0 for s in spec.shape):
            # Skip empty tensors (e.g. empty KV cache init)
            weights[spec.name] = NPUBuffer.zeros((1,), device)
        else:
            weights[spec.name] = NPUBuffer.from_numpy(
                np.zeros(spec.shape, dtype=np.float16), device, spec=None
            )
    weight_alloc_ms = (time.perf_counter() - t0) * 1000
    print(f"  Weight allocation: {weight_alloc_ms:.1f}ms ({len(weights)} buffers)")

    # Profile input buffer creation
    t0 = time.perf_counter()
    inputs = {}
    for spec in program.input_specs:
        if spec.name in decode_inputs_np:
            inputs[spec.name] = NPUBuffer.from_numpy(decode_inputs_np[spec.name], device, spec=spec)
    input_ms = (time.perf_counter() - t0) * 1000

    # Also create fake KV cache inputs
    for spec in program.input_specs:
        if spec.name not in inputs:
            inputs[spec.name] = NPUBuffer.zeros(tuple(spec.shape), device)
    print(f"  Input creation: {input_ms:.1f}ms")

    # ── Profile the run() breakdown ──
    print("\n" + "=" * 60)
    print("PROFILING run() BREAKDOWN")
    print("=" * 60)

    # (A) Pool construction + intermediate buffer allocation
    t0 = time.perf_counter()
    pool = {}
    pool.update(inputs)
    pool.update(weights)
    pool_copy_ms = (time.perf_counter() - t0) * 1000

    import ml_dtypes
    t0 = time.perf_counter()
    for alloc in program.buffer_allocations:
        dtype = np.dtype(np.float16)
        if alloc.dtype == "bfloat16":
            dtype = np.dtype(ml_dtypes.bfloat16)
        elif alloc.dtype == "int32":
            dtype = np.dtype(np.int32)
        pool[alloc.name] = NPUBuffer.zeros(
            tuple(alloc.shape), device, dtype=dtype,
            alloc_shape=tuple(alloc.alloc_shape) if alloc.alloc_shape else None,
        )
    intermediate_alloc_ms = (time.perf_counter() - t0) * 1000
    print(f"\n(A) Pool copy: {pool_copy_ms:.2f}ms")
    print(f"(A) Intermediate buffer alloc: {intermediate_alloc_ms:.1f}ms ({len(program.buffer_allocations)} buffers)")

    # (B) Output buffer allocation
    t0 = time.perf_counter()
    for spec in program.output_specs:
        if spec.name not in pool:
            pool[spec.name] = NPUBuffer.zeros(
                tuple(spec.shape), device,
                alloc_shape=tuple(spec.alloc_shape) if spec.alloc_shape else None,
            )
    output_alloc_ms = (time.perf_counter() - t0) * 1000
    print(f"(B) Output buffer alloc: {output_alloc_ms:.1f}ms")

    # (C) Command encoding
    cmd_buf = device.new_command_buffer()

    reshape_time = 0
    encoder_create_time = 0
    buffer_bind_time = 0
    param_pack_time = 0
    dispatch_time = 0
    end_encoding_time = 0
    total_encoders_created = 0

    import struct

    for call in program.kernel_calls:
        if call.kernel_name == "_reshape":
            t0 = time.perf_counter()
            in_name = call.input_buffers[0]
            out_name = call.output_buffers[0]
            if in_name in pool:
                out_shape = tuple(call.params["output_shape"])
                pool[out_name] = NPUBuffer(
                    pool[in_name].mtl_buffer, out_shape,
                    pool[in_name].dtype, device,
                    alloc_shape=pool[in_name].alloc_shape,
                )
            reshape_time += time.perf_counter() - t0
            continue

        if call.dispatch_type == "none":
            continue

        # Encoder creation
        t0 = time.perf_counter()
        encoder = cmd_buf.computeCommandEncoder()
        pipeline = executor._pipelines[call.kernel_name]
        encoder.setComputePipelineState_(pipeline)
        encoder_create_time += time.perf_counter() - t0
        total_encoders_created += 1

        # Buffer binding
        t0 = time.perf_counter()
        buf_idx = 0
        for name in call.input_buffers:
            if name in pool:
                encoder.setBuffer_offset_atIndex_(pool[name].mtl_buffer, 0, buf_idx)
            buf_idx += 1
        for name in call.output_buffers:
            if name in pool:
                encoder.setBuffer_offset_atIndex_(pool[name].mtl_buffer, 0, buf_idx)
            buf_idx += 1
        buffer_bind_time += time.perf_counter() - t0

        # Param packing
        t0 = time.perf_counter()
        params_buf = executor._pack_params(call)
        if params_buf is not None:
            encoder.setBuffer_offset_atIndex_(params_buf, 0, buf_idx)
        param_pack_time += time.perf_counter() - t0

        # Dispatch
        t0 = time.perf_counter()
        if call.dispatch_type == "1d":
            tpg = min(256, pipeline.maxTotalThreadsPerThreadgroup())
            groups = (call.total_threads + tpg - 1) // tpg
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                (groups, 1, 1), (tpg, 1, 1)
            )
        elif call.dispatch_type == "2d":
            tpg_x = min(16, call.grid_width)
            tpg_y = min(16, call.grid_height)
            groups_x = (call.grid_width + tpg_x - 1) // tpg_x
            groups_y = (call.grid_height + tpg_y - 1) // tpg_y
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                (groups_x, groups_y, 1), (tpg_x, tpg_y, 1)
            )
        elif call.dispatch_type == "3d":
            tpg_x = min(8, call.grid_width)
            tpg_y = min(8, call.grid_height)
            tpg_z = min(4, call.grid_depth)
            groups_x = (call.grid_width + tpg_x - 1) // tpg_x
            groups_y = (call.grid_height + tpg_y - 1) // tpg_y
            groups_z = (call.grid_depth + tpg_z - 1) // tpg_z
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                (groups_x, groups_y, groups_z), (tpg_x, tpg_y, tpg_z)
            )
        dispatch_time += time.perf_counter() - t0

        # End encoding
        t0 = time.perf_counter()
        encoder.endEncoding()
        end_encoding_time += time.perf_counter() - t0

    # (D) Commit & wait
    t0 = time.perf_counter()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()
    gpu_exec_ms = (time.perf_counter() - t0) * 1000

    print(f"\n(C) Command encoding breakdown ({total_encoders_created} encoders):")
    print(f"  Reshape (Python alias): {reshape_time * 1000:.1f}ms")
    print(f"  Encoder create + setPipeline: {encoder_create_time * 1000:.1f}ms")
    print(f"  Buffer binding: {buffer_bind_time * 1000:.1f}ms")
    print(f"  Param packing (struct + MTL alloc): {param_pack_time * 1000:.1f}ms")
    print(f"  Dispatch: {dispatch_time * 1000:.1f}ms")
    print(f"  endEncoding: {end_encoding_time * 1000:.1f}ms")
    total_encoding = (reshape_time + encoder_create_time + buffer_bind_time +
                      param_pack_time + dispatch_time + end_encoding_time) * 1000
    print(f"  TOTAL encoding: {total_encoding:.1f}ms")

    print(f"\n(D) GPU execution (commit + wait): {gpu_exec_ms:.1f}ms")

    total = pool_copy_ms + intermediate_alloc_ms + output_alloc_ms + total_encoding + gpu_exec_ms
    print(f"\n{'=' * 60}")
    print(f"TOTAL per decode step: ~{total:.0f}ms")
    print(f"  Buffer alloc overhead: {intermediate_alloc_ms + output_alloc_ms:.0f}ms ({(intermediate_alloc_ms + output_alloc_ms) / total * 100:.0f}%)")
    print(f"  Encoding overhead: {total_encoding:.0f}ms ({total_encoding / total * 100:.0f}%)")
    print(f"  GPU execution: {gpu_exec_ms:.0f}ms ({gpu_exec_ms / total * 100:.0f}%)")

    # ── Compare: single encoder approach ──
    print(f"\n{'=' * 60}")
    print("COMPARISON: Single encoder vs per-kernel encoder")
    print("=" * 60)

    # Re-run with single encoder
    pool2 = {}
    pool2.update(inputs)
    pool2.update(weights)
    for alloc in program.buffer_allocations:
        dtype = np.dtype(np.float16)
        if alloc.dtype == "bfloat16":
            dtype = np.dtype(ml_dtypes.bfloat16)
        elif alloc.dtype == "int32":
            dtype = np.dtype(np.int32)
        pool2[alloc.name] = NPUBuffer.zeros(
            tuple(alloc.shape), device, dtype=dtype,
            alloc_shape=tuple(alloc.alloc_shape) if alloc.alloc_shape else None,
        )
    for spec in program.output_specs:
        if spec.name not in pool2:
            pool2[spec.name] = NPUBuffer.zeros(
                tuple(spec.shape), device,
                alloc_shape=tuple(spec.alloc_shape) if spec.alloc_shape else None,
            )

    cmd_buf2 = device.new_command_buffer()
    encoder = cmd_buf2.computeCommandEncoder()  # SINGLE encoder

    t0 = time.perf_counter()
    for call in program.kernel_calls:
        if call.kernel_name == "_reshape":
            in_name = call.input_buffers[0]
            out_name = call.output_buffers[0]
            if in_name in pool2:
                out_shape = tuple(call.params["output_shape"])
                pool2[out_name] = NPUBuffer(
                    pool2[in_name].mtl_buffer, out_shape,
                    pool2[in_name].dtype, device,
                    alloc_shape=pool2[in_name].alloc_shape,
                )
            continue

        if call.dispatch_type == "none":
            continue

        pipeline = executor._pipelines[call.kernel_name]
        encoder.setComputePipelineState_(pipeline)

        buf_idx = 0
        for name in call.input_buffers:
            if name in pool2:
                encoder.setBuffer_offset_atIndex_(pool2[name].mtl_buffer, 0, buf_idx)
            buf_idx += 1
        for name in call.output_buffers:
            if name in pool2:
                encoder.setBuffer_offset_atIndex_(pool2[name].mtl_buffer, 0, buf_idx)
            buf_idx += 1

        params_buf = executor._pack_params(call)
        if params_buf is not None:
            encoder.setBuffer_offset_atIndex_(params_buf, 0, buf_idx)

        if call.dispatch_type == "1d":
            tpg = min(256, pipeline.maxTotalThreadsPerThreadgroup())
            groups = (call.total_threads + tpg - 1) // tpg
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                (groups, 1, 1), (tpg, 1, 1)
            )
        elif call.dispatch_type == "2d":
            tpg_x = min(16, call.grid_width)
            tpg_y = min(16, call.grid_height)
            groups_x = (call.grid_width + tpg_x - 1) // tpg_x
            groups_y = (call.grid_height + tpg_y - 1) // tpg_y
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                (groups_x, groups_y, 1), (tpg_x, tpg_y, 1)
            )
        elif call.dispatch_type == "3d":
            tpg_x = min(8, call.grid_width)
            tpg_y = min(8, call.grid_height)
            tpg_z = min(4, call.grid_depth)
            groups_x = (call.grid_width + tpg_x - 1) // tpg_x
            groups_y = (call.grid_height + tpg_y - 1) // tpg_y
            groups_z = (call.grid_depth + tpg_z - 1) // tpg_z
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                (groups_x, groups_y, groups_z), (tpg_x, tpg_y, tpg_z)
            )

    single_encode_ms = (time.perf_counter() - t0) * 1000
    encoder.endEncoding()

    t0 = time.perf_counter()
    cmd_buf2.commit()
    cmd_buf2.waitUntilCompleted()
    single_gpu_ms = (time.perf_counter() - t0) * 1000

    print(f"  Single encoder: encode={single_encode_ms:.1f}ms, GPU={single_gpu_ms:.1f}ms")
    print(f"  Per-kernel encoder: encode={total_encoding:.1f}ms, GPU={gpu_exec_ms:.1f}ms")
    speedup = total_encoding / single_encode_ms if single_encode_ms > 0 else 0
    print(f"  Encoding speedup: {speedup:.1f}x")


if __name__ == "__main__":
    profile_decode_step()
