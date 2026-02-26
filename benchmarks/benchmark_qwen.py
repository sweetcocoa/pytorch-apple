"""Qwen2.5-1.5B decode scaling benchmark: Executor vs DAGExecutor.

Measures decode TPS at log-scale positions up to the model's
max_position_embeddings (32768 tokens). At each position, a KV cache
is filled with random data and a single decode step is timed.
This avoids actually running thousands of decode steps.

If existing IR files have a smaller max_cache_len than requested,
the script will automatically re-extract IR with the correct size.

Prerequisites:
    pip install transformers huggingface_hub safetensors matplotlib ml-dtypes

Usage:
    cd examples/
    python ../benchmarks/benchmark_qwen.py
    python ../benchmarks/benchmark_qwen.py --max-cache-len 4096  # smaller test
    python ../benchmarks/benchmark_qwen.py --chart scaling_chart.png
    python ../benchmarks/benchmark_qwen.py --mode executor  # executor only
    python ../benchmarks/benchmark_qwen.py --mode dag       # dag only
    python ../benchmarks/benchmark_qwen.py --repeats 5      # 5 measurements per point
    python ../benchmarks/benchmark_qwen.py --mode both --chart-dir ../docs/assets/
    python ../benchmarks/benchmark_qwen.py --comparison-chart comparison.png
    python ../benchmarks/benchmark_qwen.py --backend cuda   # CUDA backend
    python ../benchmarks/benchmark_qwen.py --backend both   # Metal vs CUDA comparison
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# Allow running from project root or examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_MAX_POSITION_EMBEDDINGS = 32768  # Qwen2.5-1.5B-Instruct architectural max
PROMPT_LENGTH = 64


def _log_scale_steps(max_steps: int) -> list[int]:
    """Generate log-scale positions: 1, 2, 4, ..., up to max_steps."""
    steps = []
    v = 1
    while v < max_steps:
        steps.append(v)
        v *= 2
    steps.append(max_steps)
    return steps


@dataclass
class ScalingPoint:
    """Benchmark result for a single decode position."""

    position: int  # KV cache fill level (decode step number)
    step_time_ms: float = 0.0  # median single-step decode time
    step_time_std_ms: float = 0.0
    tps: float = 0.0  # 1000 / step_time_ms
    ttft_ms: float = 0.0  # prefill time (constant across positions)


@dataclass
class ScalingReport:
    """Full scaling benchmark report."""

    environment: dict = field(default_factory=dict)
    compile_time_sec: float = 0.0
    weight_load_time_sec: float = 0.0
    prompt_length: int = PROMPT_LENGTH
    max_cache_len: int = 0
    executor_results: list[ScalingPoint] = field(default_factory=list)
    dag_results: list[ScalingPoint] = field(default_factory=list)


def get_environment_info() -> dict:
    """Collect execution environment details."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
    }
    try:
        chip_name = subprocess.check_output(["sysctl", "-n", "hw.chip"], text=True, stderr=subprocess.DEVNULL).strip()
        info["chipset"] = chip_name
    except Exception:
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            info["chipset"] = chip
        except Exception:
            info["chipset"] = "unknown"
    try:
        mem_bytes = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        info["ram_gb"] = mem_bytes / (1024**3)
    except Exception:
        info["ram_gb"] = 0
    try:
        from npu_runtime.device import Device

        dev = Device()
        info["metal_gpu"] = str(dev.device.name())
    except Exception:
        info["metal_gpu"] = "unknown"
    return info


def _get_ir_max_cache_len(kv_mapping_path: str) -> int | None:
    """Read max_cache_len from existing KV mapping file."""
    try:
        with open(kv_mapping_path) as f:
            return json.load(f).get("max_cache_len")
    except FileNotFoundError:
        return None


def _extract_ir_if_needed(
    prefill_ir_path: str,
    decode_ir_path: str,
    kv_mapping_path: str,
    needed_max_cache_len: int,
    prefill_seq_len: int = 128,
) -> None:
    """Re-extract IR files if they don't exist or have a smaller max_cache_len than needed."""
    current = _get_ir_max_cache_len(kv_mapping_path)
    if current is not None and current >= needed_max_cache_len:
        print(f"  Existing IR has max_cache_len={current} (>= {needed_max_cache_len}), reusing.")
        return

    print(f"  Extracting IR with max_cache_len={needed_max_cache_len} (current={current})...")
    extract_script = os.path.join(os.path.dirname(__file__), "..", "examples", "extract_qwen_ir.py")
    cmd = [
        sys.executable,
        extract_script,
        "--max-cache-len",
        str(needed_max_cache_len),
        "--prefill-seq-len",
        str(prefill_seq_len),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  IR extraction failed:\n{result.stderr}")
        raise RuntimeError("IR extraction failed")
    print("  IR extraction complete.")


# ---------------------------------------------------------------------------
# Executor benchmark — single-step at each position
# ---------------------------------------------------------------------------


def benchmark_executor_scaling(
    positions: list[int],
    prefill_executor,
    decode_executor,
    prefill_program,
    decode_program,
    prefill_weights,
    decode_weights,
    kv_map,
    device,
    input_ids: np.ndarray,
    repeats: int = 3,
) -> tuple[list[ScalingPoint], float]:
    """Measure single-step decode time at each cache position.

    Returns (results, ttft_ms).
    """
    from npu_runtime.buffer import NPUBuffer

    prefill_seq_len = prefill_program.input_specs[0].shape[1]
    first_kv_name = kv_map["layers"][0]["decode_key_input"]
    max_cache_len = next(s.shape[2] for s in decode_program.input_specs if s.name == first_kv_name)
    _NEG_INF = np.float16(-np.inf)
    _FIXED_INPUTS = {"input_ids", "attention_mask", "position_ids", "cache_position"}
    decode_kv_specs = {s.name: s for s in decode_program.input_specs if s.name not in _FIXED_INPUTS}

    # Prefill (for TTFT measurement)
    actual_len = min(len(input_ids), prefill_seq_len)
    padded_ids = np.zeros((1, prefill_seq_len), dtype=np.int64)
    padded_ids[0, :actual_len] = input_ids[:actual_len]

    causal_mask = np.triu(
        np.full((1, 1, prefill_seq_len, prefill_seq_len), _NEG_INF, dtype=np.float16),
        k=1,
    )
    causal_mask[:, :, :, actual_len:] = _NEG_INF
    causal_mask[:, :, actual_len:, :] = _NEG_INF

    np_map = {
        "input_ids": padded_ids,
        "attention_mask": causal_mask,
        "position_ids": np.arange(prefill_seq_len, dtype=np.int64).reshape(1, -1),
        "cache_position": np.arange(prefill_seq_len, dtype=np.int64),
    }
    prefill_inputs = {
        spec.name: NPUBuffer.from_numpy(np_map[spec.name], device, spec=spec) for spec in prefill_program.input_specs
    }

    t0 = time.perf_counter()
    prefill_executor.run(inputs=prefill_inputs, weights=prefill_weights)
    ttft_ms = (time.perf_counter() - t0) * 1000

    # For each position, create random KV cache and time one decode step
    results = []
    for pos in positions:
        # Create random KV cache buffers (simulating pos tokens already decoded)
        kv_buffers = {}
        for layer in kv_map["layers"]:
            for dkey in [layer["decode_key_input"], layer["decode_value_input"]]:
                spec = decode_kv_specs[dkey]
                # Random data in the KV cache shape
                kv_data = np.random.randn(*spec.shape).astype(np.float16)
                kv_buffers[dkey] = NPUBuffer.from_numpy(kv_data, device, spec=spec)

        # Prepare decode inputs at this position
        cur_pos = prefill_seq_len + pos - 1  # 0-indexed position in cache
        token_arr = np.array([[1]], dtype=np.int64)  # dummy token
        decode_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
        decode_mask[0, 0, 0, : cur_pos + 1] = 0.0

        step_np_map = {
            "input_ids": token_arr,
            "attention_mask": decode_mask,
            "position_ids": np.array([[cur_pos]], dtype=np.int64),
            "cache_position": np.array([cur_pos], dtype=np.int64),
        }

        decode_inputs = {}
        for spec in decode_program.input_specs:
            if spec.name in step_np_map:
                decode_inputs[spec.name] = NPUBuffer.from_numpy(step_np_map[spec.name], device, spec=spec)
            elif spec.name in kv_buffers:
                decode_inputs[spec.name] = kv_buffers[spec.name]

        # Measure
        times_ms = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            decode_executor.run(inputs=decode_inputs, weights=decode_weights)
            times_ms.append((time.perf_counter() - t0) * 1000)

        median_ms = float(np.median(times_ms))
        std_ms = float(np.std(times_ms))
        tps = 1000.0 / median_ms if median_ms > 0 else 0

        pt = ScalingPoint(
            position=pos,
            step_time_ms=median_ms,
            step_time_std_ms=std_ms,
            tps=tps,
            ttft_ms=ttft_ms,
        )
        results.append(pt)
        print(f"    pos={pos:>6}: {median_ms:.1f}ms (TPS={tps:.1f})")

    return results, ttft_ms


# ---------------------------------------------------------------------------
# DAGExecutor benchmark — single-step at each position
# ---------------------------------------------------------------------------


def benchmark_dag_scaling(
    positions: list[int],
    prefill_ir_path: str,
    decode_ir_path: str,
    kv_mapping_path: str,
    model_id: str,
    input_ids: np.ndarray,
    repeats: int = 3,
) -> tuple[list[ScalingPoint], float]:
    """Measure single-step DAG decode time at each cache position.

    Uses DeviceBuffer inputs (same as Executor benchmark) to measure
    pure execution time without numpy→GPU upload overhead.
    """
    import ml_dtypes  # noqa: F401

    from npu_compiler.op_support import is_op_supported
    from npu_compiler.partitioner import partition
    from npu_runtime.buffer import NPUBuffer
    from npu_runtime.dag_executor import DAGExecutor
    from npu_runtime.metal_backend import MetalBackend

    with open(prefill_ir_path) as f:
        prefill_ir_dict = json.load(f)
    with open(decode_ir_path) as f:
        decode_ir_dict = json.load(f)
    with open(kv_mapping_path) as f:
        kv_map = json.load(f)

    backend = MetalBackend()
    device = backend.device
    prefill_plan = partition(prefill_ir_dict, is_op_supported)
    decode_plan = partition(decode_ir_dict, is_op_supported)

    prefill_dag = DAGExecutor(prefill_plan, backend)
    decode_dag = DAGExecutor(decode_plan, backend)

    from run_qwen_graph import load_numpy_weights

    prefill_weights_np = load_numpy_weights(model_id, prefill_ir_dict)
    decode_weights_np = load_numpy_weights(model_id, decode_ir_dict)

    prefill_dag.load_weights(prefill_weights_np)
    decode_dag.load_weights(decode_weights_np)

    prefill_seq_len = prefill_ir_dict["graph_inputs"][0]["shape"][1]
    first_kv_name = kv_map["layers"][0]["decode_key_input"]
    max_cache_len = None
    for gi in decode_ir_dict["graph_inputs"]:
        if gi["name"] == first_kv_name:
            max_cache_len = gi["shape"][2]
            break

    _NEG_INF = np.float16(-np.inf)
    _FIXED_INPUTS = {"input_ids", "attention_mask", "position_ids", "cache_position"}

    # Build decode input spec lookup for alloc_shape
    decode_input_specs = {s.name: s for s in decode_dag.npu_programs[0].input_specs}

    # Prefill (for TTFT)
    actual_len = min(len(input_ids), prefill_seq_len)
    padded_ids = np.zeros((1, prefill_seq_len), dtype=np.int64)
    padded_ids[0, :actual_len] = input_ids[:actual_len]

    causal_mask = np.triu(
        np.full((1, 1, prefill_seq_len, prefill_seq_len), _NEG_INF, dtype=np.float16),
        k=1,
    )
    causal_mask[:, :, :, actual_len:] = _NEG_INF
    causal_mask[:, :, actual_len:, :] = _NEG_INF

    prefill_inputs = {
        "input_ids": padded_ids,
        "attention_mask": causal_mask,
        "position_ids": np.arange(prefill_seq_len, dtype=np.int64).reshape(1, -1),
        "cache_position": np.arange(prefill_seq_len, dtype=np.int64),
    }

    t0 = time.perf_counter()
    prefill_dag.execute(inputs=prefill_inputs)
    ttft_ms = (time.perf_counter() - t0) * 1000

    # Build decode KV input shapes
    decode_kv_shapes = {}
    for gi in decode_ir_dict["graph_inputs"]:
        if gi["name"] not in _FIXED_INPUTS:
            decode_kv_shapes[gi["name"]] = gi["shape"]

    # For each position, time one decode step with random KV cache
    results = []
    for pos in positions:
        cur_pos = prefill_seq_len + pos - 1

        # Create KV cache as DeviceBuffer (same as Executor benchmark)
        kv_buffers: dict[str, NPUBuffer] = {}
        for layer in kv_map["layers"]:
            for dkey in [layer["decode_key_input"], layer["decode_value_input"]]:
                shape = decode_kv_shapes[dkey]
                kv_data = np.random.randn(*shape).astype(np.float16)
                spec = decode_input_specs.get(dkey)
                kv_buffers[dkey] = NPUBuffer.from_numpy(kv_data, device, spec=spec)

        decode_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
        decode_mask[0, 0, 0, : cur_pos + 1] = 0.0

        # Build inputs as DeviceBuffer
        step_np_map = {
            "input_ids": np.array([[1]], dtype=np.int64),
            "attention_mask": decode_mask,
            "position_ids": np.array([[cur_pos]], dtype=np.int64),
            "cache_position": np.array([cur_pos], dtype=np.int64),
        }

        decode_inputs: dict[str, NPUBuffer] = {}
        for name, arr in step_np_map.items():
            spec = decode_input_specs.get(name)
            decode_inputs[name] = NPUBuffer.from_numpy(arr, device, spec=spec)
        decode_inputs.update(kv_buffers)

        times_ms = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            decode_dag.execute(inputs=decode_inputs)
            times_ms.append((time.perf_counter() - t0) * 1000)

        median_ms = float(np.median(times_ms))
        std_ms = float(np.std(times_ms))
        tps = 1000.0 / median_ms if median_ms > 0 else 0

        pt = ScalingPoint(
            position=pos,
            step_time_ms=median_ms,
            step_time_std_ms=std_ms,
            tps=tps,
            ttft_ms=ttft_ms,
        )
        results.append(pt)
        print(f"    pos={pos:>6}: {median_ms:.1f}ms (TPS={tps:.1f})")

    return results, ttft_ms


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(report: ScalingReport):
    """Print formatted scaling benchmark report."""
    env = report.environment

    print("\n" + "=" * 80)
    print("Qwen2.5-1.5B Decode Scaling Benchmark")
    print("=" * 80)

    print("\nEnvironment:")
    print(f"  Chipset:    {env.get('chipset', 'N/A')}")
    print(f"  RAM:        {env.get('ram_gb', 0):.0f} GB")
    print(f"  Metal GPU:  {env.get('metal_gpu', 'N/A')}")

    print("\nConfiguration:")
    print(f"  Prompt length:    {report.prompt_length} tokens")
    print(f"  Max cache length: {report.max_cache_len} tokens")
    print(f"  Compile time:     {report.compile_time_sec:.2f} sec")
    print(f"  Weight load time: {report.weight_load_time_sec:.2f} sec")

    def _print_table(title: str, results: list[ScalingPoint]):
        if not results:
            return
        print(f"\n{title}:")
        print(f"  TTFT: {results[0].ttft_ms:.1f} ms")
        print(f"  {'Position':>10} {'Step(ms)':>14} {'TPS':>10}")
        print("  " + "-" * 38)
        for r in results:
            print(f"  {r.position:>10} {r.step_time_ms:>7.1f}±{r.step_time_std_ms:<5.1f} {r.tps:>9.1f}")

    _print_table("Executor (Direct Metal NPU)", report.executor_results)
    _print_table("DAGExecutor (Partitioned NPU+CPU)", report.dag_results)

    if report.executor_results and report.dag_results:
        print("\nComparison (DAG / Executor):")
        print(f"  {'Position':>10} {'Executor TPS':>14} {'DAG TPS':>10} {'Ratio':>8}")
        print("  " + "-" * 46)
        dag_by_pos = {r.position: r for r in report.dag_results}
        for er in report.executor_results:
            dr = dag_by_pos.get(er.position)
            if dr:
                ratio = dr.tps / er.tps if er.tps > 0 else 0
                print(f"  {er.position:>10} {er.tps:>13.1f} {dr.tps:>9.1f} {ratio:>7.2f}x")

    print()


def save_chart(report: ScalingReport, output_path: str):
    """Generate TPS vs position chart (log-x scale)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping chart generation")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: TPS vs position
    ax = axes[0]
    if report.executor_results:
        x = [r.position for r in report.executor_results]
        y = [r.tps for r in report.executor_results]
        ax.plot(x, y, marker="o", label="Executor", color="#1f77b4")
    if report.dag_results:
        x = [r.position for r in report.dag_results]
        y = [r.tps for r in report.dag_results]
        ax.plot(x, y, marker="s", label="DAGExecutor", color="#ff7f0e")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cache Position (tokens, log2)")
    ax.set_ylabel("Decode TPS (tokens/sec)")
    ax.set_title("Decode Speed vs Cache Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chart 2: Step time vs position
    ax = axes[1]
    if report.executor_results:
        x = [r.position for r in report.executor_results]
        y = [r.step_time_ms for r in report.executor_results]
        yerr = [r.step_time_std_ms for r in report.executor_results]
        ax.errorbar(x, y, yerr=yerr, marker="o", label="Executor", color="#1f77b4", capsize=3)
    if report.dag_results:
        x = [r.position for r in report.dag_results]
        y = [r.step_time_ms for r in report.dag_results]
        yerr = [r.step_time_std_ms for r in report.dag_results]
        ax.errorbar(x, y, yerr=yerr, marker="s", label="DAGExecutor", color="#ff7f0e", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cache Position (tokens, log2)")
    ax.set_ylabel("Step Time (ms)")
    ax.set_title("Per-Step Decode Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Qwen2.5-1.5B Decode Scaling — max_cache={report.max_cache_len}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to {output_path}")
    plt.close()


def save_comparison_chart(report: ScalingReport, output_path: str):
    """Generate 3-panel Executor vs DAG comparison chart (log-x scale)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping comparison chart")
        return

    if not report.executor_results or not report.dag_results:
        print("  [WARN] Need both Executor and DAG results for comparison chart")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ex_pos = [r.position for r in report.executor_results]
    dag_pos = [r.position for r in report.dag_results]

    # Panel 1: TPS vs cache position overlay
    ax = axes[0]
    ax.plot(ex_pos, [r.tps for r in report.executor_results], marker="o", label="Executor", color="#1f77b4")
    ax.plot(dag_pos, [r.tps for r in report.dag_results], marker="s", label="DAGExecutor", color="#ff7f0e")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cache Position (tokens, log2)")
    ax.set_ylabel("Decode TPS (tokens/sec)")
    ax.set_title("Decode TPS vs Cache Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Step time vs position overlay (with error bars)
    ax = axes[1]
    ax.errorbar(
        ex_pos,
        [r.step_time_ms for r in report.executor_results],
        yerr=[r.step_time_std_ms for r in report.executor_results],
        marker="o",
        label="Executor",
        color="#1f77b4",
        capsize=3,
    )
    ax.errorbar(
        dag_pos,
        [r.step_time_ms for r in report.dag_results],
        yerr=[r.step_time_std_ms for r in report.dag_results],
        marker="s",
        label="DAGExecutor",
        color="#ff7f0e",
        capsize=3,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cache Position (tokens, log2)")
    ax.set_ylabel("Step Time (ms)")
    ax.set_title("Per-Step Decode Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: DAG/Executor TPS ratio vs position
    ax = axes[2]
    dag_by_pos = {r.position: r for r in report.dag_results}
    common_pos = [p for p in ex_pos if p in dag_by_pos]
    ratios = []
    for p in common_pos:
        ex_r = next(r for r in report.executor_results if r.position == p)
        ratios.append(dag_by_pos[p].tps / ex_r.tps if ex_r.tps > 0 else 0)
    colors = ["#2ca02c" if r >= 1.0 else "#d62728" for r in ratios]
    ax.bar(range(len(common_pos)), ratios, color=colors, alpha=0.8)
    ax.set_xticks(range(len(common_pos)))
    ax.set_xticklabels([str(p) for p in common_pos], rotation=45, ha="right", fontsize=7)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Parity (1.0)")
    for i, ratio in enumerate(ratios):
        ax.text(i, ratio, f"{ratio:.2f}x", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Cache Position")
    ax.set_ylabel("DAG TPS / Executor TPS")
    ax.set_title("DAGExecutor Efficiency Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Qwen2.5-1.5B Executor vs DAGExecutor — max_cache={report.max_cache_len}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Comparison chart saved to {output_path}")
    plt.close()


def save_json(report: ScalingReport, output_path: str):
    """Save benchmark results as JSON."""

    def _points_to_dicts(points: list[ScalingPoint]) -> list[dict]:
        return [
            {
                "position": p.position,
                "step_time_ms": p.step_time_ms,
                "step_time_std_ms": p.step_time_std_ms,
                "tps": p.tps,
                "ttft_ms": p.ttft_ms,
            }
            for p in points
        ]

    data = {
        "environment": report.environment,
        "compile_time_sec": report.compile_time_sec,
        "weight_load_time_sec": report.weight_load_time_sec,
        "prompt_length": report.prompt_length,
        "max_cache_len": report.max_cache_len,
        "executor_results": _points_to_dicts(report.executor_results),
        "dag_results": _points_to_dicts(report.dag_results),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON results saved to {output_path}")


def benchmark_dag_cuda_scaling(
    positions: list[int],
    prefill_ir_path: str,
    decode_ir_path: str,
    kv_mapping_path: str,
    model_id: str,
    input_ids: np.ndarray,
    repeats: int = 3,
) -> tuple[list[ScalingPoint], float]:
    """Measure single-step CUDA DAG decode time at each cache position.

    Requires CuPy. Uses CUDABackend + compile_subgraph.
    """
    import ml_dtypes  # noqa: F401

    from cuda_compiler import compile_subgraph
    from cuda_compiler.op_support import is_cuda_op_supported
    from cuda_runtime.cuda_backend import CUDABackend
    from npu_compiler.partitioner import partition
    from npu_runtime.dag_executor import DAGExecutor

    with open(prefill_ir_path) as f:
        prefill_ir_dict = json.load(f)
    with open(decode_ir_path) as f:
        decode_ir_dict = json.load(f)
    with open(kv_mapping_path) as f:
        kv_map = json.load(f)

    backend = CUDABackend()
    prefill_plan = partition(prefill_ir_dict, is_cuda_op_supported)
    decode_plan = partition(decode_ir_dict, is_cuda_op_supported)

    prefill_dag = DAGExecutor(prefill_plan, backend, compile_fn=compile_subgraph)
    decode_dag = DAGExecutor(decode_plan, backend, compile_fn=compile_subgraph)

    from run_qwen_graph import load_numpy_weights

    prefill_weights_np = load_numpy_weights(model_id, prefill_ir_dict)
    decode_weights_np = load_numpy_weights(model_id, decode_ir_dict)

    prefill_dag.load_weights(prefill_weights_np)
    decode_dag.load_weights(decode_weights_np)

    prefill_seq_len = prefill_ir_dict["graph_inputs"][0]["shape"][1]
    first_kv_name = kv_map["layers"][0]["decode_key_input"]
    max_cache_len = None
    for gi in decode_ir_dict["graph_inputs"]:
        if gi["name"] == first_kv_name:
            max_cache_len = gi["shape"][2]
            break

    _NEG_INF = np.float16(-np.inf)
    _FIXED_INPUTS = {"input_ids", "attention_mask", "position_ids", "cache_position"}

    # Prefill
    actual_len = min(len(input_ids), prefill_seq_len)
    padded_ids = np.zeros((1, prefill_seq_len), dtype=np.int64)
    padded_ids[0, :actual_len] = input_ids[:actual_len]

    causal_mask = np.triu(
        np.full((1, 1, prefill_seq_len, prefill_seq_len), _NEG_INF, dtype=np.float16),
        k=1,
    )
    causal_mask[:, :, :, actual_len:] = _NEG_INF
    causal_mask[:, :, actual_len:, :] = _NEG_INF

    prefill_inputs = {
        "input_ids": padded_ids,
        "attention_mask": causal_mask,
        "position_ids": np.arange(prefill_seq_len, dtype=np.int64).reshape(1, -1),
        "cache_position": np.arange(prefill_seq_len, dtype=np.int64),
    }

    # Warmup prefill (JIT compile + buffer alloc on first call)
    import cupy as _cp
    for _ in range(3):
        prefill_dag.execute(inputs=prefill_inputs)
    _cp.cuda.Device().synchronize()

    # Measure steady-state TTFT (no compilation overhead)
    # Note: no explicit GPU sync to match Metal benchmark methodology
    # (Metal executor blocks internally via waitUntilCompleted)
    ttft_times = []
    for _ in range(max(repeats, 7)):
        _cp.cuda.Device().synchronize()  # drain prior work
        t0 = time.perf_counter()
        prefill_dag.execute(inputs=prefill_inputs)
        ttft_times.append((time.perf_counter() - t0) * 1000)
    ttft_ms = float(np.median(ttft_times))

    # Build decode KV input shapes
    decode_kv_shapes = {}
    for gi in decode_ir_dict["graph_inputs"]:
        if gi["name"] not in _FIXED_INPUTS:
            decode_kv_shapes[gi["name"]] = gi["shape"]

    # Pre-allocate KV cache on GPU as CUDABuffers (avoids PCIe transfer per step)
    import cupy as cp

    from cuda_runtime.cuda_backend import CUDABuffer

    kv_gpu_buffers: dict[str, CUDABuffer] = {}
    for layer in kv_map["layers"]:
        for dkey in [layer["decode_key_input"], layer["decode_value_input"]]:
            shape = decode_kv_shapes[dkey]
            kv_gpu = cp.random.randn(*shape, dtype=cp.float32).astype(cp.float16)
            kv_gpu_buffers[dkey] = CUDABuffer(kv_gpu, logical_shape=tuple(shape))

    # Warmup decode (first call has allocation overhead)
    warmup_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
    warmup_mask[0, 0, 0, :prefill_seq_len + 1] = 0.0
    warmup_inputs: dict = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": warmup_mask,
        "position_ids": np.array([[prefill_seq_len]], dtype=np.int64),
        "cache_position": np.array([prefill_seq_len], dtype=np.int64),
    }
    warmup_inputs.update(kv_gpu_buffers)
    for _ in range(3):
        decode_dag.execute(inputs=warmup_inputs)
    _cp.cuda.Device().synchronize()

    results = []
    for pos in positions:
        cur_pos = prefill_seq_len + pos - 1

        decode_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
        decode_mask[0, 0, 0, : cur_pos + 1] = 0.0

        step_inputs: dict = {
            "input_ids": np.array([[1]], dtype=np.int64),
            "attention_mask": decode_mask,
            "position_ids": np.array([[cur_pos]], dtype=np.int64),
            "cache_position": np.array([cur_pos], dtype=np.int64),
        }
        # Pass KV cache as GPU DeviceBuffers (zero-copy, no PCIe transfer)
        step_inputs.update(kv_gpu_buffers)

        times_ms = []
        for _ in range(repeats):
            _cp.cuda.Device().synchronize()  # drain prior work
            t0 = time.perf_counter()
            decode_dag.execute(inputs=step_inputs)
            times_ms.append((time.perf_counter() - t0) * 1000)

        median_ms = float(np.median(times_ms))
        std_ms = float(np.std(times_ms))
        tps = 1000.0 / median_ms if median_ms > 0 else 0

        pt = ScalingPoint(
            position=pos,
            step_time_ms=median_ms,
            step_time_std_ms=std_ms,
            tps=tps,
            ttft_ms=ttft_ms,
        )
        results.append(pt)
        print(f"    pos={pos:>6}: {median_ms:.1f}ms (TPS={tps:.1f})")

    return results, ttft_ms


def _build_prompt_tokens(tokenizer, prompt_length: int) -> np.ndarray:
    """Generate token IDs of approximately the target length."""
    base = "The quick brown fox jumps over the lazy dog. "
    text = base * (prompt_length // 8 + 1)
    ids = tokenizer.encode(text, return_tensors="np")[0]
    if len(ids) >= prompt_length:
        return ids[:prompt_length]
    while len(ids) < prompt_length:
        ids = np.concatenate([ids, ids])
    return ids[:prompt_length]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-1.5B decode scaling benchmark: Executor vs DAGExecutor",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Measurements per position (default: 3)")
    parser.add_argument(
        "--mode",
        choices=["executor", "dag", "both"],
        default="both",
        help="Which executor(s) to benchmark (default: both)",
    )
    parser.add_argument("--chart", type=str, default=None, help="Save chart to file (PNG)")
    parser.add_argument(
        "--comparison-chart", type=str, default=None, help="Save Executor vs DAG comparison chart (PNG)"
    )
    parser.add_argument(
        "--chart-dir",
        type=str,
        default=None,
        help="Directory to auto-save benchmark_chart.png + benchmark_comparison.png",
    )
    parser.add_argument("--json", type=str, default=None, help="Save results as JSON")
    parser.add_argument("--model-id", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--prefill-ir", default="qwen2_prefill_ir.json")
    parser.add_argument("--decode-ir", default="qwen2_decode_ir.json")
    parser.add_argument("--kv-mapping", default="qwen2_kv_mapping.json")
    parser.add_argument(
        "--prompt-length", type=int, default=PROMPT_LENGTH, help="Prompt length in tokens (default: 64)"
    )
    parser.add_argument(
        "--max-cache-len",
        type=int,
        default=MODEL_MAX_POSITION_EMBEDDINGS,
        help=f"Max KV cache length (default: {MODEL_MAX_POSITION_EMBEDDINGS}, model max_position_embeddings)",
    )
    parser.add_argument(
        "--backend",
        choices=["metal", "cuda", "both"],
        default="metal",
        help="Backend to use for DAG benchmark: metal (default), cuda, or both (comparison)",
    )
    args = parser.parse_args()

    run_executor = args.mode in ("executor", "both")
    run_dag = args.mode in ("dag", "both")

    # CUDA-only mode: skip Metal executor benchmark
    if args.backend == "cuda":
        run_executor = False
        run_dag = True
    report = ScalingReport()
    report.prompt_length = args.prompt_length

    # 1. Environment
    print("Collecting environment info...")
    report.environment = get_environment_info()

    # 2. Tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    input_ids = _build_prompt_tokens(tokenizer, args.prompt_length)

    # 3. Ensure IR files exist with sufficient max_cache_len
    print("Checking IR files...")
    _extract_ir_if_needed(
        args.prefill_ir,
        args.decode_ir,
        args.kv_mapping,
        needed_max_cache_len=args.max_cache_len,
    )

    # 4. Compile
    print("Compiling IR...")
    import npu_compiler

    needs_metal = args.backend in ("metal", "both") or run_executor

    if needs_metal:
        from npu_runtime.device import Device
        from npu_runtime.executor import Executor

    t0 = time.perf_counter()
    prefill_program = npu_compiler.compile(args.prefill_ir)
    decode_program = npu_compiler.compile(args.decode_ir)
    report.compile_time_sec = time.perf_counter() - t0
    print(f"  Compile time: {report.compile_time_sec:.2f}s")

    # Determine max_cache_len from decode IR
    with open(args.kv_mapping) as f:
        kv_map = json.load(f)

    first_kv_name = kv_map["layers"][0]["decode_key_input"]
    max_cache_len = next(s.shape[2] for s in decode_program.input_specs if s.name == first_kv_name)
    prefill_seq_len = prefill_program.input_specs[0].shape[1]
    report.max_cache_len = max_cache_len

    print(f"  max_cache_len={max_cache_len}, prefill_seq_len={prefill_seq_len}")

    # Generate log-scale positions
    max_decode_steps = max_cache_len - prefill_seq_len
    positions = _log_scale_steps(max_decode_steps)
    print(f"  Test positions (log-scale): {positions}")

    # 5. Load weights
    print("Loading weights...")

    if needs_metal:
        from hf_utils import load_weights_from_hf

        device = Device()

        t0 = time.perf_counter()
        prefill_weights = load_weights_from_hf(args.model_id, prefill_program, device)
        decode_weights = load_weights_from_hf(args.model_id, decode_program, device)
    else:
        device = None
        t0 = time.perf_counter()
        prefill_weights = None
        decode_weights = None
    report.weight_load_time_sec = time.perf_counter() - t0
    print(f"  Weight load time: {report.weight_load_time_sec:.2f}s")

    # 6. Executor benchmark
    if run_executor:
        print(f"\n{'=' * 60}")
        print("Benchmarking Executor (Direct Metal NPU)")
        print(f"{'=' * 60}")

        prefill_executor = Executor(prefill_program, device)
        decode_executor = Executor(decode_program, device)

        report.executor_results, _ = benchmark_executor_scaling(
            positions=positions,
            prefill_executor=prefill_executor,
            decode_executor=decode_executor,
            prefill_program=prefill_program,
            decode_program=decode_program,
            prefill_weights=prefill_weights,
            decode_weights=decode_weights,
            kv_map=kv_map,
            device=device,
            input_ids=input_ids,
            repeats=args.repeats,
        )

    # 7. DAGExecutor benchmark (Metal)
    if run_dag and args.backend in ("metal", "both"):
        print(f"\n{'=' * 60}")
        print("Benchmarking DAGExecutor (Metal NPU+CPU)")
        print(f"{'=' * 60}")

        report.dag_results, _ = benchmark_dag_scaling(
            positions=positions,
            prefill_ir_path=args.prefill_ir,
            decode_ir_path=args.decode_ir,
            kv_mapping_path=args.kv_mapping,
            model_id=args.model_id,
            input_ids=input_ids,
            repeats=args.repeats,
        )

    # 7b. DAGExecutor benchmark (CUDA)
    if run_dag and args.backend in ("cuda", "both"):
        try:
            import cupy  # noqa: F401

            print(f"\n{'=' * 60}")
            print("Benchmarking DAGExecutor (CUDA GPU)")
            print(f"{'=' * 60}")

            cuda_dag_results, _ = benchmark_dag_cuda_scaling(
                positions=positions,
                prefill_ir_path=args.prefill_ir,
                decode_ir_path=args.decode_ir,
                kv_mapping_path=args.kv_mapping,
                model_id=args.model_id,
                input_ids=input_ids,
                repeats=args.repeats,
            )

            if args.backend == "cuda":
                # Use CUDA results as the primary DAG results
                report.dag_results = cuda_dag_results
            else:
                # Both: print CUDA results separately
                print("\nCUDA DAG Results:")
                for r in cuda_dag_results:
                    print(f"  pos={r.position:>6}: {r.step_time_ms:.1f}ms (TPS={r.tps:.1f})")
        except ImportError:
            print("  [WARN] CuPy not installed, skipping CUDA benchmark")

    # 8. Report
    print_report(report)

    # Determine chart paths
    chart_path = args.chart
    comparison_chart_path = args.comparison_chart
    if args.chart_dir:
        os.makedirs(args.chart_dir, exist_ok=True)
        if not chart_path:
            chart_path = os.path.join(args.chart_dir, "benchmark_chart.png")
        if not comparison_chart_path:
            comparison_chart_path = os.path.join(args.chart_dir, "benchmark_comparison.png")

    if chart_path:
        os.makedirs(os.path.dirname(os.path.abspath(chart_path)), exist_ok=True)
        save_chart(report, chart_path)
    if comparison_chart_path and report.executor_results and report.dag_results:
        os.makedirs(os.path.dirname(os.path.abspath(comparison_chart_path)), exist_ok=True)
        save_comparison_chart(report, comparison_chart_path)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        save_json(report, args.json)


if __name__ == "__main__":
    main()
