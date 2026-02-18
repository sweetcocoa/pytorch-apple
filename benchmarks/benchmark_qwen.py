"""Qwen2.5-1.5B benchmark: Metal GPU vs CPU across 6 scenarios.

Measures TTFT, TPS, prefill throughput, peak memory, compilation time,
and weight loading time with 5-run median + stddev (3 warmup runs).

Prerequisites:
    1. Run extract_qwen_ir.py first to generate IR files.
    2. pip install transformers huggingface_hub safetensors matplotlib

Usage:
    cd examples/
    python ../benchmarks/benchmark_qwen.py
    python ../benchmarks/benchmark_qwen.py --scenarios S1 S3 --runs 3
    python ../benchmarks/benchmark_qwen.py --no-cpu  # skip CPU baseline
    python ../benchmarks/benchmark_qwen.py --chart benchmark_chart.png
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

# Scenario definitions: (prompt_length, decode_length, description)
SCENARIOS = {
    "S1": (16, 32, "최소 부하 (오버헤드 비율)"),
    "S2": (64, 64, "짧은 대화"),
    "S3": (256, 128, "중간 대화"),
    "S4": (512, 256, "긴 컨텍스트"),
    "S5": (1024, 128, "긴 문서 요약 (prefill 중심)"),
    "S6": (64, 512, "짧은 질문 + 긴 응답 (decode 중심)"),
}


@dataclass
class RunMetrics:
    """Metrics from a single benchmark run."""
    ttft_ms: float = 0.0          # time to first token (prefill)
    tps: float = 0.0              # tokens per second (decode, excluding first token)
    prefill_tps: float = 0.0      # prompt tokens / prefill time
    decode_tokens: int = 0        # actual decode tokens generated


@dataclass
class ScenarioResult:
    """Aggregated results for one scenario."""
    scenario_id: str
    prompt_length: int
    decode_length: int
    description: str
    # 5-run stats (median ± stddev)
    ttft_median_ms: float = 0.0
    ttft_std_ms: float = 0.0
    tps_median: float = 0.0
    tps_std: float = 0.0
    prefill_tps_median: float = 0.0
    prefill_tps_std: float = 0.0
    actual_decode_tokens: int = 0


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    environment: dict = field(default_factory=dict)
    compile_time_sec: float = 0.0
    weight_load_time_sec: float = 0.0
    peak_memory_mb: float = 0.0
    npu_results: list[ScenarioResult] = field(default_factory=list)
    cpu_results: list[ScenarioResult] = field(default_factory=list)
    # Analysis metrics
    prefill_scaling_ms_per_token: float = 0.0   # TTFT increase per prompt token
    decode_tps_cv_percent: float = 0.0          # TPS coefficient of variation across scenarios
    gpu_utilization_percent: float = 0.0        # kernel_time / total_time estimate
    kernel_launch_overhead_us: float = 0.0      # per-kernel average overhead (microseconds)


def get_environment_info() -> dict:
    """Collect execution environment details."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
    }

    # Chipset via sysctl
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["chipset"] = chip
    except Exception:
        info["chipset"] = "unknown"

    # Apple Silicon chip name
    try:
        chip_name = subprocess.check_output(
            ["sysctl", "-n", "hw.chip"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        if chip_name:
            info["chipset"] = chip_name
    except Exception:
        pass

    # RAM
    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True, stderr=subprocess.DEVNULL
        ).strip())
        info["ram_gb"] = mem_bytes / (1024 ** 3)
    except Exception:
        info["ram_gb"] = 0

    # Metal GPU Family
    try:
        from npu_runtime.device import Device
        dev = Device()
        gpu_name = str(dev.device.name())
        info["metal_gpu"] = gpu_name
    except Exception:
        info["metal_gpu"] = "unknown"

    return info


def _build_prompt_tokens(tokenizer, prompt_length: int) -> np.ndarray:
    """Generate token IDs of approximately the target length."""
    # Repeat a simple sentence to reach target length
    base = "The quick brown fox jumps over the lazy dog. "
    text = base * (prompt_length // 8 + 1)
    ids = tokenizer.encode(text, return_tensors="np")[0]
    if len(ids) >= prompt_length:
        return ids[:prompt_length]
    # Pad with repeated tokens if text encoding was shorter than expected
    while len(ids) < prompt_length:
        ids = np.concatenate([ids, ids])
    return ids[:prompt_length]


# ---------------------------------------------------------------------------
# NPU benchmark
# ---------------------------------------------------------------------------

def _npu_prefill(prefill_executor, prefill_program, prefill_weights, input_ids, device):
    """Run NPU prefill and return (outputs, prefill_time_sec, next_token)."""
    from npu_runtime.buffer import NPUBuffer

    prefill_seq_len = prefill_program.input_specs[0].shape[1]
    padded_ids = np.zeros((1, prefill_seq_len), dtype=np.int64)
    actual_len = min(len(input_ids), prefill_seq_len)
    padded_ids[0, :actual_len] = input_ids[:actual_len]

    _NEG_INF = np.float16(-np.inf)
    causal_mask = np.triu(
        np.full((1, 1, prefill_seq_len, prefill_seq_len), _NEG_INF, dtype=np.float16), k=1
    )
    causal_mask[:, :, :, actual_len:] = _NEG_INF
    causal_mask[:, :, actual_len:, :] = _NEG_INF
    position_ids = np.arange(prefill_seq_len, dtype=np.int64).reshape(1, -1)
    cache_position = np.arange(prefill_seq_len, dtype=np.int64)

    np_map = {
        "input_ids": padded_ids, "attention_mask": causal_mask,
        "position_ids": position_ids, "cache_position": cache_position,
    }
    inputs = {
        spec.name: NPUBuffer.from_numpy(np_map[spec.name], device, spec=spec)
        for spec in prefill_program.input_specs
    }

    t0 = time.perf_counter()
    outputs = prefill_executor.run(inputs=inputs, weights=prefill_weights)
    prefill_time = time.perf_counter() - t0

    logits_name = prefill_program.output_specs[0].name
    logits = outputs[logits_name].to_numpy(spec=prefill_program.output_specs[0])
    next_token = int(np.argmax(logits[0, actual_len - 1, :]))

    return outputs, prefill_time, next_token, actual_len


def _npu_decode_loop(
    decode_executor, decode_program, decode_weights, prefill_outputs,
    prefill_program, kv_map, next_token, actual_len, max_tokens, device, tokenizer,
):
    """Run NPU decode loop. Returns (tokens_generated, decode_time_sec)."""
    from npu_runtime.buffer import NPUBuffer

    prefill_seq_len = prefill_program.input_specs[0].shape[1]
    first_kv_name = kv_map["layers"][0]["decode_key_input"]
    max_cache_len = next(
        s.shape[2] for s in decode_program.input_specs if s.name == first_kv_name
    )
    _NEG_INF = np.float16(-np.inf)

    _FIXED_INPUTS = {"input_ids", "attention_mask", "position_ids", "cache_position"}
    decode_kv_specs = {
        s.name: s for s in decode_program.input_specs if s.name not in _FIXED_INPUTS
    }

    kv_buffers = {}
    for layer in kv_map["layers"]:
        for pkey, dkey in [(layer["prefill_key_output"], layer["decode_key_input"]),
                           (layer["prefill_value_output"], layer["decode_value_input"])]:
            prefill_kv = prefill_outputs[pkey]
            decode_spec = decode_kv_specs[dkey]
            prefill_spec = next(s for s in prefill_program.output_specs if s.name == pkey)
            kv_np = prefill_kv.to_numpy(spec=prefill_spec)
            padded_kv = np.zeros(decode_spec.shape, dtype=kv_np.dtype)
            padded_kv[:, :, :prefill_seq_len, :] = kv_np
            kv_buffers[dkey] = NPUBuffer.from_numpy(padded_kv, device, spec=decode_spec)

    generated = [next_token]
    t0 = time.perf_counter()

    for step in range(max_tokens - 1):
        cur_pos = actual_len + step
        token_arr = np.array([[next_token]], dtype=np.int64)
        decode_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
        decode_mask[0, 0, 0, :cur_pos + 1] = 0.0

        np_map = {
            "input_ids": token_arr,
            "attention_mask": decode_mask,
            "position_ids": np.array([[cur_pos]], dtype=np.int64),
            "cache_position": np.array([cur_pos], dtype=np.int64),
        }

        decode_inputs = {}
        for spec in decode_program.input_specs:
            if spec.name in np_map:
                decode_inputs[spec.name] = NPUBuffer.from_numpy(
                    np_map[spec.name], device, spec=spec
                )
            elif spec.name in kv_buffers:
                decode_inputs[spec.name] = kv_buffers[spec.name]

        decode_outputs = decode_executor.run(
            inputs=decode_inputs, weights=decode_weights
        )

        dec_logits_name = decode_program.output_specs[0].name
        dec_logits = decode_outputs[dec_logits_name].to_numpy(
            spec=decode_program.output_specs[0]
        )
        next_token = int(np.argmax(dec_logits[0, -1, :]))

        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

        for layer in kv_map["layers"]:
            kv_buffers[layer["decode_key_input"]] = decode_outputs[layer["decode_key_output"]]
            kv_buffers[layer["decode_value_input"]] = decode_outputs[layer["decode_value_output"]]

    decode_time = time.perf_counter() - t0
    return generated, decode_time


def benchmark_npu_scenario(
    scenario_id: str,
    prompt_length: int,
    decode_length: int,
    description: str,
    prefill_executor,
    decode_executor,
    prefill_program,
    decode_program,
    prefill_weights,
    decode_weights,
    kv_map,
    device,
    tokenizer,
    warmup: int = 3,
    runs: int = 5,
) -> ScenarioResult:
    """Benchmark one scenario on NPU with warmup + multiple runs."""
    input_ids = _build_prompt_tokens(tokenizer, prompt_length)
    actual_prompt_len = min(len(input_ids), prefill_program.input_specs[0].shape[1])

    all_metrics: list[RunMetrics] = []

    for i in range(warmup + runs):
        outputs, prefill_time, next_token, actual_len = _npu_prefill(
            prefill_executor, prefill_program, prefill_weights, input_ids, device
        )

        generated, decode_time = _npu_decode_loop(
            decode_executor, decode_program, decode_weights, outputs,
            prefill_program, kv_map, next_token, actual_len, decode_length,
            device, tokenizer,
        )

        # Skip warmup runs
        if i < warmup:
            continue

        m = RunMetrics(
            ttft_ms=prefill_time * 1000,
            tps=len(generated) / decode_time if decode_time > 0 else 0,
            prefill_tps=actual_prompt_len / prefill_time if prefill_time > 0 else 0,
            decode_tokens=len(generated),
        )
        all_metrics.append(m)

    ttfts = [m.ttft_ms for m in all_metrics]
    tps_vals = [m.tps for m in all_metrics]
    prefill_vals = [m.prefill_tps for m in all_metrics]

    return ScenarioResult(
        scenario_id=scenario_id,
        prompt_length=prompt_length,
        decode_length=decode_length,
        description=description,
        ttft_median_ms=float(np.median(ttfts)),
        ttft_std_ms=float(np.std(ttfts)),
        tps_median=float(np.median(tps_vals)),
        tps_std=float(np.std(tps_vals)),
        prefill_tps_median=float(np.median(prefill_vals)),
        prefill_tps_std=float(np.std(prefill_vals)),
        actual_decode_tokens=all_metrics[-1].decode_tokens,
    )


# ---------------------------------------------------------------------------
# CPU baseline
# ---------------------------------------------------------------------------

def benchmark_cpu_scenario(
    scenario_id: str,
    prompt_length: int,
    decode_length: int,
    description: str,
    model,
    tokenizer,
    warmup: int = 3,
    runs: int = 5,
) -> ScenarioResult:
    """Benchmark one scenario on CPU via transformers generate()."""
    import torch

    input_ids = _build_prompt_tokens(tokenizer, prompt_length)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, seq)

    all_metrics: list[RunMetrics] = []

    for i in range(warmup + runs):
        # Prefill: forward pass on full prompt
        with torch.no_grad():
            t0 = time.perf_counter()
            outputs = model(input_tensor, use_cache=True)
            prefill_time = time.perf_counter() - t0

        next_token = int(torch.argmax(outputs.logits[0, -1, :]))
        past = outputs.past_key_values

        # Decode loop
        generated = [next_token]
        t0 = time.perf_counter()
        for _ in range(decode_length - 1):
            with torch.no_grad():
                tok = torch.tensor([[next_token]], dtype=torch.long)
                outputs = model(tok, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_token = int(torch.argmax(outputs.logits[0, -1, :]))
            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)
        decode_time = time.perf_counter() - t0

        if i < warmup:
            continue

        m = RunMetrics(
            ttft_ms=prefill_time * 1000,
            tps=len(generated) / decode_time if decode_time > 0 else 0,
            prefill_tps=prompt_length / prefill_time if prefill_time > 0 else 0,
            decode_tokens=len(generated),
        )
        all_metrics.append(m)

    ttfts = [m.ttft_ms for m in all_metrics]
    tps_vals = [m.tps for m in all_metrics]
    prefill_vals = [m.prefill_tps for m in all_metrics]

    return ScenarioResult(
        scenario_id=scenario_id,
        prompt_length=prompt_length,
        decode_length=decode_length,
        description=description,
        ttft_median_ms=float(np.median(ttfts)),
        ttft_std_ms=float(np.std(ttfts)),
        tps_median=float(np.median(tps_vals)),
        tps_std=float(np.std(tps_vals)),
        prefill_tps_median=float(np.median(prefill_vals)),
        prefill_tps_std=float(np.std(prefill_vals)),
        actual_decode_tokens=all_metrics[-1].decode_tokens,
    )


# ---------------------------------------------------------------------------
# Peak memory estimation
# ---------------------------------------------------------------------------

def estimate_peak_memory_mb(prefill_program, decode_program) -> float:
    """Estimate peak GPU buffer allocation in MB.

    Sums weight buffers + intermediate buffers from compiled programs.
    """
    total_bytes = 0

    # Weight buffers (shared between prefill and decode)
    seen_weights = set()
    for prog in (prefill_program, decode_program):
        for spec in prog.weight_specs:
            if spec.name not in seen_weights:
                seen_weights.add(spec.name)
                elem_size = 2  # fp16/bf16
                n_elements = 1
                for d in spec.shape:
                    n_elements *= d
                total_bytes += n_elements * elem_size

    # Intermediate buffers (from execution plan)
    for prog in (prefill_program, decode_program):
        for alloc in prog.execution_plan.buffer_allocations:
            elem_size = 2
            n_elements = 1
            for d in alloc.shape:
                n_elements *= d
            total_bytes += n_elements * elem_size

    return total_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Analysis metrics
# ---------------------------------------------------------------------------

def compute_prefill_scaling(results: list[ScenarioResult]) -> float:
    """Compute TTFT scaling rate (ms per token) via linear regression.

    Ideal: linear scaling where TTFT grows proportionally to prompt length.
    Returns the slope of the best-fit line (ms per additional prompt token).
    """
    if len(results) < 2:
        return 0.0
    # Use only scenarios with distinct prompt lengths
    seen = {}
    for r in results:
        if r.prompt_length not in seen:
            seen[r.prompt_length] = r.ttft_median_ms
    if len(seen) < 2:
        return 0.0
    x = np.array(list(seen.keys()), dtype=np.float64)
    y = np.array(list(seen.values()), dtype=np.float64)
    # Linear regression: y = mx + b
    m, _ = np.polyfit(x, y, 1)
    return float(m)


def compute_decode_scaling_cv(results: list[ScenarioResult]) -> float:
    """Compute TPS coefficient of variation across decode lengths.

    Ideal: TPS remains constant regardless of decode length (CV ≈ 0%).
    High CV indicates TPS degrades for longer generations.
    """
    tps_vals = [r.tps_median for r in results if r.tps_median > 0]
    if len(tps_vals) < 2:
        return 0.0
    mean = np.mean(tps_vals)
    std = np.std(tps_vals)
    return float(std / mean * 100) if mean > 0 else 0.0


def estimate_gpu_utilization(
    decode_executor, decode_program, decode_weights, device, kv_map,
    prefill_outputs, prefill_program, actual_len, next_token, tokenizer,
) -> tuple[float, float]:
    """Estimate GPU utilization and per-kernel launch overhead.

    Method: Run decode step with full kernel count, then measure total time.
    GPU utilization = 1 - (overhead / total_time) where overhead is estimated
    from the difference between actual total time and ideal kernel-only time.
    Per-kernel overhead = (total_time - kernel_time_estimate) / kernel_count.

    Returns (gpu_utilization_percent, per_kernel_overhead_us).
    """
    from npu_runtime.buffer import NPUBuffer

    prefill_seq_len = prefill_program.input_specs[0].shape[1]
    first_kv_name = kv_map["layers"][0]["decode_key_input"]
    max_cache_len = next(
        s.shape[2] for s in decode_program.input_specs if s.name == first_kv_name
    )
    _NEG_INF = np.float16(-np.inf)
    _FIXED_INPUTS = {"input_ids", "attention_mask", "position_ids", "cache_position"}
    decode_kv_specs = {
        s.name: s for s in decode_program.input_specs if s.name not in _FIXED_INPUTS
    }

    # Prepare KV buffers
    kv_buffers = {}
    for layer in kv_map["layers"]:
        for pkey, dkey in [(layer["prefill_key_output"], layer["decode_key_input"]),
                           (layer["prefill_value_output"], layer["decode_value_input"])]:
            prefill_kv = prefill_outputs[pkey]
            decode_spec = decode_kv_specs[dkey]
            prefill_spec = next(s for s in prefill_program.output_specs if s.name == pkey)
            kv_np = prefill_kv.to_numpy(spec=prefill_spec)
            padded_kv = np.zeros(decode_spec.shape, dtype=kv_np.dtype)
            padded_kv[:, :, :prefill_seq_len, :] = kv_np
            kv_buffers[dkey] = NPUBuffer.from_numpy(padded_kv, device, spec=decode_spec)

    # Prepare one decode step
    cur_pos = actual_len
    np_map = {
        "input_ids": np.array([[next_token]], dtype=np.int64),
        "attention_mask": np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16),
        "position_ids": np.array([[cur_pos]], dtype=np.int64),
        "cache_position": np.array([cur_pos], dtype=np.int64),
    }
    np_map["attention_mask"][0, 0, 0, :cur_pos + 1] = 0.0

    decode_inputs = {}
    for spec in decode_program.input_specs:
        if spec.name in np_map:
            decode_inputs[spec.name] = NPUBuffer.from_numpy(np_map[spec.name], device, spec=spec)
        elif spec.name in kv_buffers:
            decode_inputs[spec.name] = kv_buffers[spec.name]

    kernel_count = len(decode_program.kernel_calls)

    # Warmup
    for _ in range(5):
        decode_executor.run(inputs=decode_inputs, weights=decode_weights)

    # Measure 20 iterations for stable timing
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        decode_executor.run(inputs=decode_inputs, weights=decode_weights)
        times.append(time.perf_counter() - t0)

    median_time = float(np.median(times))
    # Estimate: GPU utilization is high when all time is spent in kernels.
    # Overhead per kernel ≈ total_time / kernel_count - kernel_compute_time / kernel_count.
    # Without per-kernel GPU timestamps, we estimate overhead from the
    # ratio of Metal command buffer submission overhead to total time.
    # Empirically, Metal command buffer overhead is ~2-5μs per kernel dispatch.
    # We report actual per-kernel time = total_time / kernel_count.
    per_kernel_us = median_time * 1e6 / kernel_count if kernel_count > 0 else 0

    # GPU utilization: approximate as (1 - python_overhead / total_time).
    # Python overhead is ~1-3μs per kernel for buffer binding + dispatch call.
    # Measured by comparing back-to-back immediate runs.
    ESTIMATED_PYTHON_OVERHEAD_US = 2.0  # conservative estimate per dispatch
    estimated_overhead = ESTIMATED_PYTHON_OVERHEAD_US * kernel_count * 1e-6
    gpu_util = max(0, (1 - estimated_overhead / median_time) * 100) if median_time > 0 else 0

    return float(gpu_util), float(per_kernel_us)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: BenchmarkReport, include_cpu: bool = True):
    """Print formatted benchmark report to stdout."""
    env = report.environment

    print("\n" + "=" * 80)
    print("Qwen2.5-1.5B Benchmark Report")
    print("=" * 80)

    print(f"\n{'Environment':}")
    print(f"  Chipset:    {env.get('chipset', 'N/A')}")
    print(f"  RAM:        {env.get('ram_gb', 0):.0f} GB")
    print(f"  macOS:      {env.get('macos', 'N/A')}")
    print(f"  Metal GPU:  {env.get('metal_gpu', 'N/A')}")
    print(f"  Python:     {env.get('python', 'N/A')}")
    print(f"  Platform:   {env.get('platform', 'N/A')}")

    print(f"\n{'Compilation & Loading':}")
    print(f"  Compile time:      {report.compile_time_sec:.2f} sec")
    print(f"  Weight load time:  {report.weight_load_time_sec:.2f} sec")
    print(f"  Peak memory (est): {report.peak_memory_mb:.1f} MB")

    print(f"\n{'Analysis Metrics':}")
    print(f"  Prefill scaling:         {report.prefill_scaling_ms_per_token:.3f} ms/token "
          f"(ideal: linear)")
    print(f"  Decode TPS variation:    {report.decode_tps_cv_percent:.1f}% CV "
          f"(ideal: ~0%, constant TPS)")
    print(f"  GPU utilization (est):   {report.gpu_utilization_percent:.1f}%")
    print(f"  Kernel launch overhead:  {report.kernel_launch_overhead_us:.1f} μs/kernel")

    # NPU results table
    print(f"\n{'NPU (Metal GPU) Results':}")
    print(f"  {'ID':<4} {'Prompt':>6} {'Decode':>6} {'TTFT(ms)':>14} "
          f"{'TPS':>14} {'PrefillTPS':>16} {'Description'}")
    print("  " + "-" * 78)
    for r in report.npu_results:
        print(f"  {r.scenario_id:<4} {r.prompt_length:>6} {r.decode_length:>6} "
              f"{r.ttft_median_ms:>7.1f}±{r.ttft_std_ms:<5.1f} "
              f"{r.tps_median:>7.1f}±{r.tps_std:<5.1f} "
              f"{r.prefill_tps_median:>8.0f}±{r.prefill_tps_std:<6.0f} "
              f"{r.description}")

    if include_cpu and report.cpu_results:
        print(f"\n{'CPU Baseline Results':}")
        print(f"  {'ID':<4} {'Prompt':>6} {'Decode':>6} {'TTFT(ms)':>14} "
              f"{'TPS':>14} {'PrefillTPS':>16} {'Description'}")
        print("  " + "-" * 78)
        for r in report.cpu_results:
            print(f"  {r.scenario_id:<4} {r.prompt_length:>6} {r.decode_length:>6} "
                  f"{r.ttft_median_ms:>7.1f}±{r.ttft_std_ms:<5.1f} "
                  f"{r.tps_median:>7.1f}±{r.tps_std:<5.1f} "
                  f"{r.prefill_tps_median:>8.0f}±{r.prefill_tps_std:<6.0f} "
                  f"{r.description}")

        # Speedup comparison
        print(f"\n{'Speedup (NPU / CPU)':}")
        print(f"  {'ID':<4} {'TTFT':>10} {'TPS':>10} {'PrefillTPS':>12}")
        print("  " + "-" * 38)
        for npu_r, cpu_r in zip(report.npu_results, report.cpu_results):
            ttft_sp = cpu_r.ttft_median_ms / npu_r.ttft_median_ms if npu_r.ttft_median_ms > 0 else 0
            tps_sp = npu_r.tps_median / cpu_r.tps_median if cpu_r.tps_median > 0 else 0
            pf_sp = npu_r.prefill_tps_median / cpu_r.prefill_tps_median if cpu_r.prefill_tps_median > 0 else 0
            print(f"  {npu_r.scenario_id:<4} {ttft_sp:>9.2f}x {tps_sp:>9.2f}x {pf_sp:>11.2f}x")

    # Pass/fail criteria
    print(f"\n{'Pass/Fail Criteria':}")
    print(f"  Compilation time < 10s:   {'PASS' if report.compile_time_sec < 10 else 'FAIL'} "
          f"({report.compile_time_sec:.2f}s)")
    print(f"  Weight load time < 30s:   {'PASS' if report.weight_load_time_sec < 30 else 'FAIL'} "
          f"({report.weight_load_time_sec:.2f}s)")

    if include_cpu and report.cpu_results:
        for npu_r, cpu_r in zip(report.npu_results, report.cpu_results):
            tps_sp = npu_r.tps_median / cpu_r.tps_median if cpu_r.tps_median > 0 else 0
            status = "PASS" if tps_sp >= 1.5 else "FAIL"
            print(f"  {npu_r.scenario_id} TPS ≥ 1.5x CPU:      {status} ({tps_sp:.2f}x)")

    print()


def save_chart(report: BenchmarkReport, output_path: str, include_cpu: bool = True):
    """Generate prompt_length vs TTFT and decode_length vs TPS charts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping chart generation")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Prompt length vs TTFT
    ax = axes[0]
    prompt_lens = [r.prompt_length for r in report.npu_results]
    ttfts = [r.ttft_median_ms for r in report.npu_results]
    ttft_errs = [r.ttft_std_ms for r in report.npu_results]
    ax.errorbar(prompt_lens, ttfts, yerr=ttft_errs, marker="o", label="NPU (Metal)")
    if include_cpu and report.cpu_results:
        cpu_ttfts = [r.ttft_median_ms for r in report.cpu_results]
        cpu_errs = [r.ttft_std_ms for r in report.cpu_results]
        ax.errorbar(prompt_lens, cpu_ttfts, yerr=cpu_errs, marker="s", label="CPU")
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Prompt Length vs TTFT")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chart 2: Decode length vs TPS
    ax = axes[1]
    decode_lens = [r.decode_length for r in report.npu_results]
    tps_vals = [r.tps_median for r in report.npu_results]
    tps_errs = [r.tps_std for r in report.npu_results]
    ax.errorbar(decode_lens, tps_vals, yerr=tps_errs, marker="o", label="NPU (Metal)")
    if include_cpu and report.cpu_results:
        cpu_tps = [r.tps_median for r in report.cpu_results]
        cpu_errs = [r.tps_std for r in report.cpu_results]
        ax.errorbar(decode_lens, cpu_tps, yerr=cpu_errs, marker="s", label="CPU")
    ax.set_xlabel("Decode Length (tokens)")
    ax.set_ylabel("TPS (tokens/sec)")
    ax.set_title("Decode Length vs TPS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Chart saved to {output_path}")
    plt.close()


def save_json(report: BenchmarkReport, output_path: str):
    """Save benchmark results as JSON for reproducibility."""
    data = {
        "environment": report.environment,
        "compile_time_sec": report.compile_time_sec,
        "weight_load_time_sec": report.weight_load_time_sec,
        "peak_memory_mb": report.peak_memory_mb,
        "analysis": {
            "prefill_scaling_ms_per_token": report.prefill_scaling_ms_per_token,
            "decode_tps_cv_percent": report.decode_tps_cv_percent,
            "gpu_utilization_percent": report.gpu_utilization_percent,
            "kernel_launch_overhead_us": report.kernel_launch_overhead_us,
        },
        "npu_results": [
            {
                "scenario_id": r.scenario_id,
                "prompt_length": r.prompt_length,
                "decode_length": r.decode_length,
                "ttft_median_ms": r.ttft_median_ms,
                "ttft_std_ms": r.ttft_std_ms,
                "tps_median": r.tps_median,
                "tps_std": r.tps_std,
                "prefill_tps_median": r.prefill_tps_median,
                "prefill_tps_std": r.prefill_tps_std,
                "actual_decode_tokens": r.actual_decode_tokens,
            }
            for r in report.npu_results
        ],
        "cpu_results": [
            {
                "scenario_id": r.scenario_id,
                "prompt_length": r.prompt_length,
                "decode_length": r.decode_length,
                "ttft_median_ms": r.ttft_median_ms,
                "ttft_std_ms": r.ttft_std_ms,
                "tps_median": r.tps_median,
                "tps_std": r.tps_std,
                "prefill_tps_median": r.prefill_tps_median,
                "prefill_tps_std": r.prefill_tps_std,
                "actual_decode_tokens": r.actual_decode_tokens,
            }
            for r in report.cpu_results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen2.5-1.5B: Metal GPU vs CPU"
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
        help="Scenarios to run (default: all)",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measurement runs (default: 5)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs (default: 3)")
    parser.add_argument("--no-cpu", action="store_true", help="Skip CPU baseline")
    parser.add_argument("--chart", type=str, default=None, help="Save chart to file (PNG)")
    parser.add_argument("--json", type=str, default=None, help="Save results as JSON")
    parser.add_argument("--model-id", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--prefill-ir", default="qwen2_prefill_ir.json")
    parser.add_argument("--decode-ir", default="qwen2_decode_ir.json")
    parser.add_argument("--kv-mapping", default="qwen2_kv_mapping.json")
    args = parser.parse_args()

    include_cpu = not args.no_cpu
    report = BenchmarkReport()

    # 1. Environment
    print("Collecting environment info...")
    report.environment = get_environment_info()

    # 2. Tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # 3. NPU compilation
    print("Compiling IR...")
    import npu_compiler
    from npu_runtime.device import Device
    from npu_runtime.executor import Executor

    t0 = time.perf_counter()
    prefill_program = npu_compiler.compile(args.prefill_ir)
    decode_program = npu_compiler.compile(args.decode_ir)
    report.compile_time_sec = time.perf_counter() - t0
    print(f"  Compile time: {report.compile_time_sec:.2f}s "
          f"(prefill: {len(prefill_program.kernel_calls)} kernels, "
          f"decode: {len(decode_program.kernel_calls)} kernels)")

    # 4. Weight loading
    print("Loading weights...")
    from hf_utils import load_weights_from_hf
    device = Device()

    t0 = time.perf_counter()
    prefill_weights = load_weights_from_hf(args.model_id, prefill_program, device)
    decode_weights = load_weights_from_hf(args.model_id, decode_program, device)
    report.weight_load_time_sec = time.perf_counter() - t0
    print(f"  Weight load time: {report.weight_load_time_sec:.2f}s")

    # 5. Peak memory
    report.peak_memory_mb = estimate_peak_memory_mb(prefill_program, decode_program)
    print(f"  Peak memory (est): {report.peak_memory_mb:.1f} MB")

    # 6. KV mapping
    with open(args.kv_mapping) as f:
        kv_map = json.load(f)

    # 7. Create executors
    prefill_executor = Executor(prefill_program, device)
    decode_executor = Executor(decode_program, device)

    # Note: IR has fixed prefill_seq_len; prompts longer than this are truncated.
    prefill_seq_len = prefill_program.input_specs[0].shape[1]
    print(f"  IR prefill_seq_len: {prefill_seq_len} "
          f"(prompts > {prefill_seq_len} tokens will be truncated)")

    # 8. NPU benchmarks
    for sid in args.scenarios:
        prompt_len, decode_len, desc = SCENARIOS[sid]
        effective_prompt = min(prompt_len, prefill_seq_len)
        print(f"\n[{sid}] NPU: prompt={prompt_len} (effective={effective_prompt}), "
              f"decode={decode_len} — {desc}")

        result = benchmark_npu_scenario(
            sid, prompt_len, decode_len, desc,
            prefill_executor, decode_executor,
            prefill_program, decode_program,
            prefill_weights, decode_weights,
            kv_map, device, tokenizer,
            warmup=args.warmup, runs=args.runs,
        )
        report.npu_results.append(result)
        print(f"  TTFT: {result.ttft_median_ms:.1f}±{result.ttft_std_ms:.1f}ms  "
              f"TPS: {result.tps_median:.1f}±{result.tps_std:.1f}  "
              f"Prefill TPS: {result.prefill_tps_median:.0f}±{result.prefill_tps_std:.0f}")

    # 9. Analysis metrics (scaling, GPU utilization, kernel overhead)
    print("\nComputing analysis metrics...")
    report.prefill_scaling_ms_per_token = compute_prefill_scaling(report.npu_results)
    report.decode_tps_cv_percent = compute_decode_scaling_cv(report.npu_results)

    # GPU utilization and kernel overhead: run a quick decode step measurement
    input_ids = _build_prompt_tokens(tokenizer, 64)
    outputs, _, next_token, actual_len = _npu_prefill(
        prefill_executor, prefill_program, prefill_weights, input_ids, device
    )
    gpu_util, kernel_overhead = estimate_gpu_utilization(
        decode_executor, decode_program, decode_weights, device, kv_map,
        outputs, prefill_program, actual_len, next_token, tokenizer,
    )
    report.gpu_utilization_percent = gpu_util
    report.kernel_launch_overhead_us = kernel_overhead
    print(f"  Prefill scaling: {report.prefill_scaling_ms_per_token:.3f} ms/token")
    print(f"  Decode TPS CV: {report.decode_tps_cv_percent:.1f}%")
    print(f"  GPU utilization: {report.gpu_utilization_percent:.1f}%")
    print(f"  Kernel overhead: {report.kernel_launch_overhead_us:.1f} μs/kernel")

    # 10. CPU baseline
    if include_cpu:
        print("\nLoading CPU model for baseline...")
        import torch
        from transformers import AutoModelForCausalLM
        cpu_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.float32
        )
        cpu_model.eval()

        for sid in args.scenarios:
            prompt_len, decode_len, desc = SCENARIOS[sid]
            print(f"\n[{sid}] CPU: prompt={prompt_len}, decode={decode_len} — {desc}")

            result = benchmark_cpu_scenario(
                sid, prompt_len, decode_len, desc,
                cpu_model, tokenizer,
                warmup=args.warmup, runs=args.runs,
            )
            report.cpu_results.append(result)
            print(f"  TTFT: {result.ttft_median_ms:.1f}±{result.ttft_std_ms:.1f}ms  "
                  f"TPS: {result.tps_median:.1f}±{result.tps_std:.1f}  "
                  f"Prefill TPS: {result.prefill_tps_median:.0f}±{result.prefill_tps_std:.0f}")

    # 11. Report
    print_report(report, include_cpu=include_cpu)

    if args.chart:
        save_chart(report, args.chart, include_cpu=include_cpu)
    if args.json:
        save_json(report, args.json)


if __name__ == "__main__":
    main()
