#!/usr/bin/env python3
"""Generate CUDA benchmark charts for documentation.

Creates:
  1. cuda_benchmark_chart.png — CUDA TPS and step time vs cache position
  2. metal_vs_cuda_chart.png — Metal vs CUDA comparison (TPS, latency, speedup)
"""

import matplotlib.pyplot as plt

# ---------- Data ----------

positions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32640]

# Metal results (from docs/benchmarks.md — Apple M4 Pro)
metal_tps = [2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.7, 2.7, 2.7, 2.6, 2.4, 2.1, 1.7, 1.2]
metal_step = [
    359.5, 360.8, 360.9, 361.8, 361.1, 361.3, 361.8, 363.5,
    365.3, 368.7, 374.9, 389.7, 417.1, 475.3, 588.4, 813.6,
]
metal_ttft = 359.5  # ms

# CUDA results (RTX 3090 Ti, after all optimizations)
cuda_tps = [30.6, 31.1, 31.0, 30.2, 30.5, 30.1, 30.5, 30.9, 30.9, 31.9, 30.9, 30.0, 30.3, 31.0, 29.8, 30.7]
cuda_step = [32.7, 32.2, 32.3, 33.1, 32.8, 33.2, 32.7, 32.4, 32.4, 31.3, 32.3, 33.3, 33.0, 32.3, 33.6, 32.5]
cuda_step_std = [1.0, 0.7, 0.6, 0.6, 0.6, 0.6, 0.5, 0.8, 0.6, 0.4, 0.3, 0.9, 0.3, 1.1, 0.4, 0.4]
cuda_ttft = 34.3  # ms

# ---------- Chart 1: CUDA Benchmark ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: TPS vs position
ax = axes[0]
ax.plot(positions, cuda_tps, marker="o", color="#76b900", linewidth=2, markersize=6, label="CUDA (RTX 3090 Ti)")
ax.axhline(y=28, color="#999999", linestyle="--", alpha=0.5, label="Target: 28 TPS")
ax.set_xscale("log", base=2)
ax.set_xlabel("Cache Position (tokens, log2)")
ax.set_ylabel("Decode TPS (tokens/sec)")
ax.set_title("Decode Speed vs Cache Position")
ax.set_ylim(0, 40)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Step time vs position
ax = axes[1]
ax.errorbar(positions, cuda_step, yerr=cuda_step_std, marker="o", color="#76b900",
            linewidth=2, markersize=6, capsize=3, label="CUDA (RTX 3090 Ti)")
ax.axhline(y=35.7, color="#999999", linestyle="--", alpha=0.5, label="Target: 35.7ms (10x Metal)")
ax.set_xscale("log", base=2)
ax.set_xlabel("Cache Position (tokens, log2)")
ax.set_ylabel("Step Time (ms)")
ax.set_title("Per-Step Decode Latency")
ax.set_ylim(0, 50)
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Qwen2.5-1.5B CUDA Backend — Decode Scaling (max_cache=32768)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("docs/assets/cuda_benchmark_chart.png", dpi=150, bbox_inches="tight")
print("Saved docs/assets/cuda_benchmark_chart.png")
plt.close()

# ---------- Chart 2: Metal vs CUDA Comparison ----------

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: TPS comparison
ax = axes[0]
ax.plot(positions, metal_tps, marker="o", color="#1f77b4", linewidth=2, markersize=6, label="Metal (M4 Pro)")
ax.plot(positions, cuda_tps, marker="s", color="#76b900", linewidth=2, markersize=6, label="CUDA (RTX 3090 Ti)")
ax.set_xscale("log", base=2)
ax.set_xlabel("Cache Position (tokens, log2)")
ax.set_ylabel("Decode TPS (tokens/sec)")
ax.set_title("TPS Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Step time comparison
ax = axes[1]
ax.plot(positions, metal_step, marker="o", color="#1f77b4", linewidth=2, markersize=6, label="Metal (M4 Pro)")
ax.plot(positions, cuda_step, marker="s", color="#76b900", linewidth=2, markersize=6, label="CUDA (RTX 3090 Ti)")
ax.set_xscale("log", base=2)
ax.set_xlabel("Cache Position (tokens, log2)")
ax.set_ylabel("Step Time (ms)")
ax.set_title("Latency Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Speedup (CUDA TPS / Metal TPS)
ax = axes[2]
speedups = [c / m for c, m in zip(cuda_tps, metal_tps)]
colors = ["#76b900" if s >= 10 else "#ff7f0e" for s in speedups]
bars = ax.bar(range(len(positions)), speedups, color=colors, alpha=0.85)
ax.set_xticks(range(len(positions)))
ax.set_xticklabels([str(p) for p in positions], rotation=45, ha="right", fontsize=8)
ax.axhline(y=10, color="#999999", linestyle="--", alpha=0.7, label="10x target")
ax.axhline(y=1, color="#cccccc", linestyle="-", alpha=0.5)
ax.set_xlabel("Cache Position (tokens)")
ax.set_ylabel("CUDA / Metal Speedup")
ax.set_title("Speedup Ratio")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Add TTFT comparison as text
ttft_speedup = metal_ttft / cuda_ttft
fig.text(0.5, -0.02,
         f"TTFT: Metal {metal_ttft:.1f}ms → CUDA {cuda_ttft:.1f}ms ({ttft_speedup:.1f}x faster)",
         ha="center", fontsize=12, style="italic")

fig.suptitle("Metal vs CUDA — Qwen2.5-1.5B Decode (max_cache=32768)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("docs/assets/metal_vs_cuda_chart.png", dpi=150, bbox_inches="tight")
print("Saved docs/assets/metal_vs_cuda_chart.png")
plt.close()

print("Done!")
