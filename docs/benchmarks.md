# Benchmarks

Performance benchmarks for Qwen2.5-1.5B-Instruct running on the Metal GPU backend.

## Methodology

- **Model**: Qwen2.5-1.5B-Instruct (BFloat16)
- **Approach**: Single-step timing at log-scale cache positions (avoids full decode loops)
- **Measurement**: 3 measurements per position, reporting median and stddev
- **Metrics**: TTFT (Time To First Token), TPS (Tokens Per Second), Step Time (ms)

At each cache position, a KV cache is filled with random data and a single
decode step is timed. This isolates decode latency from autoregressive overhead
and measures how performance scales with context length.

### Test Positions

Positions are generated on a log₂ scale from 1 to `max_cache_len - prefill_seq_len`:

```
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ...
```

## Results

!!! note "Environment"
    Apple M4 Pro, 48 GB RAM, macOS 26.2, Python 3.14.0.
    Results may vary depending on hardware.

### Scaling Results (max_cache_len=32768)

| Position | Executor TPS | Executor Step (ms) | DAG TPS | DAG Step (ms) | Ratio |
|----------|-------------|-------------------|---------|---------------|-------|
| 1 | 2.8 | 359.5 | 2.7 | 372.1 | 0.97x |
| 2 | 2.8 | 360.8 | 2.8 | 362.1 | 1.00x |
| 4 | 2.8 | 360.9 | 2.8 | 362.8 | 0.99x |
| 8 | 2.8 | 361.8 | 2.8 | 361.8 | 1.00x |
| 16 | 2.8 | 361.1 | 2.8 | 362.1 | 1.00x |
| 32 | 2.8 | 361.3 | 2.8 | 362.5 | 1.00x |
| 64 | 2.8 | 361.8 | 2.8 | 362.2 | 1.00x |
| 128 | 2.8 | 363.5 | 2.7 | 363.9 | 1.00x |
| 256 | 2.7 | 365.3 | 2.7 | 365.5 | 1.00x |
| 512 | 2.7 | 368.7 | 2.7 | 369.4 | 1.00x |
| 1024 | 2.7 | 374.9 | 2.7 | 375.4 | 1.00x |
| 2048 | 2.6 | 389.7 | 2.6 | 390.3 | 1.00x |
| 4096 | 2.4 | 417.1 | 2.4 | 420.6 | 0.99x |
| 8192 | 2.1 | 475.3 | 2.1 | 474.6 | 1.00x |
| 16384 | 1.7 | 588.4 | 1.7 | 590.3 | 1.00x |
| 32640 | 1.2 | 813.6 | 1.2 | 818.2 | 0.99x |

| Metric | Value |
|--------|-------|
| Compile time | 0.19 sec |
| Weight load time | 1.44 sec |
| DAG/Executor TPS ratio | ~1.00x |

### Charts

![Benchmark Chart](assets/benchmark_chart.png)

**Panel 1 — TPS vs Cache Position**: Shows how decode throughput scales with KV cache size (log₂ x-axis).
Both Executor and DAGExecutor lines are overlaid for comparison.

**Panel 2 — Step Time vs Cache Position**: Per-step decode latency with error bars (log₂ x-axis).

### Executor vs DAGExecutor Comparison

![Comparison Chart](assets/benchmark_comparison.png)

**Panel 1 — TPS vs Cache Position**: Executor and DAGExecutor decode speed overlay.

**Panel 2 — Step Time vs Cache Position**: Latency comparison with error bars.

**Panel 3 — Efficiency Ratio**: DAGExecutor TPS as a fraction of Executor TPS at each position.
Values near 1.0 indicate minimal partition overhead.

## Reproducing

```bash
# 1. Extract IR (requires transformers + torch)
cd examples/
python extract_qwen_ir.py --prefill-seq-len 128 --max-cache-len 32768

# 2. Run integrated benchmark (Executor + DAGExecutor)
python ../benchmarks/benchmark_qwen.py \
    --mode both \
    --chart-dir ../docs/assets/ \
    --json ../benchmarks/results/benchmark_results.json

# Or run only one executor type:
python ../benchmarks/benchmark_qwen.py --mode executor --chart chart.png
python ../benchmarks/benchmark_qwen.py --mode dag --chart chart.png
```

## JSON Results

Full results are saved to `benchmarks/results/` for automated analysis and CI integration:

- `benchmark_results.json` — Combined Executor + DAGExecutor scaling results
