# Benchmarks

Performance benchmarks for Qwen2.5-1.5B-Instruct running on the Metal GPU backend.

## Methodology

- **Model**: Qwen2.5-1.5B-Instruct (BFloat16)
- **IR config**: prefill_seq_len=1024, max_cache_len=2048
- **Measurement**: 5 runs with 3 warmup runs, reporting median and stddev
- **Metrics**: TTFT (Time To First Token), TPS (Tokens Per Second), Prefill throughput

### Scenarios

| ID | Prompt | Decode | Description |
|----|--------|--------|-------------|
| S1 | 16 | 32 | Minimal load (overhead ratio) |
| S2 | 64 | 64 | Short conversation |
| S3 | 256 | 128 | Medium conversation |
| S4 | 512 | 256 | Long context |
| S5 | 1024 | 128 | Long document summarization (prefill-heavy) |
| S6 | 64 | 512 | Short question + long response (decode-heavy) |

## Results

!!! note "Environment"
    Apple M4 Pro, 48 GB RAM, macOS 26.2, Python 3.14.0.
    Results may vary depending on hardware.

### NPU (Metal GPU)

| ID | Prompt | Decode | TTFT (ms) | Decode TPS | Prefill TPS |
|----|--------|--------|-----------|------------|-------------|
| S1 | 16 | 32 | 4497.9 ± 33.9 | 16.6 ± 0.2 | 4 |
| S2 | 64 | 64 | 4476.5 ± 10.1 | 16.6 ± 0.5 | 14 |
| S3 | 256 | 128 | 4493.2 ± 15.2 | 15.9 ± 0.0 | 57 |
| S4 | 512 | 256 | 4499.7 ± 3.0 | 15.1 ± 0.0 | 114 |
| S5 | 1024 | 128 | 4492.2 ± 19.9 | 13.9 ± 0.0 | 228 |
| S6 | 64 | 512 | 4487.8 ± 11.1 | 15.7 ± 0.1 | 14 |

| Metric | Value |
|--------|-------|
| Compile time | 0.16 sec |
| Weight load time | 5.69 sec |
| Peak memory (est) | 11,505 MB |
| Prefill scaling | 0.006 ms/token |
| Decode TPS CV | 5.9% |
| GPU utilization (est) | 95.5% |
| Kernel overhead | 44.1 us/kernel |

### Charts

![Benchmark Charts](assets/benchmark_chart.png)

**Chart 1 — TTFT vs Prompt Length**: Shows how prefill latency scales with prompt length.
Linear scaling indicates efficient chunked prefill.

**Chart 2 — Per-Step TPS**: Shows decode throughput at each step for the longest decode scenario (S6).
Flat lines indicate consistent decode performance without degradation.

**Chart 3 — Throughput by Scenario**: Compares prefill throughput and decode TPS across all scenarios.

### Analysis Metrics

| Metric | Description | Ideal |
|--------|-------------|-------|
| Prefill scaling | TTFT increase per prompt token (ms/tok) | Linear |
| Decode TPS CV | Coefficient of variation across scenarios | ~0% |
| GPU utilization | Estimated kernel time / total time | >95% |
| Kernel overhead | Per-kernel launch overhead | <10 us |

## Reproducing

```bash
# 1. Extract IR (requires transformers + torch)
cd examples/
python extract_qwen_ir.py --prefill-seq-len 1024 --max-cache-len 2048

# 2. Run benchmark
python ../benchmarks/benchmark_qwen.py --no-cpu \
    --chart ../docs/assets/benchmark_chart.png \
    --json ../benchmarks/results/benchmark_results.json

# 3. With CPU baseline (slower)
python ../benchmarks/benchmark_qwen.py \
    --chart ../docs/assets/benchmark_chart.png \
    --json ../benchmarks/results/benchmark_results.json
```

## JSON Results

Full results are saved to `benchmarks/results/benchmark_results.json` for automated analysis and CI integration.
