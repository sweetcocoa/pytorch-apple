# 벤치마크

Metal GPU 백엔드에서 실행되는 Qwen2.5-1.5B-Instruct 성능 벤치마크입니다.

## 측정 방법

- **모델**: Qwen2.5-1.5B-Instruct (BFloat16)
- **접근법**: 로그 스케일 캐시 위치에서 단일 스텝 타이밍 (전체 디코드 루프 불필요)
- **측정**: 위치당 3회 측정, 중앙값 및 표준편차 기록
- **지표**: TTFT (첫 토큰까지 시간), TPS (초당 토큰 수), Step Time (ms)

각 캐시 위치에서 KV 캐시를 랜덤 데이터로 채우고 단일 디코드 스텝의 시간을
측정합니다. 이를 통해 autoregressive 오버헤드 없이 디코드 지연 시간을 분리하고,
컨텍스트 길이에 따른 성능 스케일링을 측정합니다.

### 테스트 위치

위치는 1에서 `max_cache_len - prefill_seq_len`까지 log₂ 스케일로 생성됩니다:

```
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ...
```

## 결과

!!! note "실행 환경"
    Apple M4 Pro, 48 GB RAM, macOS 26.2, Python 3.14.0.
    결과는 하드웨어에 따라 달라질 수 있습니다.

### 스케일링 결과 (max_cache_len=32768)

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

| 지표 | 값 |
|------|-----|
| 컴파일 시간 | 0.19 sec |
| 가중치 로딩 시간 | 1.44 sec |
| DAG/Executor TPS 비율 | ~1.00x |

### 차트

![벤치마크 차트](assets/benchmark_chart.png)

**패널 1 — TPS vs 캐시 위치**: KV 캐시 크기에 따른 디코드 처리량 (log₂ x축).
Executor와 DAGExecutor가 비교를 위해 겹쳐 표시됩니다.

**패널 2 — Step Time vs 캐시 위치**: 오차 막대가 있는 스텝별 디코드 지연 시간 (log₂ x축).

### Executor vs DAGExecutor 비교

![비교 차트](assets/benchmark_comparison.png)

**패널 1 — TPS vs 캐시 위치**: Executor와 DAGExecutor 디코드 속도 오버레이.

**패널 2 — Step Time vs 캐시 위치**: 오차 막대가 있는 지연 시간 비교.

**패널 3 — 효율 비율**: 각 위치에서 Executor 대비 DAGExecutor TPS 비율.
1.0에 가까울수록 파티션 오버헤드가 적음을 의미합니다.

## CUDA 백엔드 벤치마크

!!! note "실행 환경"
    NVIDIA RTX 3090 Ti, Ubuntu (WSL2), Python 3.14.0, CuPy.
    결과는 하드웨어 및 드라이버 버전에 따라 달라질 수 있습니다.

### CUDA 스케일링 결과 (max_cache_len=32768)

| Position | CUDA TPS | CUDA Step (ms) | Metal TPS | Metal Step (ms) | Speedup |
|----------|----------|----------------|-----------|-----------------|---------|
| 1 | 30.6 | 32.7 | 2.8 | 359.5 | 10.9x |
| 2 | 31.1 | 32.2 | 2.8 | 360.8 | 11.1x |
| 4 | 31.0 | 32.3 | 2.8 | 360.9 | 11.1x |
| 8 | 30.2 | 33.1 | 2.8 | 361.8 | 10.8x |
| 16 | 30.5 | 32.8 | 2.8 | 361.1 | 10.9x |
| 32 | 30.1 | 33.2 | 2.8 | 361.3 | 10.8x |
| 64 | 30.5 | 32.7 | 2.8 | 361.8 | 10.9x |
| 128 | 30.9 | 32.4 | 2.8 | 363.5 | 11.0x |
| 256 | 30.9 | 32.4 | 2.7 | 365.3 | 11.4x |
| 512 | 31.9 | 31.3 | 2.7 | 368.7 | 11.8x |
| 1024 | 30.9 | 32.3 | 2.7 | 374.9 | 11.4x |
| 2048 | 30.0 | 33.3 | 2.6 | 389.7 | 11.5x |
| 4096 | 30.3 | 33.0 | 2.4 | 417.1 | 12.6x |
| 8192 | 31.0 | 32.3 | 2.1 | 475.3 | 14.8x |
| 16384 | 29.8 | 33.6 | 1.7 | 588.4 | 17.5x |
| 32640 | 30.7 | 32.5 | 1.2 | 813.6 | 25.6x |

| 지표 | Metal (M4 Pro) | CUDA (RTX 3090 Ti) | Speedup |
|------|----------------|---------------------|---------|
| TTFT | 359.5 ms | 34.3 ms | 10.5x |
| Peak TPS | 2.8 tok/s | 31.9 tok/s | 11.4x |
| 컴파일 시간 | 0.19 sec | 0.33 sec | — |

### CUDA 차트

![CUDA 벤치마크 차트](assets/cuda_benchmark_chart.png)

**패널 1 — TPS vs 캐시 위치**: CUDA 디코드 처리량은 캐시 크기와 무관하게 ~30 TPS를 유지합니다.
Metal은 긴 컨텍스트에서 크게 저하됩니다.

**패널 2 — Step Time vs 캐시 위치**: 모든 캐시 위치에서 스텝 지연 시간이 35ms 이하를 유지합니다.

### Metal vs CUDA 비교

![Metal vs CUDA 비교](assets/metal_vs_cuda_chart.png)

**패널 1 — TPS**: CUDA는 모든 캐시 위치에서 Metal 대비 10-25배 높은 처리량을 제공합니다.

**패널 2 — 지연 시간**: CUDA는 ~33ms의 일정한 지연 시간을 유지하는 반면 Metal은 360ms에서 814ms로 증가합니다.

**패널 3 — Speedup 비율**: CUDA의 일정한 스케일링과 Metal의 선형 증가가 대비되어 컨텍스트 길이가 길수록 격차가 커집니다. 32K 토큰에서는 25배 빠릅니다.

### 주요 최적화

CUDA 백엔드의 성능은 여러 최적화에서 비롯됩니다 (자세한 내용은 [CUDA_OPTIMIZATION.md](https://github.com/user/repo/blob/main/CUDA_OPTIMIZATION.md) 참조):

1. **CuPy zero-copy 뷰** — expand, transpose, slice가 CUDA 커널 대신 CuPy 네이티브 뷰 사용
2. **GQA 인식 BMM fusion** — Grouped Query Attention 구조 활용 (batch=2 대신 batch=12)
3. **Fused reduction 커널** — RMSNorm (6→1 dispatch), SiLU+Gate (2→1), Masked Softmax (2→1)
4. **GPU 상주 KV 캐시** — 디코드 스텝당 PCIe host→device 전송 제거
5. **NVRTC 커널 캐시** — 모듈 레벨 컴파일 캐시로 중복 JIT 방지
6. **디스패치 테이블 사전 빌드** — 핫 루프에서 isinstance 체크 제거

### CUDA 벤치마크 실행

```bash
# CUDA 전용 DAG 벤치마크
python ../benchmarks/benchmark_qwen.py --mode dag --backend cuda

# 차트 포함
python ../benchmarks/benchmark_qwen.py --backend cuda --chart-dir ../docs/assets/

# Metal vs CUDA 비교
python ../benchmarks/benchmark_qwen.py --mode dag --backend both
```

!!! note
    CUDA 벤치마크에는 CuPy (`uv sync --extra cuda`)와 NVIDIA GPU가 필요합니다.

## 재현 방법

```bash
# 1. IR 추출 (transformers + torch 필요)
cd examples/
python extract_qwen_ir.py --prefill-seq-len 128 --max-cache-len 32768

# 2. 통합 벤치마크 실행 (Executor + DAGExecutor)
python ../benchmarks/benchmark_qwen.py \
    --mode both \
    --chart-dir ../docs/assets/ \
    --json ../benchmarks/results/benchmark_results.json

# 또는 하나의 executor만 실행:
python ../benchmarks/benchmark_qwen.py --mode executor --chart chart.png
python ../benchmarks/benchmark_qwen.py --mode dag --chart chart.png

# CUDA 백엔드:
python ../benchmarks/benchmark_qwen.py --mode dag --backend cuda --chart cuda_chart.png
```

## JSON 결과

전체 결과는 `benchmarks/results/`에 저장되며, 자동화된 분석 및 CI 통합에 활용할 수 있습니다:

- `benchmark_results.json` — Executor + DAGExecutor 통합 스케일링 결과
