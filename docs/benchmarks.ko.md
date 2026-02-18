# 벤치마크

Metal GPU 백엔드에서 실행되는 Qwen2.5-1.5B-Instruct 성능 벤치마크입니다.

## 측정 방법

- **모델**: Qwen2.5-1.5B-Instruct (BFloat16)
- **IR 설정**: prefill_seq_len=1024, max_cache_len=2048
- **측정**: 3회 워밍업 후 5회 실행, 중앙값 및 표준편차 기록
- **지표**: TTFT (첫 토큰까지 시간), TPS (초당 토큰 수), Prefill 처리량

### 시나리오

| ID | 프롬프트 | 디코드 | 설명 |
|----|---------|--------|------|
| S1 | 16 | 32 | 최소 부하 (오버헤드 비율) |
| S2 | 64 | 64 | 짧은 대화 |
| S3 | 256 | 128 | 중간 대화 |
| S4 | 512 | 256 | 긴 컨텍스트 |
| S5 | 1024 | 128 | 긴 문서 요약 (prefill 중심) |
| S6 | 64 | 512 | 짧은 질문 + 긴 응답 (decode 중심) |

## 결과

!!! note "실행 환경"
    Apple M4 Pro, 48 GB RAM, macOS 26.2, Python 3.14.0.
    결과는 하드웨어에 따라 달라질 수 있습니다.

### NPU (Metal GPU)

| ID | 프롬프트 | 디코드 | TTFT (ms) | Decode TPS | Prefill TPS |
|----|---------|--------|-----------|------------|-------------|
| S1 | 16 | 32 | 4497.9 ± 33.9 | 16.6 ± 0.2 | 4 |
| S2 | 64 | 64 | 4476.5 ± 10.1 | 16.6 ± 0.5 | 14 |
| S3 | 256 | 128 | 4493.2 ± 15.2 | 15.9 ± 0.0 | 57 |
| S4 | 512 | 256 | 4499.7 ± 3.0 | 15.1 ± 0.0 | 114 |
| S5 | 1024 | 128 | 4492.2 ± 19.9 | 13.9 ± 0.0 | 228 |
| S6 | 64 | 512 | 4487.8 ± 11.1 | 15.7 ± 0.1 | 14 |

| 지표 | 값 |
|------|-----|
| 컴파일 시간 | 0.16 sec |
| 가중치 로딩 시간 | 5.69 sec |
| 피크 메모리 (추정) | 11,505 MB |
| Prefill 스케일링 | 0.006 ms/token |
| Decode TPS CV | 5.9% |
| GPU 활용률 (추정) | 95.5% |
| 커널 오버헤드 | 44.1 us/kernel |

### 차트

![벤치마크 차트](assets/benchmark_chart.png)

**차트 1 — TTFT vs 프롬프트 길이**: 프롬프트 길이에 따른 prefill 지연 시간.
선형 스케일링은 효율적인 chunked prefill을 나타냅니다.

**차트 2 — 스텝별 TPS**: 가장 긴 디코드 시나리오(S6)에서 각 스텝의 디코드 처리량.
평평한 선은 성능 저하 없이 일관된 디코드 성능을 나타냅니다.

**차트 3 — 시나리오별 처리량**: 모든 시나리오에서 prefill 처리량과 디코드 TPS를 비교합니다.

### 분석 지표

| 지표 | 설명 | 이상적 |
|------|------|--------|
| Prefill 스케일링 | 프롬프트 토큰당 TTFT 증가 (ms/tok) | 선형 |
| Decode TPS CV | 시나리오 간 변동 계수 | ~0% |
| GPU 활용률 | 커널 시간 / 전체 시간 추정 | >95% |
| 커널 오버헤드 | 커널당 실행 오버헤드 | <10 us |

## 재현 방법

```bash
# 1. IR 추출 (transformers + torch 필요)
cd examples/
python extract_qwen_ir.py --prefill-seq-len 1024 --max-cache-len 2048

# 2. 벤치마크 실행
python ../benchmarks/benchmark_qwen.py --no-cpu \
    --chart ../docs/assets/benchmark_chart.png \
    --json ../benchmarks/results/benchmark_results.json

# 3. CPU 기준선 포함 (더 느림)
python ../benchmarks/benchmark_qwen.py \
    --chart ../docs/assets/benchmark_chart.png \
    --json ../benchmarks/results/benchmark_results.json
```

## JSON 결과

전체 결과는 `benchmarks/results/benchmark_results.json`에 저장되며, 자동화된 분석 및 CI 통합에 활용할 수 있습니다.
