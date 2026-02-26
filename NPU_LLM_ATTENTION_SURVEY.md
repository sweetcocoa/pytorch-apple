# NPU/TPU의 LLM Attention 처리 방식 조사

Static shape 컴파일 환경에서 LLM의 가변 길이 attention을 어떻게 처리하는지, Qualcomm NPU, Google TPU, Apple ANE 세 플랫폼의 실제 접근 방식을 정리한다.

## 핵심 질문

LLM decode에서 KV 캐시 길이는 매 step 1씩 증가한다. 하지만 NPU/TPU는 static shape 컴파일이 기본이다. 이 모순을 각 하드웨어 벤더는 어떻게 해결하는가?

## 공통 제약: Static Shape 컴파일

세 플랫폼 모두 **컴파일 시점에 텐서 shape이 확정**되어야 한다는 근본적 제약을 공유한다.

| 플랫폼 | 컴파일러 | Dynamic Shape 지원 |
|--------|---------|-------------------|
| Qualcomm NPU | QNN AOT Compiler | 미지원 (static only) |
| Google TPU | XLA | 미지원 (shape 변경 시 재컴파일) |
| Apple ANE | CoreML (coremltools) | 제한적 (EnumeratedShapes, RangeDim) |

따라서 "유효 범위만 계산"하려면 하드웨어가 dynamic shape을 지원하는 게 아니라, **여러 static shape 그래프를 미리 컴파일하고 런타임에 선택**하는 전략이 필요하다.

---

## 1. Qualcomm NPU (Hexagon / Cloud AI 100)

### 1.1 서버: Cloud AI 100 — Network Specialization

Cloud AI 100은 **Network Specialization**이라는 방식으로 여러 (batch_size, seq_len, ctx_len) 조합을 하나의 바이너리(QPC)로 컴파일한다. 가중치는 공유되고, 각 specialization에 대한 최적화된 실행 그래프가 내장된다.

```json
{
  "specializations": [
    {"batch_size": "1", "seq_len": "32", "ctx_len": "128"},
    {"batch_size": "1", "seq_len": "1",  "ctx_len": "128"}
  ]
}
```

- 첫 번째 specialization: **prefill** (seq_len = prompt 길이)
- 두 번째 specialization: **decode** (seq_len = 1, 항상 고정)
- Prompt은 left-padding + attention mask로 고정 길이에 맞춤

**KV 캐시**: `max_context_length`로 사전 할당. `_RetainedState` 접미사를 통해 on-device DDR에 상주하며, 호스트로 전송되지 않는다. `cache_index` 스칼라가 쓰기 위치를 추적한다.

**제약**: KV 캐시 크기는 컴파일 시점에 고정. 런타임에 동적 resize 불가.

> 출처: [Cloud AI SDK LLM Documentation](https://quic.github.io/cloud-ai-sdk-pages/1.15/Getting-Started/Model-Architecture-Support/Large-Language-Models/llm/)

### 1.2 모바일: Hexagon NPU — Attention은 NPU에서 안 돌린다

ASPLOS 2025 논문 "Fast On-device LLM Inference with NPUs" (PowerNPU)에 따르면, 모바일 Hexagon NPU에서는 **attention을 NPU에서 실행하지 않는다**. Attention은 shape이 chunk 길이와 sequence 위치에 따라 변하는 "dynamic operator"로 분류되어, NPU의 static graph 요구와 호환되지 않기 때문이다.

실제 모바일 LLM 추론 파이프라인:

```
NPU: Linear layers (INT8 MatMul, 73 TOPS)
CPU/GPU: Attention + LayerNorm (FP16/FP32)
```

- **Prefill**: 가변 길이 prompt을 고정 크기 chunk (예: 256 토큰)으로 분할. Linear은 NPU, attention은 CPU/GPU에서 실행. chunk 간 비순차 스케줄링으로 NPU 유휴 시간 최소화.
- **Decode**: 메모리 대역폭 바운드이므로 GPU 또는 CPU가 오히려 NPU보다 나을 수 있음.

**추가 제약**:
- NPU 접근 가능 메모리 ~4GB (모바일)
- HMX tile 정렬: FP16 tile = 32x32 (2 KiB)
- Per-group quantization 비호환 (최대 10.7x 오버헤드)
- 커스텀 커널 API 미제공

> 출처: [ASPLOS 2025 — PowerNPU](https://arxiv.org/html/2407.05858v2), [HeteroLLM](https://arxiv.org/html/2501.14794v1)

### 1.3 Qualcomm 요약

| 환경 | Attention 처리 | KV 캐시 | Sequence 길이 |
|------|--------------|---------|-------------|
| Cloud AI 100 | NPU에서 실행 | Static 사전 할당 + RetainedState | Network Specialization (padding + mask) |
| 모바일 Hexagon | CPU/GPU에서 실행 | CPU/GPU 메모리 | Chunk 분할 + 고정 크기 |

---

## 2. Google TPU (v4 / v5 / Trillium)

### 2.1 XLA의 Static Shape 제약

XLA는 모든 텐서 shape을 컴파일 시점에 요구한다. Shape이 바뀌면 새 XLA 그래프가 컴파일된다. 초기 컴파일에 **20-30분**이 소요될 수 있으며, 컴파일된 그래프는 디스크에 캐싱된다.

### 2.2 Bucketing: 핵심 전략

TPU에서 가변 길이를 처리하는 표준 방식은 **bucketing**이다. 미리 정해진 크기 집합에 대해 XLA 그래프를 컴파일해두고, 입력을 가장 가까운 bucket으로 padding한다.

```
Prefill buckets: [8, 16, 32, 64, 128, 256, 512, 1024, ..., max_model_len]
Decode batch buckets: [8, 16, 24, 32, 40, ..., 256]
```

- Prefill은 batch_size=1로 고정하고, `prefill_len`을 bucketing
- Decode는 seq_len=1로 고정하고, `batch_size`를 bucketing
- 입력은 다음 bucket 크기로 padding, attention mask가 padding 토큰을 무효화

**Padding 낭비**: JAX Scaling Book은 "prefill은 가장 긴 시퀀스로 padding되어 많은 연산을 낭비한다"고 명시적으로 언급한다. 이 때문에 batch_size=1에서 길이별 bucketing이 선호된다.

> 출처: [JAX Scaling Book — Inference](https://jax-ml.github.io/scaling-book/inference/), [vLLM TPU Installation](https://docs.vllm.ai/en/v0.5.5/getting_started/tpu-installation.html)

### 2.3 KV 캐시 관리

**방식 A — Static 사전 할당**:
KV 캐시를 `max_seq_len`으로 사전 할당. `cache_index`로 쓰기 위치 추적. Attention mask로 미사용 위치 무효화. `jax.jit`이 한 번 컴파일한 함수를 재사용.

**방식 B — Paged KV Cache (vLLM on TPU)**:
vLLM의 TPU 백엔드는 Ragged Paged Attention (RPA)을 구현한다.

- Page table: `page_indices[batch_size, pages_per_sequence]` → 글로벌 KV 버퍼 인덱싱
- `lengths` 배열 (int32)이 배치 요소별 실제 시퀀스 길이 추적
- RPA v3 (최신): KV cache scatter를 attention 커널에 fuse하여 scatter 지연 시간을 완전히 은닉

> 출처: [JAX Paged Attention Kernel](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py), [vLLM TPU Blog](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)

### 2.4 Prefill vs Decode 분리

Google은 두 phase를 근본적으로 다른 연산 문제로 취급한다:

| Phase | 특성 | 병목 | TPU 활용률 |
|-------|------|------|-----------|
| Prefill | 전체 prompt 동시 처리 | Compute-bound | 높음 (76% MFU, PaLM 540B) |
| Decode | 토큰 1개씩 순차 생성 | Memory-bandwidth-bound | 낮음 (GEMV, systolic array 활용도 저조) |

- **JetStream** (Google의 서빙 프레임워크): prefill/decode를 별도 TPU 호스트에 배치하는 disaggregated serving 지원
- **vLLM RPA v3**: prefill-only, decode-only, mixed-batch 3개 서브커널로 컴파일

> 출처: [Efficiently Scaling Transformer Inference (MLSys 2023)](https://arxiv.org/abs/2211.05102)

### 2.5 Pallas: TPU 커스텀 커널의 Variable-Length 처리

Pallas는 JAX의 TPU 커스텀 커널 프레임워크다. **Slice 크기는 컴파일 상수**여야 하지만, **slice 시작 인덱스는 런타임에 결정** 가능하다. 이 특성이 attention 커널에서 핵심적이다: block/tile 크기는 고정이지만, 어떤 block을 로드할지는 런타임에 결정된다.

가변 길이 처리 방식:
1. **SegmentIds**: 토큰별 정수 ID로 cross-sequence 격리 (packing 시)
2. **Attention mask + static cache**: 전체 캐시에 대해 matmul 후 mask로 무효화
3. **`query_seq_lengths` / `key_value_seq_lengths`**: per-sequence 길이 파라미터

> 출처: [JAX Pallas Flash Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py)

### 2.6 Google TPU 요약

| 전략 | 설명 | 사용처 |
|------|------|-------|
| Bucketing | 미리 정한 크기 집합으로 padding + 컴파일 | 모든 TPU 추론 |
| Static KV cache | max_len 사전 할당 + index + mask | 기본 JAX 추론 |
| Paged Attention | Page table 기반 KV 관리 | vLLM TPU 백엔드 |
| Disaggregated serving | Prefill/Decode를 별도 호스트에서 실행 | JetStream |

---

## 3. Apple ANE (Neural Engine)

### 3.1 CoreML의 Shape 유연성 메커니즘

ANE는 CoreML을 통해서만 접근 가능하며, 세 가지 shape 처리 방식을 제공한다:

| 방식 | 설명 | 제약 |
|------|------|------|
| **Fixed Shape** | 모든 입력을 max 크기로 padding | 연산 낭비 |
| **EnumeratedShapes** | 최대 128개의 미리 정의된 shape 집합 | iOS 18 이전: 1개 입력만 가능 |
| **RangeDim** | 차원별 min/max 범위 지정 | Unbounded range는 ANE 비호환 |

```python
# EnumeratedShapes 예시
input_shape = ct.EnumeratedShapes(
    shapes=[[1, 3, 25, 25], [1, 3, 50, 50], [1, 3, 67, 67]],
    default=[1, 3, 67, 67]
)

# RangeDim 예시 (bounded)
input_shape = ct.Shape(shape=(1, 3,
    ct.RangeDim(lower_bound=25, upper_bound=100, default=45),
    ct.RangeDim(lower_bound=25, upper_bound=100, default=45),
))
```

> 출처: [CoreML Flexible Inputs Guide](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)

### 3.2 KV 캐시: Stateful Models (iOS 18+)

iOS 18부터 CoreML에 `StateType`이 도입되어 KV 캐시를 in-place로 업데이트할 수 있다. 이전에는 매 iteration마다 KV 텐서를 입출력으로 복사해야 했다.

```python
# CoreML 변환 시
ct.StateType(wrapped_type=ct.TensorType(shape=shape), name="k_cache")
```

업데이트 사이클: `read_state()` → 새 K/V 계산 → scatter → `coreml_update_state()`

**성능 효과**: Mistral 7B (M3 Max)에서 1.6x speedup. Llama 3.1-8B에서 0.19 → 16.26 tok/s.

**실무 요구사항** (SqueezeBits/Yetter):
- 레이어별 KV 캐시를 통합 텐서로 합쳐야 함 (56개 → 2개)
- 비-첫 번째 캐시 차원은 2의 거듭제곱이어야 함
- `pow(x, 2)`을 `mul(x, x)`로 대체 필요 (수치 안정성)

> 출처: [CoreML Stateful Models Guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html), [SqueezeBits Blog](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)

### 3.3 ANE의 Attention 한계

ANE에서 attention을 실행하는 것은 가능하지만 여러 제약이 있다:

1. **데이터 레이아웃**: 4D channels-first `(B, C, 1, S)` 필수. 마지막 축 64바이트 정렬.
2. **Linear → Conv2d 변환**: 모든 `nn.Linear`을 1x1 `nn.Conv2d`로 교체해야 ANE 효율적.
3. **Head 분리**: Batched attention이 아닌 **per-head split** 필요 (L2 캐시 활용, 멀티코어 병렬화).
4. **Long sequence 제한**: 1024+ 토큰은 SDPA query slicing 필요.
5. **Fallback 위험**: ANE 비호환 레이어가 하나라도 있으면 전체 모델이 CPU로 fallback 가능.

### 3.4 실전 추론 전략: Disaggregated Inference

가장 유망한 접근법은 **ANE에서 prefill, GPU에서 decode**를 분리 실행하는 것이다:

```
Prefill: ANE (CoreML) — 높은 처리량, 낮은 전력
Decode:  GPU (MLX)    — 낮은 지연 시간, 유연한 dynamic shape
```

Apple 자체도 Llama 3.1 온디바이스 구현에서 "메모리 대역폭 바운드인 모델은 GPU가 compute FLOPS와 메모리 대역폭의 최적 조합을 제공한다"고 밝혔다.

> 출처: [Apple — On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama), [SqueezeBits — Disaggregated Inference](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)

### 3.5 Apple ANE 요약

| 전략 | 설명 | 사용처 |
|------|------|-------|
| EnumeratedShapes | 미리 정한 shape 집합 (최대 128개) | Prefill 길이 변동 |
| RangeDim | Bounded range per dimension | 유연한 입력 크기 |
| Stateful KV cache | iOS 18+ in-place 업데이트 | Decode 시 KV 관리 |
| Multifunction model | Prefill/Decode 함수 분리, 가중치 공유 | ANEMLL, Yetter |
| Disaggregated inference | ANE prefill + GPU decode | 최신 모바일 최적화 |

---

## 4. 세 플랫폼 비교

### 4.1 Attention에서의 가변 길이 처리 방식

| | Qualcomm (Cloud AI 100) | Google TPU | Apple ANE |
|---|---|---|---|
| **컴파일 단위** | Network Specialization (다중 shape 바이너리) | XLA 그래프 (shape별 컴파일) | CoreML 모델 (EnumeratedShapes) |
| **KV 캐시** | Static 사전 할당 + RetainedState | Static 사전 할당 or Paged Attention | Stateful model (iOS 18+) |
| **Prefill 길이 처리** | Left-padding + mask | Bucketing + padding | EnumeratedShapes / RangeDim |
| **Decode** | seq_len=1 고정 specialization | Batch bucketing | Multifunction model |
| **모바일 attention** | CPU/GPU에서 실행 (NPU 비호환) | N/A (서버 전용) | ANE 가능하나 GPU 선호 |

### 4.2 "유효 범위만 계산"하는가?

**아니다.** 세 플랫폼 모두 기본적으로 padding + masking으로 고정 크기를 계산한다. 다만, **낭비를 줄이는 정도**에 차이가 있다:

| 전략 | 낭비 수준 | 구현 복잡도 |
|------|----------|-----------|
| Max padding (이 프로젝트) | 최대 (항상 32768 계산) | 최소 |
| Bucketing (TPU) | 중간 (다음 bucket까지 padding) | 중간 (bucket별 컴파일) |
| Network Specialization (Qualcomm) | 중간 (specialization별 padding) | 중간 |
| Paged Attention (vLLM/TPU) | 최소 (page 단위) | 높음 (커스텀 커널) |
| Dynamic shape (CUDA GPU) | 없음 (정확히 필요한 만큼) | 높음 (런타임 shape 결정) |

### 4.3 이 프로젝트와의 관계

이 프로젝트의 현재 접근 방식 (static shape + max padding + mask)은 NPU 시뮬레이터로서 **현실적**이다. 실제 NPU 하드웨어도 동일한 제약 아래에서 동작하며, 차이점은 padding 낭비를 줄이기 위한 **bucketing** 또는 **multiple specialization** 전략의 유무에 있다.

개선 가능한 방향:

1. **Bucketing**: 몇 가지 KV 캐시 길이 (128, 512, 2048, 8192, 32768)에 대해 미리 컴파일하고, 런타임에 `actual_pos`에 가장 가까운 것을 선택
2. **Prefill/Decode 분리 컴파일**: 이미 구현됨 (별도 IR)
3. **KV 캐시 on-device 상주**: CUDA 백엔드에서 이미 구현됨 (CUDABuffer pre-allocation)

---

## 참고 자료

### Qualcomm
- [Cloud AI SDK LLM Documentation](https://quic.github.io/cloud-ai-sdk-pages/1.15/Getting-Started/Model-Architecture-Support/Large-Language-Models/llm/)
- [Efficient-Transformers Library](https://github.com/quic/efficient-transformers)
- [ASPLOS 2025 — PowerNPU](https://arxiv.org/html/2407.05858v2)
- [HeteroLLM](https://arxiv.org/html/2501.14794v1)
- [Cloud AI 100 Ultra Product Brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Prod-Brief-QCOM-Cloud-AI-100-Ultra.pdf)

### Google TPU
- [Efficiently Scaling Transformer Inference (MLSys 2023)](https://arxiv.org/abs/2211.05102)
- [JAX Scaling Book — Inference](https://jax-ml.github.io/scaling-book/inference/)
- [vLLM TPU Blog — RPA v3](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)
- [TPU LLM Inference Handbook](https://tpu-llm.github.io/tpu-llm-inference-handbook/)
- [JAX Pallas Flash Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py)
- [JetStream Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-jetstream-pytorch)

### Apple ANE
- [Deploying Transformers on the ANE](https://machinelearning.apple.com/research/neural-engine-transformers)
- [On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- [CoreML Stateful Models Guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [CoreML Flexible Inputs Guide](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)
- [SqueezeBits — Disaggregated Inference](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- [ANEMLL](https://github.com/Anemll/Anemll)
