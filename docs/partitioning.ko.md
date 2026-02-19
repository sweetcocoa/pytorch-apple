# 그래프 파티셔닝

## 개요

모델의 모든 op이 NPU에서 실행 가능한 것은 아닙니다. IR에 미지원 op이 포함된 경우, 그래프를 NPU와 CPU 구간으로 **파티셔닝**하여 교대로 실행합니다. 파티션 경계에서 Transfer op이 텐서를 디바이스 간에 이동시킵니다.

```
IR → op_support (노드별 태깅) → partitioner (그룹핑 + transfer op) → DAGExecutor
     │                            │                                     │
     ├─ NPU 파티션 ──────────────→ compile(dict) → Backend.execute()    │
     ├─ TransferOp ──────────────→ allocate_buffer / to_numpy           │
     └─ CPU 파티션 ──────────────→ torch_ir IRExecutor (ATen fallback)  │
```

## 파이프라인

### 1. Op 지원 확인

`npu_compiler.op_support.is_op_supported(op_type)`으로 해당 op이 NPU에서 실행 가능한지 확인합니다. 지원 테이블은 `codegen.HANDLED_OPS`(50+ ops, 제로 코스트 별칭 포함)와 동일합니다. 이 테이블에 없는 op은 파티션 경로를 통해 CPU fallback으로 라우팅됩니다.

### 2. 파티셔닝

`npu_compiler.partitioner.partition(ir_dict, is_supported_fn)`이 연속 그룹핑을 수행합니다:

1. 각 노드를 `npu` 또는 `cpu`로 **태깅**
2. 동일 target의 연속 노드를 `Partition` 객체로 **그룹핑**
3. producer_node 추적으로 **경계 I/O 계산**
4. 디바이스 전환 시 **`TransferOp` 삽입**

`is_supported_fn` 콜백은 `(op_type, attrs)`를 받아 `bool`을 반환합니다. `is_op_supported`를 래핑하여 커스텀 오버라이드를 추가할 수 있습니다 (예: 디버깅을 위해 특정 op을 CPU로 강제):

```python
def support_fn(op_type, _attrs=None):
    if op_type in force_cpu_ops:
        return False
    return is_op_supported(op_type)
```

**예시** (7-노드 그래프):
```
노드:     [conv] [relu] [matmul] [add] [relu2] [concat] [softmax]
타겟:      npu    npu     npu     npu    npu      cpu      npu

→ Step 0: Partition(npu, [conv, relu, matmul, add, relu2])
  Step 1: TransferOp(to_cpu, [relu2_output])
  Step 2: Partition(cpu, [concat])
  Step 3: TransferOp(to_npu, [concat_output])
  Step 4: Partition(npu, [softmax])
```

### 3. DAG 실행

`npu_runtime.DAGExecutor`가 실행 계획을 오케스트레이션합니다:

- **NPU 파티션**: 각 NPU 파티션은 `__init__` 시 `npu_compiler.compile(sub_ir_dict)`로 컴파일 (1회 컴파일, 다수 실행). `Backend.create_executor()` → `executor.run()`으로 실행.
- **CPU 파티션**: `torch_ir.IRExecutor`로 실행 (모든 ATen op에 대한 스키마 기반 fallback).
- **Transfer op**: to_npu는 `backend.allocate_buffer(numpy_array)`, to_cpu는 `device_buffer.to_numpy()`. bfloat16 dtype이 전송 과정에서 보존됩니다.

`dag.execute()` 전에 `dag.load_weights(weights_dict)`를 호출하여 NPU 가중치 버퍼를 사전 캐시합니다 (1회 업로드, 실행 시 재사용).

### 4. bfloat16 처리

numpy는 네이티브 bfloat16을 지원하지 않습니다. 파이프라인은 `ml_dtypes.bfloat16`을 사용합니다:
- `torch.bfloat16` → `.view(torch.uint16)` → `.numpy()` → `.view(ml_dtypes.bfloat16)`
- 복원: `arr.view(np.uint16)` → `torch.from_numpy().view(torch.bfloat16)`
- NPU↔CPU 전송 과정에서 dtype 보존 (손실 있는 float32 변환 없음)

## API

```python
import json
from npu_compiler import partition, is_op_supported
from npu_runtime import DAGExecutor, MetalBackend

# 1. IR 로드
ir_dict = json.load(open("model.json"))

# 2. 파티셔닝
plan = partition(ir_dict, is_op_supported)

# 3. 컴파일 + 가중치 로드
backend = MetalBackend()
dag = DAGExecutor(plan, backend)
dag.load_weights(weights_dict)

# 4. 실행
result = dag.execute(inputs={"x": input_array})
```

## 데이터 구조

| 클래스 | 모듈 | 설명 |
|--------|------|------|
| `Partition` | `npu_compiler.partitioner` | 동일 디바이스의 연속 노드 그룹 |
| `TransferOp` | `npu_compiler.partitioner` | 디바이스 간 데이터 전송 |
| `PartitionPlan` | `npu_compiler.partitioner` | 순서화된 실행 단계 (파티션 + 전송) |
| `DAGExecutor` | `npu_runtime.dag_executor` | PartitionPlan 실행기 |

## NPU 프레임워크 비교

모든 주요 NPU/가속기 프레임워크가 동일한 4단계 패턴을 따릅니다:
`capability_check() → partition() → compile_subgraphs() → orchestrate_execution()`

| 시스템 | 컴파일 단위 | Op 지원 | Fallback | 비고 |
|--------|:----------:|:-------:|:--------:|------|
| **TensorRT** | 서브그래프 → 엔진 | Layer 지원 매트릭스 (dtype) | CUDA/CPU | FP16/INT8 자동 양자화. `INetworkDefinition` 서브그래프 분리 |
| **XLA (TPU/GPU)** | HLO IR → 플랜 | HLO op 집합 | CPU fallback | JIT 컴파일. `send`/`recv`로 디바이스 전송 |
| **ExecuTorch** | `preprocess()→blob` | Partitioner 태깅 | `call_delegate` + CPU | 모바일/엣지. `to_backend()` delegate 패턴 |
| **ONNX Runtime EP** | EP 서브그래프 | 우선순위 기반 provider | 다음 EP fallback | EP별 `GetCapability()`. 우선순위 순서 할당 |
| **Qualcomm QNN** | 그래프 → context binary | Op 지원 테이블 (SoC별) | CPU delegation | Hexagon DSP. `QnnBackend::IsNodeSupported()` |
| **AWS Neuron** | 그래프 → NEFF | Graph partitioner | CPU fallback | `torch_neuronx.trace()` 자동 파티셔닝. XLA 기반 |
| **Apple ANE** | CoreML 서브그래프 | Op 호환성 테이블 | CPU/GPU fallback | `MLComputeUnits`. 자동 ANE/GPU/CPU 분배 |
| **Intel OpenVINO** | 그래프 → blob | Plugin op 지원 | HETERO plugin | `HETERO:GPU,CPU` 우선순위 형식 |
| **pytorch-apple** | Sub-IR → npubin | `is_op_supported()` | torch_ir executor | 연속 그룹핑. Backend ABC |
