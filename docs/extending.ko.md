# 확장 가이드

NPU 시뮬레이션 파이프라인에 새로운 연산자를 추가하는 방법입니다.

## 단계

### 1. Metal 커널 추가

`metal_kernels/`에 `.metal` 파일을 생성하거나 확장합니다:

```metal
#include <metal_stdlib>
using namespace metal;

#ifdef USE_BFLOAT
typedef bfloat compute_t;
#else
typedef half compute_t;
#endif

// 저장에는 항상 compute_t, 연산에는 float 사용
kernel void my_kernel(
    device const compute_t *input  [[buffer(0)]],
    device compute_t *output       [[buffer(1)]],
    constant MyParams &p           [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    float x = float(input[tid]);
    output[tid] = compute_t(/* 계산 */);
}
```

!!! note "규칙"
    - 모든 저장에 `compute_t` typedef 사용 (float16/bfloat16 처리)
    - 정밀도 손실 방지를 위해 연산 시 `float`으로 캐스팅
    - `max(half, half)` 피하기 — bfloat16 호환을 위해 `max(float(...), 0.0f)` 패턴 사용
    - `if (tid >= total) return;`으로 경계 검사

### 2. Codegen 핸들러 추가

`npu_compiler/codegen.py`의 `_generate_single_kernel_call()`에 핸들러 추가:

```python
if node.op_type == "aten.my_op.default":
    total = int(np.prod(node.outputs[0].shape))
    return KernelCall(
        kernel_name="my_kernel",
        metal_file="my_metal_file.metal",
        input_buffers=[node.inputs[0].name],
        output_buffers=[node.outputs[0].name],
        param_buffers=["my_params"],
        params={"total": total, "some_attr": node.attrs.get("attr", 0)},
        dispatch_type="1d",
        total_threads=total,
    )
```

### 3. Executor에 Param Spec 추가

`npu_runtime/executor.py`의 `_PARAM_SPECS`에 추가:

```python
("my_kernel",): ("2I", ["total", "some_attr"]),
```

형식 문자열은 `struct.pack` 형식 코드를 사용합니다:
- `I` = uint32
- `f` = float32
- `6I` = 6개 uint32 배열 (리스트 파라미터에서 자동 언패킹)

### 4. Op 지원 테이블에 등록

`npu_compiler/codegen.py`의 `HANDLED_OPS`(constraint checker가 여기서 임포트)와 `npu_compiler/op_support.py`의 `_SUPPORTED_OPS`(그래프 파티셔너 사용) 모두에 추가:

```python
# npu_compiler/codegen.py — HANDLED_OPS 집합
HANDLED_OPS.add("aten.my_op.default")

# npu_compiler/op_support.py — _SUPPORTED_OPS 집합
_SUPPORTED_OPS.add("aten.my_op.default")
```

두 집합은 동기화되어야 합니다. `HANDLED_OPS`는 단일 프로그램 컴파일 경로를, `_SUPPORTED_OPS`는 파티션 경로를 제어합니다 (`_SUPPORTED_OPS`에 없는 op은 DAGExecutor를 통해 CPU로 fallback).

### 5. 테스트 추가

```python
class TestMyKernel:
    def test_my_op(self, device):
        x = np.random.randn(4, 64).astype(np.float32)
        ref = torch.tensor(x).my_op().numpy()

        lib = device.compile_metal_file(os.path.join(kernels_dir(), "my_file.metal"))
        buf_in = NPUBuffer.from_numpy(x, device)
        buf_out = NPUBuffer.zeros((4, 64), device)
        params = make_params(device, "I", 256)
        pipeline = device.get_pipeline(lib, "my_kernel")
        dispatch_1d(device, pipeline, [buf_in, buf_out, params], 256)

        result = buf_out.to_numpy()
        npt.assert_allclose(result, ref, atol=1e-3)
```

## 퓨전 패턴 추가

### 1. `fusion_patterns.py`에 패턴 정의

```python
if node.op_type == "aten.op_a.default":
    next_nodes = consumers.get(node.outputs[0].name, [])
    if (len(next_nodes) == 1
            and next_nodes[0].op_type == "aten.op_b.default"
            and next_nodes[0].name not in fused_node_names):
        b_node = next_nodes[0]
        fused_node_names.add(node.name)
        fused_node_names.add(b_node.name)
        result.append(FusedGroup(
            name=f"fused_{node.name}",
            kernel_type="my_fusion",
            nodes=[node, b_node],
        ))
        i += 1
        continue
```

### 2. Codegen에서 처리

`_generate_fused_kernel_call()`에서:

```python
if group.kernel_type == "my_fusion":
    return _gen_my_fused_kernel(group)
```

## 디스패치 유형

| 유형 | 용도 | 그리드 |
|------|------|--------|
| `1d` | 원소별, 리덕션 | `total_threads` |
| `2d` | 2D 연산 (임베딩, matmul) | `grid_width x grid_height` |
| `3d` | 배치 연산 (BMM) | `grid_width x grid_height x grid_depth` |
| `none` | 제로 코스트 별칭 | 디스패치 없음 |
