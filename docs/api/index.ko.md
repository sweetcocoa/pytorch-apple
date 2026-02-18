# API 레퍼런스

## 모듈

### npu_compiler

IR을 Metal 실행 계획으로 변환하는 오프라인 컴파일 파이프라인입니다.

| 모듈 | 설명 |
|------|------|
| [`ir_reader`](compiler.md#ir-reader) | torch_to_ir IR JSON 로드 |
| [`constraint_checker`](compiler.md#constraint-checker) | NPU 제약 조건 검증 |
| [`graph_optimizer`](compiler.md#graph-optimizer) | BN 폴딩, noop 제거 |
| [`fusion_patterns`](compiler.md#fusion-patterns) | 연산 퓨전 패턴 매칭 |
| [`codegen`](compiler.md#code-generator) | Metal 커널 코드 생성 |
| [`compiled_program`](compiler.md#compiled-program) | 직렬화 (.npubin) |

### npu_runtime

Metal GPU를 사용하는 온라인 실행 엔진입니다.

| 모듈 | 설명 |
|------|------|
| [`device`](runtime.md#device) | Metal 디바이스 관리 |
| [`buffer`](runtime.md#npubuffer) | GPU 메모리 (NPUBuffer) |
| [`executor`](runtime.md#executor) | 커맨드 버퍼 배칭 |
| [`weight_loader`](runtime.md#weight-loader) | safetensors 로딩 |
| [`profiler`](runtime.md#profiler) | 커널 타이밍 |

### metal_kernels

모든 지원 연산에 대한 Metal 컴퓨트 셰이더입니다.

| 파일 | 커널 |
|------|------|
| [`matmul.metal`](kernels.md#matmul) | 타일드, vec, 배치 matmul |
| [`rmsnorm.metal`](kernels.md#rmsnorm) | Fused RMSNorm |
| [`softmax.metal`](kernels.md#softmax) | Softmax, masked softmax |
| [`elementwise_extended.metal`](kernels.md#elementwise) | 단항/이항 연산, SiLU+mul |
| [`elementwise_broadcast.metal`](kernels.md#broadcast) | Broadcast 이항 연산 |
