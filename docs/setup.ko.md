# 설치

## 요구사항

- Apple Silicon (M1/M2/M3/M4) macOS
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) 패키지 매니저

## 설치

```bash
# 저장소 클론
git clone <repo-url>
cd npu-simulation/pytorch-apple

# 런타임 의존성 설치
uv sync

# 개발 의존성 설치 (테스트용)
uv sync --extra dev

# 예제 의존성 설치 (Qwen/ResNet 데모)
uv sync --extra examples

# 문서 의존성 설치 (문서 빌드)
uv sync --extra docs
```

## 프로젝트 구조

```
pytorch-apple/
├── npu_compiler/           # 오프라인 컴파일 파이프라인
│   ├── ir_reader.py        # torch_to_ir IR JSON 로드
│   ├── constraint_checker.py # NPU 제약 조건 검증
│   ├── graph_optimizer.py  # BN 폴딩, noop 제거
│   ├── fusion_patterns.py  # 연산 퓨전 패턴 매칭
│   ├── codegen.py          # Metal 커널 코드 생성
│   ├── compiled_program.py # 직렬화 (.npubin)
│   ├── op_support.py       # Op 지원 테이블 (is_op_supported)
│   └── partitioner.py      # 그래프 파티셔닝 (NPU/CPU 분할)
├── npu_runtime/            # Metal GPU 온라인 실행
│   ├── backend.py          # Backend ABC (하드웨어 독립)
│   ├── metal_backend.py    # MetalBackend 구현
│   ├── device.py           # Metal 디바이스 관리
│   ├── buffer.py           # NPUBuffer (GPU 메모리)
│   ├── executor.py         # 커맨드 버퍼 배칭 (단일 프로그램)
│   ├── dag_executor.py     # DAGExecutor (NPU + CPU 혼합)
│   ├── cpu_fallback.py     # torch_ir 기반 CPU fallback
│   ├── weight_loader.py    # safetensors → NPU 버퍼
│   └── profiler.py         # 커널 시간 측정
├── metal_kernels/          # Metal 컴퓨트 셰이더
├── tests/                  # 테스트 스위트
├── examples/               # 데모 스크립트
└── docs/                   # 이 문서
```

## 의존성

### 런타임
| 패키지 | 용도 |
|--------|------|
| `numpy` | 배열 연산 |
| `ml-dtypes` | BFloat16 지원 |
| `pyobjc-framework-Metal` | Metal API 바인딩 |
| `pyobjc-framework-MetalPerformanceShaders` | MPS matmul |
| `pytorch-ir` | IR 추출 + CPU fallback 실행 (torch_to_ir) |

### 개발
| 패키지 | 용도 |
|--------|------|
| `pytest` | 테스트 |
| `torch` | 레퍼런스 연산 |
| `ruff` | 린팅 |

## 검증

```bash
# 전체 테스트 실행
uv run pytest tests/ -v

# 예상: 180+ 통과, 1-2 스킵 (모델 다운로드)
```
