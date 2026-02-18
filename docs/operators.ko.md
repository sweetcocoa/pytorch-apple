# 연산자 지원

## 지원 연산 (50+)

### CNN 연산
| 연산 타입 | Metal 커널 | 비고 |
|----------|-----------|------|
| `aten.conv2d.default` | `conv2d_kernel` | Groups, bias, padding, stride |
| `aten.batch_norm.default` | (폴딩됨) | conv 가중치에 병합 |
| `aten.relu.default` | `elementwise_relu` | conv/add와 퓨전 가능 |
| `aten.max_pool2d.default` | `max_pool2d_kernel` | |
| `aten.adaptive_avg_pool2d.default` | `adaptive_avg_pool2d_kernel` | |

### 선형대수
| 연산 타입 | Metal 커널 | 비고 |
|----------|-----------|------|
| `aten.linear.default` | `matmul_kernel` / `matmul_vec_kernel` | MPS 가속, M=1시 vec |
| `aten.matmul.default` | `matmul_notrans_kernel` / `batched_matmul_kernel` | 2D 및 배치 3D+ |
| `aten.addmm.default` | `matmul_kernel` | bias 포함 |
| `aten.t.default` | `transpose_kernel` | 2D 전치 |

### 원소별 단항 연산
| 연산 타입 | Metal 커널 | 비고 |
|----------|-----------|------|
| `aten.silu.default` | `silu_kernel` | x * sigmoid(x) |
| `aten.neg.default` | `neg_kernel` | |
| `aten.rsqrt.default` | `rsqrt_kernel` | |
| `aten.cos.default` | `cos_kernel` | |
| `aten.sin.default` | `sin_kernel` | |
| `aten.pow.Tensor_Scalar` | `pow_scalar_kernel` | |

### 원소별 이항 연산
| 연산 타입 | Metal 커널 | 비고 |
|----------|-----------|------|
| `aten.add.Tensor` | `add_kernel` / `add_broadcast_kernel` | Broadcast 지원 |
| `aten.mul.Tensor` | `mul_kernel` / `mul_broadcast_kernel` | Broadcast 지원 |
| `aten.div.Tensor` | `div_kernel` / `div_broadcast_kernel` | Broadcast 지원 |

### 스칼라 연산
| 연산 타입 | Metal 커널 | 비고 |
|----------|-----------|------|
| `aten.add.Tensor` (스칼라) | `eltwise_add_scalar_kernel` | 단일 입력 + attr |
| `aten.mul.Tensor` (스칼라) | `eltwise_mul_scalar_kernel` | 단일 입력 + attr |
| `aten.full.default` | `add_scalar_kernel` | 상수 채우기 |

### 텐서 연산
| 연산 타입 | Metal 커널 | 비고 |
|----------|-----------|------|
| `aten.embedding.default` | `embedding_kernel` | 토큰 룩업 |
| `aten.transpose.int` | `transpose_kernel` | N차원, matmul에 폴딩 가능 |
| `aten.cat.default` | `cat_2_kernel` | 2입력 연결 |
| `aten.slice.Tensor` | `slice_kernel` | 임의 차원 |
| `aten.expand.default` | `expand_kernel` | Broadcast 복사 |
| `aten.index_copy.default` | `index_copy_kernel` | KV 캐시 갱신 |

### 리덕션
| 연산 타입 | Metal 커널 | 비고 |
|----------|-----------|------|
| `aten.softmax.int` | `softmax_kernel` | 수치 안정 |
| `aten.mean.dim` | `mean_last_dim_kernel` | 마지막 차원 리덕션 |

### 위치 인코딩
| 연산 타입 | Metal 커널 | 비고 |
|----------|-----------|------|
| `wrap_with_set_grad_enabled` | `rope_kernel` | 회전 위치 임베딩 |

### 퓨전 커널
| 패턴 | Metal 커널 | 대체 연산 |
|------|-----------|----------|
| RMSNorm | `rmsnorm_kernel` | pow, mean, add, rsqrt, expand, mul, mul |
| SiLU+Gate | `silu_mul_kernel` | silu, mul |
| Masked Softmax | `masked_softmax_kernel` | add, softmax |
| Add+ReLU | `add_relu_kernel` | add, relu |

### 제로 코스트 별칭 (디스패치 없음)
| 연산 타입 | 비고 |
|----------|------|
| `aten.reshape.default` | 버퍼 별칭 |
| `aten.view.default` | 버퍼 별칭 |
| `aten.flatten.using_ints` | 버퍼 별칭 (4D->비4D는 depad 필요) |
| `aten.contiguous.default` | 항등 |
| `aten.unsqueeze.default` | 형상 변환만 |
| `aten.alias.default` | 항등 |
| `aten.detach_.default` | 항등 |
| `aten.to.dtype` | 항등 (dtype은 경계에서 처리) |
| `aten.dropout.default` | 항등 (평가 모드) |
| `<built-in function getitem>` | 다중 출력 인덱스 |
