# Metal 커널 레퍼런스

## Matmul

**파일**: `matmul.metal`

| 커널 | 설명 | 디스패치 |
|------|------|----------|
| `matmul_kernel` | 타일드 matmul C = A @ B^T (TILE=16, 공유 메모리) | 2D: (N/16, M/16) |
| `matmul_vec_kernel` | M=1 벡터 matmul, 병렬 K-리덕션 (256 스레드) | 1D: N 스레드그룹 |
| `matmul_notrans_kernel` | C = A @ B (B 전치 안함) | 2D |
| `matmul_notrans_vec_kernel` | M=1 벡터 변형 | 1D |
| `batched_matmul_kernel` | C[b] = A[b] @ B[b] | 3D: (N, M, batch) |

**파라미터**: `MatmulParams { M, N, K, has_bias }`

## RMSNorm

**파일**: `rmsnorm.metal`

| 커널 | 설명 | 디스패치 |
|------|------|----------|
| `rmsnorm_kernel` | 퓨전: x * rsqrt(mean(x^2) + eps) * weight | 1D: rows |

**파라미터**: `RMSNormParams { rows, cols, eps }`

8개 별도 디스패치(pow, mean, add, rsqrt, expand, mul, expand, mul)를 대체합니다.

## Softmax

**파일**: `softmax.metal`

| 커널 | 설명 | 디스패치 |
|------|------|----------|
| `softmax_kernel` | 수치 안정 softmax (3-pass: max, exp+sum, normalize) | 1D: rows |
| `masked_softmax_kernel` | 퓨전 add(scores, mask) + softmax | 1D: rows |
| `masked_softmax_broadcast_kernel` | 브로드캐스트 마스크 masked softmax | 1D: rows |

**파라미터**: `SoftmaxParams { rows, cols }`

## 원소별 연산

**파일**: `elementwise_extended.metal`

### 단항
| 커널 | 연산 |
|------|------|
| `silu_kernel` | x * sigmoid(x) |
| `neg_kernel` | -x |
| `rsqrt_kernel` | 1/sqrt(x) |
| `cos_kernel` | cos(x) |
| `sin_kernel` | sin(x) |
| `pow_scalar_kernel` | x^exp |

### 이항
| 커널 | 연산 |
|------|------|
| `mul_kernel` | a * b |
| `div_kernel` | a / b |

### 퓨전
| 커널 | 연산 |
|------|------|
| `silu_mul_kernel` | silu(gate) * up |

### 스칼라
| 커널 | 연산 |
|------|------|
| `add_scalar_kernel` | 상수 채우기 |
| `eltwise_add_scalar_kernel` | x + scalar |
| `eltwise_mul_scalar_kernel` | x * scalar |

### 리덕션
| 커널 | 연산 |
|------|------|
| `mean_last_dim_kernel` | 마지막 차원 평균 |

## 브로드캐스트

**파일**: `elementwise_broadcast.metal`

스트라이드 기반 브로드캐스트 이항 연산. 별도 expand+elementwise 디스패치를 제거합니다.

| 커널 | 연산 |
|------|------|
| `mul_broadcast_kernel` | 브로드캐스트 a * b |
| `add_broadcast_kernel` | 브로드캐스트 a + b |
| `div_broadcast_kernel` | 브로드캐스트 a / b |

**파라미터**: `BroadcastBinaryParams { ndim, total, a_strides[6], b_strides[6], out_shape[6] }`

스트라이드 0은 해당 차원에서 브로드캐스트를 의미합니다.

## 텐서 연산

**파일**: `tensor_ops.metal`

| 커널 | 설명 |
|------|------|
| `transpose_kernel` | 두 차원 교환 |
| `cat_2_kernel` | 축을 따라 2개 텐서 연결 |
| `slice_kernel` | 부분 텐서 추출 |
| `expand_kernel` | 브로드캐스트 복사 |

## 기타 커널

| 파일 | 커널 | 설명 |
|------|------|------|
| `embedding.metal` | `embedding_kernel` | 토큰 임베딩 룩업 |
| `rope.metal` | `rope_kernel` | 회전 위치 임베딩 |
| `index_copy.metal` | `index_copy_kernel` | KV 캐시 위치 갱신 |
| `conv_bn_relu.metal` | `conv2d_kernel` | BN+ReLU 옵션 Conv2d |
| `add_relu.metal` | `add_kernel`, `add_relu_kernel` | ReLU 옵션 덧셈 |
| `pool.metal` | `max_pool2d_kernel`, `adaptive_avg_pool2d_kernel` | 풀링 |
