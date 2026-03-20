---
title: "[코드 분석] Mamba — state-spaces/mamba 소스코드 완전 분석"
date: 2026-03-20
tags: ["코드분석", "Mamba", "SSM", "CUDA", "Triton"]
categories: ["ML/AI"]
summary: "Mamba-1/2/3의 공식 구현체(state-spaces/mamba)를 소스코드 수준에서 분석한다. 논문의 수식이 어떻게 CUDA/Triton 커널로 구현되는지, Selective SSM의 parallel scan부터 Mamba-3의 exponential-trapezoidal 이산화까지 핵심 코드를 추적한다."
math: true
toc: true
draft: false
---

## 레포 정보

- **GitHub**: [state-spaces/mamba](https://github.com/state-spaces/mamba)
- **버전**: v2.3.1
- **저자**: Albert Gu, Tri Dao et al.
- **라이선스**: Apache 2.0

---

## 1. 디렉토리 구조

```
mamba_ssm/
├── modules/                  # PyTorch nn.Module 정의
│   ├── mamba_simple.py       # Mamba-1 블록
│   ├── mamba2.py             # Mamba-2 블록 (SSD)
│   ├── mamba3.py             # Mamba-3 블록
│   ├── ssd_minimal.py        # SSD 최소 참조 구현 (논문 Algorithm 1)
│   ├── block.py              # Block 래퍼 (LN + Mixer + Residual)
│   ├── mha.py                # Multi-Head Attention (하이브리드용)
│   └── mlp.py                # GatedMLP
├── ops/                      # GPU 커널
│   ├── selective_scan_interface.py  # Mamba-1 CUDA 인터페이스
│   ├── triton/               # Mamba-2 Triton 커널
│   │   ├── ssd_combined.py   # SSD 통합 fwd/bwd
│   │   ├── ssd_chunk_scan.py # 청크 스캔
│   │   └── mamba3/           # Mamba-3 전용 커널
│   │       ├── mamba3_siso_fwd.py    # SISO forward
│   │       ├── angle_dt.py           # 복소수 SSM용 각도 누적
│   │       └── ...
│   ├── tilelang/mamba3/      # MIMO TileLang 커널
│   └── cute/mamba3/          # CuTE DSL 커널 (H100 최적화)
├── models/
│   ├── config_mamba.py       # MambaConfig
│   └── mixer_seq_simple.py   # MambaLMHeadModel
└── csrc/selective_scan/      # Mamba-1 CUDA C++ 커널
    ├── selective_scan_fwd_kernel.cuh
    ├── selective_scan_bwd_kernel.cuh
    └── selective_scan_common.h    # SSMScanOp (parallel scan)
```

**세대별 커널 기술 스택:**
- Mamba-1: **CUDA C++** (CUB parallel scan)
- Mamba-2: **Triton** (chunk-wise SSD)
- Mamba-3: **Triton** (SISO) + **TileLang** (MIMO) + **CuTE DSL** (H100 step)

---

## 2. Mamba-1: Selective SSM 구현

### 2.1 A 파라미터 초기화

**파일**: `modules/mamba_simple.py`

논문의 $A$ 행렬은 S4D 초기화를 사용한다. 코드에서는 **log-space**로 저장하여 항상 음수를 보장:

```python
# mamba_simple.py, line 103-111
A = repeat(
    torch.arange(1, self.d_state + 1, dtype=torch.float32),
    "n -> d n", d=self.d_inner,
).contiguous()
A_log = torch.log(A)  # log-space 저장
self.A_log = nn.Parameter(A_log)

# forward에서 복원:
A = -torch.exp(self.A_log.float())  # 항상 음수
```

$A$의 고유값이 $-1, -2, \ldots, -N$으로 초기화된다. `exp` 후 부호를 뒤집어 안정성을 보장하는 트릭이다.

### 2.2 Delta(dt) 초기화

```python
# mamba_simple.py, line 82-101
dt = torch.exp(
    torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
    + math.log(dt_min)
)
inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
self.dt_proj.bias.copy_(inv_dt)
```

$\Delta$는 `softplus`를 거쳐 양수가 되므로, 바이어스를 **inverse softplus**로 초기화하여 원하는 범위 `[dt_min, dt_max]` (기본 0.001~0.1)에서 시작한다.

### 2.3 B, C의 입력 의존적 생성

```python
# mamba_simple.py, line 77-79
self.x_proj = nn.Linear(
    self.d_inner,
    self.dt_rank + self.d_state * 2,  # [dt_rank, d_state, d_state]
    bias=False
)
```

하나의 프로젝션에서 `dt`(저랭크), `B`, `C`를 동시에 추출한다. 이것이 논문의 "selection mechanism" — 입력 $x_t$에서 $B(t)$, $C(t)$, $\Delta(t)$를 모두 생성한다.

### 2.4 Step 함수: 논문 수식 → 코드

논문의 이산화된 SSM:

$$h_t = \exp(\Delta_t \cdot A) \cdot h_{t-1} + \Delta_t \cdot B_t \cdot x_t$$

$$y_t = C_t \cdot h_t + D \cdot x_t$$

코드:

```python
# mamba_simple.py, line 208-253
dt = F.softplus(dt + self.dt_proj.bias)               # Delta
dA = torch.exp(einsum("bd,dn->bdn", dt, A))           # exp(Delta * A)
dB = einsum("bd,bn->bdn", dt, B)                      # Delta * B
ssm_state = ssm_state * dA + x.unsqueeze(-1) * dB     # h = dA*h + dB*x
y = einsum("bdn,bn->bd", ssm_state, C)                # y = C*h
y = y + self.D * x                                     # + D*x (skip)
y = y * self.act(z)                                    # SiLU gating
```

**수식과 코드의 1:1 대응**이 명확하다.

### 2.5 CUDA Parallel Scan 커널

**파일**: `csrc/selective_scan/selective_scan_common.h`

SSM 재귀 $h_t = a_t \cdot h_{t-1} + b_t$는 **결합 연산자**를 가진 prefix scan으로 변환된다:

```cpp
// selective_scan_common.h, line 141-145
struct SSMScanOp<float> {
    __device__ float2 operator()(
        const float2 &ab0, const float2 &ab1
    ) const {
        return make_float2(
            ab1.x * ab0.x,              // a = a1 * a0
            ab1.x * ab0.y + ab1.y       // b = a1 * b0 + b1
        );
    }
};
```

`float2.x` = 감쇠 계수 $a$, `float2.y` = 입력 기여 $b$. 이 결합 연산자로 CUB의 `BlockScan`을 호출하여 GPU에서 병렬 prefix scan을 수행한다.

**Forward 커널** (`selective_scan_fwd_kernel.cuh`):

```cpp
// line 221: 이산화
thread_data[i] = make_float2(
    exp2f(delta * A * LOG2E),    // exp(dt * A) — exp2로 최적화
    B * delta * u                 // dt * B * x
);

// line 251-252: CUB parallel prefix scan
BlockScanT(smem).InclusiveScan(
    thread_data, thread_data, SSMScanOp()
);

// line 266: 출력 계산
out += thread_data[i].y * C;     // y = scan_result * C
```

`expf` 대신 `exp2f`를 사용하고 $A$에 $\log_2(e)$를 미리 곱해두는 최적화가 인상적이다.

---

## 3. Mamba-2: State Space Duality (SSD)

### 3.1 최소 참조 구현

**파일**: `modules/ssd_minimal.py` — 논문의 Algorithm 1 그대로.

핵심 아이디어: SSM 재귀를 **청크 내부**(행렬곱, 어텐션 유사)와 **청크 간**(상태 전달, RNN 유사)으로 분해한다.

```python
# ssd_minimal.py, line 55 — 청크 내부 (causal attention)
L = torch.exp(segsum(A))  # 하삼각 감쇠 행렬
Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

# line 59-60 — 청크 경계 상태 계산
decay_states = torch.exp(A_cumsum[:,:,:,-1:] - A_cumsum)
states = einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

# line 64-69 — 청크 간 상태 전달 (재귀)
decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:,:,:,-1], (1,0))))
new_states = einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
```

`segsum` 함수가 핵심 — 누적합을 계산하고 하삼각 마스크를 적용하여 **인과적 감쇠 행렬**을 만든다.

### 3.2 Mamba-2 모듈: Mamba-1과의 구조적 차이

**파일**: `modules/mamba2.py`

```python
# line 85: Multi-head 구조
self.nheads = d_ssm // self.headdim  # headdim=64 기본

# line 133: A는 head별 (Mamba-1은 dim별)
A = torch.empty(self.nheads)

# line 96: 단일 프로젝션으로 모든 파라미터 생성
d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads
# → [z, x, B, C, dt] 동시 추출
```

| 항목 | Mamba-1 | Mamba-2 |
|------|---------|---------|
| A 크기 | `(d_inner, d_state)` | `(nheads,)` — **대폭 축소** |
| Multi-head | 없음 | 있음 (`headdim=64`) |
| B/C 공유 | 없음 | GQA 스타일 (`ngroups`) |
| 게이팅 | `SiLU(z) * y` | `RMSNorm(y, z)` |

---

## 4. Mamba-3: 세 가지 혁신의 구현

**파일**: `modules/mamba3.py`

### 4.1 새로운 파라미터들

```python
# mamba3.py, line 77-78
d_in_proj = (2*d_inner
    + 2*d_state*num_bc_heads*mimo_rank  # B, C (MIMO 확장)
    + 3*nheads                           # dd_dt, dd_A, trap
    + num_rope_angles)                   # angle (복소수 SSM)
```

Mamba-2의 `[z, x, B, C, dt]`에서 3개가 추가:
- `dd_A`: **입력 의존적 감쇠** (Mamba-2에서는 고정)
- `trap`: **사다리꼴 보간 계수** $\lambda_t$
- `angle`: **회전 각도** (복소수 SSM용)

### 4.2 Exponential-Trapezoidal 이산화

**파일**: `ops/triton/mamba3/mamba3_siso_fwd.py`

논문의 수식:

$$h_t = \alpha_t h_{t-1} + \beta_t B_{t-1} x_{t-1} + \gamma_t B_t x_t$$

커널 코드:

```python
# mamba3_siso_fwd.py, line 278-286
trap = sigmoid_approx(trap)
shifted_gamma = dt_shifted * (1 - trap_shifted)  # beta = (1-lambda)*dt*exp(A*dt)
gamma = dt * trap                                  # gamma = lambda * dt
scale = shifted_gamma + gamma                      # 전체 스케일
```

- `trap`($\lambda_t$)은 **sigmoid**로 `[0,1]` 범위 제한
- $\lambda_t = 1$이면 Mamba-2의 Euler 이산화 복원
- $\lambda_t = 0.5$이면 고전적 사다리꼴 규칙

**Conv1d가 제거된 이유**: Exp-Trapezoidal이 **상태 입력 $B_t x_t$에 대한 width-2 컨볼루션**을 암묵적으로 수행하므로, 별도의 `Conv1d`가 불필요해졌다.

### 4.3 복소수 SSM: RoPE Trick

**파일**: `ops/triton/mamba3/angle_dt.py`

논문의 복소수 SSM은 **데이터 의존적 RoPE**로 구현된다:

```python
# angle_dt.py, line 93-108
angle_vals = tanh_approx(angle_vals) * PI    # 각도를 [-pi, pi]로 제한
vals = angle_vals * dt_vals[:, None]          # angle * dt
chunk_cumsum = tl.cumsum(vals, axis=0)        # 누적 각도
out_vals = chunk_cumsum + state[None, :]      # 이전 청크 상태 합산
out_vals = out_vals - TWO_PI * tl.floor(out_vals / TWO_PI)  # mod 2*pi
```

이 누적 각도를 B/C(=K/Q)에 회전 임베딩으로 적용:

```python
# mamba3_siso_fwd.py, line 308-330
cos_block = cos_approx(angle_block)
sin_block = sin_approx(angle_block)
# 표준 rotary embedding
k0, k1 = tl.split(tl.reshape(k_pre_block, [CHUNK_SIZE, HEADDIM_QK//2, 2]))
ko0 = k0 * cos_block - k1 * sin_block
ko1 = k0 * sin_block + k1 * cos_block
```

**핵심 통찰**: 복소수 고유값 $A + i\theta$를 실수 연산으로 구현하려면, **2x2 회전 행렬**을 블록 대각으로 쌓으면 된다. 이것이 RoPE와 동일한 구조다. 단, 표준 RoPE의 고정 주파수 대신 **데이터 의존적 각도** $\Delta_t \theta_t$를 사용한다.

### 4.4 MIMO 구현

```python
# mamba3.py, line 100-108
if self.is_mimo:
    mimo_x_init_weights = torch.ones(nheads, mimo_rank, headdim) / mimo_rank
    self.mimo_x = nn.Parameter(mimo_x_init_weights)  # 입력 투영
    self.mimo_z = nn.Parameter(...)                    # 게이트 투영
    self.mimo_o = nn.Parameter(...)                    # 출력 투영
```

MIMO는 각 헤드를 `mimo_rank`(기본 4)개의 병렬 SSM 채널로 확장한다. B, C도 rank 차원을 가진다: `B = (batch, seqlen, mimo_rank, ngroups, d_state)`.

SISO 커널은 Triton, MIMO 커널은 **TileLang**(CUTLASS 스타일)로 구현된다 — 행렬곱 최적화가 MIMO에서 특히 중요하기 때문이다.

---

## 5. 모델 전체 구조

**파일**: `models/mixer_seq_simple.py`

```python
MambaLMHeadModel = Embedding + [Block × N] + LM_Head

Block = Add → LayerNorm → Mixer(Mamba/Mamba2/Mamba3)
```

설정:

```python
# config_mamba.py
@dataclass
class MambaConfig:
    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = {}          # Mixer 설정
    attn_layer_idx: list = []   # 하이브리드: 어텐션 레이어 위치
```

`attn_layer_idx`로 특정 레이어를 Attention으로 교체하여 **하이브리드 아키텍처**를 구성할 수 있다.

---

## 6. 세대별 진화 요약

| 항목 | Mamba-1 | Mamba-2 | Mamba-3 |
|------|---------|---------|---------|
| A | dim별, 고정 | head별, 고정 | head별, **입력 의존적** |
| B, C | 입력 의존적 | 입력 의존적 | 입력 의존적 + **RMSNorm + bias** |
| d\_state | 16 | 128 | 128 |
| 이산화 | Exp-Euler (softplus dt) | Exp-Euler | **Exp-Trapezoidal** (+ sigmoid trap) |
| 복소수 | 선택적 (complex A) | 없음 | **RoPE 기반** 누적 각도 |
| Multi-head | 없음 | 있음 (headdim=64) | 있음 + **MIMO** (rank=4) |
| Conv1d | 있음 (d\_conv=4) | 있음 | **제거** |
| 게이팅 | SiLU(z) * y | RMSNorm(y, z) | SiLU(z) * y 또는 RMSNorm |
| GPU 커널 | CUDA C++ | Triton | Triton + TileLang + CuTE |

---

## 7. 핵심 코드 파일 요약

| 목적 | 파일 경로 |
|------|----------|
| Mamba-1 모듈 | `modules/mamba_simple.py` |
| Mamba-2 모듈 | `modules/mamba2.py` |
| Mamba-3 모듈 | `modules/mamba3.py` |
| SSD 최소 구현 | `modules/ssd_minimal.py` |
| Mamba-1 CUDA 커널 | `csrc/selective_scan/selective_scan_fwd_kernel.cuh` |
| Parallel Scan 연산자 | `csrc/selective_scan/selective_scan_common.h` |
| Mamba-2 Triton SSD | `ops/triton/ssd_combined.py` |
| Mamba-3 SISO 커널 | `ops/triton/mamba3/mamba3_siso_fwd.py` |
| 복소수 SSM 각도 커널 | `ops/triton/mamba3/angle_dt.py` |
| Mamba-3 CuTE Step | `ops/cute/mamba3/mamba3_step_fn.py` |
| MIMO 커널 (TileLang) | `ops/tilelang/mamba3/mamba3_mimo.py` |
| 모델/설정 | `models/mixer_seq_simple.py`, `models/config_mamba.py` |

---

## 8. 소감

Mamba 코드베이스는 **논문과 코드의 대응이 매우 명확**하다. 특히:

- `ssd_minimal.py`는 논문의 Algorithm 1을 그대로 옮겨놓은 교과서적 참조 구현
- 실제 성능을 위한 최적화(`exp2f` 사용, CUB parallel scan, Triton chunk-wise 계산)가 별도 커널 파일에 깔끔하게 분리
- Mamba-3의 세 가지 혁신(Exp-Trapezoidal, 복소수 SSM, MIMO)이 각각 독립적인 코드 경로를 가지면서도 하나의 모듈(`mamba3.py`)에서 통합
- 세대별로 커널 기술이 CUDA C++ → Triton → TileLang/CuTE로 진화한 것이 GPU 프로그래밍 생태계의 발전을 반영
