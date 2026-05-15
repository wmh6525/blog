---
title: "[논문 리뷰] CacheBlend (EuroSys 2025 Best Paper) — 비접두사 KV 캐시 재사용으로 RAG 5배 가속"
date: 2026-04-03
tags: ["논문리뷰", "RAG", "LLM", "최적화", "KV캐시"]
categories: ["ML/AI"]
summary: "CacheBlend 논문 상세 리뷰. 접두사가 아닌 위치의 KV 캐시도 재사용 가능하게 만든 선택적 재계산 기법. EuroSys 2025 Best Paper. TTFT 3.3배↓, 처리량 5배↑."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion
- **저자**: Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray 외 (University of Chicago)
- **학회**: **ACM EuroSys 2025 Best Paper Award**
- **코드**: [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache) (vLLM 통합)
- **arXiv**: 2405.16444

---

## 1. 문제: 접두사 캐싱은 RAG에서 작동하지 않는다

### Prefix Caching의 한계

기존 KV 캐시 재사용(prefix caching)은 **정확한 접두사 일치**에서만 작동한다.

```
요청 A: [시스템 프롬프트] [청크 1] [청크 2] [청크 3] [질의 A]
요청 B: [시스템 프롬프트] [청크 4] [청크 2] [청크 5] [질의 B]
                          ↑                              
                     여기서 달라짐 → 청크 2의 KV 캐시 재사용 불가!
```

청크 2가 두 요청에 모두 포함되지만, **앞에 오는 청크가 다르므로** 청크 2의 KV 캐시는 재사용할 수 없다. 왜냐하면:

- 접두사 캐싱에서 계산된 청크 2의 KV 캐시는 **청크 1에 대한 cross-attention**이 포함
- 요청 B에서는 앞에 청크 4가 오므로, 청크 2는 **청크 4에 대해 어텐드**해야 함
- 따라서 KV 캐시가 완전히 다름

**결과**: RAG에서 prefix caching의 KV 캐시 히트율은 시스템 프롬프트 부분에만 해당하여, **거의 전체를 재계산**해야 한다.

---

## 2. 핵심 관찰: 어텐션은 희소하다

CacheBlend의 핵심 발견:

> 비접두사 청크에서, 대부분의 토큰(~85-90%)은 앞의 다른 청크에 대해 **매우 낮은 어텐션 점수**를 가진다. 이 토큰들의 KV 값은 독립적으로 계산해도 거의 동일하다.

즉, **소수의 토큰(~10-15%)만** 유의미한 cross-chunk attention을 가지며, 이 토큰들의 KV만 재계산하면 된다.

---

## 3. 선택적 재계산 메커니즘

### 3.1 KV Deviation 측정

각 토큰 $j$의 레이어 $i$에서의 KV 편차:

$$\delta_{kv}(j, i) = \lVert KV_i^{\text{isolated}}[j] - KV_i^{\text{full}}[j] \rVert_2$$

- $KV^{\text{isolated}}$: 독립적으로(다른 청크 없이) 계산된 KV 캐시
- $KV^{\text{full}}$: 전체 컨텍스트와 함께 계산된 KV 캐시

### 3.2 핵심 정리

> **높은 KV 편차를 가진 토큰을 재계산하면, 전체 어텐션 편차를 가장 크게 줄인다.**

따라서 **KV 편차가 가장 큰 상위 r%의 토큰만 재계산**하는 것이 최적 전략.

### 3.3 레이어 간 점진적 필터링

1. **레이어 1**: 모든 토큰의 KV 편차 계산 → 상위 $r_1$% 선택 (HKVD 토큰)
2. **레이어 2**: HKVD 토큰만 재계산 → 다시 상위 $r_2$% 선택 ($r_2 \lt  r_1$)
3. **이후 레이어**: 점진적으로 필터링

이것이 가능한 이유: 레이어 간 KV 편차의 **Spearman 상관관계가 높다** — 레이어 1에서 편차가 큰 토큰은 레이어 2에서도 편차가 크다.

### 3.4 RoPE 위치 보정

독립 계산된 KV 캐시는 다른 절대 위치를 가지므로, 회전 행렬 보정을 적용:

$$K'_{\text{corrected}} = R(\Delta pos) \cdot K_{\text{cached}}$$

RoPE가 상대 위치에만 의존하므로 이 보정은 **수학적으로 정확**하고 계산 비용 무시 가능.

---

## 4. 파이프라이닝: 재계산을 I/O 뒤에 숨긴다

**핵심 관찰**: 15% 토큰 재계산(~3ms/레이어)이 KV 캐시 SSD 로드(~16ms/레이어)보다 **빠르다**.

```
레이어 i:   [=== KV 캐시 로드 (I/O, 16ms) ===]
            [== 재계산 (GPU, 3ms) ==]  ← I/O 뒤에 완전히 숨겨짐

레이어 i+1: [=== KV 캐시 로드 (I/O, 16ms) ===]
            [== 재계산 (GPU, 3ms) ==]
```

I/O와 GPU 연산이 다른 하드웨어를 사용하므로 **완벽한 오버랩**. 재계산 지연이 사실상 0.

**부수 효과**: KV 캐시를 더 느리고 저렴한 SSD에 저장해도 성능 저하 없음.

### 재계산 비율 결정 공식

$$r\% = \max\left(\frac{T_{\text{load}}}{T_{\text{prefill}}}, r^{\ast}\%\right)$$

여기서 $r^{\ast}\% \approx 15\%$는 품질 유지를 위한 최소 비율.

---

## 5. 실험 결과

### 하드웨어

128GB RAM, 2× NVIDIA A40, 1TB NVMe SSD (4.8 GB/s)

### 5.1 TTFT 감소

| 비교 대상 | TTFT 감소 |
|---------|---------|
| vs 전체 재계산 | **2.2-3.3배** |
| vs 접두사 캐싱 | **3.4-6.1배** |

### 5.2 처리량 향상

| 비교 대상 | 처리량 향상 |
|---------|---------|
| vs 전체 재계산 | **2.8-5배** |
| vs 접두사 캐싱 | **3.3배** |

### 5.3 품질 (F1 Score)

| 비교 | F1 차이 |
|------|---------|
| CacheBlend vs 전체 재계산 | **-0.01 ~ -0.02** (거의 동일) |
| CacheBlend vs 전체 KV 재사용 (TurboRAG 방식) | **+0.10 ~ +0.20** (품질 개선) |

**핵심**: 전체 재계산과 거의 동일한 품질을 유지하면서, 전체 KV 재사용(Independent Attention)보다 **더 높은 품질**.

---

## 6. TurboRAG vs CacheBlend 비교

| 항목 | TurboRAG | CacheBlend |
|------|---------|-----------|
| 접근법 | 전체 KV 재사용 (재계산 0%) | 선택적 재계산 (~15%) |
| Cross-attention | **완전 제거** | **부분 복원** |
| Fine-tuning | **필수** | **불필요** |
| TTFT 감소 | 8.6배 (h2d 포함) | 2.2-3.3배 |
| 품질 | Naive RAG -1.4%p | Naive RAG **-0.02%p** |
| 스토리지 | 청크당 ~30MB | 청크당 ~30MB |
| 모델 호환 | Qwen2 (RoPE 모델) | 모든 Transformer |
| 프로덕션 통합 | 독립 구현 | **vLLM/LMCache 통합** |

**TurboRAG**: 더 빠르지만 Fine-tuning 필요, 품질 약간 하락
**CacheBlend**: 약간 느리지만 Fine-tuning 불필요, 품질 거의 동일, 프로덕션 준비 완료

---

## 7. vLLM/LMCache 통합

CacheBlend는 **LMCache** 프로젝트의 핵심 기능으로, vLLM v1에 통합되어 있다.

```bash
# 환경 변수 설정만으로 활성화
export LMCACHE_ENABLE_BLENDING="True"
export LMCACHE_BLEND_RECOMPUTE_RATIOS="0.15"
export LMCACHE_BLEND_SPECIAL_STR=" # # "  # 청크 구분자
```

**엔터프라이즈 채택**: Red Hat, IBM, Google Cloud, CoreWeave.

---

## 8. 후속 연구: CacheClip (Intel, 2025.10)

CacheBlend의 한계를 보완:
- **보조 소형 LLM**(0.5B)으로 재계산할 토큰 식별 → 프리필 **1.92배** 추가 가속
- NIAH 과제에서 CacheBlend 대비 **25.2%p** 성능 개선
- 8-토큰 윈도우 그룹화로 문맥 무결성 보존

---

## 9. 한계

1. **Transformer 전용**: Mamba, Griffin 등 비Transformer 아키텍처에 미적용
2. **프리필 단계만**: 디코딩 단계에서는 효과 없음
3. **최소 토큰 수 필요**: 짧은 청크에서는 블렌딩 오버헤드가 이득보다 클 수 있음
4. **15% 최소 비율**: 재계산 비율을 15% 이하로 줄이면 품질 저하
