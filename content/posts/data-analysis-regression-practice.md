---
title: "[실전 연습] 회귀 문제 시뮬레이션 — California Housing에 결측치/이상치 주입 후 풀이"
date: 2026-04-22
tags: ["데이터분석", "회귀", "시험대비", "실전연습"]
categories: ["ML/AI"]
summary: "실제 California Housing 데이터에 인공적으로 결측치, 이상치, 노이즈를 주입한 후 단계별로 해결하는 회귀 시험 시뮬레이션. 데이터 진단부터 최종 모델까지 전체 풀이 과정 포함."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [직접 데이터로 예측해 보기]({{< relref "learn" >}}#data) · 3/5

잘 정리된 데이터만 보면 문제가 생겼을 때 어디부터 고쳐야 할지 알기 어렵다. 이 실습은 주택 데이터에 일부러 빈 값과 이상한 값을 넣고 하나씩 해결한다. 처리 전후를 같은 평가 조건에서 비교하는 이유를 따라가 보자.

**먼저 알아둘 말**

- [데이터셋·라벨]({{< relref "ai-glossary" >}}#dataset): 데이터셋은 학습이나 평가에 쓰는 예시의 모음이다. 라벨은 예시에 붙인 목표 값이나 분류다.
- [회귀·분류]({{< relref "ai-glossary" >}}#regression): 회귀는 가격처럼 수치를 예측하고, 분류는 스팸 여부처럼 범주를 예측한다. 목표의 종류에 따라 평가 방식이 달라진다.
- [검증 데이터·교차검증]({{< relref "ai-glossary" >}}#cv): 검증 데이터는 학습 중 선택한 방법의 성능을 확인하는 데이터다. 교차검증은 나누는 구간을 바꿔 여러 번 학습과 평가를 반복한다.

**이번 글의 목표:** 문제 있는 데이터가 어떻게 정리되는지 따라가기.

**이 학습 경로의 앞 글:** [\[시험 대비\] 데이터 분석 회귀 문제 완전 가이드 — 전처리, 라이브러리, 모델, 평가]({{< relref "posts/data-analysis-regression-guide" >}}).

**본문에서 먼저 볼 순서:** **Step 1: 데이터 로드 + 인공 문제 주입** → **Step 2: EDA — 문제 진단**.

첫 독서에서는 데이터가 어떻게 바뀌고 무엇으로 평가하는지 읽자. 코드를 실행하려면 Python과 필요한 라이브러리를 익힌 뒤 한 단계씩 따라가면 된다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** Python의 변수·리스트·함수, 데이터 표와 결측치, 학습/검증 분리, 평가 지표.

이 글의 출발점은 [AI 기초 설명]({{< relref "ai-basics" >}})과 위의 실습·시스템 개념이다. 학습 경로의 앞 글에서 입력·처리·출력의 흐름을 확인한 뒤 읽으면 된다.

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 시나리오

> **시험 문제**: California Housing 데이터를 받아서 주택 가격(`MedHouseVal`)을 예측하라.
> 단, 데이터에는 결측치, 이상치, 노이즈가 포함되어 있다.
> RMSE 0.55 이하를 목표로 한다.

이 실습에서는 실제 데이터에 일부러 문제를 넣고 진단부터 해결까지 단계별로 따라간다.

깨끗한 데이터에 문제를 넣는 이유는 처리 전후를 비교하기 위해서다. 결측치와 이상치가 어떤 모습으로 나타나는지 확인하고, 이를 처리한 뒤 같은 검증 조건에서 결과를 본다. 이 순서를 따라가면 점수가 올랐다는 사실뿐 아니라 어느 처리가 도움이 됐는지도 설명할 수 있다. 아래 단계별 코드는 그 비교 과정을 직접 해 보는 데 초점을 맞춘다.

---

## Step 0: 환경 준비

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
```

---

## Step 1: 데이터 로드 + 인공 문제 주입

실전 시험을 연습할 수 있도록 깨끗한 데이터에 **결측치, 이상치, 노이즈**를 넣는다.

```python
# 원본 로드
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
print(f"원본 shape: {df.shape}")
print(df.columns.tolist())
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
#  'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']

# === 인공 문제 주입 ===

# (1) 결측치 주입 (3개 컬럼)
np.random.seed(42)
n = len(df)

for col, ratio in [('HouseAge', 0.10), ('AveRooms', 0.05), ('Population', 0.15)]:
    missing_idx = np.random.choice(n, size=int(n * ratio), replace=False)
    df.loc[missing_idx, col] = np.nan

# (2) 이상치 주입 (Population에 극단값 100개)
outlier_idx = np.random.choice(n, size=100, replace=False)
df.loc[outlier_idx, 'Population'] = df['Population'].max() * 10

# (3) AveOccup에 노이즈 추가 (10% 샘플)
noise_idx = np.random.choice(n, size=int(n * 0.1), replace=False)
df.loc[noise_idx, 'AveOccup'] += np.random.normal(0, 50, size=len(noise_idx))

print("\n=== 주입 후 결측치 ===")
print(df.isnull().sum())
```

**출력 예시**:
```
=== 주입 후 결측치 ===
MedInc            0
HouseAge       2064  (10%)
AveRooms       1032  (5%)
AveBedrms         0
Population     3096  (15%)
AveOccup          0
Latitude          0
Longitude         0
MedHouseVal       0
```

---

## Step 2: EDA — 문제 진단

### 2.1 기본 정보

```python
print(df.info())
print(df.describe())
```

### 2.2 결측치 시각화

```python
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

# 비율
missing_ratio = df.isnull().sum() / len(df) * 100
print(missing_ratio[missing_ratio > 0].sort_values(ascending=False))
```

### 2.3 타겟 분포

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['MedHouseVal'], kde=True, ax=ax[0])
ax[0].set_title('타겟 분포 (원본)')
sns.boxplot(y=df['MedHouseVal'], ax=ax[1])
ax[1].set_title('타겟 박스플롯')
plt.tight_layout()
plt.show()

print(f"왜도(skewness): {df['MedHouseVal'].skew():.3f}")
# 양의 왜도 — 오른쪽으로 길게 늘어진 분포
```

### 2.4 이상치 탐지

```python
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude']

for i, col in enumerate(features):
    sns.boxplot(y=df[col], ax=axes[i//4, i%4])
    axes[i//4, i%4].set_title(col)
plt.tight_layout()
plt.show()
```

**관찰**:
- `Population`에 극단적 이상치 발견 → 주입한 것
- `AveOccup`도 의심스러운 큰 값 존재
- `MedHouseVal` (타겟)도 5에서 잘림 (clipping)

### 2.5 상관관계

```python
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# 타겟과의 상관계수
print(corr['MedHouseVal'].abs().sort_values(ascending=False))
# MedHouseVal    1.000
# MedInc         0.688  ← 가장 강한 양의 상관
# AveRooms       0.151
# Latitude       0.144
# ...
```

---

## Step 3: 결측치 처리

### 전략 결정

| 컬럼 | 결측 비율 | 전략 | 이유 |
|------|---------|------|------|
| HouseAge | 10% | **중앙값 대체** | 분포가 비슷, 비율 적당 |
| AveRooms | 5% | **중앙값 대체** | 비율 낮음 |
| Population | 15% | **중앙값 + 플래그** | 비율 높고 이상치 있음 → 정보 가치 |

### 코드

```python
from sklearn.impute import SimpleImputer

# (1) Population: 결측 플래그 먼저 생성
df['Population_was_missing'] = df['Population'].isnull().astype(int)

# (2) 모든 결측치를 중앙값으로 대체
num_cols_with_na = ['HouseAge', 'AveRooms', 'Population']
imputer = SimpleImputer(strategy='median')
df[num_cols_with_na] = imputer.fit_transform(df[num_cols_with_na])

# 확인
print(df.isnull().sum().sum())  # 0
```

### 비교 — 평균 vs 중앙값

```python
# 만약 평균을 썼다면? 이상치 때문에 왜곡됨
print(f"Population 평균: {df['Population'].mean():.0f}")
print(f"Population 중앙값: {df['Population'].median():.0f}")
# 이상치 영향으로 평균이 훨씬 큼 → 중앙값 선택 정당
```

---

## Step 4: 이상치 처리

### 4.1 IQR 방법으로 탐지

```python
def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    return lower, upper, n_outliers

for col in ['Population', 'AveOccup', 'AveRooms']:
    lower, upper, n = detect_outliers_iqr(df, col)
    print(f"{col}: lower={lower:.1f}, upper={upper:.1f}, outliers={n}")
```

### 4.2 Capping (Winsorizing)

이상치를 제거하면 정보 손실 → **경계값으로 눌러주기(capping)** 권장.

```python
def cap_outliers(df, col, k=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    df[col] = df[col].clip(lower, upper)
    return df

for col in ['Population', 'AveOccup', 'AveRooms', 'AveBedrms']:
    df = cap_outliers(df, col)

# 처리 후 확인
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, col in enumerate(['Population', 'AveOccup', 'AveRooms', 'AveBedrms']):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.show()
```

---

## Step 5: 특성 공학

### 5.1 새로운 특성 생성

```python
# 인구 밀도
df['PopPerHousehold'] = df['Population'] / df['AveOccup']

# 침실 비율
df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']

# 위치 클러스터링 (간단)
df['LatLong'] = df['Latitude'] + df['Longitude']
df['IsCoastal'] = (df['Longitude'] < -120).astype(int)  # 해안 지역

print(df.columns.tolist())
```

### 5.2 타겟 변환 (왜도 완화)

```python
# 원본 분포 확인
print(f"원본 왜도: {df['MedHouseVal'].skew():.3f}")

# log 변환 (양수 데이터에 적용)
df['MedHouseVal_log'] = np.log1p(df['MedHouseVal'])
print(f"log 왜도: {df['MedHouseVal_log'].skew():.3f}")

# 시각화
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['MedHouseVal'], kde=True, ax=ax[0])
ax[0].set_title('원본')
sns.histplot(df['MedHouseVal_log'], kde=True, ax=ax[1])
ax[1].set_title('log 변환 후')
plt.tight_layout()
plt.show()

# 왜도가 크게 줄지 않으면 원본 사용
```

---

## Step 6: 데이터 분할 + 스케일링

```python
y = df['MedHouseVal']
X = df.drop(['MedHouseVal', 'MedHouseVal_log'], axis=1)

# 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 스케일링 (선형 모델용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Step 7: 모델 학습 + 비교

### 7.1 베이스라인 모델 3개

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

models = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15,
                                          random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                            random_state=42, n_jobs=-1)
}

results = []
for name, model in models.items():
    if name == 'Ridge':
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    results.append({'model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2})
    print(f"{name:15s} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
```

**예상 결과**:
```
Ridge           | RMSE: 0.7320 | MAE: 0.5320 | R²: 0.5876
RandomForest    | RMSE: 0.5050 | MAE: 0.3290 | R²: 0.8045
XGBoost         | RMSE: 0.4720 | MAE: 0.3110 | R²: 0.8290
```

### 7.2 교차 검증

```python
from sklearn.model_selection import cross_val_score, KFold

best_model = XGBRegressor(n_estimators=300, max_depth=6,
                          learning_rate=0.1, random_state=42, n_jobs=-1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=kf,
                            scoring='neg_root_mean_squared_error', n_jobs=-1)

print(f"CV RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### 7.3 특성 중요도

```python
best_model.fit(X_train, y_train)

importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importances, x='importance', y='feature')
plt.title('Feature Importance (XGBoost)')
plt.show()
```

---

## Step 8: 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(200, 800),
    'max_depth': randint(4, 12),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV RMSE: {-search.best_score_:.4f}")

# 최종 평가
final_pred = search.best_estimator_.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, final_pred)):.4f}")
print(f"Test R²:   {r2_score(y_test, final_pred):.4f}")
```

---

## Step 9: 결과 검증

### 9.1 잔차 분석

```python
residuals = y_test - final_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 잔차 분포
sns.histplot(residuals, kde=True, ax=axes[0])
axes[0].set_title('Residual Distribution')
axes[0].axvline(0, color='r', linestyle='--')

# 예측값 vs 실제값
axes[1].scatter(y_test, final_pred, alpha=0.3)
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title('Predicted vs Actual')
plt.tight_layout()
plt.show()
```

### 9.2 도메인 검증

```python
# 음수 예측 확인 (집값은 양수여야 함)
print(f"음수 예측 수: {(final_pred < 0).sum()}")

# 비현실적 큰 값 확인
print(f"5 초과 예측 수: {(final_pred > 5).sum()}")

# 타겟이 5에서 clipping되었으므로 5로 cap
final_pred = np.clip(final_pred, 0, 5)
```

---

## Step 10: 최종 답안 (시험 제출 형식)

```python
# === 최소 코드로 다시 ===

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 로드 (시험에서는 csv로 주어질 것)
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

# (인공 문제 주입은 시험에서 이미 되어 있음)
np.random.seed(42)
n = len(df)
for col, ratio in [('HouseAge', 0.10), ('AveRooms', 0.05), ('Population', 0.15)]:
    idx = np.random.choice(n, int(n * ratio), replace=False)
    df.loc[idx, col] = np.nan
outlier_idx = np.random.choice(n, 100, replace=False)
df.loc[outlier_idx, 'Population'] = df['Population'].max() * 10

# 2. 결측치 처리
num_cols_with_na = ['HouseAge', 'AveRooms', 'Population']
df['Population_missing'] = df['Population'].isnull().astype(int)
df[num_cols_with_na] = SimpleImputer(strategy='median').fit_transform(df[num_cols_with_na])

# 3. 이상치 처리 (capping)
for col in ['Population', 'AveOccup', 'AveRooms', 'AveBedrms']:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

# 4. 특성 공학
df['PopPerHousehold'] = df['Population'] / df['AveOccup']
df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']
df['IsCoastal'] = (df['Longitude'] < -120).astype(int)

# 5. 분할
y = df['MedHouseVal']
X = df.drop('MedHouseVal', axis=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 모델
model = XGBRegressor(
    n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, random_state=42, n_jobs=-1
)
model.fit(X_tr, y_tr)

# 7. 평가
pred = np.clip(model.predict(X_te), 0, 5)
print(f"RMSE: {np.sqrt(mean_squared_error(y_te, pred)):.4f}")
print(f"R²:   {r2_score(y_te, pred):.4f}")
# 목표 RMSE < 0.55 달성!
```

---

## 풀이 요약

| 단계 | 핵심 의사결정 | 선택 이유 |
|------|------------|---------|
| **결측치** | 중앙값 대체 + Population 플래그 | 이상치 영향 적은 중앙값, 결측 자체가 정보 |
| **이상치** | IQR Capping | 정보 손실 없이 영향력만 제거 |
| **특성 공학** | 비율/밀도 특성 추가 | 도메인 지식 반영 |
| **모델** | XGBoost | 결측 자동 처리, 비선형 잘 잡음 |
| **튜닝** | RandomizedSearchCV | GridSearch보다 빠름 |
| **검증** | 5-Fold CV | 과적합 방지 |

---

## 시험에서 자주 묻는 변형

### Q1. "결측치를 평균으로 채우라"
```python
df.fillna(df.mean(), inplace=True)
# 단, 이상치가 있으면 평균이 왜곡됨 — 시험 답안에서는 둘 다 시도해보자
```

### Q2. "이상치를 제거하라" (Capping이 아닌 제거)
```python
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_clean = df[(df['col'] >= Q1 - 1.5*IQR) & (df['col'] <= Q3 + 1.5*IQR)]
# 행 수 감소 → train/test 분할 후 처리 추천
```

### Q3. "표준화 후 모델링"
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)  # transform만!
# 트리 모델은 불필요, 선형/SVM/KNN/NN은 필수
```

### Q4. "교차검증으로 평가하라"
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5,
                         scoring='neg_root_mean_squared_error')
print(f"CV RMSE: {-scores.mean():.4f}")
```

---

## 관련 블로그 포스트

- [회귀 문제 완전 가이드](data-analysis-regression-guide.md) — 이론 종합
- [분류 실전 연습](data-analysis-classification-practice.md) — 분류 시뮬레이션
- [회귀/분류 데이터셋 총정리](data-analysis-datasets-guide.md) — 데이터 출처

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 문제 있는 데이터가 어떻게 정리되는지 따라가기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[시험 대비\] 데이터 분석 분류 문제 완전 가이드 — 전처리, 라이브러리, 모델, 평가]({{< relref "posts/data-analysis-classification-guide" >}}) — 분류와 불균형에서 평가 기준이 달라지는 이유 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
