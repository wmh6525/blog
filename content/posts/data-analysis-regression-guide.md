---
title: "[시험 대비] 데이터 분석 회귀 문제 완전 가이드 — 전처리, 라이브러리, 모델, 평가"
date: 2026-04-21
tags: ["데이터분석", "회귀", "시험대비", "머신러닝", "sklearn"]
categories: ["ML/AI"]
summary: "데이터 분석 시험의 회귀(Regression) 문제 대비. 결측치 처리, 이상치 탐지, 스케일링, 인코딩부터 Linear/Ridge/XGBoost/LightGBM까지. 모델별 하이퍼파라미터와 평가 지표 전반 정리."
math: true
toc: true
draft: false
---

## 회귀 문제의 표준 파이프라인

```
데이터 로드 → EDA → 결측치 처리 → 이상치 처리 → 인코딩 → 스케일링
  → 특성 공학 → 모델 학습 → 교차검증 → 하이퍼파라미터 튜닝 → 평가 → 예측
```

---

## 1. 필수 라이브러리

```python
# 데이터 처리
import pandas as pd
import numpy as np

# 시각화 (EDA용)
import matplotlib.pyplot as plt
import seaborn as sns

# 전처리
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PowerTransformer, QuantileTransformer
)

# 모델 (선형 계열)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, BayesianRidge
)

# 모델 (트리 계열)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)

# 부스팅 라이브러리 (대회/실무 표준)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 기타
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# 평가
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score
)

# 검증
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV
)

# 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```

---

## 2. EDA (탐색적 데이터 분석)

### 2.1 데이터 파악

```python
df.head()              # 상위 5개
df.info()              # 타입, 결측 확인
df.describe()          # 수치형 통계
df.describe(include='O')  # 범주형 통계
df.shape               # (행, 열)
df.dtypes              # 컬럼 타입
df.nunique()           # 고유값 수
```

### 2.2 결측치 확인

```python
df.isnull().sum()                    # 컬럼별 결측치 수
df.isnull().sum() / len(df) * 100    # 결측치 비율

# 시각화
sns.heatmap(df.isnull(), cbar=False)
```

### 2.3 타겟 변수 분석 (회귀 필수)

```python
# 분포 확인 — 정규성 체크
sns.histplot(df['target'], kde=True)
plt.show()

# 왜도/첨도
print("Skewness:", df['target'].skew())
print("Kurtosis:", df['target'].kurt())

# 왜도가 크면(|skew| > 1) → log 변환 고려
df['target_log'] = np.log1p(df['target'])
```

### 2.4 상관관계

```python
# 수치형 변수 간 상관관계
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

# 타겟과의 상관계수 상위 10개
corr['target'].abs().sort_values(ascending=False).head(10)
```

---

## 3. 결측치 처리 방법

### 3.1 결측치 메커니즘 이해

| 유형 | 의미 | 처리 |
|------|------|------|
| **MCAR** (완전 랜덤) | 결측이 데이터와 무관 | 제거/대체 모두 OK |
| **MAR** (조건부 랜덤) | 다른 변수로 설명 가능 | 모델 기반 대체 권장 |
| **MNAR** (비랜덤) | 결측 자체가 정보 | 플래그 변수 추가 |

### 3.2 제거 방법

```python
# 결측치 있는 행 제거 (비율 낮을 때만)
df.dropna()

# 특정 컬럼만
df.dropna(subset=['important_col'])

# 50% 이상 결측인 컬럼 제거
df = df.loc[:, df.isnull().mean() < 0.5]
```

### 3.3 단순 대체

```python
# 수치형 — 평균/중앙값
df['col'].fillna(df['col'].mean(), inplace=True)
df['col'].fillna(df['col'].median(), inplace=True)  # 이상치에 강건

# 범주형 — 최빈값
df['cat_col'].fillna(df['cat_col'].mode()[0], inplace=True)

# 또는 "Missing" 이라는 새 카테고리로
df['cat_col'].fillna('Missing', inplace=True)
```

### 3.4 SimpleImputer (sklearn)

```python
from sklearn.impute import SimpleImputer

# 수치형
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# 범주형
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# 상수 값으로
const_imputer = SimpleImputer(strategy='constant', fill_value=0)
```

### 3.5 KNN Imputer (이웃 기반)

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

유사한 행들의 평균으로 대체. **수치형만** 적용 가능.

### 3.6 Iterative Imputer (MICE, 반복 회귀)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = imputer.fit_transform(df)
```

각 결측치 컬럼을 다른 컬럼들로 예측해서 채움. **가장 정교하지만 느림**.

### 3.7 결측 플래그 추가 (중요)

```python
# 결측 자체가 정보일 수 있음 → 표시 변수 추가
df['col_was_missing'] = df['col'].isnull().astype(int)
df['col'].fillna(df['col'].median(), inplace=True)
```

### 3.8 시험 답안 템플릿

```python
# 수치형: 중앙값 (이상치에 강건)
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 범주형: 최빈값
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
```

---

## 4. 이상치 처리

### 4.1 IQR 방법 (가장 표준적)

```python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 제거
df_clean = df[(df['col'] >= lower) & (df['col'] <= upper)]

# Capping (Winsorizing) — 값을 경계로 눌러줌
df['col'] = df['col'].clip(lower, upper)
```

### 4.2 Z-score 방법

```python
from scipy import stats

z_scores = np.abs(stats.zscore(df['col']))
df_clean = df[z_scores < 3]  # |z| > 3은 이상치
```

### 4.3 Isolation Forest (다변량)

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(df[num_cols])
df_clean = df[outliers == 1]
```

### 4.4 시각화로 확인

```python
# 박스플롯
df[num_cols].boxplot(figsize=(15, 5))

# 분포도
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(num_cols[:6]):
    sns.boxplot(y=df[col], ax=axes[i//3, i%3])
```

---

## 5. 인코딩 (범주형 → 수치형)

### 5.1 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['encoded'] = le.fit_transform(df['cat_col'])
```

**주의**: 순서가 없는 범주에 쓰면 모델이 잘못 학습할 수 있음 (트리 모델은 OK).

### 5.2 One-Hot Encoding

```python
# pandas
df_encoded = pd.get_dummies(df, columns=['cat_col'], drop_first=True)

# sklearn
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(df[['cat_col']])
```

**언제 쓰는가**: 선형 모델, NN에 필수. 고유값 수가 적을 때(<20).

### 5.3 Ordinal Encoding (순서 있는 범주)

```python
from sklearn.preprocessing import OrdinalEncoder

# 순서 명시
categories = [['Low', 'Medium', 'High']]
oe = OrdinalEncoder(categories=categories)
df['encoded'] = oe.fit_transform(df[['level']])
```

### 5.4 Target Encoding (타겟 평균)

```python
# 각 카테고리의 타겟 평균으로 치환
target_mean = df.groupby('cat_col')['target'].mean()
df['cat_encoded'] = df['cat_col'].map(target_mean)

# 또는 category_encoders 라이브러리
# pip install category_encoders
from category_encoders import TargetEncoder
te = TargetEncoder()
df['encoded'] = te.fit_transform(df['cat_col'], df['target'])
```

**주의**: 데이터 누수 방지 — train으로 fit, test에 transform만.

### 5.5 Frequency Encoding

```python
freq = df['cat_col'].value_counts(normalize=True)
df['freq_encoded'] = df['cat_col'].map(freq)
```

---

## 6. 스케일링 (수치형 정규화)

### 6.1 StandardScaler (Z-score)

$$x' = \frac{x - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**언제**: 가장 일반적. 선형 모델, SVM, NN에 필수.

### 6.2 MinMaxScaler ([0, 1] 범위)

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

**언제**: NN, 거리 기반. 이상치에 민감.

### 6.3 RobustScaler (중앙값, IQR 기반)

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
```

**언제**: 이상치가 많을 때. 중앙값과 IQR 사용 → 이상치의 영향 적음.

### 6.4 PowerTransformer (왜도 완화)

```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson: 음수 값 가능
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)

# Box-Cox: 양수 값만
pt = PowerTransformer(method='box-cox')
```

**언제**: 데이터가 크게 왜곡되어 있을 때 → 정규분포에 가깝게.

### 6.5 Log 변환 (간단 버전)

```python
# 타겟이 오른쪽으로 왜곡된 경우
y_log = np.log1p(y)  # log(1+y), y=0 방어

# 예측 후 역변환
pred = np.expm1(pred_log)
```

### 6.6 트리 모델은 스케일링 불필요

```
Tree-based (RF, XGBoost, LightGBM, CatBoost): 스케일링 불필요
Linear (LR, Ridge, Lasso): 필요
Distance-based (KNN, SVM, NN): 필수
```

---

## 7. 특성 공학

### 7.1 다항 특성

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False)
X_poly = poly.fit_transform(X)
```

### 7.2 구간화 (Binning)

```python
# 동일 간격
df['age_bin'] = pd.cut(df['age'], bins=5)

# 동일 빈도
df['age_bin'] = pd.qcut(df['age'], q=5)

# 수동 경계
df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 100],
                       labels=['young', 'adult', 'middle', 'senior'])
```

### 7.3 날짜 특성 분해

```python
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
```

### 7.4 상호작용 특성

```python
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
df['price_per_sqft'] = df['price'] / df['sqft']
```

---

## 8. 회귀 모델

### 8.1 모델별 특징 비교

| 모델 | 특징 | 언제 |
|------|------|------|
| **LinearRegression** | 가장 기본, 선형 관계 | 베이스라인 |
| **Ridge** | L2 규제, 다중공선성 완화 | 특성 많고 상관 높을 때 |
| **Lasso** | L1 규제, 특성 선택 효과 | 일부 특성만 중요할 때 |
| **ElasticNet** | L1+L2 혼합 | Ridge/Lasso 중간 |
| **DecisionTree** | 해석 용이, 과적합 위험 | 교육용/기준선 |
| **RandomForest** | 강건, 과적합 덜함 | **시험 필수** |
| **GradientBoosting** | sklearn 기본 부스팅 | 보통 성능 |
| **XGBoost** | 성능 좋음, 결측 자동 처리 | **대회/실무 표준** |
| **LightGBM** | XGBoost보다 빠름 | 대용량 데이터 |
| **CatBoost** | 범주형 자동 처리 | 범주 많을 때 |

### 8.2 기본 사용법

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
```

### 8.3 주요 모델 학습 코드

```python
# Ridge
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)

# Lasso (특성 선택)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)
# 선택된 특성
selected = X.columns[model.coef_ != 0]

# RandomForest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

# XGBoost
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    early_stopping_rounds=50,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# LightGBM
from lightgbm import LGBMRegressor
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)
```

### 8.4 특성 중요도

```python
# 트리 모델
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=importances.head(15), x='importance', y='feature')
```

---

## 9. 평가 지표 (회귀)

### 9.1 주요 지표

| 지표 | 수식 | 특징 |
|------|------|------|
| **MSE** | $\frac{1}{n}\sum(y - \hat{y})^2$ | 이상치에 민감 |
| **RMSE** | $\sqrt{\text{MSE}}$ | 원 단위, **가장 일반적** |
| **MAE** | $\frac{1}{n}\sum|y - \hat{y}|$ | 이상치에 강건 |
| **MAPE** | $\frac{100}{n}\sum\left|\frac{y-\hat{y}}{y}\right|$ | % 단위, y=0 주의 |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | 설명력, 1에 가까울수록 좋음 |

### 9.2 코드

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score
)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R²:   {r2:.4f}")
```

### 9.3 지표 선택 가이드

- **일반적**: RMSE
- **이상치 많음**: MAE
- **상대적 오차 중요**: MAPE (단, y가 0 근처면 불안정)
- **모델 비교**: R² (음수면 평균 예측보다 나쁨)

---

## 10. 교차 검증

### 10.1 K-Fold

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf,
                         scoring='neg_root_mean_squared_error')
print(f"CV RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

### 10.2 여러 지표 동시에

```python
from sklearn.model_selection import cross_validate

scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']
results = cross_validate(model, X, y, cv=5, scoring=scoring)

print(f"RMSE: {-results['test_neg_root_mean_squared_error'].mean():.4f}")
print(f"MAE:  {-results['test_neg_mean_absolute_error'].mean():.4f}")
print(f"R²:   {results['test_r2'].mean():.4f}")
```

---

## 11. 하이퍼파라미터 튜닝

### 11.1 GridSearch

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best score:", -grid.best_score_)
```

### 11.2 RandomizedSearch (빠름)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'learning_rate': uniform(0.01, 0.3),
}

random_search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42,
)
```

### 11.3 Optuna (베이지안 최적화)

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    }
    model = LGBMRegressor(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=5,
                           scoring='neg_root_mean_squared_error').mean()
    return -score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(study.best_params)
```

---

## 12. 파이프라인 (Pipeline)

전처리 + 모델을 하나로 묶어 데이터 누수 방지:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_cols = ['age', 'income']
cat_cols = ['city', 'gender']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

---

## 13. 시험 답안 템플릿 (최소 코드)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로드
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. 타겟 분리
y = train['target']
X = train.drop(['target', 'id'], axis=1)

# 3. 결측치 처리
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(include=['object']).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna('Missing')
test[num_cols] = test[num_cols].fillna(X[num_cols].median())
test[cat_cols] = test[cat_cols].fillna('Missing')

# 4. 인코딩
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], test[col]])
    le.fit(combined)
    X[col] = le.transform(X[col])
    test[col] = le.transform(test[col])

# 5. 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. 모델
model = RandomForestRegressor(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# 7. 평가
pred_val = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, pred_val))
r2 = r2_score(y_val, pred_val)
print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

# 8. 예측 + 제출
pred_test = model.predict(test)
submission = pd.DataFrame({'id': test['id'], 'target': pred_test})
submission.to_csv('submission.csv', index=False)
```

---

## 14. 체크리스트 (시험 전 점검)

```
데이터 이해:
☐ info(), describe() 확인
☐ 타겟 분포 확인 (왜곡 시 log 변환)
☐ 결측치 비율 파악
☐ 수치형/범주형 구분

전처리:
☐ 결측치 처리 (median/most_frequent)
☐ 이상치 확인 (IQR/boxplot)
☐ 범주형 인코딩 (Label/OneHot)
☐ 필요시 스케일링 (선형 모델 사용 시)

모델링:
☐ train/val 분할 또는 K-Fold
☐ 베이스라인 모델 (LinearRegression 또는 RandomForest)
☐ 성능 향상 모델 (XGBoost/LightGBM)
☐ 하이퍼파라미터 튜닝

평가:
☐ RMSE/MAE/R² 계산
☐ 특성 중요도 확인
☐ 예측값 분포 확인 (음수 불가 등 도메인 제약)

제출:
☐ test 데이터도 동일 전처리
☐ 예측 파일 형식 맞추기
```

---

## 15. 자주 하는 실수

1. **Train/Test에 다른 전처리 적용**: fit은 train, transform은 양쪽에
2. **누수(Leakage)**: 전체 데이터로 스케일링 후 분할 → train만으로 fit
3. **타겟 변환 후 역변환 잊음**: `np.log1p` 했으면 `np.expm1`로 복원
4. **범주형에 Label Encoding**: 선형 모델에 치명적 (순서가 있다고 학습)
5. **이상치 무조건 제거**: 도메인상 중요할 수 있음, 신중히
6. **랜덤 시드 미고정**: `random_state=42` 재현성 확보
7. **MAPE에서 y=0 오류**: 0 포함 시 MAPE 불안정

---

## 16. 관련 블로그 포스트

- [데이터 분석 분류 문제 가이드](data-analysis-classification-guide.md)
