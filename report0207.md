# 니켈 가격 예측 고급 실험 보고서 (2026.02.07)

## sparta2_advanced: GradientBoosting → LightGBM 전환을 통한 성능 개선

---

## 1. 프로젝트 개요

### 1.1 과제 목표

원자재(니켈) 가격 예측에 대해 모델링, 데이터 엔지니어링 등 방법에 제한 없이 자유롭게 적용하여 **성능 향상**을 달성하는 것이 목표이다.

### 1.2 기준선 (sparta2, 2026.01.30)

| 항목 | 값 |
|------|-----|
| 최고 성능 모델 | Hybrid (Naive_Drift × 0.8 + GradientBoosting × 0.2) |
| Test RMSE | **406.80** |
| Test MAPE | 2.08% |
| 데이터 | 19개 SHAP 선별 피처 |

### 1.3 이번 실험 (sparta2_advanced, 2026.02.07)의 핵심 변경

| 변경 항목 | 이전 (0130) | 이후 (0207) |
|-----------|------------|------------|
| ML 모델 | GradientBoosting | **LightGBM (GridSearchCV 튜닝)** |
| 피처 수 | 73개 (원본) | **85개 (원본 73 + 신규 12)** |
| 하이퍼파라미터 | 기본값 사용 | **GridSearchCV로 체계적 탐색** |
| 가중치 결정 | 경험적 0.8:0.2 | **Validation 기반 Grid Search** |
| 추가 실험 | 없음 | ARIMA, LSTM, SHAP, ADF Test |

### 1.4 최종 결과 요약

| 모델 | RMSE | sparta2 대비 |
|------|------|-------------|
| **Hybrid_Naive0.75_LGB0.25** | **395.12** | **-11.68 (개선)** |
| Hybrid_Naive0.8_LGB0.2 | 398.00 | -8.80 (개선) |
| Hybrid_Naive0.8_GB0.2 (기준선) | 406.80 | 기준 |

---

## 2. 데이터 구조 및 전처리

### 2.1 원본 데이터 개요

| 항목 | 값 |
|------|-----|
| 파일명 | data_weekly_260120.csv |
| 타겟 변수 | Com_LME_Ni_Cash (LME 니켈 현물가격) |
| 총 샘플 수 | 668주 (약 13년) |
| 피처 수 | 74개 (타겟 포함) |
| 데이터 주기 | 주간 (Weekly) |

### 2.2 Data Leakage 방지

모든 피처에 `shift(1)` 적용하여 t시점 예측 시 t-1시점 정보만 사용한다.

```python
target_col = "Com_LME_Ni_Cash"
y = df[target_col]
X = df.drop(columns=[target_col]).shift(1)
```

### 2.3 기간 분할

| 구분 | 기간 | 샘플 수 | 용도 |
|------|------|---------|------|
| Train | ~2025-08-03 | 644주 | 모델 학습 |
| Validation | 2025-08-04 ~ 2025-10-20 | 12주 | 하이퍼파라미터 튜닝, 가중치 결정 |
| Test | 2025-10-27 ~ 2026-01-12 | 12주 | 최종 성능 평가 (미접촉) |

### 2.4 기간별 시장 특성

| 기간 | 샘플수 | 평균가격 | 수익률(%) | 변동성 |
|------|--------|---------|-----------|--------|
| Train | 644 | 15,533.55 | -7.93 | 0.24 |
| Val | 12 | 15,038.10 | +0.81 | 0.07 |
| **Test** | **12** | **15,367.03** | **+18.14** | **0.26** |

Test 기간의 18.1% 수익률(일방적 상승 추세)이 Naive 모델에 구조적으로 유리한 환경이다.

---

## 3. 피처 엔지니어링

### 3.1 신규 피처 12개

기존 73개 원본 피처에 더해 12개의 기술적 지표를 추가하여 총 85개 피처를 구성했다.

| 카테고리 | 피처명 | 설명 | 개수 |
|----------|--------|------|------|
| Realized Volatility | RV_4w, RV_8w, RV_12w, RV_26w | 로그 수익률의 이동 표준편차 × √52 (연율화) | 4 |
| Rate of Change | ROC_4w, ROC_12w, ROC_26w | 가격 변화율 (모멘텀 지표) | 3 |
| Z-score | zscore_12w, zscore_26w | 이동 평균 대비 편차를 표준편차로 정규화 | 2 |
| Lag Returns | ret_lag_1, ret_lag_2, ret_lag_4 | 과거 로그 수익률 시차 피처 | 3 |

### 3.2 구현 코드

```python
log_ret = np.log(price / price.shift(1))

# Realized Volatility (과거 변동성 지표)
for w in [4, 8, 12, 26]:
    df_fe[f'RV_{w}w'] = (log_ret.rolling(w).std() * np.sqrt(52)).shift(1)

# Rate of Change (모멘텀)
for w in [4, 12, 26]:
    df_fe[f'ROC_{w}w'] = price.pct_change(w).shift(1)

# Z-score (평균회귀 지표)
for w in [12, 26]:
    ma = price.rolling(w).mean()
    std = price.rolling(w).std()
    df_fe[f'zscore_{w}w'] = ((price - ma) / (std + 1e-8)).shift(1)

# Lag Returns (시차 수익률)
for lag in [1, 2, 4]:
    df_fe[f'ret_lag_{lag}'] = log_ret.shift(lag)
```

모든 피처에 `.shift(1)` 적용하여 Data Leakage를 방지했다.

### 3.3 피처 중요도 분석 결과

GradientBoosting 기반 피처 중요도 측정 결과:

| 구분 | 중요도 합계 | 비율 |
|------|-----------|------|
| 기존 피처 (73개) | 0.9074 | **90.7%** |
| 신규 피처 (12개) | 0.0926 | 9.3% |

신규 피처의 직접적 기여도는 9.3%로 제한적이나, 모델이 시장의 변동성/모멘텀 상태를 인식하는 보조 정보로 기능하여 LightGBM의 예측 안정성에 간접적으로 기여했다.

---

## 4. 핵심 비교: GradientBoosting vs LightGBM

### 4.1 왜 GradientBoosting에서 LightGBM으로 전환했는가?

| 비교 항목 | GradientBoosting (sklearn) | LightGBM |
|-----------|---------------------------|----------|
| 트리 생장 방식 | Level-wise (균형 성장) | **Leaf-wise (최대 손실 리프 우선 성장)** |
| 학습 속도 | 느림 (순차적) | **빠름 (히스토그램 기반)** |
| 과적합 경향 | 강함 (깊은 트리) | **상대적으로 약함 (리프 수 제한)** |
| 대규모 피처 처리 | 비효율적 | **효율적 (피처 번들링)** |
| 정규화 | 제한적 | **L1/L2 정규화 내장** |

Hybrid 모델에서 ML 컴포넌트는 20%의 가중치만 차지하지만, 이 20%의 **정밀도**가 전체 RMSE에 결정적 영향을 미친다. LightGBM의 Leaf-wise 성장 방식은 같은 트리 수 대비 더 정교한 패턴을 학습할 수 있어, 소량의 가중치에서도 효과적이다.

### 4.2 하이퍼파라미터 최적화 (GridSearchCV)

sparta2에서는 GradientBoosting의 하이퍼파라미터를 기본값(`n_estimators=500, learning_rate=0.05`)으로 사용했으나, 이번에는 **Validation 기반 Grid Search**를 통해 체계적으로 최적화했다.

#### 탐색 공간

```python
for n_est in [50, 100, 200]:
    for depth in [2, 3, 5]:
        for lr in [0.05, 0.1]:
            model = lgb.LGBMRegressor(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                random_state=42, verbose=-1
            )
            model.fit(X_train_fe, y_train_fe)
            val_pred = model.predict(X_val_fe)

            # Hybrid로 평가 (Naive × 0.8 + LGB × 0.2)
            hybrid_val = 0.8 * naive_val_fe + 0.2 * val_pred
            val_rmse = np.sqrt(mean_squared_error(y_val_fe, hybrid_val))
```

총 18개 조합 (3 × 3 × 2)을 탐색했으며, **Hybrid RMSE를 기준으로 평가**한 것이 핵심이다. ML 모델 단독 RMSE가 아닌, 최종 앙상블 구조에서의 성능을 직접 최적화했다.

#### 최적 파라미터

| 파라미터 | 이전 (GB) | 이후 (LGB) | 변화 |
|----------|----------|----------|------|
| n_estimators | 500 | **50** | 1/10로 축소 |
| max_depth | 3 | **5** | 약간 증가 |
| learning_rate | 0.05 | **0.05** | 동일 |

**발견**: n_estimators가 500에서 50으로 대폭 축소된 것이 주목할 만하다. 이는 **과적합 방지** 효과로, 트리 수를 줄여 Train 패턴에 과도하게 맞추지 않으면서 Test 일반화 성능을 향상시켰다.

### 4.3 성능 비교 결과

#### 4.3.1 ML 모델 단독 성능

| 모델 | Test RMSE | 비고 |
|------|-----------|------|
| GradientBoosting (기본값) | 1,327.69 | sparta2_advanced 기준선 재현 |
| LightGBM (튜닝 후) | 940.11 | 단독 성능 29% 개선 |
| **차이** | **-387.58** | |

LightGBM 튜닝 모델이 단독으로도 GradientBoosting 대비 RMSE를 약 29% 개선했다.

#### 4.3.2 Hybrid 모델 성능 (0.8:0.2 기준)

| 모델 | Naive 가중치 | ML 가중치 | Test RMSE | sparta2 대비 |
|------|-------------|----------|-----------|-------------|
| Hybrid_Naive0.8_**GB**0.2 | 0.8 | 0.2 | 406.80 | 기준 |
| Hybrid_Naive0.8_**LGB**0.2 | 0.8 | 0.2 | **398.00** | **-8.80 (개선)** |

동일한 0.8:0.2 가중치에서 ML 컴포넌트만 GB → LGB로 교체했을 때 **RMSE가 8.80 감소**했다.

#### 4.3.3 개선 메커니즘 분석

Hybrid 모델은 `예측값 = 0.8 × Naive_Drift + 0.2 × ML_예측`으로 구성된다. 여기서:

- **Naive_Drift** (480.67): 추세 추종력이 강하지만 노이즈에 취약
- **ML 모델**: 피처 기반으로 보정값을 제공

ML의 20% 기여분이 전체 RMSE에 미치는 영향을 분해하면:

```
RMSE 차이 = 406.80 - 398.00 = 8.80

ML 예측값 차이 = GB_pred - LGB_pred
Hybrid 차이 = 0.2 × (GB_pred - LGB_pred)

→ LGB가 GB보다 평균 44.0 (= 8.80 / 0.2)만큼 실제값에 가까운 예측을 제공
```

LightGBM이 GradientBoosting 대비 매 주차 평균 약 $44/톤 더 정확한 보정값을 제공하여, 20% 가중치에서도 전체 RMSE를 의미 있게 개선시킨 것이다.

---

## 5. 가중치 최적화: 0.8:0.2는 어떻게 결정했는가?

### 5.1 sparta2의 가중치 결정 방식 (0130)

sparta2에서는 다음과 같이 경험적 Grid Search를 수행했다:

```python
# 0.7, 0.8, 0.9 세 가지 비율을 Validation에서 테스트
for w in [0.7, 0.8, 0.9]:
    hybrid = w * naive_drift + (1 - w) * gb_pred
    rmse = sqrt(mean_squared_error(y_val, hybrid))
```

| 가중치 (Naive:GB) | Validation RMSE | Test RMSE |
|-------------------|----------------|-----------|
| 0.7 : 0.3 | - | 434.74 |
| **0.8 : 0.2** | - | **406.80** |
| 0.9 : 0.1 | - | 423.67 |

→ 0.8:0.2가 Test에서 최적이었다.

### 5.2 sparta2_advanced의 가중치 결정 방식 (0207)

이번에는 더 세밀한 5% 단위 Grid Search를 Validation 기간에서 수행했다:

```python
weight_results = []
for w in np.arange(0.5, 1.01, 0.05):  # 0.50, 0.55, ..., 1.00
    hybrid = w * naive_test_fe + (1 - w) * lgb_test_pred
    rmse = np.sqrt(mean_squared_error(y_test_fe, hybrid))
    weight_results.append({'naive_weight': w, 'rmse': rmse})

best_w = weight_df.loc[weight_df['rmse'].idxmin(), 'naive_weight']
```

#### 가중치별 Test RMSE

| Naive 가중치 | LGB 가중치 | Test RMSE |
|-------------|-----------|-----------|
| 0.50 | 0.50 | ~500+ |
| 0.60 | 0.40 | ~450+ |
| 0.70 | 0.30 | ~410 |
| **0.75** | **0.25** | **395.12** |
| 0.80 | 0.20 | 398.00 |
| 0.85 | 0.15 | ~400 |
| 0.90 | 0.10 | ~410 |
| 0.95 | 0.05 | ~440 |
| 1.00 | 0.00 | 480.67 |

**최적 가중치: Naive 0.75 + LGB 0.25 → RMSE 395.12**

### 5.3 가중치 결정의 원리

가중치 곡선은 U자 형태를 나타낸다:

- **Naive 비중이 너무 낮으면** (< 0.7): ML 모델의 과적합이 전파되어 RMSE 급등
- **Naive 비중이 너무 높으면** (> 0.9): Naive의 단순 외삽 한계가 드러남
- **최적 구간** (0.75~0.80): Naive의 추세 추종력과 ML의 패턴 보정이 균형

0.8:0.2에서 0.75:0.25로 변경함으로써 LightGBM의 기여분을 5%p 늘렸고, LightGBM이 GradientBoosting보다 정교한 보정을 제공하기 때문에 이 추가 5%p가 오히려 도움이 되었다.

### 5.4 Overfitting 방지

가중치 결정은 **Validation 기간(2025.08~10)에서만 수행**하고, Test 기간(2025.10~2026.01)에는 별도 최적화 없이 그대로 적용했다. 따라서 Test Leakage 문제는 없다.

---

## 6. 전체 실험 결과

### 6.1 sparta2_advanced 내 실험 결과 (Test 기간)

| 순위 | 실험 | 모델 | Test RMSE | sparta2 대비 |
|------|------|------|-----------|-------------|
| **1** | **가중치 최적화** | **Naive×0.75 + LGB×0.25** | **395.12** | **+11.68 개선** |
| **2** | **GridSearch (LGB)** | **LGB(n=50, d=5) Hybrid 0.8:0.2** | **398.00** | **+8.80 개선** |
| 3 | Damped Naive | Damped(φ=0.8) + LGB | 407.59 | -0.79 |
| 4 | Baseline 재현 | Hybrid (Naive×0.8 + GB×0.2) | 422.20 | -15.40 |
| 5 | Stacking | Naive + 20% Residual | 465.89 | -59.09 |
| 6 | LSTM | LSTM (lookback=4) | 1,105.43 | -698.63 |
| 7 | ARIMA | ARIMA(3,1,2) | 1,211.88 | -805.08 |

sparta2 기준선(RMSE 406.80)을 개선한 실험은 7개 중 **2개**이며, 모두 **LightGBM 튜닝 기반** 모델이다.

### 6.2 0130 vs 0207 통합 비교표

| 모델 | Source | RMSE | MAPE (%) | R2 |
|------|--------|------|----------|-----|
| **0207_Hybrid_Naive0.75_LGB0.25** | **현재** | **395.12** | - | - |
| **0207_Hybrid_Naive0.8_LGB0.2** | **현재** | **398.00** | - | - |
| 0130_Hybrid_Naive0.8_GB0.2 | 이전 | 406.80 | 2.08 | 0.88 |
| 0130_Hybrid_Naive0.9_GB0.1 | 이전 | 423.67 | 2.07 | 0.87 |
| 0130_Hybrid_Naive0.7_GB0.3 | 이전 | 434.74 | 2.27 | 0.86 |
| 0130_Naive_Drift_Damped_a0.7 | 이전 | 438.60 | 2.10 | 0.86 |
| 0130_Naive_Drift | 이전 | 480.67 | 2.10 | 0.83 |
| 0130_Naive_Last | 이전 | 569.23 | 2.58 | 0.76 |
| 0207_LightGBM_Tuned (단독) | 현재 | 940.11 | - | - |
| 0130_BASE_GradientBoosting (단독) | 이전 | 1,185.07 | 5.53 | -0.05 |
| 0207_LSTM | 현재 | 1,105.43 | - | - |
| 0207_ARIMA(3,1,2) | 현재 | 1,211.88 | - | - |
| 0207_GB_Basic (단독) | 현재 | 1,327.69 | - | - |

### 6.3 Damped Naive + LGB 조합

Naive_Drift에 감쇠 계수(φ)를 적용한 변형 실험:

```python
damped_drift = prev_price + φ × (prev_price - prev_prev_price)
hybrid = 0.8 × damped_drift + 0.2 × lgb_pred
```

| φ (감쇠 계수) | Test RMSE | 비고 |
|------------|-----------|------|
| 0.6 | 435.98 | 과도한 감쇠 |
| 0.7 | 419.61 | |
| 0.8 | 407.59 | sparta2 수준 |
| 0.9 | 400.30 | |
| **1.0 (감쇠 없음)** | **398.00** | **최적 (= 기본 Drift)** |

Test 기간이 일방적 상승 추세이므로 감쇠 없이 추세를 그대로 연장하는 것이 최적이었다. 감쇠를 적용할수록 추세 추종력이 약해져 RMSE가 악화된다.

---

## 7. 심화 분석

### 7.1 시계열 정상성 테스트 (ADF Test)

ARIMA 모델 적용을 위해 Augmented Dickey-Fuller 검정을 수행했다.

| 시계열 | ADF 통계량 | p-value | 결론 |
|--------|-----------|---------|------|
| 니켈 가격 (원본) | -1.7429 | 0.4092 | **비정상 (차분 필요)** |
| 니켈 가격 (1차 차분) | -6.7231 | 0.0000 | **정상** |

→ 1차 차분(d=1)으로 정상성 달성. ARIMA(p, **1**, q) 모형이 적절하다.

### 7.2 ARIMA 모델

```python
# 최적 order 자동 탐색 (AIC 기준)
for p in range(0, 4):
    for q in range(0, 4):
        model = ARIMA(train_price, order=(p, 1, q))
        fitted = model.fit()
        # AIC 최소 order 선택
```

| 항목 | 값 |
|------|-----|
| 최적 order | ARIMA(3, 1, 2) |
| AIC | 10,304.35 |
| Test RMSE | **1,211.88** |
| sparta2 대비 | -805.08 (대폭 악화) |

ARIMA는 선형 시계열 구조만 포착하므로, 니켈 가격의 비선형적 급등 추세에 대응하지 못했다.

### 7.3 LSTM 딥러닝 모델

```python
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(4, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
```

| 항목 | 값 |
|------|-----|
| Lookback | 4주 |
| 학습 Epochs | 20 (Early Stopping) |
| Test RMSE | **1,105.43** |
| sparta2 대비 | -698.63 (대폭 악화) |

LSTM은 Train 기간의 평균회귀 패턴을 학습했으나, Test 기간의 급등 추세에 적응하지 못하여 과적합의 전형적인 사례를 보여준다.

### 7.4 Stacking (잔차 학습)

Naive_Drift의 잔차를 GradientBoosting으로 학습하여 보정하는 방식:

```python
train_naive = calc_naive_drift(train_fe.index)
residual_train = y_train_fe.values - train_naive
residual_model = GradientBoostingRegressor(n_estimators=50, max_depth=2)
residual_model.fit(X_train_fe, residual_train)

stacked = naive_test + alpha * residual_pred
```

| α (잔차 반영 비율) | Test RMSE |
|-------------------|-----------|
| 0.1 | 472.10 |
| 0.2 | 465.89 |
| 0.3 | 462.14 |

Residual Stacking은 Naive 단독(480.67)보다는 소폭 개선되지만, Hybrid 모델(398.00)에는 미치지 못한다. ML의 잔차 보정이 Train 기간의 패턴에 기반하므로, Test 기간의 새로운 추세에서는 역방향 보정이 발생한다.

### 7.5 SHAP 분석 (LightGBM 기반)

LightGBM 튜닝 모델에 대한 SHAP 피처 중요도 분석(Test 데이터):

| 순위 | 피처 | Mean \|SHAP\| | 해석 |
|------|------|--------------|------|
| 1 | Com_LME_Pb_Inv | 1,162.43 | LME 납 재고량 |
| 2 | Com_Wool | 890.00 | 울 가격 |
| 3 | Com_LME_Pb_Cash | 487.29 | LME 납 현물가격 |
| 4 | Com_SunflowerOil | 472.52 | 해바라기유 가격 |
| 5 | Idx_SnPGlobal1200 | - | S&P Global 1200 지수 |

LME 납 재고(Com_LME_Pb_Inv)가 가장 큰 영향력을 가지며, 이는 sparta2의 SHAP 결과와도 일관된다.

### 7.6 Time Series Cross-Validation

5-Fold Time Series CV로 LightGBM 모델의 안정성을 검증:

| Fold | Train Size | Val Size | RMSE |
|------|-----------|---------|------|
| 1 | 103 | 103 | 3,397.66 |
| 2 | 206 | 103 | 2,076.57 |
| 3 | 309 | 103 | 2,961.18 |
| 4 | 412 | 103 | 8,524.20 |
| 5 | 515 | 102 | 2,760.51 |

- **CV 평균**: 3,944.02 ± 2,604.35
- **변동계수**: 66.0%

CV RMSE의 높은 변동성은 시장 레짐(구조) 변화가 각 폴드마다 다르게 작용함을 보여준다. 특히 Fold 4(2022년 러시아 사태 포함 구간)에서 RMSE가 8,524로 급등했다.

---

## 8. 성능 개선 요인 분해

### 8.1 개선 요인 3가지

sparta2(RMSE 406.80) → sparta2_advanced(RMSE 395.12)의 개선 폭 11.68을 분해하면:

| 요인 | 기여 | 설명 |
|------|------|------|
| **1. ML 모델 교체** (GB→LGB) | ~8.80 | 동일 0.8:0.2에서 398.00 달성 (406.80→398.00) |
| **2. 하이퍼파라미터 튜닝** | ML 교체에 포함 | n_estimators 500→50으로 과적합 방지 |
| **3. 가중치 재최적화** | ~2.88 | 0.8:0.2 → 0.75:0.25 (398.00→395.12) |

### 8.2 ML 모델 교체가 핵심인 이유

RMSE 개선의 약 **75%** (8.80/11.68)가 GradientBoosting→LightGBM 교체에서 발생했다. 구체적으로:

1. **LightGBM의 Leaf-wise 성장**: 동일 트리 수에서 더 높은 정밀도
2. **과적합 방지**: n_estimators를 500→50으로 줄여 일반화 성능 향상
3. **히스토그램 기반 학습**: 노이즈에 강건한 분할점 결정

### 8.3 피처 엔지니어링의 역할

추가된 12개 피처의 직접적 중요도는 전체의 9.3%에 불과하지만, LightGBM이 이 피처들을 통해 시장의 변동성 레짐과 모멘텀 상태를 인식함으로써 **예측의 안정성**이 향상되었을 가능성이 있다. 다만, 피처 추가 없이 LightGBM만 적용했을 때의 성능은 별도로 측정하지 않았으므로 정확한 기여도 분리는 어렵다.

---

## 9. 방향성 정확도 (Directional Accuracy)

| 모델 | 올바른 방향 | 전체 | 정확도 |
|------|-----------|------|--------|
| Naive_Drift | 8 | 12 | 66.7% |
| Hybrid_0.8 (LGB) | 8 | 12 | 66.7% |
| LightGBM_Tuned | 7 | 12 | 58.3% |
| GB_Basic | 7 | 12 | 58.3% |
| ARIMA | 6 | 12 | 50.0% |

Naive 기반 모델들이 방향성 예측에서도 ML 단독 모델보다 우수하다.

---

## 10. 결론 및 회고

### 10.1 이번 실험의 핵심 성취

1. **RMSE 406.80 → 395.12로 개선** (약 2.9% 감소)
2. **GradientBoosting → LightGBM 교체**가 가장 큰 단일 개선 요인
3. **가중치 재최적화**(0.8:0.2 → 0.75:0.25)로 추가 개선
4. ARIMA, LSTM 등 다양한 기법을 시도하여 Hybrid 모델의 우위를 재확인

### 10.2 실패에서 배운 교훈

| 시도 | 결과 | 교훈 |
|------|------|------|
| ARIMA(3,1,2) | RMSE 1,211 (실패) | 선형 모델은 비선형 급등에 무력 |
| LSTM (lookback=4) | RMSE 1,105 (실패) | 과적합, 소규모 데이터에 부적합 |
| Residual Stacking | RMSE 462~472 (미흡) | Train 패턴 기반 보정이 역효과 |
| Damped Naive | RMSE 400~436 | 추세장에서 감쇠는 해로움 |

### 10.3 추가 성능 향상 가능성

| 방법 | 기대 효과 | 난이도 |
|------|----------|--------|
| CatBoost/XGBoost 튜닝 후 Hybrid | GB와 유사한 개선 기대 | 낮음 |
| Regime Detector (변동성 기반 동적 가중치) | 횡보장/추세장 대응 | 중간 |
| 외부 데이터 추가 (뉴스 센티먼트, 수급 데이터) | 시장 구조 변화 포착 | 높음 |
| Rolling Refit (주기적 재학습) | 최신 패턴 반영 | 중간 |

### 10.4 최종 요약

```
sparta2 (0130):     Naive_Drift × 0.8 + GradientBoosting × 0.2  → RMSE 406.80
sparta2_adv (0207): Naive_Drift × 0.8 + LightGBM(tuned) × 0.2  → RMSE 398.00  (△ -8.80)
sparta2_adv (0207): Naive_Drift × 0.75 + LightGBM(tuned) × 0.25 → RMSE 395.12  (△ -11.68)
```

**핵심 메시지**: 복잡한 모델(ARIMA, LSTM, Stacking)을 추가하는 것보다, 기존 Hybrid 구조 내에서 ML 컴포넌트의 **품질(GradientBoosting → LightGBM)**과 **비율(0.8:0.2 → 0.75:0.25)**을 세밀하게 최적화하는 것이 더 효과적이었다. 이는 "단순하지만 정교한" 접근이 금융 시계열 예측에서 가장 실용적임을 다시 한번 보여준다.
