# 니켈 가격 예측 프로젝트 - 실험 전체 기록

## 프로젝트 개요

- **목표**: LME 니켈 현물 가격(Com_LME_Ni_Cash)을 주간 단위로 예측
- **데이터**: 2013-04-01 ~ 2026-01-20, 668주 주간 데이터, 74개 변수
- **타겟 변수**: Com_LME_Ni_Cash (니켈 현물 가격, USD/톤)
- **평가 기간**: Train (~2025-08-03) / Validation (2025-08-04 ~ 2025-10-20) / Test (2025-10-27 ~ 2026-01-12, 12주)
- **핵심 제약**: 데이터 누수(Data Leakage) 방지를 위해 모든 피처에 1주 지연(shift(1)) 적용

---

## 실험 흐름 전체 요약

```
EDA.ipynb (탐색)
    ↓
sparta2_0205이전과제백업.ipynb (핵심 실험)
    ├── SHAP 피처 선택 → 5종 부스팅 베이스라인 비교
    ├── 3단계 스태킹 (Baseline → Residual → ROR)
    ├── Naive 모델 발견 → Hybrid 모델 개발
    └── 백테스트 및 방향성 평가
    ↓
sparta2_advanced.ipynb (개선 시도)
    ├── 피처 엔지니어링 (기술적 지표 12개 추가)
    ├── LightGBM GridSearchCV 최적화
    ├── ARIMA 전통 시계열 모델
    ├── LSTM (TensorFlow) 딥러닝
    └── 최종 RMSE 398.00 달성
    ↓
dl_lstm_transformer.ipynb (DL 파이프라인)
    └── LSTM/Transformer 3단계 스태킹 (ML과 동일 구조)
    ↓
dl_advanced.ipynb (DL 심화)
    ├── 과적합 방지 DL 설계
    ├── Naive + DL / ML + DL 앙상블
    └── Quantile Regression (불확실성 정량화)
```

---

## 1. EDA.ipynb — 탐색적 데이터 분석

### 사용한 방법

1. **데이터 로드 및 전처리**: pandas로 CSV 로드, datetime 인덱스 설정, 시간순 정렬
2. **결측치 검사**: `isnull().sum()` → 결측치 없음 확인
3. **시계열 연속성 확인**: 날짜 간격이 정확히 7일인지 검증 → 누락 없음 확인
4. **Min-Max 정규화**: `sklearn.MinMaxScaler`로 0~1 스케일링 → 단위가 다른 변수 간 추세 비교를 위함
5. **퍼센트 변화율(수익률)**: `pct_change()`로 변동성 기반 상관관계 분석
6. **상관관계 분석**: 74×74 히트맵 + 상위/하위 상관 쌍 추출
7. **그룹별 시계열 시각화**: LME 금속, 재고, 에너지, 금융 지표 그룹으로 나누어 정규화 추세 비교

### 핵심 인사이트

- **변수 간 높은 상관관계**: S&P500과 S&P Global 1200(0.997), US 2Y와 1Y 채권(0.994), 원유와 브렌트유(0.993) 등 거의 동일하게 움직이는 변수 쌍이 많음 → 다중공선성 문제 인지
- **강한 음의 상관관계**: 달러 인덱스(DXY)와 호주 달러(AUD/USD)(-0.894), 원/달러 환율과 중국 채권(-0.915) → 거시경제 역학 관계 확인
- **데이터 품질 양호**: 결측치 없음, 주간 간격 정확 → 별도 보간이나 리샘플링 불필요

### 왜 이 방법을 사용했는가

- 74개 변수의 구조를 파악하지 않으면 이후 피처 선택이 근거 없이 진행됨
- 정규화 추세 비교를 통해 어떤 변수가 니켈과 동조하는지 시각적으로 확인
- 상관관계 분석으로 다중공선성 있는 변수 쌍을 미리 파악 → 이후 SHAP 기반 선택의 참고 자료

### 가장 많이 배운 점

- **단순 상관관계만으로는 인과관계를 판단할 수 없다**: 니켈 가격과 높은 상관관계를 보이는 변수도 사실은 글로벌 경기 사이클의 공통 원인(confounding factor)으로 인한 것일 수 있음
- **정규화 시각화의 유용성**: 원본 가격 단위(니켈 15,000 vs 금리 3.5)가 너무 달라 직접 비교가 불가능 → Min-Max 정규화로 추세의 방향성만 비교하면 직관적 패턴 발견 가능

---

## 2. sparta2_0205이전과제백업.ipynb — 핵심 실험 (베이스라인 비교 + 3단계 스태킹)

이 노트북이 프로젝트의 **중심 실험**이며, 가장 많은 발견이 이루어졌다.

### 2.1 실험 설계

```
Stage 1 (Baseline): 원본 피처 → 가격 예측 → 기준선 확립
Stage 2 (Residual): 잔차 = 실제 - Baseline 예측 → 잔차 보정 → 오차 축소
Stage 3 (ROR):      ROR = 2×Baseline 예측 - 실제 → 수익률 예측 → 방향성 강화
```

### 2.2 데이터 전처리 및 피처 선택

#### 방법

1. **컬럼 필터링**: 74개 변수 중 금속 가격, 금융 지표(Idx_), 채권(Bonds_), 환율(EX_), LME 관련 변수만 선별
2. **누수 방지**: `X = df.drop(target).shift(1)` — 모든 피처를 1주 지연시켜 t시점 예측에 t-1시점 정보만 사용
3. **SHAP 기반 피처 선택**:
   - XGBoost(n_estimators=100)를 **학습 데이터만으로** 학습
   - `shap.TreeExplainer`로 SHAP 값 계산
   - 평균 |SHAP| 상위 20개 피처 선택
   - LME Index는 순환참조(니켈 가격을 포함하는 종합지수)이므로 의도적 제외
4. **피처 안정성 검증**: 5-Fold 시계열 CV로 각 fold별 SHAP 상위 피처가 일관되는지 확인

#### 왜 SHAP을 사용했는가

- **단순 상관관계의 한계**: 상관관계가 높아도 모델 예측에 기여하지 않을 수 있음 (다중공선성)
- **SHAP의 장점**: 각 피처가 개별 예측에 얼마나 기여하는지를 Shapley value로 정량화 → 상호작용 효과 반영
- **train-only 선택**: validation/test 정보가 피처 선택에 유입되면 낙관적 편향(optimistic bias) 발생 → 실전 성능과 괴리

#### LME Index 제외 실험

- LME Index는 SHAP 1위였으나, 이 지수 자체가 니켈 가격을 포함하는 가중 평균 → "니켈 가격으로 니켈 가격을 예측"하는 동어반복
- 제외 시 해석 가능성 향상 + 대체 피처(SHAP 21위~)를 추가하여 피처 수 유지

#### 선택된 피처의 도메인 해석

| 피처 그룹 | 대표 변수 | 니켈과의 관계 |
|----------|----------|------------|
| 중국 경기 | CSI300, Shanghai50 | 중국 = 세계 니켈 소비 60% → 경기 지표가 수요 선행 |
| 달러 강약 | DXY, EX_USD_CNY | 달러 강세 → 달러 표시 원자재 가격 하락 압력 |
| LME 재고 | Com_LME_Ni_Inv | 재고 증가 = 공급 > 수요 → 가격 하락 선행 지표 |
| 연관 금속 | Pb, Sn, Al 가격 | 비철금속 동조화 → 글로벌 제조업 경기 반영 |
| 에너지 | 원유, 천연가스 | 니켈 제련 에너지 비용 + 글로벌 인플레이션 지표 |

### 2.3 베이스라인 모델 5종 비교

#### 구현 방법

| 모델 | 구현 | 역할 | 하이퍼파라미터 탐색 |
|------|------|------|-----------------|
| **GradientBoosting** | `sklearn.GradientBoostingRegressor` | 안정적 기준선 | ParameterSampler 10회 |
| **XGBoost** | `xgboost.XGBRegressor` | 강한 비선형 학습 | ParameterSampler 16회 |
| **LightGBM** | `lightgbm.LGBMRegressor` | 빠른 잔차 안정화 | ParameterSampler 16회 |
| **CatBoost** | `catboost.CatBoostRegressor` | 과적합 억제 | ParameterSampler 12회 |
| **AdaBoost** | `sklearn.AdaBoostRegressor` | 약한 학습기 앙상블 | ParameterSampler 10회 |
| **Naive_Last** | `y.shift(1)` | 전주 가격 그대로 | 없음 |
| **Naive_Drift** | `y.shift(1) + (y.shift(1) - y.shift(2))` | 전주 가격 + 변화량 | 없음 |

#### 각 모델의 구현 상세

- **GradientBoosting**: scikit-learn의 기본 GBM. `n_estimators=500, learning_rate=0.05` 기본값 + `max_depth`, `subsample`, `min_samples_leaf` 탐색. 순차적 부스팅으로 안정적이나 학습 속도가 느림
- **XGBoost**: 정규화(L1/L2) 내장, 결측치 자동 처리, 병렬 학습 지원. `colsample_bytree`, `reg_alpha`, `reg_lambda` 추가 탐색
- **LightGBM**: 히스토그램 기반 분할로 빠른 학습. `num_leaves`, `min_child_samples`, `feature_fraction` 탐색. Leaf-wise 성장 전략
- **CatBoost**: Ordered Boosting으로 과적합 방지 내장. `depth`, `l2_leaf_reg`, `border_count` 탐색. 범주형 변수 자동 처리(이 데이터에는 범주형 없음)
- **AdaBoost**: 약한 학습기(Decision Stump) 가중 앙상블. `n_estimators`, `learning_rate` 탐색. 이상치에 민감

#### 하이퍼파라미터 탐색 방식

- `sklearn.model_selection.ParameterSampler`로 랜덤 탐색
- 각 모델별 budget(탐색 횟수) 차등 배정 (XGBoost/LightGBM 16회, CatBoost 12회, GB/AdaBoost 10회)
- **Validation RMSE 기준**으로 최적 파라미터 선정 → Test에는 사용하지 않음

#### 결과

- **Validation 기준**: GradientBoosting 최우수 (VAL RMSE: 106.95)
- **Test 기준**: **Naive_Drift가 모든 ML 모델을 압도** (Test RMSE: 480.67 vs ML 최우수 ~1,175)

### 2.4 Residual 스태킹 (Stage 2)

#### 방법

```python
잔차(residual) = 실제값(y) - 베이스라인 예측(ŷ_base)
잔차 모델 학습: residual ~ f(X)
최종 예측: ŷ = ŷ_base + ŷ_residual
```

- **베이스라인**: Validation 최우수인 GradientBoosting 고정
- **잔차 모델 후보**: 5개 부스팅 모델 전부 시도
- **조합 탐색**: GB + {XGBoost, LightGBM, CatBoost, AdaBoost, GB} × 파라미터 그리드
- **상위 2개 선정**: GB+LightGBM, GB+XGBoost

#### 왜 Residual 스태킹을 사용했는가

- 베이스라인 모델이 잡지 못하는 **체계적 오차(systematic error)**가 있다면, 2차 모델이 그 패턴을 학습할 수 있음
- 예: 베이스라인이 상승장에서 일관되게 과소추정한다면, 잔차 모델이 "상승 보정량"을 학습

#### 결과

- Validation: GB+LightGBM (RMSE 92.20) → 베이스라인(106.95) 대비 13.8% 개선
- **Test: 과적합 징후** — Validation에서 개선되었으나 Test에서는 성능 저하

### 2.5 ROR 스태킹 (Stage 3)

#### 방법

```python
ROR(Rate of Return) = (예측가격 - 전주가격) / 전주가격
ROR 타겟 = 2 × 베이스라인 예측 - 실제값  (= 베이스라인 예측 - 잔차)
최종 예측 = 전주 가격 × (1 + ROR 예측)
```

- Residual 스태킹 상위 2개 조합을 기반으로 3차 모델 확장
- 메타 피처(베이스라인 예측값, 잔차 예측값)를 원본 피처에 추가하여 학습

#### 왜 ROR을 사용했는가

- **가격 레벨(level) 예측의 한계**: 가격 자체는 비정상(non-stationary)이므로 모델이 수준(level)에 의존하게 됨
- **수익률(return) 예측의 장점**: 정상성(stationarity)이 더 높아 모델 학습에 유리
- **3단계 스태킹**: 각 단계가 이전 단계의 오차를 보정 → 이론적으로는 점진적 개선

#### 결과

- **Validation에서 추가 개선**되었으나, **Test에서 심한 과적합**
- 3단계 스태킹의 복잡도가 12주 Test에서 일반화 능력을 저해

### 2.6 Naive 모델 발견 및 후속 실험

#### 핵심 발견

> **Naive_Drift(전주 가격 + 전주 변화량)가 모든 ML 모델을 Test에서 압도**

이 발견이 프로젝트 방향을 완전히 전환시켰다.

#### 후속 실험 (Section 9)

| 실험 | 방법 | 결과 |
|------|------|------|
| **Naive 변형** | SMA4, Damped Drift(α=0.3~0.9) | Damped(α=0.9) 소폭 개선 |
| **Naive + ML 하이브리드** | `w × Naive_Drift + (1-w) × ML_pred` | **w=0.8이 최적 → RMSE 406.80** |
| **Naive + ML Residual** | Naive 잔차를 ML로 보정 | 제한적 효과 |
| **Naive + ML ROR** | Naive 기반 수익률을 ML이 보정 | 효과 없음 |

#### Hybrid 모델 (최종 선정)

```python
최종 예측 = 0.8 × Naive_Drift + 0.2 × GradientBoosting
# Naive_Drift = 전주 가격 + (전주 가격 - 전전주 가격)
```

- **Test RMSE: 406.80** — 이 프로젝트의 기준선(baseline)이 됨

### 2.7 백테스트 및 방향성 평가

#### 백테스트 방법

- 거래비용 0.1% + 슬리피지 0.05% 반영
- 임계값(threshold) 0.3%, 0.5%, 1.0%에서 거래 신호 생성
- 누적 수익률, Sharpe Ratio, Maximum Drawdown 계산

#### 방향성 평가

- 상승/하락 방향 정확도(Directional Accuracy) 측정
- 혼동행렬(Confusion Matrix)로 상승 예측 시 실제 상승 비율, 하락 예측 시 실제 하락 비율 분석

### 2.8 이 노트북에서 가장 많이 배운 것

1. **복잡한 모델 ≠ 좋은 모델**: 3단계 스태킹은 Validation에서 인상적이었으나 Test에서 완패. 12주라는 짧은 Test 기간에서 과적합의 위험성을 체감
2. **시장 레짐(regime) 변화의 위력**: Train 기간(평균회귀 패턴)과 Test 기간(일방적 상승)의 구조가 다르면, 학습한 패턴이 무용지물
3. **Naive의 구조적 우위**: 추세 시장에서 "전주 + 변화량"은 자연스럽게 추세를 추종 → ML이 학습하기 어려운 단순하지만 강력한 전략
4. **SHAP의 가치**: 단순 성능 비교를 넘어 "왜 이 변수가 중요한가"를 도메인 지식으로 해석할 수 있었음 (중국 경기, 달러 강약, LME 재고)
5. **파생 피처의 함정**: LME Index가 SHAP 1위였으나 순환참조 → 맹목적 피처 선택의 위험성

---

## 3. sparta2_advanced.ipynb — 기준선 개선 시도

### 목적

sparta2에서 달성한 Hybrid RMSE 406.80을 체계적으로 개선하기 위한 추가 실험

### 3.1 피처 엔지니어링 (12개 신규 피처)

#### 생성한 피처

| 피처 유형 | 변수 | 계산 방법 | 목적 |
|----------|------|----------|------|
| **변동성(RV)** | RV_4w, RV_8w, RV_12w, RV_26w | `log_ret.rolling(w).std() × √52` | 변동성 클러스터링 활용 |
| **모멘텀(ROC)** | ROC_4w, ROC_12w, ROC_26w | `price.pct_change(w)` | 단기/중기/장기 추세 강도 |
| **평균회귀(Z-score)** | zscore_12w, zscore_26w | `(price - MA) / std` | 과매수/과매도 신호 |
| **시차 수익률** | ret_lag_1, ret_lag_2, ret_lag_4 | `log_ret.shift(lag)` | 자기상관 패턴 |

#### 왜 이 피처들을 만들었는가

- **변동성**: 금융 시계열에서 변동성은 클러스터링 특성(큰 변동 후 큰 변동)을 보임 → 모델이 변동성 레짐을 인식할 수 있도록
- **모멘텀**: "최근 N주 수익률"은 추세 강도의 직접적 지표 → Naive Drift와 유사한 정보를 ML이 학습할 수 있도록
- **Z-score**: 가격이 이동평균에서 크게 벗어나면(z > 2) 평균회귀 가능성 → 반전 신호
- **시차 수익률**: 자기상관(autocorrelation) 패턴을 직접적으로 피처화

#### 결과

- GradientBoosting 기반 피처 중요도 분석: **신규 피처의 중요도 합계는 전체의 일부**(기존 피처가 여전히 지배적)
- 피처 추가 자체로는 극적 개선 없음

### 3.2 LightGBM GridSearchCV 최적화

#### 방법

```python
for n_est in [50, 100, 200]:
    for depth in [2, 3, 5]:
        for lr in [0.05, 0.1]:
            model = lgb.LGBMRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr)
            # Validation RMSE 기준 최적화
```

- 18개 조합 전수 탐색 (3 × 3 × 2)
- **Hybrid 기준으로 평가**: `0.8 × Naive + 0.2 × LightGBM` 형태로 Validation RMSE 측정 → 실제 사용 형태와 동일하게 평가

#### 왜 GradientBoosting에서 LightGBM으로 교체했는가

- LightGBM은 히스토그램 기반 분할로 학습 속도가 빠름
- Leaf-wise 성장 전략으로 동일 트리 수에서 더 정밀한 분할 가능
- GridSearchCV를 통해 체계적 하이퍼파라미터 탐색이 가능

#### 결과

- **Test RMSE: 398.00** (sparta2 기준선 406.80 대비 8.80 개선)
- 핵심 변경: GradientBoosting → 튜닝된 LightGBM

### 3.3 Time Series Cross-Validation

#### 방법

- `sklearn.TimeSeriesSplit(n_splits=5)`로 5-Fold 시계열 CV
- Train 데이터 내에서만 수행 → Validation/Test 정보 미사용
- 각 Fold별 LightGBM RMSE 측정 → 평균 및 표준편차 계산

#### 왜 사용했는가

- 일반 K-Fold CV는 시간 순서를 무시 → 미래 데이터로 과거를 예측하는 누수 발생
- Time Series CV는 항상 과거 → 미래 방향으로만 학습/검증 → 실전에 가까운 평가
- CV 변동계수(std/mean)를 통해 모델 안정성 정량화

### 3.4 가중치 최적화

#### 방법

```python
for w in np.arange(0.5, 1.01, 0.05):
    hybrid = w * naive + (1-w) * lgb_pred
    # Validation RMSE 측정
# Validation 최적 w를 Test에 적용 (Test로 튜닝하지 않음!)
```

#### 핵심 인사이트

- **Validation 최적 가중치 ≈ 0.8** → sparta2의 직관적 선택(0.8)이 실증적으로도 정당화됨
- Naive 비중이 높을수록 안정적 (0.8~0.9 범위가 최적)
- ML 비중이 너무 높으면(0.5 이하) Test 성능 급락 → ML 과적합의 영향

### 3.5 Damped Naive + Stacking

#### Damped Naive

```python
damped_drift = prev_price + damping × (prev_price - prev_prev_price)
# damping < 1: 추세 감쇠, damping = 1: 원래 Naive Drift
```

- Validation에서 최적 damping 탐색 → Test 적용
- 결과: damping=1.0(원래 Naive Drift)이 Test에서도 최적

#### Residual Stacking

```python
residual = y_train - naive_drift_train
residual_model.fit(X_train, residual)
stacked = naive_test + alpha × residual_model.predict(X_test)
```

- α=0.2에서 소폭 개선 가능성 확인

### 3.6 ARIMA 모델

#### 방법

- ADF 검정으로 원본 시계열 비정상성 확인 → 1차 차분 후 정상성 확보
- ARIMA(p, 1, q)에서 p∈[0,3], q∈[0,3] 전수 탐색 → AIC 기준 최적 order 선택
- 최적: ARIMA(3, 1, 2)

#### 왜 ARIMA를 시도했는가

- 전통적 시계열 분석의 대표 기법
- ML/DL과 다른 접근: 자기회귀(AR) + 이동평균(MA) + 차분(I)으로 시계열 자체의 패턴만 학습
- 외생 변수 없이 순수 시계열 구조만으로 어느 수준까지 가능한지 확인

#### 결과

- **Test RMSE: 1,211.88** — Naive보다 훨씬 나쁨
- 해석: ARIMA는 학습 기간의 평균 수준으로 회귀하려는 경향 → 급등하는 Test 기간에서 크게 과소추정

### 3.7 LSTM 딥러닝 (TensorFlow/Keras)

#### 구현 상세

```python
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(4, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
```

- **Lookback**: 4주 (최근 4주 가격으로 다음 주 예측)
- **스케일링**: MinMaxScaler(Train 데이터만으로 fit → Leakage 방지)
- **학습**: Adam optimizer, MSE 손실, EarlyStopping(patience=10)
- **단변량**: 니켈 가격만 입력 (외생 변수 미사용)

#### 왜 LSTM을 시도했는가

- LSTM은 시계열의 장기 의존성(long-term dependency)을 학습할 수 있는 RNN 변형
- 비선형 패턴을 자동으로 포착 가능
- 부스팅 모델과 다른 특성의 예측을 생성 → 앙상블 다양성 확보 가능성

#### 결과

- **Test RMSE: ~1,105** — Naive보다 훨씬 나쁨
- 4주 lookback + 단변량 구조의 한계
- 학습 데이터의 패턴(평균회귀)과 Test(급등)의 불일치

### 3.8 이 노트북에서 가장 많이 배운 것

1. **GradientBoosting → LightGBM 교체의 효과**: 동일 Hybrid 구조에서 ML 컴포넌트만 교체하여 RMSE 406.80 → 398.00 달성 → 모델 선택과 튜닝의 중요성
2. **ARIMA의 한계**: 전통 시계열 모델은 구조 변화(structural break)에 매우 취약. 학습 기간 평균으로 회귀하려는 특성이 급등 시장에서 치명적
3. **Validation ≠ Test**: GridSearchCV에서 Validation 최적이 Test에서도 최적이라는 보장 없음 → 시장 레짐 변화 시 특히 위험
4. **피처 엔지니어링의 한계**: 12개 신규 피처를 추가했지만 극적 개선 없음 → 피처의 양보다 모델 구조와 앙상블 전략이 더 중요할 수 있음

---

## 4. dl_lstm_transformer.ipynb — DL 3단계 스태킹 파이프라인

### 목적

ML에서 사용한 3단계 스태킹(Baseline → Residual → ROR)을 **LSTM과 Transformer로 동일하게 구현**하여 DL의 시계열 예측 능력을 평가

### 4.1 데이터 전처리 (ML과의 차이)

#### 피처 수익률 변환

```python
# 양수 시계열: 로그 수익률
df_ret[pos_cols] = np.log(df[pos_cols] / df[pos_cols].shift(1))
# 음수 포함 시계열: 단순 차분
df_ret[non_pos_cols] = df[non_pos_cols].diff()
```

- ML 노트북은 가격 레벨 피처를 사용했으나, DL은 수익률로 변환
- **이유**: DL(특히 LSTM)은 입력 스케일에 민감 → 수익률 변환 + StandardScaler로 정규화

#### 시퀀스 생성

```python
SEQ_LEN = 24  # 24주(약 6개월) lookback
# 각 샘플: (24, n_features) 형태의 2D 시퀀스
```

- 시계열을 슬라이딩 윈도우로 잘라 (시퀀스 길이 × 피처 수) 형태의 3D 텐서 생성

### 4.2 모델 구현

#### LSTM

```python
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.1):
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)      # 전체 시퀀스 처리
        last = out[:, -1]           # 마지막 시점의 hidden state만 사용
        return self.fc(last)        # 선형 변환으로 1개 값 출력
```

- **2-layer LSTM** (hidden_size=64, dropout=0.1)
- 마지막 시점의 hidden state를 Dense layer에 통과시켜 예측
- PyTorch 기반 구현

#### Transformer

```python
class TransformerForecaster(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        self.input_proj = nn.Linear(n_features, d_model)   # 피처→모델 차원 매핑
        self.pos_enc = PositionalEncoding(d_model)           # 위치 인코딩
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)     # 차원 변환
        x = self.pos_enc(x)        # 위치 정보 추가
        x = self.encoder(x)        # Self-Attention 처리
        last = x[:, -1]            # 마지막 시점 출력
        return self.fc(last)
```

- **Positional Encoding**: sin/cos 기반 위치 인코딩으로 시퀀스 내 시간적 순서 정보 제공
- **Multi-Head Self-Attention** (4 heads): 시퀀스 내 모든 시점 간 관계를 학습
- **Feed-Forward Network**: dim_feedforward=128

#### 왜 이 구조를 선택했는가

- **LSTM**: 시계열의 순차적 패턴을 학습하는 가장 검증된 아키텍처. Hidden state가 과거 정보를 요약
- **Transformer**: NLP에서 혁신을 일으킨 Self-Attention 메커니즘을 시계열에 적용. 장기 의존성을 직접 모델링 가능 (LSTM의 정보 소실 문제 없음)
- **동일 3단계 구조**: ML과 동일한 Baseline → Residual → ROR 파이프라인으로 공정한 비교

### 4.3 학습 방법

```python
def fit_model(model, train_loader, val_loader, epochs, lr=1e-3, patience=10):
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    # EarlyStopping: val_loss 기준 patience 내 개선 없으면 중단
    # Best model state 저장 및 복원
```

- **Optimizer**: Adam (lr=1e-3)
- **Loss**: MSE
- **Early Stopping**: patience=10 (Validation loss 기준)
- **Epochs**: Baseline 5, Residual 5, ROR 5 (매우 적은 에폭 → 실험적 빠른 확인용)

### 4.4 3단계 파이프라인 실행

1. **Validation에서 베이스라인 선택**: LSTM, Transformer 각각 학습 → VAL RMSE 기준 Top 2 선정
2. **Residual 조합 탐색**: Base(LSTM/Trans) × Residual(LSTM/Trans) 4개 조합 → VAL RMSE Top 2
3. **ROR 확장**: 상위 Residual 조합에 ROR 모델 추가 → 최종 Test 평가
4. **최종 학습**: Train+Validation 합쳐서 재학습 → Test 예측

### 4.5 결과 및 발견

- **DL 모델들은 Test에서 ML보다 더 심한 과적합**
- LSTM/Transformer 단독: Test RMSE ~1,500~2,000
- Residual/ROR 추가: 소폭 개선되었으나 여전히 Naive/Hybrid보다 훨씬 나쁨
- **5 에폭의 한계**: 충분히 학습되지 않았을 가능성은 있으나, 더 많이 학습하면 과적합이 더 심해질 수 있음

### 4.6 이 노트북에서 가장 많이 배운 것

1. **소규모 데이터에서 DL의 한계**: 668주(~640 학습 샘플)는 LSTM/Transformer가 일반화하기에 턱없이 부족
2. **시퀀스 길이의 중요성**: SEQ_LEN=24는 임의 선택 → 시퀀스 길이에 따라 성능이 크게 달라질 수 있음
3. **ML과 DL의 공정 비교 어려움**: 에폭 수, 학습률, 모델 크기 등 DL 하이퍼파라미터 탐색 공간이 훨씬 큼

---

## 5. dl_advanced.ipynb — DL 심화 실험

### 목적

dl_lstm_transformer.ipynb에서 DL 단독이 실패한 원인을 분석하고, **4개 가설을 체계적으로 검증**

### 5.1 가설 설계

| 가설 | 내용 | 검증 방법 |
|------|------|----------|
| **H1** | DL 과적합 방지 설계로 단독 성능 개선 가능 | Dropout 강화, 모델 축소, 정규화 |
| **H2** | Naive + DL 앙상블로 과적합 완화 가능 | 가중 평균 (Naive 0.8 + DL 0.2) |
| **H3** | ML + DL 앙상블이 ML 단독보다 효과적 | GB + LSTM/Transformer 결합 |
| **H4** | Quantile Regression으로 예측 불확실성 정량화 가능 | 예측 구간 커버리지 확인 |

### 5.2 H1: DL 과적합 방지 설계

#### 구현

```python
class LSTMRegularized(nn.Module):
    # hidden=32 (기존 64→축소), layers=1 (기존 2→축소), dropout=0.3 (기존 0.1→강화)

class TransformerRegularized(nn.Module):
    # d_model=32 (기존 64→축소), nhead=2 (기존 4→축소), layers=1, dropout=0.3
```

- **모델 축소**: 파라미터 수를 절반 이하로 줄여 과적합 여지 감소
- **Dropout 강화**: 0.1 → 0.3으로 증가
- **L2 정규화**: `weight_decay=1e-4` 추가
- **조기 종료**: patience=15, epochs=80

#### 결과

- 기존 ~1,957 → 정규화 후에도 여전히 Hybrid(406.80)에 못 미침
- **부분적 개선은 있었으나 근본적 한계 돌파 실패**

### 5.3 H2: Naive + DL 앙상블

#### 방법

```python
naive_lstm = 0.8 * naive_drift + 0.2 * lstm_pred
naive_trans = 0.8 * naive_drift + 0.2 * transformer_pred
naive_dl_avg = 0.8 * naive_drift + 0.1 * lstm_pred + 0.1 * trans_pred
```

- ML Hybrid와 동일한 가중 평균 전략을 DL에 적용
- Naive의 추세 추종력이 DL의 과적합을 보완할 수 있는지 검증

#### 결과

- DL 단독보다 크게 개선 (Naive의 안정성이 지배)
- 하지만 **Naive + ML(GB) Hybrid보다 나쁨** → DL의 20% 기여가 ML의 20% 기여보다 열등

### 5.4 H3: ML + DL 앙상블

#### 방법

```python
ml_lstm = 0.7 * gb_pred + 0.3 * lstm_pred
three_way = 0.6 * naive + 0.2 * gb + 0.1 * lstm + 0.1 * transformer
```

- GB와 DL의 예측 다양성이 보완적일 수 있다는 가설
- 3-way 앙상블(Naive + ML + DL)도 시도

#### 결과

- **ML+DL 앙상블이 ML 단독보다 열등** → DL이 노이즈를 추가
- 3-way 앙상블도 기존 Hybrid(Naive+ML)를 넘지 못함

### 5.5 H4: Quantile Regression

#### 구현

```python
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    qr = QuantileRegressor(quantile=q, alpha=0.1, solver='highs')
    qr.fit(X_train_scaled, y_train)
    qr_preds[q] = qr.predict(X_test_scaled)
```

- **목적**: 점 예측(point prediction)이 아닌 예측 구간(prediction interval) 생성
- 10%, 25%, 50%, 75%, 90% 분위수별 별도 모델 학습
- 80% 예측 구간: Q10~Q90, 50% 예측 구간: Q25~Q75

#### 왜 Quantile Regression을 시도했는가

- 급등 시장에서 "정확한 가격"보다 "가격이 이 범위 안에 있을 확률 80%"가 의사결정에 더 유용
- 불확실성 정량화는 리스크 관리의 핵심
- DL/ML이 점 예측에서 실패해도, 구간 예측이 유효하면 실무적 가치 있음

#### 결과

- 80% 구간 커버리지: 실제 측정 → 목표 80%에 근접 여부 확인
- Median(Q50) RMSE: ML/DL보다 나쁨 (선형 모델의 한계)
- **구간 예측 자체는 유효** → 불확실성 커뮤니케이션 도구로서 가치 있음

### 5.6 최종 가설 검증 결과

| 가설 | 결과 | 해석 |
|------|------|------|
| H1 | 부분적 개선 | 정규화로 과적합 완화되었으나 근본적 한계 존재 |
| H2 | Naive+DL < Naive+ML | DL의 노이즈가 ML보다 큼 |
| H3 | ML+DL < ML 단독 | DL이 앙상블에 부정적 기여 |
| H4 | 유효 | 점 예측 외 불확실성 정량화에 가치 |

### 5.7 이 노트북에서 가장 많이 배운 것

1. **DL은 만능이 아니다**: 소규모 금융 시계열(668주)에서 DL은 ML/Naive를 넘지 못함. 데이터 규모와 모델 복잡도의 균형이 핵심
2. **과적합 방지만으로는 부족**: Dropout/L2/모델 축소로 과적합을 줄여도, 학습 패턴과 Test 패턴이 다르면 무의미
3. **앙상블의 조건**: 앙상블이 효과적이려면 개별 모델이 서로 다른 강점을 가져야 함. DL이 ML과 같은 방향으로 틀리면 앙상블 가치 없음
4. **Quantile Regression의 실무적 가치**: "이 가격일 것이다"보다 "80% 확률로 이 범위"가 더 actionable한 정보일 수 있음

---

## 전체 실험에서의 종합 발견

### 모델별 최종 Test RMSE 비교

| 순위 | 모델 | RMSE | 출처 |
|------|------|------|------|
| 1 | Hybrid (Naive×0.8 + LightGBM_Tuned×0.2) | **~398** | sparta2_advanced |
| 2 | Hybrid (Naive×0.8 + GradientBoosting×0.2) | **406.80** | sparta2_backup |
| 3 | Naive_Drift | ~480.67 | sparta2_backup |
| 4 | Naive_Last | ~550+ | sparta2_backup |
| 5 | LSTM (TensorFlow, 단변량) | ~1,105 | sparta2_advanced |
| 6 | ARIMA(3,1,2) | ~1,212 | sparta2_advanced |
| 7 | GradientBoosting 단독 | ~1,175~1,328 | sparta2_backup |
| 8 | LightGBM 단독 | ~940 | sparta2_advanced |
| 9 | LSTM/Transformer (PyTorch, 다변량) | ~1,500~2,000 | dl_lstm_transformer |

### 프로젝트 전체에서 가장 중요한 교훈 5가지

1. **Naive 모델의 구조적 우위를 먼저 확인하라**: 어떤 복잡한 모델이든 Naive를 이기지 못하면, 시장 구조(추세/평균회귀)를 먼저 파악해야 함. 이 프로젝트의 Test 기간은 일방적 상승이었고, Naive Drift는 이를 자연스럽게 추종

2. **Hybrid 전략의 핵심은 "무엇을 얼마나 섞느냐"**: Naive 80% + ML 20%가 최적이었으며, 이는 ML이 "약간의 보정"만 담당하고 대부분의 예측을 Naive에 위임하는 구조. ML 비중을 높이면 과적합 위험 증가

3. **Validation과 Test의 괴리를 항상 경계하라**: 3단계 스태킹은 Validation에서 인상적이었지만 Test에서 완패. 이는 시계열 예측의 본질적 어려움 — 미래는 과거와 다를 수 있음

4. **모델 복잡도와 데이터 규모의 균형**: 668주라는 데이터로 LSTM/Transformer를 학습시키는 것은 과적합의 레시피. 부스팅 모델(+SHAP 피처 선택)이 이 규모에서 가장 실용적

5. **해석 가능성의 가치**: SHAP을 통해 "왜 이 예측인가"를 설명할 수 있는 것은 블랙박스 DL에서 불가능. 실무에서는 예측 정확도만큼 해석 가능성이 중요 (중국 경기→니켈 수요, 달러 강약→원자재 약세 등)
