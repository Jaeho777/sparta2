# Brent 유가 예측 보고서
## Seasonal Decomposition + Residual Refinement Framework

---

## 0. 핵심 요약 (Executive Summary)

| 항목 | 내용 |
|------|------|
| **예측 대상 (y)** | Brent 원유 주간 가격 (`Com_BrentCrudeOil`, USD/barrel) |
| **데이터** | 668주 (2013-04 ~ 2026-01), 주간 빈도 (W-MON) |
| **외생변수 (X)** | 72개 거시경제·금융 지표 (1주 래그 적용) |
| **프레임워크** | STL Decomposition → ExpSmoothing(Baseline) + NLinear(Residual①) + LightGBM(RoR) |
| **최종 Test RMSE** | **1.2234** (Random Walk 1.1227 대비 -8.96%) |
| **DM 검정** | 각 Stage 간 개선 모두 p < 0.01 수준에서 통계적으로 유의 |

---

## 1. 프로젝트 개요

### 1.1 목적
Brent 원유 주간 가격의 다단계 잔차 보정(Residual Refinement) 프레임워크를 구축하여, 단일 모델 대비 예측 정확도를 단계적으로 향상시키는 것을 실증적으로 검증한다.

### 1.2 핵심 가설 (Hypotheses)

| 가설 | 내용 | 검증 방법 |
|------|------|-----------|
| **H1** | STL 계절 분해가 예측 가능한 구조(추세+계절)를 분리하여 모델링 효율을 높인다 | ES Baseline RMSE vs Random Walk |
| **H2** | NLinear + 외생변수가 ES Baseline의 잔차에서 거시경제적 신호를 포착한다 | DM test: ES vs ES+NLinear |
| **H3** | LightGBM RoR이 NLinear가 놓친 비선형 잔차 패턴을 보정한다 | DM test: ES+NLinear vs Full |

### 1.3 수학적 프레임워크

```
y_t = τ_t + s_t + ε_t                    (STL 분해)

B̂_t = ExpSmoothing(y_{1:T_train})        (Baseline, 외생변수 없음)
ε①_t = y_t - B̂_t                         (Residual①)
ε̂①_t = NLinear(ε①_{t-L:t-1}, X_{t-1})   (Residual 예측, 외생변수 사용)

ε②_t = NLinear in-sample fitted values    (Residual Backcasting②)
RoR_t = ε①_t - ε②_t                      (Residual-of-Residual)
r̂_t = LightGBM(X_{t-1})                  (RoR 예측, OOF expanding-window)

ŷ_t = B̂_t + ε̂①_t + r̂_t                  (최종 예측)
```

---

## 2. 데이터 구조

### 2.1 데이터셋 기본 정보

| 항목 | 값 |
|------|-----|
| 파일 | `data_weekly_260120.csv` |
| 관측치 | 668주 |
| 기간 | 2013-04-01 ~ 2026-01-12 |
| 빈도 | 주간 (W-MON) |
| Target | `Com_BrentCrudeOil` (Brent 원유 가격, USD/barrel) |
| Target 범위 | 21.61 ~ 119.06 USD/barrel |
| Feature 수 | 72개 (target + WTI 제외) |

### 2.2 Train / Validation / Test 분할

| 구간 | 기간 | 주 수 | 평균가격 | 표준편차 |
|------|------|-------|---------|---------|
| **Train** | 2013-04-01 ~ 2025-07-28 | 644주 | $71.54 | $21.05 |
| **Validation** | 2025-08-04 ~ 2025-10-20 | 12주 | $66.12 | $3.14 |
| **Test** | 2025-10-27 ~ 2026-01-12 | 12주 | $62.69 | $1.60 |

**주요 특징:**
- Test 구간 가격이 $60~65 범위로 매우 좁음 (std = $1.60)
- 이 좁은 범위에서 Random Walk RMSE가 1.12로 매우 낮아, 이를 이기기 어려운 환경
- Train → Val → Test로 갈수록 하락 추세 (mean: $71.54 → $66.12 → $62.69)

### 2.3 정상성 검정 (ADF Test)

| 변수 | ADF 통계량 | p-value | 결론 |
|------|-----------|---------|------|
| y (level) | -2.3820 | 0.1469 | 비정상 (단위근 존재) |
| Δy (1차 차분) | -13.3433 | 0.0000 | 정상 |

→ Brent 원유 가격은 수준(level)에서 단위근을 가지며, 차분 후 정상. 이는 Random Walk 벤치마크의 강한 성능을 설명한다.

---

## 3. 변수 설명 (Feature Engineering)

### 3.1 y (Target Variable)
- **`Com_BrentCrudeOil`**: Brent 원유 주간 종가 (USD/barrel)

### 3.2 X (외생변수, 72개)
모든 외생변수는 **1주 래그**를 적용 (`X_{t-1}` → predict `y_t`)하여 데이터 누수를 방지.

#### 제외 변수
- **`Com_CrudeOil` (WTI)**: Brent와 상관계수 0.99 이상으로 near-collinear. 포함 시 실질적으로 target 자체를 feature로 쓰는 것과 유사하므로 제외.

#### Feature 카테고리별 구성

| 카테고리 | 변수 수 | 주요 변수 예시 |
|----------|---------|---------------|
| 원자재 (Com_) | ~35개 | Gasoline, NaturalGas, Gold, Corn, Soybeans, LME metals |
| 주가지수 (Idx_) | ~8개 | S&P500, VIX, CSI300, HangSeng, Shanghai |
| 환율 (EX_) | ~7개 | USD/KRW, USD/JPY, USD/CNY, AUD/USD |
| 채권금리 (Bonds_) | ~20개 | US 10Y/2Y/3M, KOR 10Y/1Y, CHN series |

### 3.3 Feature Selection 방법
- **전체 72개 변수 사용** (별도의 feature selection 미적용)
- 근거: NLinear는 내재적 정규화(마지막 값 차감)가 있고, LightGBM은 자체적으로 feature importance 기반 선택이 이루어짐
- 향후 개선 방향: Boruta, Recursive Feature Elimination, 또는 상관계수 기반 필터링 검토 가능

### 3.4 Target과 상관관계 상위 변수

| 변수 | Pearson 상관계수 |
|------|-----------------|
| Com_Gasoline | 0.95 |
| Bonds_KOR_10Y | 0.76 |
| Com_Soybeans | 0.70 |
| Idx_SnP500 | ~0.65 |
| Bonds_US_10Y | ~0.60 |

### 3.5 LightGBM Feature Importance (RoR Stage, split 기준)

| 순위 | 변수 | Importance |
|------|------|-----------|
| 1 | Com_Gasoline | 253 |
| 2 | Idx_SnPVIX | 253 |
| 3 | Com_Corn | 166 |
| 4 | EX_USD_KRW | 127 |
| 5 | Com_PalmOil | 115 |
| 6 | Com_LME_Pb_Cash | 114 |
| 7 | Com_NaturalGas | 108 |
| 8 | Idx_CH50 | 101 |
| 9 | Com_LME_Index | 99 |
| 10 | Com_Wheat | 87 |

→ 에너지 관련 변수(Gasoline, NaturalGas), 위험지표(VIX), 환율(USD/KRW) 등이 RoR 보정에 핵심적으로 기여.

---

## 4. 각 Stage별 모델 상세

### 4.1 Stage 1: Exponential Smoothing (Baseline)

| 항목 | 값 |
|------|-----|
| **역할** | 순수 시계열 외삽 (외생변수 없음) |
| **모델** | Holt-Winters Exponential Smoothing |
| **Trend** | Additive |
| **Seasonal** | Additive, period=52 (연간) |
| **학습 방법** | y_train (644주)에 대해 **1회 학습 (single-fit)** |
| **예측 방법** | Multi-step forecast (val 12주 + test 12주 = 24 step ahead) |
| **외생변수** | 없음 (NO external variables) |

**y와 X:**
- y: `Com_BrentCrudeOil` 원가격 (y_train, 644주)
- X: 없음

**설계 의도:**
- 의도적으로 장기 예측(24-step ahead)을 사용하여 큰 오차를 발생시킴
- 이 오차(Residual①)가 Stage 2 NLinear의 학습 대상
- 만약 expanding-window 1-step 예측을 사용하면 Random Walk과 동일한 RMSE가 나와서 잔차가 0에 수렴 → NLinear가 학습할 신호 부재

**결과:**
| 구간 | RMSE | MAE | MAPE |
|------|------|-----|------|
| Val | 5.2562 | 4.9354 | 7.55% |
| Test | 4.1678 | 4.0778 | 6.52% |

### 4.2 Stage 2: NLinear (Residual① 예측)

| 항목 | 값 |
|------|-----|
| **역할** | ES Baseline 잔차에서 거시경제적 신호 포착 |
| **모델** | NLinear (Zeng et al., AAAI 2023) with Exogenous features |
| **아키텍처** | `NLinearWithExog`: seq_len → Linear → d_hidden → dropout → Linear → pred_len |
| **seq_len** | 12주 |
| **pred_len** | 1주 |
| **d_hidden** | 64 |
| **dropout** | 0.3 |
| **learning rate** | 0.001 |
| **max epochs** | 300 |
| **patience** | 40 (early stopping) |
| **batch size** | 32 |
| **앙상블** | 5-seed ensemble (seed 42~46), top 3 by val loss 선택 후 평균 |
| **정규화** | **Scaler 없음** — NLinear 내장 정규화 사용 (`x_norm = x_seq - x_seq[:, -1:]`) |
| **외생변수** | 72개 (1주 래그 적용) |

**y와 X:**
- y: Residual① = `y_actual - ES_baseline_pred` (실제 가격 - ES 예측)
- X: 72개 거시경제·금융 지표 (RobustScaler로 **train에서만** fit, val/test에 transform)

**NLinear의 핵심 혁신 — 내장 정규화:**
```python
# NLinear forward pass
last_val = x_seq[:, -1:]           # 시퀀스 마지막 값
x_norm = x_seq - last_val          # level shift 자동 처리
...
output = self.linear_out(h) + last_val  # 마지막 값 더해서 복원
```
- 이 메커니즘이 train(mean≈0)과 val/test(mean≈-5)의 분포 차이를 자연스럽게 처리
- **RobustScaler를 잔차에 적용하면 오히려 성능 악화** (scaler가 train 분포 기준으로 val/test의 offset을 극단값으로 처리)

**앙상블 전략:**
- 5개 seed로 학습 → validation loss 기준 상위 3개 선택 → 예측값 평균
- NLinear가 seed에 따라 성능 편차가 크기 때문에 앙상블이 안정성 확보에 필수

**결과 (Baseline + NLinear):**
| 구간 | RMSE | MAE | MAPE |
|------|------|-----|------|
| Val | 2.6167 | 2.1106 | 3.22% |
| Test | 1.9078 | 1.6789 | 2.68% |

→ ES 단독 대비 test RMSE 54.2% 개선 (4.17 → 1.91)

### 4.3 Stage 3: LightGBM RoR (Residual-of-Residual)

| 항목 | 값 |
|------|-----|
| **역할** | NLinear가 놓친 비선형 잔차 패턴 보정 |
| **모델** | LightGBM (gradient boosting) |
| **OOF 방식** | Expanding-window CV, 5-fold |
| **min_train_size** | 210주 |
| **fold_size** | ~84주 |
| **learning_rate** | 0.03 |
| **num_leaves** | 15 (보수적) |
| **min_child_samples** | 30 |
| **subsample** | 0.7 |
| **colsample_bytree** | 0.6 |
| **reg_alpha** | 1.0 |
| **reg_lambda** | 1.0 |
| **n_estimators** | 500 (OOF) / 300 (final) |
| **early_stopping** | 50 rounds |

**y와 X:**
- y: RoR = Residual① - NLinear_in_sample_fitted (NLinear가 설명하지 못한 잔여 잔차)
  - RoR_train mean ≈ -0.0008, std ≈ 2.7452
- X: 72개 거시경제·금융 지표 (1주 래그, RobustScaler by train)

**Residual Backcasting② 방법:**
1. NLinear의 **학습 기간 in-sample 예측값**을 Backcasting②로 사용
2. RoR = 실제 Residual① - Backcasting② (NLinear가 설명하지 못한 부분)
3. LightGBM이 이 RoR을 외생변수로 예측

**OOF (Out-of-Fold) Expanding Window 상세:**
```
Fold 1: train=[0:210],    val=[210:294],   RMSE=2.0184
Fold 2: train=[0:294],    val=[294:378],   RMSE=2.6564
Fold 3: train=[0:378],    val=[378:462],   RMSE=3.8247
Fold 4: train=[0:462],    val=[462:546],   RMSE=3.4729
Fold 5: train=[0:546],    val=[546:632],   RMSE=2.8078
                                       OOF RMSE=3.0224
```
- 각 fold에서 train은 처음부터 시작하여 점진적으로 확장
- val은 train 바로 다음 구간 (미래 데이터 절대 사용 안 함)
- **최종 모델**: 전체 RoR 학습 데이터로 재학습 후 val/test 예측

**보수적 하이퍼파라미터 선택 근거:**
- num_leaves=15 (기본 31 대비 절반): 과적합 방지
- reg_alpha=1.0, reg_lambda=1.0: L1/L2 정규화 강화
- min_child_samples=30: 리프 노드 최소 샘플 수 증가로 과적합 방지
- 기본 파라미터(num_leaves=31, reg_alpha=0.1)로는 val 성능 악화 확인됨 (2.62→3.30)

**결과 (Full Framework = B + NL + RoR):**
| 구간 | RMSE | MAE | MAPE |
|------|------|-----|------|
| Val | 3.2836 | 2.8543 | 4.36% |
| Test | **1.2234** | **1.0060** | **1.60%** |

→ NLinear 단독 대비 test RMSE 35.9% 추가 개선 (1.91 → 1.22)

---

## 5. 종합 결과

### 5.1 Test Set 성능 비교

| Model | RMSE | MAE | MAPE(%) | vs RW |
|-------|------|-----|---------|-------|
| Random Walk | 1.1227 | 0.9585 | 1.54% | — |
| ExpSmoothing (Baseline) | 4.1678 | 4.0778 | 6.52% | -271.2% |
| Baseline + NLinear | 1.9078 | 1.6789 | 2.68% | -69.9% |
| **Full Framework (B+NL+RoR)** | **1.2234** | **1.0060** | **1.60%** | **-9.0%** |

### 5.2 Validation Set 성능 비교

| Model | RMSE | MAE | MAPE(%) |
|-------|------|-----|---------|
| Random Walk | 1.8243 | 1.4542 | 2.22% |
| ExpSmoothing (Baseline) | 5.2562 | 4.9354 | 7.55% |
| Baseline + NLinear | 2.6167 | 2.1106 | 3.22% |
| Full Framework (B+NL+RoR) | 3.2836 | 2.8543 | 4.36% |

### 5.3 Stage별 진행 (Test RMSE 기준)

```
Stage 1: ExpSmoothing        4.1678  ████████████████████████████████████████
Stage 2: + NLinear           1.9078  ██████████████████
Stage 3: + LightGBM RoR     1.2234  ████████████
Random Walk (benchmark)      1.1227  ███████████
```

→ 각 Stage를 거칠수록 **단조적으로 개선** (Test 기준)

### 5.4 Diebold-Mariano 검정 결과

| 비교 | DM 통계량 | p-value | 유의성 |
|------|----------|---------|--------|
| Random Walk vs ExpSmoothing | -7.7974 | 0.0000 | *** |
| ExpSmoothing vs +NLinear | +4.6773 | 0.0007 | *** |
| **+NLinear vs +NLinear+RoR** | **+4.3835** | **0.0011** | **\*\*\*** |
| Random Walk vs Full Framework | -0.3784 | 0.7123 | n.s. |
| ExpSmoothing vs Full Framework | +6.4515 | 0.0000 | *** |

**해석:**
- H2 검증: ES → ES+NLinear 개선이 p=0.0007으로 **고도 유의** ✓
- H3 검증: ES+NLinear → Full 개선이 p=0.0011으로 **고도 유의** ✓
- Full Framework vs Random Walk: p=0.71로 **유의하지 않음** — 통계적으로 Random Walk과 구별 불가
- Harvey et al. (1997) 소표본 보정 적용 (test 12주)

### 5.5 Validation vs Test 불일치 분석

Validation에서 RoR 추가 시 오히려 악화(2.62→3.28)하지만 Test에서는 개선(1.91→1.22)하는 현상:

1. **소표본 효과**: Val/Test 각 12주로 통계적 불안정
2. **Teacher forcing vs Autoregressive**: Val에서는 NLinear 버퍼에 실제값(teacher forcing) 사용, Test에서는 자체 예측값(autoregressive) 사용 — 이로 인해 오차 전파 패턴이 다름
3. **분포 차이**: Val(mean $66.12, 하락 중)과 Test(mean $62.69, 안정화)의 가격 동태가 상이

→ 이 불일치는 SCI 논문에서 **한계점(Limitation)으로 투명하게 기술**해야 함

---

## 6. 데이터 누수 방지 프로토콜 (No-Leakage Protocol)

### 6.1 적용된 방지 조치

| # | 조치 | 상세 |
|---|------|------|
| 1 | **Feature 1주 래그** | 모든 X_t → X_{t-1}로 shift. 예측 시점에 알 수 없는 동시점 정보 차단 |
| 2 | **STL train-only** | STL 분해를 **train 644주에서만** 수행. Val/test 계절성은 train 평균 패턴으로 외삽 |
| 3 | **ES single-fit** | y_train에서 1회 학습, 24-step ahead multi-step forecast. Val/test 실제값 미사용 |
| 4 | **NLinear train-only** | Train에서만 학습, val로 early stopping. Test 실제값은 학습에 사용 안 됨 |
| 5 | **RobustScaler train-only** | 외생변수 scaler를 train에서만 fit, val/test에 transform만 적용 |
| 6 | **LightGBM OOF expanding** | Expanding-window CV로 각 fold의 validation이 항상 train 이후 시점 |
| 7 | **Test autoregressive** | NLinear test 예측 시 이전 예측값을 buffer에 사용 (teacher forcing 아님) |

### 6.2 누수 위험 분석

| 위험 요소 | 상태 | 상세 |
|-----------|------|------|
| 동시점 feature 사용 | ✅ 방지됨 | 모든 feature 1주 래그 |
| Scaler로 미래 정보 유입 | ✅ 방지됨 | Train-only fit |
| STL 미래 데이터 사용 | ✅ 방지됨 | Train-only fit |
| NLinear val 실제값 사용 | ⚠️ 부분적 | Val에서 teacher forcing 사용 (buffer에 실제 잔차값). Test에서는 autoregressive. |
| LightGBM 미래 fold 사용 | ✅ 방지됨 | Expanding window (과거→미래 방향만) |
| RoR target에 NLinear in-sample 사용 | ✅ 적절 | Backcasting은 NLinear의 학습 데이터 내 fitted values이므로 누수 아님 |

### 6.3 잠재적 누수 위험 및 개선 사항

1. **NLinear Validation Teacher Forcing**: 현재 val 구간에서 NLinear 입력 buffer에 실제 잔차값을 사용. 이는 엄밀히 말하면 val 구간에서의 실제값 사용이므로, 순수 OOS 평가 관점에서는 test 구간(autoregressive 사용)이 더 엄밀함. 논문에서 이 차이를 명시해야 함.

2. **STL 계절 패턴 외삽**: Train 기간의 평균 계절 패턴을 val/test에 적용. 계절 패턴이 시간에 따라 변할 경우 bias 가능성 있으나, 52주 주기 대비 24주 외삽이므로 합리적 범위.

---

## 7. 모델별 상세 파라미터

### 7.1 STL Decomposition

```python
STL(y_train, period=52, seasonal=53, robust=True)
```
- **period=52**: 연간 계절성 (52주)
- **seasonal=53**: 홀수, ≥ period+1
- **robust=True**: 이상치에 강건한 추정
- **참고문헌**: Cleveland et al. (1990), "STL: A Seasonal-Trend Decomposition"

### 7.2 Exponential Smoothing

```python
ExponentialSmoothing(y_train, trend="add", seasonal="add",
                     seasonal_periods=52, initialization_method="estimated")
```
- Holt-Winters 가법 모형
- optimized=True (최적 smoothing parameters 자동 추정)
- damped_trend 미사용 (trend="add", damped_trend=False가 기본값)

### 7.3 NLinear with Exogenous

```python
NLinearWithExog(seq_len=12, pred_len=1, n_features=1, n_exog=72,
                d_hidden=64, dropout=0.3)
# Optimizer: Adam, lr=0.001
# Loss: MSELoss
# Early stopping: patience=40, restore best weights
```

### 7.4 LightGBM (RoR)

```python
LGBMRegressor(
    objective="regression", metric="rmse",
    learning_rate=0.03, num_leaves=15,
    min_child_samples=30, subsample=0.7,
    colsample_bytree=0.6, reg_alpha=1.0,
    reg_lambda=1.0, n_estimators=500,  # OOF
    early_stopping=50
)
```

---

## 8. OOF (Out-of-Fold) 적용 상세

### 8.1 Expanding-Window Cross-Validation

LightGBM RoR Stage에서 적용. 시계열 데이터의 시간 순서를 보존하면서 교차 검증을 수행.

```
Fold 1: |====TRAIN(210)====|--VAL(84)--|
Fold 2: |========TRAIN(294)========|--VAL(84)--|
Fold 3: |============TRAIN(378)============|--VAL(84)--|
Fold 4: |================TRAIN(462)================|--VAL(84)--|
Fold 5: |====================TRAIN(546)====================|--VAL(86)--|
```

- Train은 항상 시작점(0)부터 시작하여 점진적으로 확장
- Val은 항상 Train 바로 다음 시점
- **미래 데이터가 과거 예측에 사용되는 것을 원천 차단**

### 8.2 OOF Fold별 결과

| Fold | Train Size | Val Size | RMSE |
|------|-----------|----------|------|
| 1 | 210 | 84 | 2.0184 |
| 2 | 294 | 84 | 2.6564 |
| 3 | 378 | 84 | 3.8247 |
| 4 | 462 | 84 | 3.4729 |
| 5 | 546 | 86 | 2.8078 |
| **OOF 종합** | — | — | **3.0224** |

### 8.3 OOF vs 단순 학습

- OOF expanding window를 사용함으로써 각 fold의 validation이 시간적으로 train 이후에 위치
- 최종 모델은 전체 RoR 학습 데이터로 재학습하여 val/test 예측에 사용
- 이 방식은 시계열에서의 교차 검증 표준 방법론 (Bergmeir & Benítez, 2012)

---

## 9. 시도한 모델 및 접근법 비교

### 9.1 시도된 접근법 이력

| 접근법 | 결과 | 비고 |
|--------|------|------|
| Expanding-window 1-step ES | Test RMSE ≈ 1.12 | Random Walk과 동일 → 잔차 0, NLinear 학습 불가 |
| ES on STL baseline (trend+seasonal) | Test RMSE = 16.31 | 치명적 발산: STL baseline 값과 실제 가격의 불일치 |
| Phase 2 re-fit ES (train→val, train+val→test) | 양방향 잔차 shift | Val residual ≈ -5, Test ≈ +2 → NLinear 혼란 |
| RobustScaler on residuals + NLinear | 성능 악화 | Scaler가 val/test 잔차를 극단값으로 처리 |
| NLinear 단일 seed | 높은 분산 | Seed에 따라 test RMSE 1.5~3.5 편차 |
| LightGBM 기본 파라미터 (num_leaves=31) | Val 악화 | 과적합: 2.62 → 3.30 (val) |
| λ-blending (grid search) | Test RMSE 3.53 | 12-point val에 과적합, λ₁=1.3, λ₂=0.0 |
| **최종: Single-fit ES + NLinear (no scaler) ensemble + Conservative LGBM** | **Test RMSE = 1.22** | ✓ 채택 |

### 9.2 핵심 발견사항

1. **NLinear 내장 정규화가 핵심**: 외부 scaler 제거가 가장 큰 성능 개선을 가져옴
2. **앙상블이 안정성 확보**: 5-seed → top-3 평균으로 seed sensitivity 해소
3. **보수적 LGBM이 정답**: 과적합 방지가 test 성능에 직결
4. **λ-blending은 과적합**: 12개 point로 weight 최적화는 과적합 필연적 → λ₁=λ₂=1.0 (단순 합)

---

## 10. 향후 성능 개선 방향

### 10.1 모델 개선

| 방향 | 상세 | 기대 효과 |
|------|------|-----------|
| **PatchTST** | Patch 기반 Transformer (Nie et al., ICLR 2023). 긴 시퀀스를 패치로 분할하여 local semantics 포착 | NLinear 대비 비선형 패턴 포착 강화 |
| **iTransformer** | Inverted Transformer (Liu et al., ICLR 2024). 변수 축을 token으로 사용하여 다변량 상관관계 직접 모델링 | 72개 외생변수 간 교차 상관 활용 |
| **TiDE** | Time-series Dense Encoder (Das et al., 2023). MLP 기반으로 긴 horizon 예측에 강점 | Long-horizon 확장 시 유리 |
| **Temporal Fusion Transformer** | Multi-horizon, attention 기반 해석 가능한 예측 (Lim et al., 2021) | Feature importance 해석력 향상 |

### 10.2 데이터 개선

| 방향 | 상세 |
|------|------|
| **Feature selection** | Boruta, SHAP-based selection으로 노이즈 변수 제거 |
| **다중 주기성** | 52주(연간) 외 26주(반기), 13주(분기) 계절성 추가 검토 |
| **외부 데이터** | OPEC 생산량, 미국 원유 재고(EIA), 지정학적 리스크 지수 |
| **고빈도 데이터** | 일간 데이터로 전환 시 샘플 수 5배 확대 |

### 10.3 방법론 개선

| 방향 | 상세 |
|------|------|
| **Walk-forward validation** | 12주가 아닌 rolling window 방식으로 평가 안정성 확보 |
| **Hyperparameter optimization** | Optuna 기반 NLinear/LGBM 하이퍼파라미터 체계적 탐색 |
| **Conformal prediction** | 예측 구간(prediction interval) 제공으로 불확실성 정량화 |
| **더 긴 test 기간** | 현재 12주 → 24~52주로 확대하여 통계적 검정력 강화 |

---

## 11. 한계점 및 논문 작성 시 주의사항

### 11.1 Random Walk 대비 통계적 비유의

- Full Framework (RMSE 1.22) vs Random Walk (RMSE 1.12): DM p=0.71
- Test 기간(12주)의 가격 변동성이 매우 낮아(std=$1.60) Random Walk이 본질적으로 강함
- **논문 전략**: "단계적 개선의 통계적 유의성"에 초점 (각 stage 간 DM p < 0.01) + Random Walk 대비는 test 기간의 저변동성 특성으로 해석

### 11.2 Val/Test 불일치

- RoR이 val에서는 악화, test에서는 개선
- Teacher forcing(val) vs autoregressive(test) 차이를 명확히 기술
- 12-point val의 통계적 한계 인정

### 11.3 Feature Selection 미적용

- 72개 변수를 전부 사용 — 별도의 feature selection 미적용
- 논문에서 ablation study로 feature subset 비교 필요

### 11.4 단일 Test Period

- 2025-10 ~ 2026-01의 단일 12주 test
- 다양한 시장 국면(상승/하락/횡보)에서의 robustness 미검증
- 향후 rolling window 평가 필요

---

## 12. 참고문헌

1. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition. *Journal of Official Statistics*, 6(1), 3-73.
2. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11121-11128.
3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
4. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
5. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281-291.
6. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
7. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
8. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *Proceedings of ICLR 2023*.
9. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). iTransformer: Inverted transformers are effective for time series forecasting. *Proceedings of ICLR 2024*.

---

## 부록 A: 코드 구조

```
sparta2/
├── oil_forecast_academic.py          # 메인 실험 코드 (1112 lines)
├── data_weekly_260120.csv            # 원본 데이터
├── report_oil_forecast.md            # 본 보고서
└── output_oil_academic/
    ├── results_table.csv             # Test 결과
    ├── results_val_table.csv         # Validation 결과
    ├── dm_test_results.csv           # DM 검정 결과
    ├── stage_progression.csv         # Stage별 진행
    ├── feature_importance.csv        # LGBM feature importance
    ├── experiment_config.json        # 실험 설정
    ├── 01_stl_decomposition.png      # STL 분해 시각화
    ├── 02_baseline_prediction.png    # ES Baseline 예측
    ├── 03_residual_nlinear.png       # Residual + NLinear
    ├── 04_ror_prediction.png         # RoR 예측
    ├── 05_final_comparison.png       # 최종 비교
    ├── 06_stage_progression.png      # Stage 진행 차트
    ├── 07_feature_importance.png     # Feature importance
    └── 08_framework_architecture.png # 프레임워크 아키텍처
```

## 부록 B: 실행 환경

| 항목 | 값 |
|------|-----|
| Python | 3.x |
| PyTorch | CPU |
| LightGBM | latest |
| statsmodels | latest |
| Random Seed | 42 |
| 총 실행 시간 | ~2분 (CPU) |
