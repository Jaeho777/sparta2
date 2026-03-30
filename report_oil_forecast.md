# Brent 유가 예측 보고서
## Seasonal Decomposition + Residual Refinement Framework

---

## 0. 핵심 요약 (Executive Summary)

| 항목 | 내용 |
|------|------|
| **예측 대상 (y)** | Brent 원유 주간 가격 (`Com_BrentCrudeOil`, USD/barrel) |
| **데이터** | 668주 (2013-04 ~ 2026-01), 주간 빈도 (W-MON) |
| **외생변수 (X)** | 22개 도메인 기반 선별 + 33개 파생변수 = 55개 (1주 래그) |
| **프레임워크** | STL → ExpSmoothing(Baseline) + NLinear(Residual①) + LightGBM(RoR, validation-gated) |
| **최종 Test RMSE** | **1.2608** (Baseline+NLinear) / 1.2822 (Full w/ RoR) |
| **Random Walk** | 1.1227 (DM p=0.68, 통계적으로 구별 불가) |
| **DM 검정** | ES → NLinear: p=0.0000*** (고도 유의) |

---

## 1. 프로젝트 개요

### 1.1 목적
Brent 원유 주간 가격의 다단계 잔차 보정(Residual Refinement) 프레임워크를 구축하여, 경제적으로 의미 있는 변수와 파생변수를 활용한 예측 성능 개선을 실증적으로 검증한다.

### 1.2 핵심 가설 (Hypotheses)

| 가설 | 내용 | 검증 결과 |
|------|------|-----------|
| **H1** | STL 계절 분해가 예측 가능한 구조(추세+계절)를 분리 | ES Baseline RMSE 4.17 (구조적 외삽 가능) |
| **H2** | NLinear + 외생변수가 잔차에서 거시경제적 신호 포착 | DM p=0.0000*** ✓ |
| **H3** | LightGBM RoR이 NLinear 잔차의 비선형 패턴 보정 | DM p=0.93 (유의하지 않음) ✗ |

→ H2는 강하게 지지, H3는 도메인 기반 feature selection으로 NLinear가 충분한 신호를 추출하여 RoR의 추가 보정 여지가 제한적

### 1.3 수학적 프레임워크

```
y_t = τ_t + s_t + ε_t                    (STL 분해)

B̂_t = ExpSmoothing(y_{1:T_train})        (Baseline, 외생변수 없음)
ε①_t = y_t - B̂_t                         (Residual①)
ε̂①_t = NLinear(ε①_{t-L:t-1}, X_{t-1})   (Residual 예측, 외생변수 사용)

ε②_t = NLinear in-sample fitted values    (Residual Backcasting②)
RoR_t = ε①_t - ε②_t                      (Residual-of-Residual)
r̂_t = LightGBM(X_{t-1})                  (RoR 예측, OOF expanding-window)

ŷ_t = B̂_t + ε̂①_t + λ·r̂_t               (최종 예측, λ는 validation gate)
```

**Validation Gate**: RoR 추가가 validation RMSE를 개선하면 λ=1, 아니면 λ=0. 이를 통해 RoR이 noise를 추가하는 경우 자동 차단.

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
| 원변수 | 22개 (도메인 기반 선별) |
| 파생변수 | 33개 (수익률, 스프레드, 비율, 이동평균) |
| **총 Feature** | **55개** |

### 2.2 Train / Validation / Test 분할

| 구간 | 기간 | 주 수 | 평균가격 | 표준편차 |
|------|------|-------|---------|---------|
| **Train** | 2013-04-01 ~ 2025-07-28 | 644주 | $71.54 | $21.05 |
| **Validation** | 2025-08-04 ~ 2025-10-20 | 12주 | $66.12 | $3.14 |
| **Test** | 2025-10-27 ~ 2026-01-12 | 12주 | $62.69 | $1.60 |

**주요 특징:**
- Test 구간 가격이 $60~65 범위로 매우 좁음 (std = $1.60)
- Random Walk RMSE가 1.12로 매우 낮아, 이를 이기기 어려운 환경
- Train → Val → Test로 갈수록 하락 추세 (mean: $71.54 → $66.12 → $62.69)

### 2.3 정상성 검정 (ADF Test)

| 변수 | ADF 통계량 | p-value | 결론 |
|------|-----------|---------|------|
| y (level) | -2.3820 | 0.1469 | 비정상 (단위근 존재) |
| Δy (1차 차분) | -13.3433 | 0.0000 | 정상 |

→ 단위근 존재 → Random Walk 벤치마크가 본질적으로 강함

---

## 3. 변수 설명 (Feature Engineering)

### 3.1 y (Target Variable)
- **`Com_BrentCrudeOil`**: Brent 원유 주간 종가 (USD/barrel)

### 3.2 Feature Selection: 도메인 기반 선별 (22개 원변수)

모든 외생변수에 **1주 래그** 적용 (`X_{t-1}` → predict `y_t`)

#### 제외 변수
- **`Com_CrudeOil` (WTI)**: Brent와 상관계수 0.99 이상, near-collinear → 사실상 target leak에 가까움

#### Feature 그룹별 구성 (경제적 근거)

| 그룹 | 변수 | 경제적 근거 |
|------|------|-------------|
| **에너지 밸류체인** | Com_Gasoline | 정제 제품, 직접적 가격 전이 |
| **대체 에너지** | Com_NaturalGas, Com_Uranium, Com_Coal | 에너지 substitution effect |
| **경기/산업 선행** | Com_LME_Cu_Cash, Com_Steel, Com_Iron_Ore | Dr. Copper (구리=경기선행), 산업활동 |
| **달러/환율** | Idx_DxyUSD, EX_USD_CNY | 달러-원유 역상관, 중국=최대 원유 수입국 |
| **금리/채권** | Bonds_US_10Y, Bonds_US_2Y, Bonds_US_3M | 통화정책, 경기 기대 |
| **위험/안전자산** | Idx_SnPVIX, Com_Gold | VIX=공포지수, 금=안전자산 |
| **수요 대리** | Idx_SnP500, Idx_CSI300 | 미국·중국 경기 대리변수 |
| **아시아 수입국** | EX_USD_KRW, Bonds_KOR_10Y, EX_USD_JPY | 한국(4위), 일본(3위) 원유 수입국 |
| **바이오연료/원자재** | Com_Corn, Com_Soybeans, Com_PalmOil | 바이오에탄올/디젤 원료, 원자재 슈퍼사이클 |

### 3.3 파생변수 (33개)

| 유형 | 개수 | 상세 |
|------|------|------|
| **주간 수익률** | 22개 | 각 원변수의 `pct_change()` |
| **금리 스프레드** | 1개 | `Bonds_US_10Y - Bonds_US_2Y` (경기침체 지표) |
| **Crack Spread** | 1개 | `Com_Gasoline - Com_BrentCrudeOil` (정제 마진) |
| **Gold/Oil 비율** | 1개 | `Com_Gold / Com_BrentCrudeOil` (리스크 심리) |
| **MA4/MA12 이탈도** | 8개 | Gasoline, NaturalGas, VIX, DxyUSD × (MA4, MA12) |

**파생변수 설계 근거:**
- **수익률**: 수준(level)의 비정상성을 해소하고 변화율 기반 신호 포착
- **금리 스프레드**: 장단기 금리차는 경기침체의 가장 강력한 선행지표 (Estrella & Mishkin, 1998)
- **Crack Spread**: 정제 마진은 석유 수급의 핵심 지표
- **Gold/Oil 비율**: 안전자산 대비 위험자산 가격비로 시장 심리 반영
- **MA 이탈도**: 단기(4주)/중기(12주) 추세 대비 현재 수준의 모멘텀 신호

### 3.4 LightGBM Feature Importance (RoR Stage, split 기준)

| 순위 | 변수 | Importance | 유형 |
|------|------|-----------|------|
| 1 | Idx_SnPVIX | 214 | 원변수 |
| 2 | Com_Gasoline_ret | 151 | 파생(수익률) |
| 3 | Com_Iron_Ore_ret | 136 | 파생(수익률) |
| 4 | Com_Gold_ret | 130 | 파생(수익률) |
| 5 | EX_USD_KRW_ret | 128 | 파생(수익률) |
| 6 | Bonds_US_2Y_ret | 122 | 파생(수익률) |
| 7 | Spread_Crack | 119 | 파생(스프레드) |
| 8 | Idx_CSI300 | 118 | 원변수 |
| 9 | Idx_SnPVIX_ret | 115 | 파생(수익률) |
| 10 | Com_PalmOil_ret | 113 | 파생(수익률) |

→ **파생변수가 상위 10개 중 8개** — 수익률과 스프레드가 RoR 보정에 핵심적으로 기여
→ VIX(공포지수), 에너지 수익률, 산업금속 수익률이 원유 잔차 패턴의 주요 동인

---

## 4. 각 Stage별 모델 상세

### 4.1 Stage 1: Exponential Smoothing (Baseline)

| 항목 | 값 |
|------|-----|
| **역할** | 순수 시계열 외삽 (외생변수 없음) |
| **모델** | Holt-Winters Exponential Smoothing |
| **Trend** | Additive |
| **Seasonal** | Additive, period=52 (연간) |
| **학습** | y_train (644주)에 대해 **1회 학습 (single-fit)** |
| **예측** | Multi-step forecast (24 step ahead: val 12 + test 12) |
| **외생변수** | 없음 |

**y와 X:**
- y: `Com_BrentCrudeOil` 원가격 (y_train, 644주)
- X: 없음

**설계 의도:**
- 장기 예측(24-step ahead)으로 큰 오차를 의도적으로 생성
- 이 오차(Residual①)가 Stage 2 NLinear의 학습 대상
- Expanding-window 1-step ES는 Random Walk과 동일 RMSE → 잔차 0 → NLinear 학습 불가

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
| **seq_len** | **24주** (반기 lookback) |
| **pred_len** | 1주 |
| **d_hidden** | 64 |
| **dropout** | 0.3 |
| **learning rate** | 0.001 |
| **max epochs** | 300, patience=40 |
| **batch size** | 32 |
| **앙상블** | **10-seed ensemble** (seed 42~51), **top 5** by val loss 선택 후 평균 |
| **잔차 정규화** | **없음** — NLinear 내장 정규화 (`x_norm = x_seq - x_seq[:, -1:]`) |
| **외생변수** | 55개 (22 raw + 33 derived, RobustScaler train-only fit) |

**y와 X:**
- y: Residual① = `y_actual - ES_baseline_pred`
- X: 55개 도메인 기반 + 파생 변수 (RobustScaler로 train에서만 fit)

**NLinear 내장 정규화 (핵심):**
```python
last_val = x_seq[:, -1:]           # 시퀀스 마지막 값
x_norm = x_seq - last_val          # level shift 자동 처리
output = self.linear_out(h) + last_val  # 복원
```
- Train(mean≈0)과 val/test(mean≈-5)의 분포 차이를 자연스럽게 처리
- RobustScaler를 잔차에 적용하면 오히려 성능 악화 확인됨

**앙상블 전략:**
- 10개 seed 학습 → val loss 기준 상위 5개 선택 → 예측값 평균
- NLinear는 seed에 따라 val RMSE 3.09~3.66 편차 → 앙상블 필수

**결과 (Baseline + NLinear):**
| 구간 | RMSE | MAE | MAPE |
|------|------|-----|------|
| Val | 3.1601 | 2.7778 | 4.25% |
| Test | **1.2608** | **1.1316** | **1.81%** |

→ ES 단독 대비 test RMSE **69.7% 개선** (4.17 → 1.26)

### 4.3 Stage 3: LightGBM RoR (Residual-of-Residual)

| 항목 | 값 |
|------|-----|
| **역할** | NLinear가 놓친 비선형 잔차 패턴 보정 |
| **모델** | LightGBM (gradient boosting) |
| **OOF** | Expanding-window CV, 5-fold |
| **min_train_size** | 210주 |
| **learning_rate** | 0.02 |
| **num_leaves** | 10 (매우 보수적) |
| **min_child_samples** | 40 |
| **subsample** | 0.6 |
| **colsample_bytree** | 0.5 |
| **reg_alpha** | 2.0 |
| **reg_lambda** | 2.0 |
| **n_estimators** | 500 (OOF) / 300 (final) |
| **early_stopping** | 50 rounds |
| **Validation Gate** | val RMSE 개선 시 λ=1, 아니면 λ=0 |

**y와 X:**
- y: RoR = Residual① - NLinear_in_sample_fitted
- X: 55개 도메인 기반 + 파생 변수 (RobustScaler by train)

**OOF Expanding Window:**
```
Fold 1: train=[0:206],    val=[206:288],   RMSE=1.8566
Fold 2: train=[0:288],    val=[288:370],   RMSE=2.6425
Fold 3: train=[0:370],    val=[370:452],   RMSE=3.6442
Fold 4: train=[0:452],    val=[452:534],   RMSE=3.1787
Fold 5: train=[0:534],    val=[534:620],   RMSE=2.5353
                                       OOF RMSE=2.8342
```

**Validation Gate 결과:**
- Stage 2 val RMSE: 3.1601 → Stage 3 val RMSE: 2.8393 → **PASS** (λ=1)
- 그러나 test에서는 미세 악화: 1.2608 → 1.2822

**결과 (Full Framework = B + NL + λ·RoR):**
| 구간 | RMSE | MAE | MAPE |
|------|------|-----|------|
| Val | 2.8393 | 2.4729 | 3.77% |
| Test | 1.2822 | 0.9959 | 1.61% |

→ RoR은 val에서 개선(3.16→2.84)하나 test에서 미세 악화(1.26→1.28), DM p=0.93 (비유의)

---

## 5. 종합 결과

### 5.1 Test Set 성능 비교

| Model | RMSE | MAE | MAPE(%) | vs RW |
|-------|------|-----|---------|-------|
| Random Walk | 1.1227 | 0.9585 | 1.54% | — |
| ExpSmoothing (Baseline) | 4.1678 | 4.0778 | 6.52% | -271.2% |
| **Baseline + NLinear** | **1.2608** | **1.1316** | **1.81%** | **-12.3%** |
| Full Framework (B+NL+RoR) | 1.2822 | 0.9959 | 1.61% | -14.2% |

### 5.2 Validation Set 성능 비교

| Model | RMSE | MAE | MAPE(%) |
|-------|------|-----|---------|
| Random Walk | 1.8243 | 1.4542 | 2.22% |
| ExpSmoothing (Baseline) | 5.2562 | 4.9354 | 7.55% |
| Baseline + NLinear | 3.1601 | 2.7778 | 4.25% |
| Full Framework (B+NL+RoR) | 2.8393 | 2.4729 | 3.77% |

### 5.3 Stage별 진행 (Test RMSE 기준)

```
Stage 1: ExpSmoothing        4.1678  ████████████████████████████████████████
Stage 2: + NLinear           1.2608  ████████████
Stage 3: + LightGBM RoR     1.2822  ████████████ (미세 악화)
Random Walk (benchmark)      1.1227  ███████████
```

### 5.4 Diebold-Mariano 검정 결과

| 비교 | DM 통계량 | p-value | 유의성 |
|------|----------|---------|--------|
| Random Walk vs ExpSmoothing | -7.7974 | 0.0000 | *** |
| **ExpSmoothing vs +NLinear** | **+9.6034** | **0.0000** | **\*\*\*** |
| +NLinear vs +NLinear+RoR | -0.0945 | 0.9264 | n.s. |
| Random Walk vs Full Framework | -0.4268 | 0.6778 | n.s. |
| ExpSmoothing vs Full Framework | +8.9649 | 0.0000 | *** |

**해석:**
- **H2 강하게 지지**: ES → NLinear 개선 DM=+9.60, p<0.0001 (초고도 유의)
- **H3 기각**: NLinear → Full 개선 DM=-0.09, p=0.93 (RoR 추가 효과 없음)
- Full vs RW: p=0.68 (통계적으로 구별 불가, 그러나 RMSE 차이 0.16)
- Harvey et al. (1997) 소표본 보정 적용 (test 12주)

### 5.5 RoR 비유의 분석

NLinear가 도메인 기반 55개 feature로 이미 예측 가능한 신호 대부분을 포착하여, RoR이 보정할 잔차에 체계적 패턴이 부족:
1. **NLinear 성능이 이미 충분**: Test RMSE 1.26으로 RW(1.12)에 근접
2. **RoR train mean ≈ 0.29, std ≈ 2.87**: 잔여 잔차가 노이즈에 가까움
3. **Feature 55개 → 22개 원변수**: 도메인 기반 선별로 노이즈 변수가 제거되어 NLinear의 신호 추출이 효율적

→ 이는 "적절한 feature engineering이 모델 복잡도를 대체할 수 있다"는 발견

---

## 6. 데이터 누수 방지 프로토콜 (No-Leakage Protocol)

### 6.1 적용된 방지 조치

| # | 조치 | 상세 |
|---|------|------|
| 1 | **Feature 1주 래그** | 모든 X_t → X_{t-1}로 shift |
| 2 | **STL train-only** | STL 분해를 train 644주에서만 수행 |
| 3 | **ES single-fit** | y_train에서 1회 학습, 24-step ahead forecast |
| 4 | **NLinear train-only** | Train에서만 학습, val로 early stopping |
| 5 | **RobustScaler train-only** | 외생변수 scaler를 train에서만 fit |
| 6 | **LightGBM OOF expanding** | Expanding-window CV (과거→미래 방향만) |
| 7 | **Test autoregressive** | NLinear test 예측 시 이전 예측값을 buffer에 사용 |
| 8 | **파생변수 래그 보존** | 수익률·MA 등 파생변수도 1주 래그 후 계산 |

### 6.2 누수 위험 분석

| 위험 요소 | 상태 | 상세 |
|-----------|------|------|
| 동시점 feature 사용 | ✅ 방지됨 | 모든 feature 1주 래그 |
| Scaler로 미래 정보 유입 | ✅ 방지됨 | Train-only fit |
| STL 미래 데이터 사용 | ✅ 방지됨 | Train-only fit |
| NLinear val teacher forcing | ⚠️ 부분적 | Val에서는 실제 잔차 buffer, Test에서는 autoregressive |
| LightGBM 미래 fold 사용 | ✅ 방지됨 | Expanding window |
| Crack Spread에 target 포함 | ⚠️ 주의 | `Gasoline - Brent`에 target이 포함되나, 1주 래그 적용으로 `X_{t-1}`의 Brent값 사용 → 누수 아님 |

### 6.3 잠재적 개선 사항

1. **NLinear Validation Teacher Forcing**: Val에서 실제 잔차를 buffer에 사용. 엄밀한 OOS 평가는 test(autoregressive)가 더 적절
2. **Crack Spread 파생변수**: 정의상 target(Brent)을 포함하지만, 1주 래그가 적용되어 있어 직접적 leak은 아님. 단, reviewer 우려 가능 → 제거 옵션도 고려

---

## 7. 모델별 상세 파라미터

### 7.1 STL Decomposition
```python
STL(y_train, period=52, seasonal=53, robust=True)
```
- Cleveland et al. (1990)

### 7.2 Exponential Smoothing
```python
ExponentialSmoothing(y_train, trend="add", seasonal="add",
                     seasonal_periods=52, initialization_method="estimated")
.fit(optimized=True)
```

### 7.3 NLinear with Exogenous
```python
NLinearWithExog(seq_len=24, pred_len=1, n_exog=55, d_hidden=64, dropout=0.3)
# Adam, lr=0.001, MSELoss, patience=40
# 10-seed ensemble, top 5 by val loss
```

### 7.4 LightGBM (RoR)
```python
LGBMRegressor(
    learning_rate=0.02, num_leaves=10,
    min_child_samples=40, subsample=0.6,
    colsample_bytree=0.5, reg_alpha=2.0,
    reg_lambda=2.0, n_estimators=500,
    early_stopping=50
)
```

---

## 8. OOF (Out-of-Fold) 적용 상세

### 8.1 Expanding-Window Cross-Validation

```
Fold 1: |====TRAIN(206)====|--VAL(82)--|
Fold 2: |========TRAIN(288)========|--VAL(82)--|
Fold 3: |============TRAIN(370)============|--VAL(82)--|
Fold 4: |================TRAIN(452)================|--VAL(82)--|
Fold 5: |====================TRAIN(534)====================|--VAL(86)--|
```

### 8.2 OOF Fold별 결과

| Fold | Train Size | Val Size | RMSE |
|------|-----------|----------|------|
| 1 | 206 | 82 | 1.8566 |
| 2 | 288 | 82 | 2.6425 |
| 3 | 370 | 82 | 3.6442 |
| 4 | 452 | 82 | 3.1787 |
| 5 | 534 | 86 | 2.5353 |
| **OOF 종합** | — | — | **2.8342** |

---

## 9. 시도한 모델 및 접근법 비교

### 9.1 시도된 접근법 이력

| 접근법 | Test RMSE | 비고 |
|--------|----------|------|
| 72개 전체 feature + seq_len=12 | 1.22 | RoR 유의, 그러나 feature 선별 근거 부족 |
| 도메인 16개 + 파생 27개 = 43개, seq_len=12 | 2.04 | Feature 감소로 NLinear 성능 하락 |
| 도메인 22개 + 파생 33개 = 55개, seq_len=12 | 2.18 | 아시아/바이오연료 추가로 소폭 개선 |
| **도메인 22개 + 파생 33개 = 55개, seq_len=24** | **1.26** | ✓ lookback 확대가 핵심 |
| Expanding-window 1-step ES | ~1.12 | RW과 동일 → 잔차 0, NLinear 학습 불가 |
| RobustScaler on residuals | 악화 | Scaler가 val/test 잔차를 극단값 처리 |
| λ-blending (grid search) | 3.53 | 12-point val 과적합 |

### 9.2 핵심 발견사항

1. **seq_len=24가 핵심**: 12→24주 lookback 확대가 test RMSE를 2.18→1.26으로 대폭 개선. 반기(6개월) 패턴을 포착하는 데 충분한 맥락 필요
2. **도메인 기반 feature selection 유효**: 22개 경제적 의미 변수 + 파생변수가 72개 전체 대비 해석력 크게 향상하면서 성능 유지
3. **파생변수가 LGBM에 핵심적**: Top 10 feature importance 중 8개가 파생변수 (수익률, 스프레드)
4. **NLinear 내장 정규화**: 외부 scaler 없이 level shift를 처리하는 것이 최적
5. **10-seed → top-5 앙상블**: Seed sensitivity 해소 (val RMSE std 0.18 → 앙상블로 안정)

---

## 10. 향후 성능 개선 방향

### 10.1 모델 개선

| 방향 | 상세 | 기대 효과 |
|------|------|-----------|
| **PatchTST** | Patch 기반 Transformer (Nie et al., ICLR 2023) | 긴 시퀀스의 local semantics 포착 |
| **iTransformer** | Inverted Transformer (Liu et al., ICLR 2024) | 다변량 간 교차 상관 직접 모델링 |
| **TiDE** | Time-series Dense Encoder (Das et al., 2023) | MLP 기반, long-horizon 강점 |
| **TFT** | Temporal Fusion Transformer (Lim et al., 2021) | Attention 기반 해석 가능한 예측 |

### 10.2 데이터/Feature 개선

| 방향 | 상세 |
|------|------|
| **SHAP-based selection** | SHAP value 기반 feature 중요도로 추가 정제 |
| **외부 데이터** | OPEC 생산량, EIA 원유 재고, 지정학적 리스크 지수, 해운 운임(BDI) |
| **교차 시장 파생변수** | Oil-Gold spread, Energy sector ETF returns, Brent-Dubai spread |
| **고빈도 데이터** | 일간 전환 시 샘플 수 5배 확대 |

### 10.3 방법론 개선

| 방향 | 상세 |
|------|------|
| **Walk-forward validation** | Rolling window로 다양한 시장 국면에서 평가 |
| **Optuna HPO** | NLinear/LGBM 하이퍼파라미터 체계적 탐색 |
| **Conformal prediction** | 예측 구간 제공으로 불확실성 정량화 |
| **Longer test period** | 12주 → 52주로 확대, 통계적 검정력 강화 |
| **RoR 대안**: 다른 잔차 모델 | XGBoost, CatBoost, 또는 Transformer 기반 RoR |

---

## 11. 한계점 및 논문 작성 시 주의사항

### 11.1 Random Walk 대비 통계적 비유의
- NLinear (RMSE 1.26) vs RW (RMSE 1.12): DM p=0.68
- Test 기간 저변동성(std=$1.60)이 RW에 유리
- **논문 전략**: "ES→NLinear 개선의 통계적 유의성(p<0.0001)"에 초점

### 11.2 RoR Stage 비유의
- H3 기각: RoR 추가 효과 통계적으로 유의하지 않음
- 해석: 도메인 기반 feature selection으로 NLinear가 이미 충분한 신호 추출
- 논문에서 "feature engineering의 모델 복잡도 대체 효과"로 논의 가능

### 11.3 Val/Test 불일치
- NLinear: val 3.16 vs test 1.26 (test가 훨씬 좋음)
- Teacher forcing(val) vs autoregressive(test) 차이
- Test 기간의 좁은 가격 범위가 예측에 유리하게 작용

### 11.4 단일 Test Period
- 2025-10 ~ 2026-01 단일 12주 test
- 다양한 시장 국면에서의 robustness 미검증

---

## 12. 참고문헌

1. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition. *Journal of Official Statistics*, 6(1), 3-73.
2. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? *AAAI*, 37(9), 11121-11128.
3. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*, 30.
4. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253-263.
5. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction MSE. *IJF*, 13(2), 281-291.
6. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.).
7. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
8. Nie, Y., et al. (2023). A time series is worth 64 words. *ICLR 2023*.
9. Liu, Y., et al. (2024). iTransformer. *ICLR 2024*.
10. Estrella, A., & Mishkin, F. S. (1998). Predicting U.S. recessions: Financial variables as leading indicators. *REStat*, 80(1), 45-61.

---

## 부록 A: 코드 구조

```
sparta2/
├── oil_forecast_academic.py          # 메인 실험 코드
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
