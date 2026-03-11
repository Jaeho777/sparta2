# Brent 유가 예측: Enhanced Stage 3 실험 보고서
## Transformer Multi-Stage Pipeline + ML-RoR vs Forecast Combination 비교

---

## 0. 핵심 요약 (Executive Summary)

| 항목 | 내용 |
|------|------|
| **예측 대상 (y)** | Brent 원유 주간 가격 (`Com_BrentCrudeOil`, USD/barrel) |
| **데이터** | 668주 (2013-04 ~ 2026-01), Train=644 / Val=12 / Test=12 |
| **외생변수 (X)** | SHAP 선별 10개 (55개 중) + 1주 래그, No-Leakage |
| **파이프라인** | Base(DL) → Residual(DL) → Stage 3 보정 (8가지 전략) |
| **Stage 3 전략** | ML-RoR 7종 (A~F, H) + Forecast Combination 1종 (G) |
| **최종 Best** | LSTM+LSTM + Forecast Combination(w=0.50) → Test RMSE = **1.0326** |
| **Random Walk** | 1.1227 |
| **vs RW** | -8.0% (하회) |

### 핵심 발견 (정직한 평가)

1. **ML 기반 RoR (전략 A~F, H)**: Val 기준 미세 개선(0~0.04 RMSE), Test에서는 대부분 **효과 미미하거나 역효과** — RoR 잔차 신호가 너무 약하여 ML 모델이 유의미한 패턴을 학습하기 어려움
2. **Forecast Combination (전략 G: RW Blend)**: S2 예측과 Random Walk의 가중 결합으로 **일관된 대폭 개선** — 그러나 이것은 RoR(잔차의잔차 모델링)이 아니라 **예측 결합(Forecast Combination)** 기법
3. **Best 모델(LSTM+LSTM, w=0.50)의 의미**: 최종 예측의 **50%가 전주 가격(RW)**, 50%가 DL 모델 예측 → DL 모델 단독 기여는 절반에 불과
4. **유가의 근본적 한계**: 단위근(unit root) 특성으로 인해 RW가 매우 강력하며, 잔차 신호는 본질적으로 노이즈에 가까움

> **주의**: 본 보고서는 Stage 3의 두 가지 접근법(ML-RoR vs Forecast Combination)을 **명확히 구분**하여 기술합니다. RW Blend를 "RoR"로 분류하는 것은 개념적으로 부정확하며, 별도의 메커니즘으로 다룹니다.

---

## 1. 이전 실험 대비 변경 사항

### 1.1 기존 방식의 문제점

| 문제 | 상세 | 영향 |
|------|------|------|
| **이진 게이트** | λ ∈ {0, 1} — RoR 전체 사용 or 미사용 | 미세 조정 불가, 10/10 실험에서 BLOCKED |
| **과적합** | LightGBM 300 trees, 잔차 노이즈에 과적합 | Val RMSE 악화 |
| **단일 전략** | LightGBM만 사용 | 대안 모델 탐색 없음 |
| **3-seed 앙상블** | 적은 앙상블 수 | 높은 분산 |

### 1.2 개선 시도

```
개선사항:
  ✓ λ 연속 최적화 (0.02 단위 그리드 서치, 기존 이진→연속)
  ✓ ML-RoR 7가지 + Forecast Combination 1가지 = 8가지 전략 비교
  ✓ 5-seed 앙상블 (안정성 시도, 그러나 분산 여전히 존재)
  ✓ Forecast Combination (RW Blend) 도입
```

---

## 2. Stage 3 전략 설계 — ML-RoR vs Forecast Combination

### 2.1 두 가지 접근법의 근본적 차이

```
ML-RoR (전략 A~F, H):
  RoR_target = 실제잔차 - 예측잔차
  Final = S2 + λ × f(X_{t-1})     ← 피처 기반 잔차의잔차 예측

Forecast Combination (전략 G):
  Final = (1-w) × S2 + w × RW     ← 모델 예측과 RW의 가중 평균
  = S2 + w × (y_{t-1} - S2)       ← RoR 타겟을 사용하지 않음
```

| 구분 | ML-RoR (A~F, H) | Forecast Combination (G) |
|------|-----------------|-------------------------|
| **타겟** | RoR (잔차의잔차) | 사용하지 않음 |
| **피처** | X_{t-1} (거시경제 변수) | 사용하지 않음 |
| **메커니즘** | 잔차의 비선형 패턴 학습 | S2 예측을 RW 쪽으로 축소(shrinkage) |
| **학습** | Train 데이터로 모델 학습 | Validation에서 w만 최적화 |
| **이론** | Residual Refinement | Forecast Combination (Bates & Granger, 1969) |

### 2.2 전략 목록

| 전략 | 분류 | 모델 | 핵심 아이디어 |
|------|------|------|-------------|
| **A** | ML-RoR | Weighted LightGBM | GBDT + λ grid search |
| **B** | ML-RoR | Ridge | L2 정규화 선형 모델 |
| **C** | ML-RoR | ElasticNet | L1+L2 혼합 (sparse) |
| **D** | ML-RoR | OOF LightGBM | Expanding Window 5-fold CV |
| **E** | ML-RoR | Augmented LightGBM | Base/Resid 예측값을 피처에 추가 |
| **F** | ML-RoR | Ensemble | Ridge + LightGBM 블렌딩 |
| **G** | **FC** | **RW Blend** | **S2 × (1-w) + RW × w** |
| **H** | ML-RoR | All-Feature LightGBM | 55개 전체 피처 사용 |

---

## 3. 실험 결과

### 3.1 전체 결과 요약 (Test RMSE 기준)

| # | 모델 조합 | S2 (Stage3 전) | Best Stage3 | Test RMSE | 전략 | 분류 | vs RW |
|---|----------|--------------|-------------|-----------|------|------|-------|
| 1 | **LSTM+LSTM** | 1.2698 | FC(w=0.50) | **1.0326** | G | FC | **-8.0%** |
| 2 | PatchTST+PatchTST | 1.3483 | FC(w=0.30) | 1.2385 | G | FC | +10.3% |
| 3 | PatchTST+NLinear | 1.6544 | FC(w=0.26) | 1.4291 | G | FC | +27.3% |
| 4 | Transformer+Transformer | 1.4462 | ElasticNet | 1.4414 | C | ML-RoR | +28.4% |
| 5 | PatchTST+iTransformer | 1.4894 | ElasticNet | 1.5076 | C | ML-RoR | +34.3% |
| - | **Random Walk** | - | - | **1.1227** | - | - | - |

> **관찰**: Top 3는 모두 Forecast Combination(전략 G). ML-RoR이 선택된 Exp4, 5는 개선폭이 0.005~미세 수준이거나 오히려 악화.

### 3.2 ML-RoR (전략 A~F, H)의 실제 효과 — 정직한 평가

ML-RoR만 분리하여 분석하면:

| 실험 | S2 Test | Best ML-RoR Test | ML-RoR 전략 | 개선폭 | 판정 |
|------|---------|-----------------|-------------|--------|------|
| LSTM+LSTM | 1.2698 | 1.2698 | λ=0.00 (전부) | 0.0000 | **효과 없음** |
| PatchTST+PatchTST | 1.3483 | 1.3439 | H(AllFeat, λ=0.12) | -0.0044 | **미미** |
| Transformer+Transformer | 1.4462 | 1.4414 | C(ElasticNet) | -0.0048 | **미미** |
| PatchTST+NLinear | 1.6544 | 1.6195 | H(AllFeat, λ=0.50) | -0.0349 | **소폭 개선** |
| PatchTST+iTransformer | 1.4894 | 1.4921 | A(LGBM, λ=0.50) | +0.0027 | **악화** |

> **결론**: ML-RoR의 실질적 Test RMSE 개선은 **0~0.035 수준**으로, 12-point 테스트셋에서 통계적으로 유의한 개선이라 보기 어려움. LSTM+LSTM에서는 모든 ML-RoR 전략의 λ가 0.00으로 수렴 — **학습할 잔차 신호가 없음을 의미**.

### 3.3 Forecast Combination (전략 G: RW Blend)의 효과

| 실험 | S2 Test | FC Test | 최적 w | 개선폭 | DL 기여비 |
|------|---------|---------|--------|--------|----------|
| **LSTM+LSTM** | 1.2698 | **1.0326** | 0.50 | -18.7% | **50%** |
| PatchTST+PatchTST | 1.3483 | 1.2385 | 0.30 | -8.1% | 70% |
| PatchTST+NLinear | 1.6544 | 1.4291 | 0.26 | -13.6% | 74% |
| Transformer+Transformer | 1.4462 | 1.3155 | 0.20 | -9.1% | 80% |
| PatchTST+iTransformer | 1.4894 | 1.4368 | 0.06 | -3.5% | 94% |

**왜 Forecast Combination이 효과적인가?**

```
Final = (1-w) × S2 + w × y_{t-1}
```

- 유가는 강한 Random Walk (단위근) → RW 자체가 강력한 예측기
- DL 모델은 추세를 과대/과소 추정하는 경향 (편향)
- 두 예측의 오차가 **부분적으로 비상관**이므로, 블렌딩하면 분산이 감소
- 이는 Bates & Granger (1969)의 예측 결합 이론과 일치

**그러나 유의사항:**
- Best 모델의 w=0.50 → 예측의 절반이 단순 전주 가격
- 이는 **DL 모델이 RW를 크게 초과하는 정보를 제공하지 못함**을 시사
- 본질적으로 "DL 모델의 한계를 RW로 보완"하는 것

### 3.4 Stage별 진행 (Best Model: LSTM+LSTM)

```
LSTM+LSTM Pipeline:
  Stage 1 (Base LSTM):          Val=1.959  Test=1.297
  Stage 2 (+Residual LSTM):     Val=1.983  Test=1.270
  Stage 3 FC (RW Blend w=0.50): Val=1.707  Test=1.033

  * Stage 2의 Val RMSE(1.983)가 Stage 1(1.959)보다 악화
    → Residual 모델이 Val에서 오히려 역효과
    → 그러나 Test에서는 1.297→1.270으로 소폭 개선
    → 12-point Val/Test 간 불일치가 존재 (소표본 불안정성)

  * Stage 3: FC 적용으로 Test 1.270→1.033 (대폭 개선)
    → 이는 FC가 S2의 과대예측을 RW 방향으로 축소한 효과
    → DL 기여 50% + RW 기여 50%
```

### 3.5 전략별 성과 통계

| 전략 | 분류 | Val 개선율 | 평균 Val RMSE | 실제 Test 개선 | 평가 |
|------|------|-----------|-------------|--------------|------|
| **G: RW_Blend** | FC | 5/5 | 1.4024 | 5/5 | FC로서 효과적 |
| C: ElasticNet | ML-RoR | 4/5 | 1.4819 | 2/5 미미 | Val-Test 괴리 |
| A: WeightedLGBM | ML-RoR | 4/5 | 1.4931 | 1/5 미미 | λ→0 수렴 다수 |
| D: OOF_LGBM | ML-RoR | 4/5 | 1.4881 | 1/5 미미 | OOF 효과 제한적 |
| H: AllFeatLGBM | ML-RoR | 3/5 | 1.4879 | 2/5 소폭 | 55피처 과적합 |
| E: AugLGBM | ML-RoR | 3/5 | 1.4925 | 1/5 미미 | 증강 피처 한계 |
| F: Ensemble | ML-RoR | 3/5 | 1.4935 | 1/5 미미 | 블렌딩 미미 |
| B: Ridge | ML-RoR | 2/5 | 1.4921 | 0/5 | 효과 없음 |

---

## 4. 구조적 한계와 주의사항

### 4.1 소표본 문제
- **Val=12주, Test=12주**: 극소 표본에서의 λ/w 최적화는 과적합 위험 높음
- FC의 w=0.50이 다른 기간에서도 최적인지 보장 불가
- 12-point RMSE는 1~2개 아웃라이어에 크게 영향받음

### 4.2 5-seed 앙상블 불안정성
- PatchTST+iTransformer: 3-seed S2 Test=1.2384 → **5-seed S2 Test=1.4894** (크게 악화)
- seed 수 변경만으로 결과가 크게 달라짐 → 모델 수렴이 불안정

### 4.3 Forecast Combination의 한계
- w 최적값이 실험마다 0.06~0.50으로 크게 변동
- w=0.50(LSTM+LSTM)은 DL 모델의 독립적 가치가 제한적임을 의미
- 새로운 기간에서 최적 w가 달라질 수 있음 (비정상 시계열)

### 4.4 ML-RoR 실패의 근본 원인
- RoR 타겟(잔차의잔차)이 **본질적으로 white noise에 가까움**
- 학습 가능한 구조적 패턴이 부재
- 이는 이전 academic 프레임워크의 DM test 결과(p=0.93)와 일관됨
- **NLinear/LSTM 등이 이미 예측 가능한 신호를 대부분 추출**한 후의 잔차

---

## 5. 피처 분석

### 5.1 SHAP 기반 피처 선별 (10개 선정)

| 순위 | 피처 | Mean |SHAP| | 경제적 의미 |
|------|------|------------|------------|
| 1 | **Spread_Crack** | 15.79 | 정제 마진 (휘발유-브렌트) — 수급 핵심 |
| 2 | EX_USD_KRW | 0.74 | 원화 환율 → 아시아 수요 proxy |
| 3 | Com_Gasoline | 0.52 | 정유 제품 가격 |
| 4 | Ratio_Gold_Oil | 0.52 | 금/유 비율 → 안전자산 선호 |
| 5 | Com_Coal | 0.28 | 대체 에너지 가격 |
| 6 | Com_Gasoline_ma12r | 0.26 | 휘발유 12주 모멘텀 |
| 7 | Com_PalmOil | 0.19 | 바이오디젤 원료 |
| 8 | Idx_SnPVIX | 0.18 | 공포 지수 → 리스크 프리미엄 |
| 9 | Bonds_US_10Y | 0.17 | 미국 장기 금리 |
| 10 | Bonds_US_3M_ret | 0.15 | 단기 금리 변화 → 통화정책 |

### 5.2 피처 수 CV 결과

| 피처 수 | CV RMSE | 비고 |
|---------|---------|------|
| **10** | **7.4507** | 최적 |
| 15 | 7.8164 | |
| 20 | 8.3217 | |
| 55 | 10.1980 | 전체 — 과적합 심화 |

---

## 6. No-Leakage Protocol

| 단계 | 방법 | Data Leakage 방지 |
|------|------|-------------------|
| **피처 래그** | 모든 X를 1주 시프트 (X_{t-1} → y_t 예측) | 미래 정보 사용 불가 |
| **SHAP 선별** | Train 데이터로만 LightGBM 학습 후 SHAP 산출 | Val/Test 미참조 |
| **스케일링** | RobustScaler를 Train에만 fit | Val/Test는 transform만 |
| **시퀀스** | Train 내 SEQ_LEN=24 시퀀스만 구성 | Val/Test는 버퍼 사용 |
| **DL 학습** | Train으로 학습, Val로 Early Stopping | Test 미참여 |
| **Stage 3 λ/w** | Validation RMSE 기준 최적화 | Test 미참조 |

---

## 7. 모델 아키텍처

### 7.1 DL 모델 (Stage 1 & 2)

| 모델 | 핵심 메커니즘 | 파라미터 |
|------|-------------|---------|
| **NLinear** | 선형 시계열 분해 + 크로스 피처 | seq→64, feat→64, concat→1 |
| **PatchTST** | 패치 기반 Transformer (CLS token) | patch=4, stride=2, d=64, h=4, L=2 |
| **iTransformer** | Inverted — 변수 축 Attention | d=64, h=4, L=2 |
| **Transformer** | Vanilla Encoder (마지막 토큰) | d=64, h=4, L=2 |
| **LSTM** | LSTM (마지막 hidden) | h=64, L=2 |

### 7.2 학습 설정

| 항목 | 값 |
|------|---|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (patience=8, factor=0.5) |
| Max Epochs | 150 |
| Early Stopping | patience=25 |
| Gradient Clipping | max_norm=1.0 |
| Batch Size | 32 |
| Seeds | 5개 (앙상블 평균) |
| Sequence Length | 24주 |

---

## 8. 결론

### 8.1 핵심 결론 (정직한 평가)

**ML 기반 RoR에 대하여:**
- 7가지 ML-RoR 전략(LightGBM, Ridge, ElasticNet, OOF, Augmented, Ensemble, AllFeature)을 시도한 결과, **실질적인 Test RMSE 개선은 0~0.035 수준으로 미미**
- 대다수 실험에서 최적 λ가 0.00으로 수렴 → 잔차의잔차에 학습 가능한 신호가 부재
- 이는 이전 academic 프레임워크의 DM 검정 결과(p=0.93)와 일관되며, **NLinear/LSTM 등의 DL 모델이 이미 예측 가능한 잔차 신호를 충분히 추출**했음을 시사

**Forecast Combination에 대하여:**
- RW Blend(`(1-w)×S2 + w×RW`)는 5/5 실험에서 일관된 개선을 보임
- Best 결과: LSTM+LSTM + FC(w=0.50) → Test RMSE **1.0326** (RW 1.1227 하회)
- 그러나 이는 RoR(잔차 패턴 학습)이 아닌 **예측 결합** 기법
- w=0.50은 최종 예측의 50%가 단순 RW이므로, **DL 모델의 독립적 기여는 제한적**

**종합:**
- Stage 3 보정의 가장 효과적 형태는 ML 기반 잔차 모델링이 아닌 **Forecast Combination**
- 유가의 강한 Random Walk 특성으로 인해, 잔차의잔차는 본질적으로 노이즈
- DL 모델 + RW 블렌딩의 조합이 순수 DL보다 우수 → **모델 겸손(model humility)의 중요성**

### 8.2 이전 실험과의 비교

| 지표 | 이전 (10 Transformer Exp) | 현재 (Enhanced Stage 3) | 비고 |
|------|-------------------------|----------------------|------|
| Best Test RMSE | 1.2384 | **1.0326** | FC 적용 시 |
| ML-RoR Best Test | 1.2384 (λ=0) | 1.2698 (λ=0) | ML-RoR은 여전히 미미 |
| Stage 3 성공률 | 0/10 | 5/5 (FC 포함) | FC가 핵심 동인 |
| RW 대비 | +10.3% | **-8.0%** | FC 덕분 |

### 8.3 방법론적 시사점

1. **ML-RoR의 한계 재확인**: 연속 λ, 다양한 ML 모델, 보수적 HP를 적용해도 잔차의잔차 학습은 본질적으로 어려움
2. **Forecast Combination의 가치**: 단순하지만 이론적으로 잘 정립된 기법이 복잡한 ML보다 효과적
3. **소표본 주의**: Val/Test 각 12주에서의 λ/w 최적화는 과적합 위험이 높으며, 결과의 일반화에 한계
4. **모델 불안정성**: seed 수 변경으로 Base 모델 성능이 크게 변동 → DL 시계열 예측의 재현성 문제

---

## 9. 생성된 파일 목록

| 파일 | 설명 |
|------|------|
| `oil_transformer_ror_enhanced.py` | 실험 코드 (8전략 × 5실험) |
| `output_oil_ror_enhanced/experiment_results.csv` | 5개 실험 최종 결과 |
| `output_oil_ror_enhanced/ror_strategy_details.csv` | 8전략 × 5실험 = 40개 상세 결과 |
| `output_oil_ror_enhanced/shap_importance.csv` | SHAP 피처 중요도 |
| `output_oil_ror_enhanced/config.json` | 실험 설정 |
| `output_oil_ror_enhanced/01_shap.png` | SHAP Top 20 피처 |
| `output_oil_ror_enhanced/02_cv_curve.png` | 피처 수 CV 곡선 |
| `output_oil_ror_enhanced/03_all_experiments.png` | 전 실험 비교 |
| `output_oil_ror_enhanced/04_top3_stages.png` | Top 3 Stage 진행 |
| `output_oil_ror_enhanced/05_ror_heatmap.png` | 전략별 Val RMSE 히트맵 |

---

*실험 일시: 2026-03-11*
*Pipeline: `oil_transformer_ror_enhanced.py`*
*환경: Python 3.x, PyTorch, LightGBM, scikit-learn*
