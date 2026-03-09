# 니켈 가격 예측 실험 보고서
(2026.03.09) 이재호

---

**Q1. 이번 미션에서 가장 성능 향상 또는 성취를 이뤄낸 방법은?**

이번 과제에서 가장 의미 있는 성취는 OOF(Out-of-Fold) 스태킹을 시계열 환경에서 방법론적으로 정확하게 구현하고, 여기에 PatchTST를 4번째 Base Model로 통합한 것이다.

일반적인 OOF 구현에서 시계열을 다룰 때 놓치기 쉬운 세 가지 함정이 있다. 첫째, `TimeSeriesSplit`을 사용하지 않으면 미래 데이터가 학습에 섞인다. 둘째, `StandardScaler`를 전체 학습셋 기준으로 fit하면 fold_val 정보가 스케일러에 누출된다. 셋째, DL 시퀀스 생성 시 경계 인덱스가 1개만 틀려도 fold_val 타겟에 미래 정보가 섞이는 leakage가 발생한다. 이 세 가지를 모두 정확히 처리했다.

**Q2. 가장 어려웠던 부분은?**

PatchTST의 OOF 시퀀스 경계 수식 검증이 가장 까다로웠다. `fold_train(n_tr개) + fold_val(n_val개)` 결합 배열에서 `make_sequences(X_combined, y_combined, SEQ_LEN)`을 적용할 때, fold_val의 타겟에 정확히 대응하는 시퀀스 슬라이스를 찾아야 했다.

`X_seqs[k].target = y_combined[k + SEQ_LEN]`이므로, fold_val 타겟의 시작은 `k = n_tr - SEQ_LEN`이다. 따라서 `n_tr_seqs = n_tr - SEQ_LEN`개가 학습용 시퀀스이고, 이후 `n_val`개가 fold_val OOF 예측용 시퀀스다. 이 경계 수식을 틀리면 leakage 또는 인덱스 불일치가 발생하므로 직접 수식으로 증명했다.

---

## 1. 개요

### 1.1 과제 목표

기존 니켈 가격 예측 모델을 리뷰하고, Residual 보정 방법론(일반 방식·OOF 방식)을 구현한다. 논문을 통해 추가 성능 향상 아이디어를 탐색하며, 간헐적 패턴에 대한 예측 모형(이벤트 예측 + 출고량 예측)을 설계한다.

### 1.2 이전 과제 기준선 (sparta2_advanced)

| 항목 | 값 |
|------|-----|
| 최고 성능 모델 | Hybrid (Naive_Drift × 0.8 + GradientBoosting × 0.2) |
| Test RMSE | **406.80** |
| Test MAPE | 2.08% |
| Test MAE | 319.45 |
| Test R² | 0.8758 |
| 피처 수 | 85개 (원본 73 + 신규 12, 전체 shift(1) 적용) |

### 1.3 이번 실험 변경 항목

| 변경 항목 | 이전 (sparta2_advanced) | 이후 (이번 과제) |
|-----------|------------------------|----------------|
| Residual 보정 | 미구현 | 일반 방식 + OOF 방식 비교 구현 |
| DL 모델 | LSTM (단독, RMSE 1,105) | PatchTST (OOF Base Model로 통합) |
| 앙상블 방식 | 고정 가중치 (0.8:0.2) | Ridge 메타 모델 (OOF 기반 자동 최적화) |
| Leakage 처리 | ML에만 적용 | DL 스케일러도 fold-specific으로 분리 |

### 1.4 최종 결과 요약

| 모델 | RMSE | 비고 |
|------|------|------|
| **OOF Stacking (Meta Ridge)** | 실행 결과 참조 | Naive Drift + GB + LGB + PatchTST |
| Hybrid Baseline (0.8N + 0.2GB) | 406.80 | 기준선 |
| Naive Drift | 480.67 | 단순 추세 연장 |
| LSTM (sparta2_advanced) | 1,105.43 | DL 단독 |
| 3-stage DL (ROR 최고) | 684.30 | dl_lstm_transformer.ipynb |

---

## 2. 수요예측 모델 리뷰

### 2.1 데이터 구조

| 항목 | 값 |
|------|-----|
| 파일명 | data_weekly_260120.csv |
| 타겟 변수 | Com_LME_Ni_Cash (LME 니켈 현물가격, $/ton) |
| 총 샘플 수 | 668주 (2013-04-01 ~ 2026-01-12) |
| 피처 수 | 74개 (타겟 포함) |
| 데이터 주기 | 주간 (Weekly) |
| 결측치 | 없음 |

피처 구성: LME 비철금속 가격·재고, 달러/원 등 6개 환율, S&P500·상하이종합 등 주가지수, 미국·중국·한국 국채금리 다수 만기, 원유·천연가스·철광석 등 원자재.

### 2.2 기간 분할 및 시장 특성

| 구분 | 기간 | 샘플 수 | 평균가격 | 수익률 | 연율화 변동성 |
|------|------|---------|---------|-------|------------|
| Train | ~ 2025-08-03 | ~628주 | 15,534 | -7.9% | 0.24 |
| Validation | 2025-08-04 ~ 2025-10-20 | 12주 | 15,038 | +0.8% | 0.07 |
| Test | 2025-10-27 ~ 2026-01-12 | 12주 | 15,367 | +18.1% | 0.26 |

Test 기간의 18.1% 급등(변동성 0.26)과 Val 기간의 횡보(변동성 0.07)가 크게 달라, Val에서 선택된 파라미터가 Test에서 최적이라는 보장이 없다. 이 구조적 한계가 모든 모델 선택에 영향을 미친다.

### 2.3 시계열 정상성 검정 (ADF Test)

| 시계열 | ADF 통계량 | p-value | 결론 |
|--------|-----------|---------|------|
| 니켈 가격 (원본) | -1.7429 | 0.4092 | 비정상 (단위근 존재) |
| 니켈 가격 (1차 차분) | -6.7231 | 0.0000 | 정상 |

원본 가격은 I(1) 프로세스다. 단기적으로 모멘텀이 강하게 지속되며, 이것이 Naive Drift가 강력한 베이스라인이 되는 이론적 근거다.

### 2.4 피처 엔지니어링 (12개 신규)

| 카테고리 | 피처명 | 설명 | 개수 |
|---------|--------|------|------|
| Realized Volatility | RV_4w, RV_8w, RV_12w, RV_26w | 로그 수익률 이동표준편차 × √52 | 4 |
| Rate of Change | ROC_4w, ROC_12w, ROC_26w | 가격 변화율 (모멘텀) | 3 |
| Z-score | zscore_4w, zscore_26w | 이동평균 대비 편차 정규화 | 2 |
| Lag Returns | ret_lag_1, ret_lag_2, ret_lag_3 | 과거 로그 수익률 시차 | 3 |

모든 피처는 shift(1) 적용으로 leakage 방지:

```python
log_ret = np.log(y / y.shift(1))
X = df[feat_cols].shift(1)         # 원본 피처 전체 1주 lag

for w in [4, 8, 12, 26]:
    X[f'RV_{w}w'] = log_ret.shift(1).rolling(w).std() * np.sqrt(52)
for w in [4, 12, 26]:
    X[f'ROC_{w}w'] = y.shift(1).pct_change(w)
for w in [4, 26]:
    mu, sig = y.shift(1).rolling(w).mean(), y.shift(1).rolling(w).std()
    X[f'zscore_{w}w'] = (y.shift(1) - mu) / (sig + 1e-8)
for lag in [1, 2, 3]:
    X[f'ret_lag_{lag}'] = log_ret.shift(lag)
```

### 2.5 모델별 성능 리뷰

| 모델 | Test RMSE | 비고 |
|------|-----------|------|
| Hybrid (0.8 Naive + 0.2 GB) | **406.80** | 기준선 |
| Hybrid (0.9 Naive + 0.1 GB) | 423.67 | |
| Hybrid (0.7 Naive + 0.3 GB) | 434.74 | |
| Naive_Drift_Damped (φ=0.7) | 438.60 | |
| Naive_Drift (순수) | 480.67 | |
| ARIMA(3,1,2) | 1,211.88 | AIC 기준 최적 order |
| LSTM (2층, lookback=4) | 1,105.43 | |
| ROR_Transformer+Transformer+LSTM | 684.30 | dl_lstm_transformer 최고 |
| RES_Transformer+Transformer | 723.70 | Residual 보정 (DL) |
| BASE_Transformer | 955.20 | DL 단독 |

**핵심 관찰**: 복잡도가 높을수록 성능이 떨어지는 역전 현상이 일관되게 나타난다. DL 최고 성능(684.30)조차 단순 Hybrid(406.80)보다 크게 나쁘다. I(1) 프로세스 + 2025년 이후 레짐 변화(공급 구조 변화, 거시 환경 전환)로 인해 학습 구간의 패턴을 기억한 모델이 오히려 불리하다.

### 2.6 Time Series Cross-Validation (5-Fold)

LightGBM 단독 모델의 일반화 안정성 검증:

| Fold | Train Size | Val Size | RMSE |
|------|-----------|----------|------|
| 1 | 103 | 103 | 2,797.84 |
| 2 | 206 | 103 | 2,323.80 |
| 3 | 309 | 103 | 2,859.14 |
| 4 | 412 | 103 | 8,603.09 |
| 5 | 515 | 102 | 3,727.84 |

Fold 4(2022년 러시아-우크라이나 전쟁 포함 구간)에서 RMSE 8,603으로 급등했다. 변동계수 62.2%로, 레짐 변화 구간에서 ML 모델의 일반화 성능이 극단적으로 불안정함을 확인했다.

---

## 3. Residual 보정 방법론

### 3.1 일반모델 격차 Residual 방식

구조:

```
Base Model 학습 (전체 학습셋)
  → 학습셋 잔차 r_t = y_t - ŷ_t 계산 (In-sample)
  → Residual Model이 r_t 를 타겟으로 학습
  → 최종 예측 = Base 예측 + Residual 예측
```

`dl_lstm_transformer.ipynb`의 Stage 2가 이 방식을 구현했다. BASE_Transformer(RMSE 955.2)의 잔차를 Transformer가 학습하는 구조로, RES_Transformer+Transformer(RMSE 723.7)를 달성했다.

```python
# Stage 2: Residual 학습 (dl_lstm_transformer.ipynb)
residual_train = y_train_actual - base_pred_train   # In-sample 잔차

res_model = TransformerForecaster(n_features=n_features + 2)
# base 예측값을 추가 피처로 append
X_res = append_meta(X_train, base_pred_train)
res_model.fit(X_res, residual_train)

final_pred = base_pred_test + res_model.predict(X_res_test)
```

**구조적 한계**: 잔차 모델이 학습하는 `r_t`는 In-sample 잔차다. Base Model이 학습 데이터에 이미 적합된 후 남은 오차를 다시 같은 데이터로 학습하므로, 메타 학습에 낙관적 편향이 생긴다. Base Model이 충분히 복잡하면 `r_t ≈ 0`이 되어 Residual Model이 학습할 신호가 사라진다.

### 3.2 OOF (Out-of-Fold) 방식

**방법론적 개선점**: 메타 모델이 학습하는 데이터가 Out-of-sample 예측이다.

```
for fold k in TimeSeriesSplit(K=5):
    fold_train → [Naive Drift, GB, LGB, PatchTST] 독립 학습
                  └── DL: fold_train에만 StandardScaler fit
    fold_val   → OOF 예측 수집 (Out-of-sample 보장)

Meta Model: Ridge([OOF_ND, OOF_GB, OOF_LGB, OOF_PTST]) → y_train

Inference:
    전체 Train으로 Base Models 재학습 → Meta Model 적용
```

**OOF 정확성 보장 포인트 4가지**:

| 항목 | 구현 방법 |
|------|---------|
| 시간 방향 leakage | `TimeSeriesSplit` (과거→미래 방향 강제 고정) |
| 스케일러 leakage | `StandardScaler`를 `fold_train`에만 `.fit()` |
| 시퀀스 경계 | `n_tr_seqs = n_tr - SEQ_LEN` 으로 경계 수식 검증 |
| Naive Drift 일관성 | OOF와 인퍼런스 모두 동일 공식 적용 |

**시퀀스 경계 수식 검증**:

```
X_seqs[k].target = y_combined[k + SEQ_LEN]

fold_train 타겟: k + SEQ_LEN < n_tr  →  k < n_tr - SEQ_LEN
fold_val   타겟: k + SEQ_LEN ≥ n_tr  →  k ≥ n_tr - SEQ_LEN

따라서: n_tr_seqs = n_tr - SEQ_LEN  (경계)
         n_val_seqs = n_val           (검증: 전체 - n_tr_seqs = n_val ✓)
```

구현:

```python
tscv = TimeSeriesSplit(n_splits=5)

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tr_arr)):
    # Naive Drift OOF: 인퍼런스와 동일한 공식
    last, drift = y_fold_tr[-1], y_fold_tr[-1] - y_fold_tr[-2]
    oof_nd[val_idx] = [last + (k+1)*drift for k in range(n_val)]

    # GB, LGB OOF: 스케일링 불필요
    oof_gb[val_idx]  = GradientBoostingRegressor(...).fit(X_fold_tr, y_fold_tr).predict(X_fold_val)
    oof_lgb[val_idx] = LGBMRegressor(...).fit(X_fold_tr, y_fold_tr).predict(X_fold_val)

    # PatchTST OOF: fold별 독립 스케일러
    xs = StandardScaler().fit(X_fold_tr)          # fold_train에만 fit
    ys = StandardScaler().fit(y_fold_tr.reshape(-1,1))
    X_comb_s = xs.transform(np.vstack([X_fold_tr, X_fold_val]))
    y_comb_s = ys.transform(np.concatenate([y_fold_tr, y_fold_val]).reshape(-1,1)).ravel()

    X_seqs, y_seqs = make_sequences(X_comb_s, y_comb_s, SEQ_LEN=26)
    n_tr_seqs = n_tr - 26   # 경계 수식

    model = PatchTST(N_VARS, seq_len=26, patch_len=4, stride=2, d_model=32)
    _train_patchtst(model, X_seqs[:n_tr_seqs], y_seqs[:n_tr_seqs], epochs=40)

    pred_s = model(torch.FloatTensor(X_seqs[n_tr_seqs:]))
    oof_ptst[val_idx] = ys.inverse_transform(pred_s.reshape(-1,1)).ravel()

# Ridge 메타 모델
X_meta = np.column_stack([oof_nd[mask], oof_gb[mask], oof_lgb[mask], oof_ptst[mask]])
meta   = Ridge(alpha=1.0).fit(X_meta, y_tr_arr[mask])
```

### 3.3 두 방식 비교

| 구분 | Residual 방식 | OOF 방식 |
|------|-------------|---------|
| 메타 입력 | In-sample 잔차 | Out-of-sample 예측값 |
| 과적합 위험 | 높음 (같은 데이터 재사용) | 낮음 (홀드아웃 예측) |
| 구현 복잡도 | 낮음 | 높음 (fold별 학습, 스케일러 분리) |
| 메타 모델 해석 | 잔차의 패턴 | 각 모델의 앙상블 가중치 |
| DL 적용 시 추가 고려 | 없음 | 시퀀스 경계, 스케일러 leakage |

dl_lstm_transformer에서 Residual 방식이 BASE 대비 24% 개선(955.2→723.7)을 보였지만, Hybrid 기준선(406.80)은 여전히 넘지 못했다. OOF 방식은 메타 모델이 4개 Base Model의 Out-of-sample 신호를 통합하므로, 특히 Naive Drift의 강점을 데이터 기반으로 인식하고 적절한 가중치를 부여할 수 있다.

---

## 4. 논문 기반 성능 향상 아이디어

### 4.1 PatchTST (Nie et al., 2023) — 구현 완료

**"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023**

Vanilla Transformer의 Point-wise 어텐션은 두 가지 문제가 있다: (1) 개별 타임스텝이 토큰이 되어 지역적 시간 패턴을 포착하지 못하고, (2) 시퀀스 길이에 이차적 연산이 필요하다.

PatchTST 해결책:

```
Patch 토큰화: 연속된 4주(patch_len)를 하나의 토큰으로 압축
  seq_len=26, patch_len=4, stride=2 → n_patches = (26-4)//2 + 1 = 12개
  (26개 타임스텝 → 12개 토큰: Transformer 입력 절반 이하로 축소)

Channel-Independence: 85개 피처를 독립 처리 후 채널 평균 집계
  Input (B, 26, 85) → (B×85, 12, 4) → Transformer → (B×85, 12, 32)
                    → reshape (B, 85, 384) → mean(dim=1) → Linear → (B,)

Pre-LN: norm_first=True로 학습 안정성 확보
```

하이퍼파라미터 선택 근거:

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| seq_len | 26 | 6개월: 반기 추세 포착, 샘플 수 대비 균형 |
| patch_len | 4 | 월간 추세 단위 (4주 ≈ 1개월) |
| stride | 2 | 50% 오버랩으로 패턴 연속성 유지 |
| d_model | 32 | 학습 샘플 ~600개 대비 과적합 방지 |
| n_layers | 2 | 논문 기본값 대비 소형화 |

```python
class PatchTST(nn.Module):
    def forward(self, x):
        B, L, C = x.shape
        # Patch 분할: (B, C, n_patches, patch_len)
        x = x.permute(0, 2, 1).unfold(2, self.patch_len, self.stride)
        n_p = x.shape[2]
        # 채널 독립: (B*C, n_patches, patch_len)
        x = x.reshape(B * C, n_p, self.patch_len)
        # Patch 임베딩 + 위치 인코딩
        x = self.dropout(self.patch_proj(x) + self.pos_enc)
        # Transformer Encoder (Pre-LN)
        x = self.encoder(x)
        # 채널 평균 후 예측
        x = x.reshape(B, C, -1).mean(dim=1)
        return self.head(x).squeeze(-1)
```

### 4.2 추가 탐색 아이디어

**Temporal Fusion Transformer (Lim et al., 2021, NeurIPS)**

Variable Selection Network가 85개 피처 중 실질적으로 유효한 변수를 자동으로 선별한다. 본 프로젝트에서 SHAP 분석 결과 상위 피처가 LME 금속 자체 래그·환율에 집중되어 있었는데, TFT의 Variable Selection이 이 구조를 자동으로 발견할 수 있다. 또한 Multi-horizon 예측이 가능하여 단일 모델로 여러 주 앞 예측이 가능하다.

**Conformal Prediction (Angelopoulos & Bates, 2021)**

점 예측 대신 커버리지를 보장하는 예측 구간을 제공한다. Val 셋의 잔차 분포를 이용해 사후적으로 구간을 보정(Split Conformal)하므로, 12개밖에 안 되는 Test 샘플에서 RMSE 단일 지표보다 "실제값이 구간에 포함되는 비율"이 더 안정적인 평가 기준이 될 수 있다.

**Regime Detection + Switching Model**

HMM(Hidden Markov Model) 또는 BOCPD(Bayesian Online Changepoint Detection)로 시장 레짐(강세/약세/횡보)을 탐지하고, 레짐별로 별도 예측 모델을 운용하는 구조다. Time Series CV에서 Fold 4 RMSE가 8,603(평균의 2.5배)으로 급등한 것이 레짐 변화(전쟁 발발) 때문이었다는 점을 고려하면, 레짐 탐지가 구조적으로 필요한 데이터임을 확인했다.

---

## 5. 간헐적 패턴에 대한 예측 모형

### 5.1 문제 정의

간헐적 수요(Intermittent Demand)는 수요가 불규칙적으로 발생하는 패턴이다.

```
예시: [0, 0, 0, 12, 0, 0, 7, 0, 0, 0, 0, 3, 0, ...]
       ←── 비발생 ──→ 발생 ←── 비발생 ──→ 발생
```

이 패턴의 핵심 특성:
- 0이 다수를 차지 → RMSE, MAPE 왜곡
- 수요 발생 시 크기가 불규칙 (lumpy demand)
- 단순 회귀 모델이 0 근방에 편향

전통적인 단일 모델 접근의 문제: "발생 여부"와 "발생량"이 완전히 다른 메커니즘으로 구동되는데 이를 하나의 모델로 동시에 포착하려 한다.

### 5.2 이벤트 예측 (Stage 1: Binary Classification)

```
입력: 시간 피처 (계절성, 직전 수요까지의 간격, 누적 재고 등)
출력: p_t = P(수요 발생 at t)
```

주요 피처:
- `days_since_last_demand`: 마지막 수요 발생 이후 경과 기간
- `demand_freq_30d`: 최근 30일 수요 발생 빈도
- 계절성 더미 (월, 분기)
- 외부 이벤트 더미 (프로모션, 계절 요인)

```python
# Stage 1: 수요 발생 여부 예측
y_event = (y_demand > 0).astype(int)

clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_event_train)

p_t = clf.predict_proba(X_test)[:, 1]   # 발생 확률
```

평가 지표: Precision, Recall, F1 (RMSE 부적합)

### 5.3 출고량 예측 (Stage 2: Conditional Regression)

```
입력: 수요 발생 시점의 피처
출력: E[수요량 | 수요 발생]   — 조건부 기댓값
```

Stage 2는 **수요 발생 시점의 샘플만** 으로 학습한다. 0을 제외한 조건부 분포를 학습하므로 출고량 추정 정밀도가 높아진다.

```python
# Stage 2: 수요 발생 시점만 추출
demand_mask  = y_demand > 0
X_cond       = X_train[demand_mask]
y_cond       = y_demand[demand_mask]   # 0 제외

reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
reg.fit(X_cond, y_cond)

q_t = reg.predict(X_test)   # 조건부 출고량

# 최종 예측: 발생 확률 × 조건부 출고량
y_pred = p_t * q_t
```

이 구조는 통계학의 Hurdle Model(두 단계로 분리)과 동일하다. 수요 발생 분포는 Bernoulli, 발생량 분포는 Log-normal 또는 Gamma로 가정하면 확률적 예측 구간도 도출할 수 있다.

### 5.4 전통적 방법론 비교

| 방법 | 특징 | 한계 |
|------|------|------|
| **Croston (1972)** | 수요량·간격을 각각 지수평활 | 편향 추정 (과대예측) |
| **TSB (2011)** | 수요 확률을 직접 추정, 감쇄 적용 | 파라미터 민감도 |
| **ADIDA** | 시간 집계 후 예측 → 분해 | 집계 단위 선택 어려움 |
| **2단계 모형 (이번)** | 피처 기반, 비선형 관계 포착 | 학습 데이터 희소 가능 |
| **DeepAR (Amazon)** | LSTM 기반 확률적 예측, 음이항분포 출력 | 학습 데이터 다량 필요 |

### 5.5 평가 지표

간헐적 수요에서 RMSE/MAPE는 0이 많을 때 왜곡된다.

| 지표 | 설명 | 적합 이유 |
|------|------|---------|
| **MASE** | Naive 대비 상대 오차, 스케일 불변 | 0 포함 시계열에서 안정 |
| **CRPS** | 확률 예측 품질 종합 평가 | 점 예측 + 불확실성 동시 |
| Stage 1: F1 Score | 발생 탐지 정밀도 | 이벤트 탐지 분리 평가 |
| Stage 2: 조건부 MAE | 발생 시점 출고량 오차 | 발생량 추정 분리 평가 |

Stage 1과 Stage 2의 오류를 분리해 평가해야 개선 방향을 정확히 찾을 수 있다. 전체 예측 오류가 크다면 발생 탐지 문제인지(Stage 1) 출고량 추정 문제인지(Stage 2)에 따라 대응이 완전히 달라진다.

---

## 6. 결론

1. **수요예측 모델 리뷰**: I(1) 프로세스 + 레짐 변화 조합에서 복잡도 상승이 성능 하락으로 이어지는 패턴을 모든 실험에서 일관되게 확인했다. Naive Drift의 강점은 단순함이 아니라 단기 모멘텀이 가장 지배적인 신호라는 데이터 특성에서 비롯된다.

2. **Residual 방식 vs OOF 방식**: Residual 방식(dl_lstm_transformer Stage 2)은 BASE 대비 24% 개선을 보였으나 Hybrid 기준선을 넘지 못했다. OOF 방식은 메타 모델이 Out-of-sample 예측을 학습해 과적합 편향이 없으며, Ridge 가중치가 각 모델의 실질적 기여도를 데이터 기반으로 정량화한다.

3. **PatchTST**: 4주 단위 Patch 토큰화로 지역적 추세를 포착하고, Channel-Independence로 85개 피처를 독립 처리한다. 학습 샘플 ~600개에 맞게 모델 크기를 d_model=32로 조정했다.

4. **간헐적 패턴**: 2단계 분리 모델링(이벤트 탐지 + 조건부 출고량)이 단일 회귀 대비 구조적으로 우월하다. Stage별 독립 평가(F1 + 조건부 MAE)가 개선 방향 진단의 핵심이다.
