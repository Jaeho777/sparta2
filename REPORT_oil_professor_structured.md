# Brent 유가 실험로그 통합 정리 보고서

본 문서는 `REPORT_oil_professor.md`와 관련 산출물에 흩어져 있던 실험 로그를 교수님 공유용 포맷으로 다시 정리한 버전이다. 동일한 모델 이름이라도 피처 수, seed 수, Stage 3 보정 방식이 다르면 별도 로그로 분리했다.

## 전체 요약

| 순서 | 실험 로그 | Bench-mark | 실험모델 | Test RMSE | Test MAPE (%) | 해석 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | PatchTST 단독 vs PatchTST+NLinear | PatchTST | PatchTST+NLinear | 1.4494 -> 1.8609 | 1.8341 -> 2.1539 | Test 악화 |
| 2 | PatchTST+NLinear vs Stage 3 보정 | PatchTST+NLinear (S2) | PatchTST+NLinear+RW Blend | 1.6544 -> 1.4291 | 기록 미보존 | Test 개선, 단 FC 효과 |
| 3 | PatchTST vs PatchTST+iTransformer | PatchTST | PatchTST+iTransformer | 1.5339 -> 1.2308 | 1.5997 -> 1.5054 | 단일 분할 Test 개선 |
| 4 | PatchTST+Transformer vs +ElasticNet RoR | PatchTST+Transformer | PatchTST+Transformer+ElasticNet | 2.2752 -> 2.3577 | 3.0008 -> 3.1342 | Validation 개선, Test 악화 |
| 5 | ExpSmoothing+NLinear vs +LightGBM RoR | ExpSmoothing+NLinear | ExpSmoothing+NLinear+LightGBM | 1.2648 -> 1.2275 | 1.8192 -> 1.4708 | 단일 분할 Test 소폭 개선 |

## 실험 1. PatchTST + NLinear (단독 실험)

### 01. 핵심쟁점

- `기본모델`인 `PatchTST` 대비 `PatchTST + NLinear`이 실제 Test 구간에서도 성능 향상이 있는지 확인
- 추가 Stage 3인 `LightGBM RoR`까지 붙였을 때 추가 이득이 있는지 확인

### 02. 데이터 및 모델 세팅

- **예측 타깃:**
  - Brent 원유 현물 가격 (`Com_BrentCrudeOil`)
- **예측 단위:**
  - 주간 예측
- **데이터 구간 세팅:**
  - TrainSet 기간: `2013-04-01 ~ 2025-07-28` (총 `644주`)
  - ValidationSet 기간: `2025-08-04 ~ 2025-10-20` (총 `1개 Fold`)
    - Cross-Validation Fold 1: `2025-08-04 ~ 2025-10-20` (총 `12주`)
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
- **피처리스트**
  - 기본모델 / 실험모델 공통 (총 `55개`, 모두 `1주 lag` 적용)

    ```python
    'Com_Gasoline', 'Com_NaturalGas', 'Com_Uranium', 'Com_Coal',
    'Com_LME_Cu_Cash', 'Com_Steel', 'Com_Iron_Ore', 'Idx_DxyUSD', 'EX_USD_CNY',
    'Bonds_US_10Y', 'Bonds_US_2Y', 'Bonds_US_3M', 'Idx_SnPVIX', 'Com_Gold',
    'Idx_SnP500', 'Idx_CSI300', 'EX_USD_KRW', 'Bonds_KOR_10Y', 'EX_USD_JPY',
    'Com_Corn', 'Com_Soybeans', 'Com_PalmOil', 'Com_Gasoline_ret',
    'Com_NaturalGas_ret', 'Com_Uranium_ret', 'Com_Coal_ret',
    'Com_LME_Cu_Cash_ret', 'Com_Steel_ret', 'Com_Iron_Ore_ret',
    'Idx_DxyUSD_ret', 'EX_USD_CNY_ret', 'Bonds_US_10Y_ret', 'Bonds_US_2Y_ret',
    'Bonds_US_3M_ret', 'Idx_SnPVIX_ret', 'Com_Gold_ret', 'Idx_SnP500_ret',
    'Idx_CSI300_ret', 'EX_USD_KRW_ret', 'Bonds_KOR_10Y_ret', 'EX_USD_JPY_ret',
    'Com_Corn_ret', 'Com_Soybeans_ret', 'Com_PalmOil_ret', 'Spread_US_10Y_2Y',
    'Spread_Crack', 'Ratio_Gold_Oil', 'Com_Gasoline_ma4_ratio',
    'Com_Gasoline_ma12_ratio', 'Com_NaturalGas_ma4_ratio',
    'Com_NaturalGas_ma12_ratio', 'Idx_SnPVIX_ma4_ratio',
    'Idx_SnPVIX_ma12_ratio', 'Idx_DxyUSD_ma4_ratio', 'Idx_DxyUSD_ma12_ratio'
    ```

- **모델 세팅**
  - 기본모델 `PatchTST`

    ```python
    patchtst_params = {
        "seq_len": 24,
        "seeds": 3,
        "epochs": 120,
        "patience": 20,
        "lr": 1e-3,
    }
    ```

  - 실험모델 `PatchTST + NLinear (+ LightGBM RoR check)`

    ```python
    nlinear_params = {
        "seq_len": 24,
        "pred_len": 1,
        "hidden": 64,
        "lr": 1e-3,
        "epochs": 300,
        "batch_size": 32,
        "patience": 40,
        "seeds": 10,
        "top_k": 5,
    }

    lightgbm_ror_params = {
        "n_folds": 5,
        "learning_rate": 0.02,
        "num_leaves": 10,
        "min_child_samples": 40,
        "subsample": 0.6,
        "colsample_bytree": 0.5,
        "reg_alpha": 2.0,
        "reg_lambda": 2.0,
        "ror_lambda": 0.0,
    }
    ```

### 03. 실험 설계 및 적용

- Stage 1에서 `PatchTST`가 Brent 가격 수준을 직접 예측
- Stage 2에서 `NLinear`가 Stage 1 잔차를 예측하도록 설계
- Validation gate 결과 `LightGBM RoR`는 채택되지 않았고, 최종적으로 `ror_lambda = 0.0`이라 Stage 3 이득은 없음

### 04. 실험(모델링) 결과

#### 04-01. 결과 요약 (MAPE 기준)

|  | Bench-mark (%) | 실험모델 (%) | 증감 (%p) |
| --- | --- | --- | --- |
| Brent | 1.83 | 2.15 | +0.32 |

#### 04-02. 세부 결과

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

    |  | RMSE | MAPE (%) | nRMSE (%) (mean-normalized) | MAE | R2 |
    | --- | --- | --- | --- | --- | --- |
    | Bench-mark: PatchTST | 1.4494 | 1.8341 | 2.3118 | 1.1445 | -0.1419 |
    | 실험모델: PatchTST+NLinear | 1.8609 | 2.1539 | 2.9683 | 1.3548 | -0.8826 |
    | 참고: PatchTST+NLinear+LightGBM | 1.8609 | 2.1539 | 2.9683 | 1.3548 | -0.8826 |

- **Validation 참고**
  - Validation RMSE는 `2.0242 -> 1.5342`로 좋아졌지만, Test에서는 반대로 악화됨

### 05. 결론 및 얻게 된 인사이트

- `PatchTST+NLinear`는 Validation에서는 좋아 보였지만 Test에서는 `PatchTST` 단독보다 확실히 나빠짐
- 이 로그에서는 `NLinear residual correction`이 일반화되지 못했고, `LightGBM RoR`도 validation gate를 통과하지 못함
- 발표 시에는 이 실험을 `PatchTST+NLinear 단독 적용은 Test 개선을 만들지 못한 케이스`로 정리하는 것이 안전함

### 06. 향후 Action Plan

- `PatchTST+NLinear`는 단독 주력모델이 아니라 비교용 로그로 두기
- 같은 구조를 재검증하려면 rolling-origin으로 다시 확인하기
- 다음 실험부터는 Stage 3 미채택 사유(`lambda=0`)를 결과표에 같이 남기기

## 실험 2. PatchTST + NLinear + Stage 3 보정 (Enhanced RoR 로그)

### 01. 핵심쟁점

- `PatchTST+NLinear (S2)` 대비 Stage 3 보정이 실제로 성능 향상을 만드는지 확인
- Stage 3 개선이 `순수 ML-RoR` 때문인지, 아니면 `RW Blend(예측결합)` 때문인지 구분

### 02. 데이터 및 모델 세팅

- **예측 타깃:**
  - Brent 원유 현물 가격 (`Com_BrentCrudeOil`)
- **예측 단위:**
  - 주간 예측
- **데이터 구간 세팅:**
  - TrainSet 기간: `2013-04-01 ~ 2025-07-28` (총 `644주`)
  - ValidationSet 기간: `2025-08-04 ~ 2025-10-20` (총 `1개 Fold`)
    - Cross-Validation Fold 1: `2025-08-04 ~ 2025-10-20` (총 `12주`)
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
- **피처리스트**
  - Stage 1 / Stage 2 공통 (SHAP + TimeSeriesCV 선별 `10개`)

    ```python
    'Spread_Crack', 'EX_USD_KRW', 'Com_Gasoline', 'Ratio_Gold_Oil',
    'Com_Coal', 'Com_Gasoline_ma12r', 'Com_PalmOil', 'Idx_SnPVIX',
    'Bonds_US_10Y', 'Bonds_US_3M_ret'
    ```

  - Stage 3 참고
    - `Strategy G: RW_Blend`는 외생변수 없이 `S2`와 `Random Walk`를 가중결합
    - `Strategy H: AllFeatLGBM`는 실험 1과 동일한 `55개 전체 피처` 사용

- **모델 세팅**
  - 기본모델 `PatchTST + NLinear (S2)`

    ```python
    common_params = {
        "seq_len": 24,
        "n_seeds": 5,
    }

    patchtst_params = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "patch_len": 4,
        "stride": 2,
    }

    nlinear_params = {
        "hidden": 64,
        "dropout": 0.3,
    }
    ```

  - 실험모델 `PatchTST + NLinear + Stage 3`

    ```python
    stage3_best_params = {
        "selected_strategy": "RW_Blend",
        "w": 0.26,
    }

    stage3_best_ml_ror = {
        "selected_strategy": "AllFeatLGBM",
        "lambda": 0.50,
    }
    ```

### 03. 실험 설계 및 적용

- Stage 1 `PatchTST`, Stage 2 `NLinear` 뒤에 Stage 3 후보 `A~H` 8개 전략을 모두 비교
- 최종 채택은 Validation RMSE 기준이며, 이 실험의 최종 승자는 `Strategy G = RW_Blend(w=0.26)`
- 순수 ML-RoR만 놓고 보면 `Strategy H = AllFeatLGBM(lambda=0.50)`가 가장 나았지만 개선폭은 작았음

### 04. 실험(모델링) 결과

#### 04-01. 결과 요약 (RMSE 기준; 상세 MAPE 로그 미보존)

|  | Bench-mark RMSE | 실험모델 RMSE | 증감 |
| --- | --- | --- | --- |
| Brent | 1.6544 | 1.4291 | -0.2253 |

#### 04-02. 세부 결과

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

    |  | RMSE | MAPE (%) | nRMSE (%) | MAE | R2 |
    | --- | --- | --- | --- | --- | --- |
    | Bench-mark: PatchTST+NLinear (S2) | 1.6544 | 기록 미보존 | 기록 미보존 | 기록 미보존 | 기록 미보존 |
    | 참고: Best ML-RoR = +AllFeatLGBM(H) | 1.6195 | 기록 미보존 | 기록 미보존 | 기록 미보존 | 기록 미보존 |
    | 실험모델(최종): +RW Blend(w=0.26) | 1.4291 | 기록 미보존 | 기록 미보존 | 기록 미보존 | 기록 미보존 |

- **Validation 참고**
  - S2 Val RMSE: `1.4589`
  - Best ML-RoR(H) Val RMSE: `1.4405`
  - Final RW Blend Val RMSE: `1.3978`

### 05. 결론 및 얻게 된 인사이트

- 이 로그에서 Stage 3의 최종 개선은 있었지만, 그 주된 원인은 `RoR 학습`이 아니라 `RW Blend`였다
- 순수 ML-RoR 최고치인 `AllFeatLGBM`는 Test RMSE를 `1.6544 -> 1.6195`로만 줄였고, 개선폭이 매우 작음
- 발표 시에는 `PatchTST+NLinear가 RoR로 크게 좋아졌다`고 말하기보다, `RW shrinkage/forecast combination이 일부 오류를 완화했다`고 표현하는 편이 정확함

### 06. 향후 Action Plan

- 발표 자료에서 `Forecast Combination`과 `ML-RoR`를 반드시 분리 표기하기
- 다음 로그부터는 Final 선택 전략별 `MAE / MAPE / nRMSE / R2`를 함께 저장하기
- `PatchTST+NLinear` 계열은 rolling-origin에서도 같은 개선이 재현되는지 확인하기

## 실험 3. PatchTST + iTransformer

### 01. 핵심쟁점

- `PatchTST` 기본예측 대비 `iTransformer` 잔차 보정이 Test 기준으로 개선을 만드는지 확인
- 단일 분할 개선이 반복 평가에서도 유지되는지 확인

### 02. 데이터 및 모델 세팅

- **예측 타깃:**
  - Brent 원유 현물 가격 (`Com_BrentCrudeOil`)
- **예측 단위:**
  - 주간 예측
- **데이터 구간 세팅:**
  - TrainSet 기간: `2013-04-01 ~ 2025-07-28` (총 `644주`)
  - ValidationSet 기간: `2025-08-04 ~ 2025-10-20` (총 `1개 Fold`)
    - Cross-Validation Fold 1: `2025-08-04 ~ 2025-10-20` (총 `12주`)
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
- **피처리스트**
  - 기본모델 / 실험모델 공통 (`10개`, SHAP + TimeSeriesCV 선별)

    ```python
    'Spread_Crack', 'EX_USD_KRW', 'Com_Gasoline', 'Ratio_Gold_Oil',
    'Com_Coal', 'Com_Gasoline_ma12r', 'Com_PalmOil', 'Idx_SnPVIX',
    'Bonds_US_10Y', 'Bonds_US_3M_ret'
    ```

- **모델 세팅**
  - 기본모델 `PatchTST`

    ```python
    patchtst_params = {
        "seq_len": 24,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "patch_len": 4,
        "stride": 2,
        "seeds": 3,
        "epochs": 120,
        "patience": 20,
        "optimizer": "Adam(lr=1e-3, weight_decay=1e-5)",
    }
    ```

  - 실험모델 `PatchTST + iTransformer`

    ```python
    itransformer_params = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "seeds": 3,
        "epochs": 120,
        "patience": 20,
        "optimizer": "Adam(lr=1e-3, weight_decay=1e-5)",
    }
    ```

### 03. 실험 설계 및 적용

- 25개 조합을 1차 screening 한 뒤 상위 8개를 confirmatory 재학습
- `PatchTST+iTransformer`는 공식 선정모델은 아니었지만 단일 분할 Test에서 가장 좋은 조합 중 하나로 확인됨
- 다만 반복 원점 평균 성능표에서는 `PatchTST+iTransformer` 평균 Test RMSE가 `3.8544`로 불안정함

### 04. 실험(모델링) 결과

#### 04-01. 결과 요약 (MAPE 기준)

|  | Bench-mark (%) | 실험모델 (%) | 증감 (%p) |
| --- | --- | --- | --- |
| Brent | 1.60 | 1.51 | -0.09 |

#### 04-02. 세부 결과

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

    |  | RMSE | MAPE (%) | nRMSE (%) (mean-normalized) | MAE | R2 |
    | --- | --- | --- | --- | --- | --- |
    | Bench-mark: PatchTST | 1.5339 | 1.5997 | 2.4467 | 1.0027 | -0.2791 |
    | 실험모델: PatchTST+iTransformer | 1.2308 | 1.5054 | 1.9633 | 0.9460 | 0.1764 |

- **Validation 참고**
  - Validation RMSE는 `1.5235 -> 1.3534`로 개선
  - 다만 top 3 validation 조합에는 들지 못해 공식 선정모델은 아님

### 05. 결론 및 얻게 된 인사이트

- 단일 분할 기준으로는 `PatchTST+iTransformer`가 `PatchTST`보다 확실히 좋았음
- 특히 Test RMSE가 `1.5339 -> 1.2308`, R2가 `-0.2791 -> 0.1764`로 개선되어 발표용 단일 분할 사례로는 강함
- 그러나 반복 평가 평균에서는 불안정하므로 `단일 분할 최저 오차 조합`으로 소개하고, `가장 안정적인 최종모델`이라고 말하는 것은 피해야 함

### 06. 향후 Action Plan

- 이 조합은 `single-split best case`로 정리하고 rolling-origin 재검증을 별도로 수행하기
- seed 수와 학습 예산을 늘린 뒤 평균 성능 분산을 다시 확인하기
- 발표에서는 `좋은 단일 구간 사례`와 `재현성`을 분리해서 설명하기

## 실험 4. PatchTST + Transformer + ElasticNet RoR

### 01. 핵심쟁점

- `PatchTST+Transformer` 2단계 모델에 `ElasticNet RoR`를 추가했을 때 실제 Test 향상이 있는지 확인
- Validation 기준 공식 선정모델이 Test에서도 유지되는지 확인

### 02. 데이터 및 모델 세팅

- **예측 타깃:**
  - Brent 원유 현물 가격 (`Com_BrentCrudeOil`)
- **예측 단위:**
  - 주간 예측
- **데이터 구간 세팅:**
  - TrainSet 기간: `2013-04-01 ~ 2025-07-28` (총 `644주`)
  - ValidationSet 기간: `2025-08-04 ~ 2025-10-20` (총 `1개 Fold`)
    - Cross-Validation Fold 1: `2025-08-04 ~ 2025-10-20` (총 `12주`)
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
- **피처리스트**
  - Stage 1 / Stage 2 공통 (`10개`, SHAP + TimeSeriesCV 선별)

    ```python
    'Spread_Crack', 'EX_USD_KRW', 'Com_Gasoline', 'Ratio_Gold_Oil',
    'Com_Coal', 'Com_Gasoline_ma12r', 'Com_PalmOil', 'Idx_SnPVIX',
    'Bonds_US_10Y', 'Bonds_US_3M_ret'
    ```

  - Stage 3

    ```python
    '2차 잔차 u_t'를 대상으로 ElasticNet(alpha=0.01, l1_ratio=0.10, lambda=0.50) 적용
    ```

- **모델 세팅**
  - 기본모델 `PatchTST + Transformer`

    ```python
    patchtst_params = {
        "seq_len": 24,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "patch_len": 4,
        "stride": 2,
        "seeds": 3,
        "epochs": 120,
        "patience": 20,
    }

    transformer_params = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "seeds": 3,
        "epochs": 120,
        "patience": 20,
    }
    ```

  - 실험모델 `PatchTST + Transformer + ElasticNet RoR`

    ```python
    elasticnet_ror_params = {
        "alpha": 0.01,
        "l1_ratio": 0.10,
        "lambda": 0.50,
    }
    ```

### 03. 실험 설계 및 적용

- 25개 조합 screening -> 상위 8개 confirmatory -> 상위 3개 Stage 3 RoR 적용
- `PatchTST+Transformer+ElasticNet`은 Validation RMSE가 가장 낮아 공식 선정모델이 되었음
- 그러나 반복 원점 평균에서는 `PatchTST+Transformer`의 평균 Test RMSE `4.7979`, `+ElasticNetRoR`의 평균 Test RMSE `4.8990`으로 오히려 악화

### 04. 실험(모델링) 결과

#### 04-01. 결과 요약 (MAPE 기준)

|  | Bench-mark (%) | 실험모델 (%) | 증감 (%p) |
| --- | --- | --- | --- |
| Brent | 3.00 | 3.13 | +0.13 |

#### 04-02. 세부 결과

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

    |  | RMSE | MAPE (%) | nRMSE (%) (mean-normalized) | MAE | R2 |
    | --- | --- | --- | --- | --- | --- |
    | Bench-mark: PatchTST+Transformer | 2.2752 | 3.0008 | 3.6290 | 1.8934 | -1.8139 |
    | 실험모델: +ElasticNet RoR | 2.3577 | 3.1342 | 3.7607 | 1.9776 | -2.0219 |

- **Validation 참고**
  - Validation RMSE는 `1.2580 -> 1.2443`로 소폭 개선

### 05. 결론 및 얻게 된 인사이트

- 이 조합은 `validation-selected official model`이라는 의미는 있지만 `test-best model`은 아님
- Validation에서는 소폭 좋아졌지만 Test에서는 모든 핵심 지표가 악화되었고, 반복 평가 평균에서도 개선이 재현되지 않음
- 발표 시에는 `공식 선정 기준을 통과한 모델`로만 소개하고, 성능 headline으로 사용하지 않는 것이 타당함

### 06. 향후 Action Plan

- RoR 채택 기준을 Validation 단일 값이 아니라 rolling-origin 평균으로 강화하기
- Stage 3의 선택 근거와 Test 결과를 같은 슬라이드에 함께 제시하기
- `official selected`와 `single-split best`를 명확히 분리해 보고서 체계를 유지하기

## 실험 5. ExpSmoothing + NLinear + LightGBM RoR (STL 기반)

### 01. 핵심쟁점

- `ExpSmoothing + NLinear` 대비 `LightGBM RoR` 추가가 실제로 의미 있는 개선을 만드는지 확인
- 구조 분해 기반 접근이 Transformer 계열과 별개로 설명 가능한 개선을 보이는지 확인

### 02. 데이터 및 모델 세팅

- **예측 타깃:**
  - Brent 원유 현물 가격 (`Com_BrentCrudeOil`)
- **예측 단위:**
  - 주간 예측
- **데이터 구간 세팅:**
  - TrainSet 기간: `2013-04-01 ~ 2025-07-28` (총 `644주`)
  - ValidationSet 기간: `2025-08-04 ~ 2025-10-20` (총 `1개 Fold`)
    - Cross-Validation Fold 1: `2025-08-04 ~ 2025-10-20` (총 `12주`)
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
- **피처리스트**
  - 기본모델 `ExpSmoothing`: Brent 단변량 시계열만 사용
  - 실험모델 `+NLinear`, `+LightGBM RoR`: 실험 1과 동일한 `55개 lagged macro-financial feature set` + 잔차 sequence 사용

- **모델 세팅**
  - 기본모델 `Exponential Smoothing + NLinear`

    ```python
    stl_params = {
        "seasonal": 53,
        "trend": None,
    }

    expsmoothing_params = {
        "trend": "add",
        "seasonal": "add",
        "seasonal_periods": 52,
        "initialization_method": "estimated",
        "use_boxcox": False,
    }

    nlinear_params = {
        "seq_len": 24,
        "pred_len": 1,
        "d_hidden": 64,
        "lr": 1e-3,
        "epochs": 300,
        "batch_size": 32,
        "patience": 40,
        "seeds": 10,
        "top_k": 5,
    }
    ```

  - 실험모델 `ExpSmoothing + NLinear + LightGBM RoR`

    ```python
    lightgbm_ror_params = {
        "n_folds": 5,
        "learning_rate": 0.02,
        "num_leaves": 10,
        "min_child_samples": 40,
        "subsample": 0.6,
        "colsample_bytree": 0.5,
        "reg_alpha": 2.0,
        "reg_lambda": 2.0,
        "n_estimators": 300,
    }
    ```

### 03. 실험 설계 및 적용

- STL로 시계열을 분해한 뒤 `Exponential Smoothing`으로 baseline 생성
- Stage 2에서 `NLinear`가 1차 residual을 예측하고, Stage 3에서 `LightGBM`이 residual-of-residual을 OOF 방식으로 보정
- 반복 원점 평균에서는 `ExpSmoothing+NLinear` 평균 Test RMSE `2.4743`, `+LightGBM` 평균 Test RMSE `2.5089`로 평균 기준 이득은 재현되지 않음

### 04. 실험(모델링) 결과

#### 04-01. 결과 요약 (MAPE 기준)

|  | Bench-mark (%) | 실험모델 (%) | 증감 (%p) |
| --- | --- | --- | --- |
| Brent | 1.82 | 1.47 | -0.35 |

#### 04-02. 세부 결과

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

    |  | RMSE | MAPE (%) | nRMSE (%) (mean-normalized) | MAE | R2 |
    | --- | --- | --- | --- | --- | --- |
    | Bench-mark: ExpSmoothing+NLinear | 1.2648 | 1.8192 | 2.0175 | 1.1353 | 0.1304 |
    | 실험모델: +LightGBM RoR | 1.2275 | 1.4708 | 1.9580 | 0.9144 | 0.1809 |

- **Validation 참고**
  - Validation RMSE는 `3.1619 -> 2.7126`으로 개선

### 05. 결론 및 얻게 된 인사이트

- 단일 분할 기준으로는 `LightGBM RoR`가 `ExpSmoothing+NLinear`보다 소폭 더 좋았음
- 특히 MAE와 MAPE 개선폭이 RMSE보다 더 커서, 큰 오차 몇 개보다 평균적인 절대오차 감소에 더 기여한 구조로 볼 수 있음
- 다만 반복 원점 평균에서는 개선이 안정적으로 재현되지 않으므로, 발표에서는 `구조적으로 해석 가능한 개선 사례` 정도로 표현하는 편이 맞음

### 06. 향후 Action Plan

- STL 계열은 해석 가능성 장점이 있으므로 발표 보조축으로 유지하기
- `단일 분할 개선`과 `반복 평균 미재현`을 함께 명시하기
- 향후에는 DM test와 rolling-origin 결과를 같은 표에 묶어서 저장하기

## 발표용 한 줄 정리

- `PatchTST+NLinear` 단독 적용은 Test 개선을 만들지 못했다.
- `PatchTST+NLinear`의 Stage 3 개선은 `RoR`보다는 `RW Blend(예측결합)`의 기여가 컸다.
- `PatchTST+iTransformer`는 단일 분할 기준으로는 가장 인상적인 Transformer 조합이다.
- `PatchTST+Transformer+ElasticNet`은 공식 선정모델이지만 Test headline으로 쓰기 어렵다.
- `ExpSmoothing+NLinear+LightGBM`는 단일 분할에서는 깔끔한 개선 사례지만, 평균 재현성은 추가 확인이 필요하다.
