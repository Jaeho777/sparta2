# 유가 가격 예측 실험 보고서

## 결론
- 본 보고서의 모든 실험은 동일한 시계열 분할을 사용하였다. Train은 `2013-04-01`~`2025-07-28` 644주, Validation은 `2025-08-04`~`2025-10-20` 12주, Test는 `2025-10-27`~`2026-01-12` 12주이다.
- 메인 브랜치의 니켈 예측 방식인 `Naive Drift + GradientBoosting Hybrid`를 유가에 그대로 적용한 결과, 최고 baseline은 `Hybrid_Naive0.7_GB0.3`였으며 test RMSE는 `1.3440`였다.
- Advanced 파이프라인의 타깃 `y_t`는 `Com_BrentCrudeOil_t`이며, 입력 `X_t`는 22개 원변수와 33개 파생변수를 생성한 뒤 전부 `shift(1)`한 55개 후보 변수로 구성하였다. 따라서 예측 시점 `t`에서 `t-1`까지의 정보만 사용하였다.
- Feature selection은 train 구간에서만 수행하였다. LightGBM SHAP importance와 `TimeSeriesCV(5-fold)`를 결합한 결과 최적 변수 수는 10개였고, 선정 변수는 `Spread_Crack, EX_USD_KRW, Com_Gasoline, Ratio_Gold_Oil, Com_Coal, Com_Gasoline_ma12r, Com_PalmOil, Idx_SnPVIX, Bonds_US_10Y, Bonds_US_3M_ret`였다.
- Stage 1-Stage 2 조합은 일부 조합이 아니라 `5 x 5 = 25개` 전체를 screening하였다. 이후 screening 상위 8개 조합만 동일 분할에서 보다 큰 학습 예산으로 confirmatory rerun하였다.
- 현재 selection rule에 따른 공식 채택 모델은 `PatchTST+Transformer + ElasticNet RoR`이다. 이 조합은 validation 기준 최종 RMSE `1.2443`로 1위를 기록하였다.
- 반면 confirmatory stage에서 가장 낮은 holdout test RMSE는 `PatchTST+iTransformer`의 `1.2308`였다. 다만 해당 조합은 `Stage 2 validation top 3` 규칙 밖에 있었으므로, 본 보고서에서는 이를 `사후 holdout candidate`로만 해석하였다.
- Residual과 RoR는 각각 1차 오차와 2차 오차에 예측 가능한 구조가 남아 있는지 확인하기 위한 보정 단계이다. 그러나 이번 결과는 이 보정이 항상 holdout 성능 향상으로 이어지지 않는다는 점도 함께 보여주었다.

## 1. 실험 목적
본 실험의 목적은 다음 두 가지를 검증하는 데 있다.
1. 메인 브랜치의 니켈 예측 구조를 유가 데이터에 그대로 적용했을 때 확보되는 baseline 성능을 확인한다.
2. SHAP 기반 변수 선택, 교차검증, 다양한 transformer 계열 조합, residual 및 RoR 보정을 체계적으로 결합했을 때 성능 향상이 재현 가능한지 평가한다.

## 2. 데이터와 분할
### 2.1 데이터 개요
- 파일: `data_weekly_260120.csv`
- 기간: 2013-04-01 ~ 2026-01-12
- 빈도: 주간(월요일 기준)
- 총 관측치: 668주
- 타깃 변수: `Com_BrentCrudeOil`

### 2.2 입력과 타깃의 정의
- 타깃 `y_t`: 시점 `t`의 Brent 유가 `Com_BrentCrudeOil`
- 입력 `X_t`: 원변수 22개와 파생변수 33개를 생성한 뒤 `shift(1)`을 적용한 값
- 파생변수: 수익률(`*_ret`), 금리 스프레드(`Spread_10Y_2Y`), 크랙 스프레드(`Spread_Crack`), 금/유 비율(`Ratio_Gold_Oil`), 이동평균 비율(`*_ma4r`, `*_ma12r`)

### 2.3 Train / Validation / Test 설정

**표 1. Train, Validation, Test 기간과 용도**
| Split | Target 기간 | 샘플 수 | 용도 |
| --- | --- | ---: | --- |
| Train | 2013-04-01 ~ 2025-07-28 | 644 | 모델 학습, SHAP, 변수 수 선택 |
| Validation | 2025-08-04 ~ 2025-10-20 | 12 | 조합 선택, lambda 선택, top 3 선정 |
| Test | 2025-10-27 ~ 2026-01-12 | 12 | 최종 holdout 평가 |

### 2.4 Sequence 입력 구성
Advanced 모델은 24주 시퀀스를 사용하였다.
- Validation target을 만들 때는 train 마지막 24주(`2025-02-17`~`2025-07-28`)를 입력 buffer로만 사용하였다.
- Test target을 만들 때는 train+validation 마지막 24주(`2025-05-12`~`2025-10-20`)를 입력 buffer로만 사용하였다.
- 따라서 validation/test의 입력은 직전 24주의 과거 정보만 참조하며, 평가 target 자체는 train과 중복되지 않는다.

### 2.5 실험 구조 도식

**그림 1. 전체 실험 절차**
```text
원자료(주간 패널)
  -> 타깃 y = Brent 유가
  -> 원변수 + 파생변수 생성
  -> 모든 X에 shift(1) 적용
  -> Train-only SHAP ranking
  -> TimeSeriesCV로 변수 수 선택
  -> 25개 조합 screening
  -> 상위 8개 confirmatory rerun
  -> Validation top 3만 RoR 적용
  -> 최종 holdout 평가
```

설명: 변수 선택, 조합 선택, RoR 채택 여부 결정은 모두 train 또는 validation 정보만으로 수행되며, test는 최종 holdout 평가에만 사용된다.

**그림 2. Residual 및 RoR 보정 구조**
```text
Stage 1 Base Model
  -> yhat_base
  -> Residual target r_t = y_t - yhat_base,t
  -> Stage 2 Residual Model
  -> yhat_stage2 = yhat_base + rhat_t
  -> RoR target q_t = r_t - rhat_t
  -> RoR Model
  -> yhat_final = yhat_stage2 + lambda * qhat_t
```

설명: Residual 단계는 1차 오차의 예측 가능성을 검토하고, RoR 단계는 residual model이 남긴 2차 오차의 예측 가능성을 추가로 검토한다.

**그림 3. 공식 채택 모델과 사후 holdout candidate의 구분**
```text
Validation ranking
  -> 공식 채택 모델 선정

Test ranking
  -> 사후 holdout candidate 해석

주의:
  test 성능은 해석용이지 선택용이 아님
```

설명: 본 보고서는 validation 기준 공식 채택 모델과, 사후적으로 holdout test가 가장 낮게 나온 조합을 명시적으로 구분하여 서술한다.

## 3. Baseline 재현: 니켈 메인 방식의 유가 적용
### 3.1 방법
- Naive Drift와 GradientBoostingRegressor를 가중 평균하여 hybrid를 구성하였다.
- GradientBoostingRegressor 파라미터는 `n_estimators=500`, `learning_rate=0.05`, `max_depth=3`으로 고정하였다.
- Hybrid 가중치는 `0.7/0.3`, `0.8/0.2`, `0.9/0.1`을 비교하였다.
- 모든 입력 변수에는 `shift(1)`을 적용하였고, 결측은 train fit 기반 median imputation으로 처리하였다.

### 3.2 결과

**표 2. 니켈 메인 방식 baseline 성능 비교**
| Model | Validation_RMSE | RMSE | MAE | MAPE(%) |
| --- | --- | --- | --- | --- |
| Naive_Drift |  | 1.5125 | 1.2740 | 2.0383 |
| GradientBoosting |  | 2.4451 | 2.1321 | 3.4385 |
| Hybrid_Naive0.7_GB0.3 | 2.3013 | 1.3440 | 1.1317 | 1.8162 |
| Hybrid_Naive0.8_GB0.2 | 2.4552 | 1.3474 | 1.1522 | 1.8476 |
| Hybrid_Naive0.9_GB0.1 | 2.6261 | 1.4056 | 1.2056 | 1.9305 |

해석: `Hybrid_Naive0.7_GB0.3`가 동일 분할에서 test RMSE `1.3440`로 가장 우수한 baseline이었다. 이후의 모든 advanced 결과는 이 기준선과 함께 해석하는 것이 타당하다.

## 4. Feature selection: SHAP + TimeSeriesCV
### 4.1 절차
1. Train 구간의 lagged candidate 55개를 구성하였다.
2. LightGBM(`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`)을 train에만 적합하였다.
3. Train SHAP mean absolute value로 변수를 정렬하였다.
4. 상위 `10/15/20/25/30/40/55`개 변수 조합에 대해 `TimeSeriesCV(5-fold)` 평균 RMSE를 계산하였다.
5. 평균 RMSE가 가장 낮은 변수 개수를 최종 채택하였다.

### 4.2 CV 결과

**표 3. Feature 수별 TimeSeriesCV 결과**
| n_features | cv_rmse |
| --- | --- |
| 10 | 7.4507 |
| 15 | 7.8164 |
| 20 | 8.3217 |
| 25 | 9.9492 |
| 30 | 10.0514 |
| 40 | 10.1836 |
| 55 | 10.2080 |

### 4.3 최종 선정 변수

**표 4. SHAP 기준 최종 선정 변수 10개**
| feature | mean_abs_shap |
| --- | --- |
| Spread_Crack | 15.7882 |
| EX_USD_KRW | 0.7411 |
| Com_Gasoline | 0.5219 |
| Ratio_Gold_Oil | 0.5150 |
| Com_Coal | 0.2761 |
| Com_Gasoline_ma12r | 0.2611 |
| Com_PalmOil | 0.1877 |
| Idx_SnPVIX | 0.1836 |
| Bonds_US_10Y | 0.1718 |
| Bonds_US_3M_ret | 0.1548 |

해석: 변수 수가 10개를 초과하면 CV RMSE가 일관되게 악화되었다. 본 데이터에서는 많은 변수를 사용하는 것보다, train 구간에서 안정적으로 설명력을 보인 소수 변수 집합이 더 적절하였다.

## 5. Advanced 파이프라인과 선정 규칙
### 5.1 후보 모델

**표 5. Advanced 파이프라인 후보 모델과 핵심 파라미터**
| 모델 | 설명 | 핵심 파라미터 |
| --- | --- | --- |
| `NLinear` | 시간축 선형 패턴과 마지막 시점 feature를 함께 사용하는 간결한 모델 | hidden 64, dropout 0.3 |
| `PatchTST` | 시퀀스를 patch로 나누어 transformer encoder로 처리하는 구조 | d_model 64, heads 4, layers 2, patch_len 4, stride 2 |
| `iTransformer` | 변수 축을 token처럼 다루어 cross-feature 상호작용을 학습하는 구조 | d_model 64, heads 4, layers 2 |
| `Transformer` | 시간축 self-attention encoder | d_model 64, heads 4, layers 2 |
| `LSTM` | Recurrent hidden state로 순차 정보를 누적하는 모델 | hidden 64, layers 2 |

### 5.2 공통 학습 설정
- optimizer: `Adam(lr=1e-3, weight_decay=1e-5)`
- scheduler: `ReduceLROnPlateau(patience=8, factor=0.5)`
- batch size: `32`
- gradient clipping: `1.0`
- target은 train 평균/표준편차로 정규화하였다.

### 5.3 Selection rule
- 1차 screening: 25개 조합 전체, `seeds=1`, `epochs=60`, `patience=12`
- 2차 confirmatory rerun: screening 상위 8개, `seeds=3`, `epochs=120`, `patience=20`
- 공식 top 3 선정 기준: confirmatory `S2_Val_RMSE`
- RoR 적용 범위: `Stage 2 validation top 3` 조합에 한정

이 규칙에 따라 `PatchTST+iTransformer`는 test에서 가장 낮은 RMSE를 기록했더라도, S2 validation 순위가 5위였으므로 RoR 단계 대상이 아니었다.

## 6. 조합 실험 결과
### 6.1 Screening 결과

**표 6. 25개 조합 screening 결과**
| Experiment | S2_Val_RMSE | S2_Test_RMSE | Seeds | Epochs |
| --- | --- | --- | --- | --- |
| Transformer+iTransformer | 1.1475 | 1.7200 | 1 | 60 |
| LSTM+PatchTST | 1.2821 | 1.3509 | 1 | 60 |
| PatchTST+iTransformer | 1.2848 | 1.7348 | 1 | 60 |
| PatchTST+Transformer | 1.3128 | 2.0704 | 1 | 60 |
| Transformer+PatchTST | 1.3371 | 1.8776 | 1 | 60 |
| Transformer+Transformer | 1.3897 | 2.1475 | 1 | 60 |
| PatchTST+LSTM | 1.4245 | 1.5973 | 1 | 60 |
| PatchTST+PatchTST | 1.5113 | 1.5374 | 1 | 60 |
| PatchTST+NLinear | 1.5354 | 3.2821 | 1 | 60 |
| Transformer+LSTM | 1.6629 | 1.3904 | 1 | 60 |
| LSTM+iTransformer | 1.8730 | 1.8685 | 1 | 60 |
| LSTM+Transformer | 1.9803 | 1.7189 | 1 | 60 |
| NLinear+iTransformer | 2.0395 | 5.4941 | 1 | 60 |
| Transformer+NLinear | 2.0922 | 1.5706 | 1 | 60 |
| NLinear+NLinear | 2.1923 | 6.3189 | 1 | 60 |
| NLinear+PatchTST | 2.2044 | 6.7506 | 1 | 60 |
| NLinear+Transformer | 2.3402 | 6.7266 | 1 | 60 |
| LSTM+LSTM | 2.6254 | 2.3973 | 1 | 60 |
| iTransformer+Transformer | 2.7563 | 3.4119 | 1 | 60 |
| LSTM+NLinear | 2.8150 | 2.7488 | 1 | 60 |
| NLinear+LSTM | 2.8872 | 4.1890 | 1 | 60 |
| iTransformer+PatchTST | 2.9149 | 4.0342 | 1 | 60 |
| iTransformer+iTransformer | 3.3947 | 3.5602 | 1 | 60 |
| iTransformer+LSTM | 4.2452 | 5.0900 | 1 | 60 |
| iTransformer+NLinear | 4.6643 | 6.4417 | 1 | 60 |

해석: Screening 단계의 validation 순위는 confirmatory rerun에서 일부 변동되었다. 따라서 screening 결과만으로 최종 결론을 내리는 것은 적절하지 않다.

### 6.2 Confirmatory 상위 8개

**표 7. Confirmatory rerun 상위 8개 결과**
| Experiment | Base_Val_RMSE | Base_Test_RMSE | S2_Val_RMSE | S2_Test_RMSE |
| --- | --- | --- | --- | --- |
| PatchTST+Transformer | 1.5235 | 1.5339 | 1.2580 | 2.2752 |
| PatchTST+LSTM | 1.5235 | 1.5339 | 1.2839 | 2.7910 |
| LSTM+PatchTST | 1.8687 | 1.2776 | 1.2870 | 1.7378 |
| Transformer+PatchTST | 1.3204 | 1.5563 | 1.2922 | 1.5927 |
| PatchTST+iTransformer | 1.5235 | 1.5339 | 1.3534 | 1.2308 |
| Transformer+iTransformer | 1.3204 | 1.5563 | 1.4456 | 1.4526 |
| PatchTST+PatchTST | 1.5235 | 1.5339 | 1.4880 | 1.5401 |
| Transformer+Transformer | 1.3204 | 1.5563 | 1.4914 | 1.4453 |

### 6.3 Confirmatory stage의 test 순위

**표 8. Confirmatory stage의 holdout test 순위**
| Experiment | S2_Val_RMSE | S2_Test_RMSE |
| --- | --- | --- |
| PatchTST+iTransformer | 1.3534 | 1.2308 |
| Transformer+Transformer | 1.4914 | 1.4453 |
| Transformer+iTransformer | 1.4456 | 1.4526 |
| PatchTST+PatchTST | 1.4880 | 1.5401 |
| Transformer+PatchTST | 1.2922 | 1.5927 |
| LSTM+PatchTST | 1.2870 | 1.7378 |
| PatchTST+Transformer | 1.2580 | 2.2752 |
| PatchTST+LSTM | 1.2839 | 2.7910 |

### 6.4 조합별 해석

**표 9. 주요 조합의 단계별 성능과 해석**
| Experiment | Base_Val | S2_Val | Final_Val | Base_Test | S2_Test | Final_Test | 해석 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PatchTST+Transformer | 1.5235 | 1.2580 | 1.2443 | 1.5339 | 2.2752 | 2.3577 | 선정 규칙상 공식 채택 대상이다. 다만 holdout test에서는 base보다 성능이 저하되었다. |
| PatchTST+iTransformer | 1.5235 | 1.3534 | 1.3534 | 1.5339 | 1.2308 | 1.2308 | confirmatory 8개 중 base 대비 validation과 test를 모두 개선한 유일한 조합이다. 그러나 validation top 3 규칙 밖이므로 공식 채택 대상은 아니다. |
| LSTM+PatchTST | 1.8687 | 1.2870 | 1.2524 | 1.2776 | 1.7378 | 1.5402 | RoR 적용 후 Stage 2 대비 test가 일부 회복되었으나, 최종 test는 LSTM base보다 높다. |
| PatchTST+LSTM | 1.5235 | 1.2839 | 1.2789 | 1.5339 | 2.7910 | 2.7255 | validation 순위는 높지만 holdout test에서는 residual과 RoR 모두 base를 상회하지 못하였다. |
| Transformer+Transformer | 1.3204 | 1.4914 | 1.4914 | 1.5563 | 1.4453 | 1.4453 | validation은 악화되었으나 test는 base 대비 소폭 개선되었다. 추가 검증이 필요한 후보이다. |
| Transformer+iTransformer | 1.3204 | 1.4456 | 1.4456 | 1.5563 | 1.4526 | 1.4526 | test는 base 대비 소폭 개선되었으나 validation이 악화되어 현재 규칙에서는 채택 근거가 부족하다. |

종합 해석:
- `PatchTST+Transformer`는 selection rule상 공식 채택 모델이지만, holdout test에서는 base보다 성능이 저하되었다.
- `PatchTST+iTransformer`는 confirmatory 8개 중 base 대비 validation과 test를 모두 개선한 유일한 조합이었다. 다만 현재 규칙상 공식 채택 모델은 아니다.
- `LSTM+PatchTST`의 RoR는 Stage 2 대비 test를 일부 회복하였으나, 최종 test는 LSTM base보다 높았다.
- `Transformer+Transformer`, `Transformer+iTransformer`는 holdout test에서 base 대비 소폭 개선되었지만 validation이 악화되어, 추가 rolling validation 없이는 채택 근거로 사용하기 어렵다.

## 7. Residual, RoR, OOF의 의미와 한계
### 7.1 Residual 보정
Residual 단계는 base 모델이 설명하지 못한 1차 오차 `r_t = y_t^z - yhat_base_t^z`에 예측 가능한 구조가 남아 있는지 확인하기 위한 절차이다.

### 7.2 RoR 보정
RoR 단계는 residual 모델이 설명하지 못한 2차 오차 `q_t = r_t - rhat_t`를 다시 한 번 모델링하는 절차이다. 최종 예측은 `yhat_final_t = yhat_stage2_t + lambda * qhat_t * sigma_y`로 계산하였다.

### 7.3 RoR 결과

**표 10. 공식 top 3 조합의 RoR 보정 결과**
| Experiment | RoR_Strategy | RoR_Description | S2_Val_RMSE | S2_Test_RMSE | Final_Val_RMSE | Final_Test_RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| PatchTST+Transformer | C | ElasticNet(a=0.01,l1=0.1,l=0.50) | 1.2580 | 2.2752 | 1.2443 | 2.3577 |
| LSTM+PatchTST | G | AllFeatLGBM(l=0.50) | 1.2870 | 1.7378 | 1.2524 | 1.5402 |
| PatchTST+LSTM | G | AllFeatLGBM(l=0.50) | 1.2839 | 2.7910 | 1.2789 | 2.7255 |

해석:
- `PatchTST+Transformer`에서는 ElasticNet RoR가 validation은 개선했으나 test는 추가로 악화시켰다.
- `LSTM+PatchTST`에서는 All-feature LightGBM RoR가 Stage 2 test를 일부 회복하였다. 다만 base보다 우수하지는 않았다.
- `PatchTST+LSTM`에서는 validation 개선 폭이 작고 test는 여전히 높았다.

따라서 Residual과 RoR는 방법론적으로 의미가 있으나, 본 split에서는 `항상 성능을 높이는 일반 해법`으로 해석할 수 없다.

## 8. 누수 점검
- 모든 feature는 생성 후 `shift(1)`을 적용하였다.
- SHAP importance와 변수 수 선택은 train에서만 수행하였다.
- Scaler는 train에만 fit하였다.
- Sequence buffer는 과거 문맥으로만 사용하였으며, target과 중복되지 않는다.
- Screening, confirmatory rerun, top 3 선정, RoR lambda 선택은 validation 또는 train 내부 정보로만 수행하였다.
- Test는 최종 holdout 보고용으로만 사용하였다.

판정: 코드 기준으로 직접적인 데이터 누수는 확인되지 않았다. 현재 남아 있는 주요 위험은 leakage가 아니라 `12주 validation block의 짧음`에 따른 selection variance이다.

## 9. 향후 보완 방향
1. `Nested rolling-origin validation`을 적용하여 여러 validation block에서 조합 순위를 집계할 필요가 있다.
2. `PatchTST+iTransformer`는 새로운 confirmatory window에서 재평가할 필요가 있다.
3. `Transformer+Transformer`, `Transformer+iTransformer`와 같이 holdout에서는 양호하나 validation이 약한 조합은 반복 검증이 필요하다.
4. RoR는 `Stage 2 대비`와 `base 대비`를 동시에 만족하는 경우에만 채택하도록 규칙을 강화할 필요가 있다.
5. SSM/Mamba 계열과 direct multi-horizon decoder를 추가하여 residual two-stage 구조 자체를 대체할 후보도 검토할 수 있다.

## 10. 최종 요약
1. Baseline과 advanced 실험은 동일한 train/validation/test 분할을 사용하였다.
2. 니켈 메인 방식 hybrid baseline의 최고 성능은 test RMSE `1.3440`였다.
3. Advanced 실험에서는 SHAP + TimeSeriesCV로 55개 후보 변수를 10개로 축소하였다.
4. 25개 조합을 screening하고 상위 8개를 confirmatory rerun하였다.
5. Selection rule상 공식 채택 모델은 `PatchTST+Transformer + ElasticNet RoR`였다.
6. Confirmatory holdout test 최저 조합은 `PatchTST+iTransformer`였으며, test RMSE는 `1.2308`였다.
7. 따라서 본 보고서의 결론은 `공식 채택 모델`과 `사후 holdout candidate`를 구분하여 제시하는 것이 타당하다.
