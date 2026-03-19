# 01. 핵심쟁점

---

- `기본모델(PatchTST)`과 `실험모델(Residual Correction)`을 비교했을 때, **동일한 단변량 horizon-12 benchmark에서 성능 향상이 있는지**
- `NLinear`, `XGBoost`, `LightGBM` 중 어떤 잔차보정모형이 `ts-cv` 평균 기준과 `최종 holdout` 기준에서 가장 안정적인지

# 02. 데이터 및 모델 세팅

---

- **예측 타깃**
  - `WTI Oil` 현물 가격 (`Com_CrudeOil`)
  - `Brent Oil` 현물 가격 (`Com_BrentCrudeOil`)
- **예측 단위**
  - 주간 예측
- **시계열 구성**
  - 각 타깃은 **별도의 단변량 시계열**로 구성

- **실험에 사용된 전체 구간**
  - `2013-04-01 ~ 2026-01-12` (총 `668개` 주간 관측치)

- **평가 프로토콜 공통 설정**
  - `horizon = 12`
  - `step size = 4`
  - `number of windows = 24`
  - `final holdout = 12`
  - `input size = 48`
  - `season length = 52`
  - 최근 `116주`가 실질적인 평가 영역이며, 이는 `ts-cv 104주 + final holdout 12주 = 52*2 + 12`로 계산

- **데이터 구간 세팅**
  - TrainSet 기간: `2013-04-01 ~ 2023-10-23` (총 `552주`)
  - ValidationSet 기간: `2023-10-30 ~ 2025-10-20` (총 `24개 Fold`, 고유 평가일수 `104주`)
    - Cross-Validation Fold 1: `2023-10-30 ~ 2024-01-15` (총 `12주`)
    - Cross-Validation Fold 2: `2023-11-27 ~ 2024-02-12` (총 `12주`)
    - Cross-Validation Fold 3: `2023-12-25 ~ 2024-03-11` (총 `12주`)
    - Cross-Validation Fold 4: `2024-01-22 ~ 2024-04-08` (총 `12주`)
    - Cross-Validation Fold 5: `2024-02-19 ~ 2024-05-06` (총 `12주`)
    - Cross-Validation Fold 6: `2024-03-18 ~ 2024-06-03` (총 `12주`)
    - Cross-Validation Fold 7: `2024-04-15 ~ 2024-07-01` (총 `12주`)
    - Cross-Validation Fold 8: `2024-05-13 ~ 2024-07-29` (총 `12주`)
    - Cross-Validation Fold 9: `2024-06-10 ~ 2024-08-26` (총 `12주`)
    - Cross-Validation Fold 10: `2024-07-08 ~ 2024-09-23` (총 `12주`)
    - Cross-Validation Fold 11: `2024-08-05 ~ 2024-10-21` (총 `12주`)
    - Cross-Validation Fold 12: `2024-09-02 ~ 2024-11-18` (총 `12주`)
    - Cross-Validation Fold 13: `2024-09-30 ~ 2024-12-16` (총 `12주`)
    - Cross-Validation Fold 14: `2024-10-28 ~ 2025-01-13` (총 `12주`)
    - Cross-Validation Fold 15: `2024-11-25 ~ 2025-02-10` (총 `12주`)
    - Cross-Validation Fold 16: `2024-12-23 ~ 2025-03-10` (총 `12주`)
    - Cross-Validation Fold 17: `2025-01-20 ~ 2025-04-07` (총 `12주`)
    - Cross-Validation Fold 18: `2025-02-17 ~ 2025-05-05` (총 `12주`)
    - Cross-Validation Fold 19: `2025-03-17 ~ 2025-06-02` (총 `12주`)
    - Cross-Validation Fold 20: `2025-04-14 ~ 2025-06-30` (총 `12주`)
    - Cross-Validation Fold 21: `2025-05-12 ~ 2025-07-28` (총 `12주`)
    - Cross-Validation Fold 22: `2025-06-09 ~ 2025-08-25` (총 `12주`)
    - Cross-Validation Fold 23: `2025-07-07 ~ 2025-09-22` (총 `12주`)
    - Cross-Validation Fold 24: `2025-08-04 ~ 2025-10-20` (총 `12주`)
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

- **피처리스트**
  - 기본모델

    ```python
    [target_series_only]
    ```

  - 실험모델
    - 동일한 단변량 입력(`48주 history`)을 사용하고, baseline의 inner OOF residual을 `12-step residual target`으로 학습

- **모델 세팅**
  - 기본모델 `PatchTST`

    ```python
    patchtst_params = {
        "hidden_size": 128,
        "attention_heads": 16,
        "linear_hidden_size": 256,
        "patch_len": 16,
        "stride": 8,
        "dropout": 0.2,
        "encoder_layers": 3,
        "attn_dropout": 0.0,
        "fc_dropout": 0.2,
        "max_steps": 5000,
        "learning_rate": 0.0001,
        "scaler_type": "identity",
        "batch_size": 32,
        "patience": 12,
    }
    ```

  - 실험모델 `Residual Correction`

    ```python
    residual_models = {
        "NLinear": {"hidden_size": 128, "max_steps": 300, "learning_rate": 0.001},
        "XGBoost": {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4},
        "LightGBM": {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4},
    }
    ```

# 03. 실험 설계 및 적용

---

- 모든 비교는 **단변량 direct 12-step forecasting protocol**에서 수행했다.
- outer 평가는 `expanding window ts-cv 24개`와 `최종 holdout 12주`로 분리했다.
- 잔차보정모형은 baseline의 `in-sample residual`이 아니라, 각 outer train 안에서 생성한 **inner OOF PatchTST residual**을 학습했다.
- `ts-cv` 결과는 overlapping forecast가 존재하므로, 같은 날짜에 대한 예측을 평균낸 뒤 `RMSE`, `MAE`, `MAPE`, `NRMSE`를 계산했다.
- `NRMSE`는 각 타깃의 전체 range(`2013-04-01 ~ 2026-01-12`)로 고정 정규화했다.
- 따라서 본 결과는 이전 exploratory `rolling 1-step` 결과와 직접 비교하면 안 되며, **이번 문단에서 정의한 strict benchmark의 결과값**으로만 해석해야 한다.

# 04. 실험(모델링) 결과

### 04-01. 결과 요약

---

- **TS-CV 평균**
  - `WTI Oil`: `PatchTST + NLinear`가 가장 우수
  - `Brent Oil`: `PatchTST + NLinear`가 가장 우수
- **최종 Holdout 12주**
  - `WTI Oil`: `PatchTST + XGBoost`가 가장 우수
  - `Brent Oil`: `PatchTST + LightGBM`가 가장 우수

- **핵심 해석**
  - `NLinear` 잔차보정은 `ts-cv 평균 재현성`에서 가장 안정적인 개선을 보였음
  - `최종 holdout 12주`에서는 `WTI=XGBoost`, `Brent=LightGBM`이 가장 좋았고, 두 타깃 모두 baseline보다 개선되었음
  - 다만 `Naive(last value)` reference와 비교하면, 이번 실험의 모든 learned model은 아직 `Naive`를 넘지 못했음

- **핵심 Leaderboard (TS-CV, date-averaged)**

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Brent Oil | PatchTST | `NLinear` | **6.879** | **6.037** | **8.193** | **0.071** |
| Brent Oil | PatchTST | `-` | 8.460 | 7.021 | 9.097 | 0.087 |
| Brent Oil | PatchTST | `LightGBM` | 10.586 | 9.317 | 12.945 | 0.109 |
| Brent Oil | PatchTST | `XGBoost` | 11.361 | 9.953 | 13.858 | 0.117 |
| WTI Oil | PatchTST | `NLinear` | **6.611** | **5.543** | **7.676** | **0.066** |
| WTI Oil | PatchTST | `LightGBM` | 9.857 | 8.561 | 12.552 | 0.098 |
| WTI Oil | PatchTST | `XGBoost` | 10.225 | 8.855 | 12.996 | 0.102 |
| WTI Oil | PatchTST | `-` | 10.264 | 8.377 | 11.164 | 0.102 |

### 04-02. 세부 결과

---

- **TS-CV Leaderboard**
  - 기준 파일: [leaderboard_tscv.csv](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_univariate_direct_residual_0320/leaderboard_tscv.csv)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Brent Oil | PatchTST | `NLinear` | **6.879** | **6.037** | **8.193** | **0.071** |
| Brent Oil | PatchTST | `-` | 8.460 | 7.021 | 9.097 | 0.087 |
| Brent Oil | PatchTST | `LightGBM` | 10.586 | 9.317 | 12.945 | 0.109 |
| Brent Oil | PatchTST | `XGBoost` | 11.361 | 9.953 | 13.858 | 0.117 |
| WTI Oil | PatchTST | `NLinear` | **6.611** | **5.543** | **7.676** | **0.066** |
| WTI Oil | PatchTST | `LightGBM` | 9.857 | 8.561 | 12.552 | 0.098 |
| WTI Oil | PatchTST | `XGBoost` | 10.225 | 8.855 | 12.996 | 0.102 |
| WTI Oil | PatchTST | `-` | 10.264 | 8.377 | 11.164 | 0.102 |

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
  - 기준 파일: [leaderboard_holdout.csv](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_univariate_direct_residual_0320/leaderboard_holdout.csv)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Brent Oil | PatchTST | `LightGBM` | **9.525** | **9.272** | **14.863** | **0.098** |
| Brent Oil | PatchTST | `XGBoost` | 10.726 | 10.458 | 16.750 | 0.110 |
| Brent Oil | PatchTST | `-` | 12.578 | 12.516 | 20.016 | 0.129 |
| Brent Oil | PatchTST | `NLinear` | 13.197 | 13.128 | 20.988 | 0.135 |
| WTI Oil | PatchTST | `XGBoost` | **8.328** | **8.095** | **13.827** | **0.083** |
| WTI Oil | PatchTST | `LightGBM` | 8.369 | 8.112 | 13.865 | 0.084 |
| WTI Oil | PatchTST | `-` | 9.348 | 9.282 | 15.839 | 0.093 |
| WTI Oil | PatchTST | `NLinear` | 9.561 | 9.495 | 16.181 | 0.095 |

- **Naive Reference Sanity Check**
  - 기준 파일: [naive_reference.csv](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_univariate_direct_residual_0320/naive_reference.csv)

| Target | Eval Split | Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Brent Oil | TS-CV | `Naive(last value)` | **4.531** | **3.591** | **4.849** | **0.046** |
| WTI Oil | TS-CV | `Naive(last value)` | **4.933** | **4.040** | **5.745** | **0.049** |
| Brent Oil | Holdout | `Naive(last value)` | **1.818** | **1.383** | **2.247** | **0.019** |
| WTI Oil | Holdout | `Naive(last value)` | **1.648** | **1.279** | **2.212** | **0.016** |

- **Plot**
  - plot manifest: [plot_manifest.csv](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_univariate_direct_residual_0320/plot_manifest.csv)

![WTI Oil TS-CV](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_univariate_direct_residual_0320/plots/tscv_wti_oil_actual_vs_pred.png)

![Brent Oil Holdout](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_univariate_direct_residual_0320/plots/holdout_brent_oil_actual_vs_pred.png)

# 05. 결론 및 얻게 된 인사이트

---

- 사용자가 정의한 `strict univariate direct horizon-12 benchmark` 기준에서는 `NLinear`가 `ts-cv 평균`에서 가장 일관되게 좋았다.
- `최종 holdout 12주`에서는 `WTI`는 `XGBoost`, `Brent`는 `LightGBM`이 가장 좋았고, 두 타깃 모두 `PatchTST baseline`보다는 개선되었다.
- 즉, 이번 exact setting 결과는 `평균 재현성(ts-cv)`과 `최종 최근 구간(holdout)`의 최적 residual model이 다를 수 있음을 보여준다.
- 그러나 같은 protocol의 `Naive(last value)`가 여전히 모든 learned model보다 좋았으므로, 현재 결론은 `residual correction이 custom PatchTST baseline 대비 개선`이라는 수준으로만 해석해야 한다.
- 이전의 `1점대 성능`은 다른 프로토콜이나 단일 confirmatory 세팅에서 나온 값일 가능성이 높고, 본 결과와 직접 비교하면 안 된다.

# 06. 향후 Action Plan

---

- 발표본 메인 결과는 이 strict benchmark 결과만 사용한다.
- `ts-cv best model`과 `holdout best model`을 분리해서 설명한다.
- learned model 결과를 제시할 때는 반드시 `Naive` reference를 함께 보여준다.
- 후속 실험은 `공식 PatchTST 구현 사용 또는 현재 구현 재검증`, `residual learner 입력 구조 보강`, `single-target confirmatory rerun` 순서로 진행한다.
- 이전 `rolling 1-step` 및 `multivariate+exogenous` 결과는 부록/reference로만 남긴다.
