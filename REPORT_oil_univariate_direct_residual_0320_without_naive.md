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
        "NLinear": {
            "architecture": "Linear(48 -> 12)",
            "last_value_normalization": True,
            "max_steps": 300,
            "learning_rate": 0.001,
        },
        "XGBoost": {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4},
        "LightGBM": {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4},
    }
    ```

# 03. 실험 설계 및 적용

---

- 모든 비교는 **단변량 direct 12-step forecasting protocol**에서 수행했다.
- outer 평가는 `expanding window ts-cv 24개`와 `최종 holdout 12주`로 분리했다.
- 잔차보정모형은 baseline의 `in-sample residual`이 아니라, 각 outer train 안에서 생성한 **inner OOF PatchTST residual**을 학습했다.
- `NLinear`는 구현 audit 후 hidden-layer MLP가 아닌 **선형 direct residual head**로 교정했다.
- 본 문서는 `PatchTST baseline`과 `Residual Correction`의 **learned-model 내부 비교본**이다.

# 04. 실험(모델링) 결과

### 04-01. 결과 요약 (MAPE 기준)

---

- **TS-CV 평균**
  - `WTI Oil`: `PatchTST baseline`이 가장 우수
  - `Brent Oil`: `PatchTST baseline`이 가장 우수
- **최종 Holdout 12주**
  - `WTI Oil`: `PatchTST + XGBoost`가 가장 우수
  - `Brent Oil`: `PatchTST + LightGBM`가 가장 우수

| Target | Bench-mark (PatchTST, %) | 실험모델 (Best Residual, %) | 증감 (%) |
| --- | --- | --- | --- |
| WTI Oil | 11.164 | 11.504 (`NLinear`) | +0.340 |
| Brent Oil | 9.097 | 9.972 (`NLinear`) | +0.875 |

- **핵심 Leaderboard**
  - 기준 파일: [leaderboard_tscv.csv](output_oil_univariate_direct_residual_0320/leaderboard_tscv.csv)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Brent Oil | PatchTST | `-` | 8.460 | 7.021 | 9.097 | 0.087 |
| Brent Oil | PatchTST | `NLinear` | 9.250 | 7.688 | 9.972 | 0.095 |
| Brent Oil | PatchTST | `LightGBM` | 10.586 | 9.317 | 12.945 | 0.109 |
| Brent Oil | PatchTST | `XGBoost` | 11.361 | 9.953 | 13.858 | 0.117 |
| WTI Oil | PatchTST | `LightGBM` | 9.857 | 8.561 | 12.552 | 0.098 |
| WTI Oil | PatchTST | `XGBoost` | 10.225 | 8.855 | 12.996 | 0.102 |
| WTI Oil | PatchTST | `-` | 10.264 | 8.377 | 11.164 | 0.102 |
| WTI Oil | PatchTST | `NLinear` | 10.728 | 8.620 | 11.504 | 0.107 |

### 04-02. 세부 결과

---

- **Test Set Metric**
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
  - 기준 파일: [leaderboard_holdout.csv](output_oil_univariate_direct_residual_0320/leaderboard_holdout.csv)

| Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Brent Oil | PatchTST | `LightGBM` | **9.525** | **9.272** | **14.863** | **0.098** |
| Brent Oil | PatchTST | `XGBoost` | 10.726 | 10.458 | 16.750 | 0.110 |
| Brent Oil | PatchTST | `-` | 12.578 | 12.516 | 20.016 | 0.129 |
| Brent Oil | PatchTST | `NLinear` | 14.683 | 14.533 | 23.212 | 0.151 |
| WTI Oil | PatchTST | `XGBoost` | **8.328** | **8.095** | **13.827** | **0.083** |
| WTI Oil | PatchTST | `LightGBM` | 8.369 | 8.112 | 13.865 | 0.084 |
| WTI Oil | PatchTST | `-` | 9.348 | 9.282 | 15.839 | 0.093 |
| WTI Oil | PatchTST | `NLinear` | 13.073 | 12.851 | 21.935 | 0.131 |

- **Plot**
  - plot manifest: [plot_manifest.csv](output_oil_univariate_direct_residual_0320/plot_manifest.csv)

![WTI Oil TS-CV](output_oil_univariate_direct_residual_0320/plots/tscv_wti_oil_actual_vs_pred.png)

![Brent Oil Holdout](output_oil_univariate_direct_residual_0320/plots/holdout_brent_oil_actual_vs_pred.png)

# 05. 결론 및 얻게 된 인사이트

---

- 구현 audit 후 교정된 `NLinear`는 `TS-CV`와 `Holdout` 모두에서 baseline을 넘지 못했다.
- learned-model 내부 비교 기준에서는 `TS-CV`에서 `PatchTST baseline` 자체가 가장 좋았고, `Holdout 12주`에서는 `WTI=XGBoost`, `Brent=LightGBM`가 최적 residual이었다.
- 즉, 이번 strict benchmark에서는 `tree-based residual correction은 holdout에서만 일부 개선`, `NLinear correction은 개선을 만들지 못했다`.

# 06. 향후 Action Plan

---

- 발표에서 learned-model 내부 비교가 필요할 때는 이 `Naive 제외 버전`을 사용한다.
- 다만 메인 결론은 반드시 `Naive 포함 버전`과 함께 제시한다.
- `NLinear 우세`로 보였던 이전 내부 로그는 교정 전 구현과 직접 비교하면 안 된다.
- 후속 실험은 `공식 PatchTST 구현 검증`, `residual learner 입력 구조 보강`, `single-target confirmatory rerun` 순서로 진행한다.
