# Oil Canonical Residual Benchmark 0319

본 문서는 [oil_canonical_residual_benchmark_0319.py](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/oil_canonical_residual_benchmark_0319.py) 기준의 최종 benchmark 보고서이다.  
이번 버전은 기존 `_0318` exploratory 결과와 달리 `OOF residual`, `단일 rolling 1-step outer protocol`, `date-averaged ts-cv leaderboard`, `고정 range NRMSE`를 사용해 **실험 구조 자체의 일관성**을 맞춘 canonical 비교본이다.

# 01. 핵심쟁점

---

- `기본모델(PatchTST)`과 `실험모델(Residual Correction)`을 비교했을 때, **같은 canonical protocol 안에서 성능 향상이 있는지**
- `NLinear`, `iTransformer`, `XGBoost`, `LightGBM` 중 어떤 residual model이 **구조적으로 일관된 benchmark**에서 가장 낫는지
- `단변량`과 `다변량+외생변수` setting에서 residual correction의 효과가 **타깃별로 어떻게 달라지는지**

# 02. 데이터 및 모델 세팅

---

- **전체 데이터 구간**
  - `2013-04-01 ~ 2026-01-12` (총 `668주`)
- **예측 타깃**
  - `WTI Oil` (`Com_CrudeOil`)
  - `Brent Oil` (`Com_BrentCrudeOil`)
- **예측 단위**
  - 주간 1-step rolling forecast

- **평가 프로토콜 공통 설정**
  - `input size = 48`
  - `outer eval window = 12`
  - `step size = 4`
  - `number of windows = 24`
  - `final holdout = 12`
  - `season length = 52`
  - `inner OOF splits = 5`
  - `inner validation size = 24`

- **데이터 구간 세팅**
  - TrainSet 기간: `2013-04-01 ~ 2023-10-23` (총 `552주`)
  - ValidationSet 기간: `2023-10-30 ~ 2025-10-20` (총 `24개 Fold`)
  - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

- **비교 setting**
  - `Univariate`
    - feature count: `1`
    - target series only
  - `Multivariate+Exogenous`
    - feature count: `59`
    - `lagged exogenous` 방식의 raw + engineered features

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
        "max_steps": 300,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "patience": 20,
    }
    ```

  - 실험모델 `Residual Correction`

    ```python
    residual_models = {
        "NLinear": {"max_steps": 200, "learning_rate": 0.001, "batch_size": 64},
        "iTransformer": {"max_steps": 200, "learning_rate": 0.001, "batch_size": 64},
        "XGBoost": {"n_estimators": 100, "learning_rate": 0.03},
        "LightGBM": {"n_estimators": 100, "learning_rate": 0.03},
    }
    ```

# 03. 실험 설계 및 적용

---

- 모든 비교는 **같은 outer rolling 1-step ts-cv + holdout protocol**에서 수행했다.
- 2단계 residual model은 baseline의 `in-sample fitted residual`이 아니라, **inner OOF PatchTST residual**로 학습했다.
- `ts-cv` leaderboard는 raw overlapping row를 그대로 합치지 않고, **date-averaged prediction** 기준으로 metric을 계산했다.
- `NRMSE`는 각 타깃의 전체 범위로 고정 정규화해 holdout 12주 길이에 따른 왜곡을 줄였다.
- 따라서 본 보고서의 수치는 이전 exploratory `_0318` 결과와 직접 이어서 해석하면 안 되며, **canonical benchmark의 최종 기준값**으로만 사용한다.

# 04. 실험(모델링) 결과

### 04-01. 결과 요약

---

- **Univariate / TS-CV**
  - `WTI`: `iTransformer` best
  - `Brent`: `iTransformer` best
- **Univariate / Holdout**
  - `WTI`: `NLinear` best
  - `Brent`: `NLinear` best

- **Multivariate+Exogenous / TS-CV**
  - `WTI`: `iTransformer` best
  - `Brent`: `iTransformer` best
- **Multivariate+Exogenous / Holdout**
  - `WTI`: `iTransformer` best
  - `Brent`: baseline `PatchTST` best

- **핵심 메시지**
  - canonical protocol 기준으로는 `iTransformer`가 `ts-cv 평균`에서 가장 강했고,
  - `NLinear`는 특히 `univariate holdout`에서 강했다.
  - `Brent exogenous holdout`에서는 residual correction이 baseline을 넘지 못했다.

### 04-02. 세부 결과

---

- **TS-CV Leaderboard**
  - 기준 파일: [leaderboard_tscv.csv](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_canonical_residual_0319/leaderboard_tscv.csv)

| Setting | Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Univariate | WTI Oil | PatchTST | `iTransformer` | **6.056** | **4.776** | **6.566** | **0.060** |
| Univariate | WTI Oil | PatchTST | `LightGBM` | 6.523 | 5.290 | 7.458 | 0.065 |
| Univariate | WTI Oil | PatchTST | `XGBoost` | 6.871 | 5.651 | 8.026 | 0.069 |
| Univariate | WTI Oil | PatchTST | `-` | 8.075 | 6.349 | 8.720 | 0.081 |
| Univariate | WTI Oil | PatchTST | `NLinear` | 7.877 | 6.422 | 9.146 | 0.079 |
| Univariate | Brent Oil | PatchTST | `iTransformer` | **5.424** | **4.321** | **5.693** | **0.056** |
| Univariate | Brent Oil | PatchTST | `NLinear` | 5.801 | 4.558 | 6.086 | 0.060 |
| Univariate | Brent Oil | PatchTST | `-` | 6.172 | 4.815 | 6.263 | 0.063 |
| Univariate | Brent Oil | PatchTST | `XGBoost` | 5.818 | 4.869 | 6.496 | 0.060 |
| Univariate | Brent Oil | PatchTST | `LightGBM` | 6.032 | 5.014 | 6.752 | 0.062 |
| Multivariate+Exogenous | WTI Oil | PatchTST | `iTransformer` | **5.193** | **4.076** | **5.781** | **0.052** |
| Multivariate+Exogenous | WTI Oil | PatchTST | `LightGBM` | 6.311 | 5.152 | 7.433 | 0.063 |
| Multivariate+Exogenous | WTI Oil | PatchTST | `XGBoost` | 6.428 | 5.164 | 7.462 | 0.064 |
| Multivariate+Exogenous | WTI Oil | PatchTST | `-` | 8.569 | 7.447 | 10.079 | 0.086 |
| Multivariate+Exogenous | WTI Oil | PatchTST | `NLinear` | 9.399 | 7.120 | 9.912 | 0.094 |
| Multivariate+Exogenous | Brent Oil | PatchTST | `iTransformer` | **7.008** | **5.701** | **7.535** | **0.072** |
| Multivariate+Exogenous | Brent Oil | PatchTST | `XGBoost` | 7.123 | 6.054 | 8.181 | 0.073 |
| Multivariate+Exogenous | Brent Oil | PatchTST | `LightGBM` | 7.192 | 6.038 | 8.106 | 0.074 |
| Multivariate+Exogenous | Brent Oil | PatchTST | `-` | 8.510 | 7.062 | 9.116 | 0.087 |
| Multivariate+Exogenous | Brent Oil | PatchTST | `NLinear` | 8.658 | 7.344 | 9.842 | 0.089 |

- **Holdout Leaderboard**
  - 기준 파일: [leaderboard_holdout.csv](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_canonical_residual_0319/leaderboard_holdout.csv)

| Setting | Target | Baseline Model | Residual Model | RMSE | MAE | MAPE | NRMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Univariate | WTI Oil | PatchTST | `NLinear` | **7.805** | **7.074** | **12.097** | **0.078** |
| Univariate | WTI Oil | PatchTST | `iTransformer` | 9.094 | 8.804 | 15.019 | 0.091 |
| Univariate | WTI Oil | PatchTST | `-` | 10.248 | 9.970 | 17.006 | 0.102 |
| Univariate | WTI Oil | PatchTST | `LightGBM` | 10.713 | 10.447 | 17.818 | 0.107 |
| Univariate | WTI Oil | PatchTST | `XGBoost` | 11.183 | 10.278 | 17.571 | 0.112 |
| Univariate | Brent Oil | PatchTST | `NLinear` | **2.401** | **2.127** | **3.374** | **0.025** |
| Univariate | Brent Oil | PatchTST | `-` | 2.838 | 2.291 | 3.637 | 0.029 |
| Univariate | Brent Oil | PatchTST | `iTransformer` | 3.103 | 2.688 | 4.268 | 0.032 |
| Univariate | Brent Oil | PatchTST | `LightGBM` | 4.808 | 4.465 | 7.080 | 0.049 |
| Univariate | Brent Oil | PatchTST | `XGBoost` | 5.020 | 4.830 | 7.671 | 0.052 |
| Multivariate+Exogenous | WTI Oil | PatchTST | `iTransformer` | **4.462** | **3.719** | **6.328** | **0.045** |
| Multivariate+Exogenous | WTI Oil | PatchTST | `NLinear` | 5.333 | 3.763 | 6.442 | 0.053 |
| Multivariate+Exogenous | WTI Oil | PatchTST | `LightGBM` | 6.265 | 5.876 | 9.970 | 0.063 |
| Multivariate+Exogenous | WTI Oil | PatchTST | `XGBoost` | 6.562 | 6.189 | 10.498 | 0.066 |
| Multivariate+Exogenous | WTI Oil | PatchTST | `-` | 9.442 | 8.983 | 15.214 | 0.094 |
| Multivariate+Exogenous | Brent Oil | PatchTST | `-` | **3.631** | **2.885** | **4.669** | **0.037** |
| Multivariate+Exogenous | Brent Oil | PatchTST | `NLinear` | 5.994 | 5.381 | 8.655 | 0.062 |
| Multivariate+Exogenous | Brent Oil | PatchTST | `XGBoost` | 7.402 | 6.768 | 10.868 | 0.076 |
| Multivariate+Exogenous | Brent Oil | PatchTST | `LightGBM` | 7.877 | 7.406 | 11.895 | 0.081 |
| Multivariate+Exogenous | Brent Oil | PatchTST | `iTransformer` | 9.637 | 9.329 | 14.957 | 0.099 |

- **Plot**
  - plot manifest: [plot_manifest.csv](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_canonical_residual_0319/plot_manifest.csv)
  - 예시

![Univariate WTI ts-cv](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_canonical_residual_0319/plots/univariate_tscv_wti_oil_actual_vs_pred.png)

![Exogenous Brent holdout](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/output_oil_canonical_residual_0319/plots/multivariate_plus_exogenous_holdout_brent_oil_actual_vs_pred.png)

# 05. 결론 및 얻게 된 인사이트

---

- canonical protocol 기준으로 `ts-cv 평균`에서는 `iTransformer`가 가장 강했다.
- 그러나 `holdout`에서는 setting과 타깃에 따라 승자가 달랐다.
  - `Univariate`에서는 `WTI`, `Brent` 모두 `NLinear`가 최종 holdout best
  - `Multivariate+Exogenous`에서는 `WTI`는 `iTransformer`, `Brent`는 baseline `PatchTST`가 best
- 즉, `평균 재현성(ts-cv)`과 `최종 최근 구간(holdout)`의 best residual model은 완전히 같지 않았다.
- 특히 `Brent exogenous holdout`에서는 residual correction이 baseline을 넘지 못했으므로, 외생변수를 넣는다고 항상 보정이 유리한 것은 아니었다.
- 이번 canonical 결과는 이전 exploratory `_0318`보다 보수적이지만, 구조적으로는 훨씬 방어 가능하다.

# 06. 향후 Action Plan

---

- 발표본 메인 결과는 이 canonical benchmark만 사용한다.
- `ts-cv` 기준 추천 모델과 `holdout` 기준 추천 모델을 분리해서 설명한다.
- `WTI`는 `Univariate-NLinear`와 `Exogenous-iTransformer`를 후속 후보로 보고, `Brent`는 `Univariate-NLinear`와 `Exogenous-baseline`을 함께 검토한다.
- 이전 `_0318` 결과와 `1점대 confirmatory` 결과는 부록/reference로만 둔다.
