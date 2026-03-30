# Brent 유가 잔차보정모형 중심 결과보고서

교수님 로그를 발표용으로 다시 정리할 때는, 여러 모델 전체를 넓게 나열하기보다 `잔차보정모형` 자체에 초점을 맞춰 `NLinear`, `LightGBM`, `XGBoost` 순으로 설명하는 편이 가장 명확하다.  
아래 보고서는 그 기준으로 다시 묶은 버전이며, **최신 교수님용 패키지 결과**와 **과거 exploratory 로그**를 구분해서 정리했다.

# 01. 핵심쟁점

---

- `기본모델` 대비 `잔차보정모형`을 붙였을 때, **실제 Test 구간 성능 향상이 있었는지**
- `NLinear`, `LightGBM`, `XGBoost` 중 **어떤 잔차보정모형이 가장 설득력 있는 개선을 보였는지**
- 단일 분할 결과가 아니라 **rolling-origin 같은 반복 평가에서도 재현되는지**

# 02. 데이터 및 모델 세팅

---

- **예측 타깃:**
  - Brent 원유 현물 가격 (`Com_BrentCrudeOil`)
- **예측 단위:**
  - 주간 예측
- **데이터 구간 세팅:**
  - 최신 교수님용 공통 단일 분할
    - TrainSet 기간: `2013-04-01 ~ 2025-07-28` (총 `644주`)
    - ValidationSet 기간: `2025-08-04 ~ 2025-10-20` (총 `12주`)
    - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)
  - rolling-origin 참고 구간
    - Cross-Validation Fold 1 Test 시작점: `2024-11-25`
    - Cross-Validation Fold 2 Test 시작점: `2025-02-17`
    - Cross-Validation Fold 3 Test 시작점: `2025-05-12`
    - Cross-Validation Fold 4 Test 시작점: `2025-08-04`
    - Cross-Validation Fold 5 Test 시작점: `2025-10-27`
  - XGBoost 참고 로그는 별도 legacy 세팅
    - 전체 기간: `2014-03-31 ~ 2026-01-12` (총 `616주`)
    - TestSet 기간: `2025-10-27 ~ 2026-01-12` (총 `12주`)

- **피처리스트**
  - 최신 교수님용 공통 입력피처 (총 55개)
    - 원시 변수 `22개` + 파생 변수 `33개`

    ```python
    # raw features (22)
    'Com_Gasoline', 'Com_NaturalGas', 'Com_Uranium', 'Com_Coal',
    'Com_LME_Cu_Cash', 'Com_Steel', 'Com_Iron_Ore',
    'Idx_DxyUSD', 'EX_USD_CNY',
    'Bonds_US_10Y', 'Bonds_US_2Y', 'Bonds_US_3M',
    'Idx_SnPVIX', 'Com_Gold',
    'Idx_SnP500', 'Idx_CSI300',
    'EX_USD_KRW', 'Bonds_KOR_10Y', 'EX_USD_JPY',
    'Com_Corn', 'Com_Soybeans', 'Com_PalmOil'
    ```

    ```python
    # derived features (33)
    '[each raw feature]_ret',
    'Spread_US_10Y_2Y', 'Spread_Crack', 'Ratio_Gold_Oil',
    'Com_Gasoline_ma4_ratio', 'Com_Gasoline_ma12_ratio',
    'Com_NaturalGas_ma4_ratio', 'Com_NaturalGas_ma12_ratio',
    'Idx_SnPVIX_ma4_ratio', 'Idx_SnPVIX_ma12_ratio',
    'Idx_DxyUSD_ma4_ratio', 'Idx_DxyUSD_ma12_ratio'
    ```

  - PatchTST 공정 비교용 피처
    - 위 55개 후보에서 `SHAP + TimeSeriesCV`로 고른 `10개 selected feature`
  - XGBoost legacy 로그
    - 노트북 출력상 총 `99개 feature`
    - 단, 독립적인 최종 feature list와 standalone 결과표는 저장되지 않음

- **모델 세팅**
  - 기본모델 1: `ExpSmoothing`

    ```python
    baseline_model = "ExponentialSmoothing"
    split = {
        "train_end": "2025-07-28",
        "val_start": "2025-08-04",
        "test_start": "2025-10-27",
    }
    ```

  - 실험모델 1: `ExpSmoothing + NLinear`

    ```python
    nlinear_params = {
        "seq_len": 24,
        "pred_len": 1,
        "d_hidden": 64,
        "dropout": 0.3,
        "lr": 1e-3,
        "epochs": 300,
        "patience": 40,
        "batch_size": 32,
        "seeds": 10,
        "top_k": 5,
    }
    ```

  - 실험모델 2: `ExpSmoothing + NLinear + LightGBM RoR`

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

  - 실험모델 3: `PatchTST + NLinear`

    ```python
    patchtst_params = {
        "seq_len": 24,
        "patch_len": 4,
        "stride": 2,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.2,
        "epochs": 120,
        "patience": 20,
        "batch_size": 32,
        "lr": 1e-3,
    }
    ```

  - 실험모델 4: `XGBoost residual ensemble`

    ```python
    xgboost_note = {
        "status": "legacy exploratory only",
        "available_result": "weighted LightGBM + XGBoost residual ensemble",
        "standalone_final_table": "not archived",
    }
    ```

# 03. 실험 설계 및 적용

---

- `NLinear`은 1차 잔차보정모형으로 두고, `ExpSmoothing` 또는 `PatchTST`가 남긴 residual을 직접 예측했다.
- `LightGBM`는 `NLinear` 이후에도 남는 residual-of-residual을 보정하는 2차 보정모형으로 적용했다.
- `XGBoost`는 최신 최종 패키지의 주력 실험이 아니라, 과거 notebook에서 `LightGBM + XGBoost weighted ensemble` 형태로만 확인된다.
- 해석 원칙은 아래와 같이 두었다.
  - 최신 CSV가 남아 있는 `NLinear`, `LightGBM` 결과를 우선 근거로 사용
  - `PatchTST + NLinear`는 NLinear 일반화 여부를 보는 반례로 사용
  - `XGBoost`는 최신 패키지와 동급 evidence가 아니므로 `legacy 참고 로그`로만 사용

# 04. 실험(모델링) 결과

### 04-01. 결과 요약 (MAPE 기준)

---

발표용 표는 아래처럼 `Baseline`과 `Residual Model`을 분리해 두는 편이 좋다.  
이렇게 두면 같은 baseline에서 `residual 미적용`과 `residual 적용` 결과를 한 표 안에서 바로 비교할 수 있다.

| Target | Baseline | Residual Model | Eval Split | RMSE | MAE | MAPE (%) | NRMSE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Brent Oil | ExpSmoothing | `-` | Single Test | 4.1754 | 4.0857 | 6.5276 | 6.6601 |
| Brent Oil | ExpSmoothing | `NLinear` | Single Test | 1.2648 | 1.1353 | 1.8192 | 2.0175 |
| Brent Oil | ExpSmoothing + NLinear | `-` | Single Test | 1.2648 | 1.1353 | 1.8192 | 2.0175 |
| Brent Oil | ExpSmoothing + NLinear | `LightGBM RoR` | Single Test | 1.2275 | 0.9144 | 1.4708 | 1.9580 |
| Brent Oil | PatchTST | `-` | Single Test | 1.5339 | 1.0027 | 1.5997 | 2.4467 |
| Brent Oil | PatchTST | `NLinear` | Single Test | 1.9251 | 1.3985 | 2.2132 | 3.0707 |
| Brent Oil | Persistence / Random Walk | `-` | Legacy Test | 1.1227 | 기록 미보존 | 기록 미보존 | 기록 미보존 |
| Brent Oil | Persistence / Random Walk | `LightGBM + XGBoost ensemble` | Legacy Test | 1.1579 | 기록 미보존 | 기록 미보존 | 기록 미보존 |

### 04-02. 세부 결과

---

- **Baseline 대비 residual 투입 전후 비교표**

  | 비교쌍 | Target | Baseline | Residual Model | RMSE Before | RMSE After | MAPE Before (%) | MAPE After (%) | ΔMAPE (%p) | 1차 판단 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `ExpSmoothing -> NLinear` | Brent Oil | ExpSmoothing | NLinear | 4.1754 | 1.2648 | 6.5276 | 1.8192 | -4.7084 | **강한 개선** |
  | `ExpSmoothing + NLinear -> LightGBM RoR` | Brent Oil | ExpSmoothing + NLinear | LightGBM RoR | 1.2648 | 1.2275 | 1.8192 | 1.4708 | -0.3484 | 단일 분할 소폭 개선 |
  | `PatchTST -> NLinear` | Brent Oil | PatchTST | NLinear | 1.5339 | 1.9251 | 1.5997 | 2.2132 | +0.6135 | **악화** |
  | `Persistence -> LGB+XGB ensemble` | Brent Oil | Persistence / Random Walk | LightGBM + XGBoost ensemble | 1.1227 | 1.1579 | 기록 미보존 | 기록 미보존 | 비교 불가 | Test 악화 |

- **rolling-origin 평균 비교표**

  | Target | Baseline | Residual Model | Eval Split | RMSE | MAE | MAPE (%) | NRMSE (%) |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Brent Oil | ExpSmoothing + NLinear | `-` | Rolling Mean | 2.4743 | 2.0201 | 2.9664 | 3.6189 |
  | Brent Oil | ExpSmoothing + NLinear | `LightGBM RoR` | Rolling Mean | 2.5089 | 2.0283 | 2.9828 | 3.6746 |

- **해석**
  - `ExpSmoothing -> NLinear`가 최신 교수님용 패키지에서 가장 강한 잔차보정 성공 사례다.
  - `ExpSmoothing -> NLinear`는 DM 검정 `p = 1.08e-06`으로 통계적 설득력도 가장 높다.
  - `LightGBM RoR`는 단일 Test split에서는 좋아졌지만, rolling-origin 평균과 DM 검정 `p = 0.8697` 기준으로는 재현성이 약하다.
  - `PatchTST -> NLinear`는 오히려 악화되어, `NLinear`가 모든 baseline에서 자동으로 좋아지는 것은 아님을 보여준다.
  - `XGBoost`는 standalone 최신 결과표가 없고 legacy ensemble 로그만 남아 있어, 발표 본문에서는 보조 참고로만 쓰는 것이 안전하다.

# 05. 결론 및 얻게 된 인사이트

---

- `NLinear`는 이번 Brent 실험에서 가장 설득력 있는 잔차보정모형이었다. 특히 `ExpSmoothing -> NLinear` 구간은 Test 성능과 DM 검정 모두에서 근거가 강했다.
- `LightGBM`는 `NLinear` 위에 한 번 더 보정하는 2차 residual model로는 의미가 있었지만, 단일 분할 개선이 rolling-origin 평균까지 이어지지는 않았다.
- `NLinear`의 성능은 baseline 의존적이었다. `PatchTST` 위에서는 오히려 성능이 악화되어, residual 구조가 다르면 같은 보정모형도 작동 방식이 달라진다는 점을 확인했다.
- `XGBoost`는 최신 최종 패키지의 독립 evidence가 아니라 legacy notebook 로그에 가까워, 이번 발표의 주력 메시지로 사용하기에는 근거가 부족하다.

# 06. 향후 Action Plan

---

- 발표 첫 메시지는 `잔차보정모형 중 가장 강한 근거를 보인 것은 NLinear`로 두는 것이 적절하다.
- `LightGBM`는 `NLinear 이후 추가 보정 가능성`을 보여주는 보조 결과로 배치하고, 재현성 한계도 같이 설명하는 편이 안전하다.
- `XGBoost`는 본론보다 부록 또는 참고 슬라이드로 내리고, 최신 공통 split에서 재실행한 결과가 확보되기 전까지는 강한 주장에 쓰지 않는 것이 좋다.
- 후속 실험은 `같은 split / 같은 feature universe / 같은 metric table`로 `LightGBM vs XGBoost`를 다시 맞춰 돌려야 발표 설득력이 올라간다.
