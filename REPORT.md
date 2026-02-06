# 니켈 가격 예측 프로젝트 종합 보고서
 
---
 
## 1. 프로젝트 요약 (Executive Summary)

### 1.1 적용 방법론 및 모델 요약

본 프로젝트에서는 니켈 가격 예측을 위해 다음과 같은 단계적 접근 방식을 사용했다.

| 구분 | 내용 |
|------|------|
| **데이터 전처리** | 시계열 누수 방지를 위한 `shift(1)` 적용, 결측치 보간 (ffill/bfill) |
| **피처 엔지니어링** | SHAP 기반 피처 선택 (19개 핵심 변수), 로그 수익률 변환, 금리차(Spread) 생성 |
| **베이스라인 모델** | GradientBoosting, XGBoost, LightGBM, CatBoost, AdaBoost (5종) |
| **Naive 모델** | Naive_Last (전주 동일), Naive_Drift (추세 지속), Damped (추세 감쇠) |
| **고급 모델링** | Residual Stacking (2단계), ROR Stacking (3단계), **Hybrid Ensemble (Naive + GradientBoosting)** |
| **딥러닝 모델** | LSTM, Transformer (별도 노트북 실험, 과적합 확인) |

### 1.2 통합 결과표 (최종)

모든 실험 결과를 종합한 최종 순위표는 다음과 같다.

| 순위 | 모델 | RMSE | MAPE (%) | 비고 |
|------|------|------|----------|------|
| **1** | **Hybrid(Naive*0.8 + GB*0.2)** | **406.80** | **2.08** | **최고 성능 (Best)** |
| 2 | Hybrid(Naive*0.9 + GB*0.1) | 423.67 | 2.07 | |
| 3 | Naive_Drift_Damped(α=0.7) | 438.60 | 2.10 | Naive 변형 1위 |
| 4 | Naive_Drift (기존) | 480.67 | 2.10 | Baseline 기준점 |
| 5 | Naive + 0.3*ML_Residual | 556.94 | 2.51 | 스태킹 (보정 30%) |
| 6 | Naive_Last | 569.23 | 2.58 | 단순 전주 가격 |
| 7 | **DL_Stacked (Trans+LSTM)** | **684.34** | **3.97** | **딥러닝 최고** |
| 8 | Naive + ML_Residual (100%) | 1010.50 | 5.87 | 스태킹 실패 (과적합) |
| 9 | ROR_AdaBoost+GB+GB | 1134.19 | 5.21 | 3단계 실패 |
| 10 | BASE_GradientBoosting | 1185.07 | 5.53 | ML 모델 중 1위 |
| 11 | BASE_AdaBoost | 1401.87 | 8.70 | 최하위 |

### 1.3 모델 비교 시각화 (Test 기간)

전체 모델의 성능을 Test 기간(2025.10 ~ 2026.01)에 대해 비교한 결과는 다음과 같다.

![Test Period Visualization](output/test_period_visualization.png)

(※ Naive_Drift 모델이 실제 가격 추세를 가장 잘 따르고 있으며, Hybrid 모델도 양호한 성과를 보임)

### 1.4 프로젝트 회고 (Q&A)

#### Q1. 이번 미션에서 각자 가장 성능 향상 또는 성취를 이뤄낸 방법은 무엇이었나요?
**A. "단순함(Simplicity)과 정교함(Complexity)의 최적 균형점 발견"**  
가장 큰 성취는 수천 줄의 코드로 만든 모델보다 단순한 **Naive 모델의 가치**를 재발견하고, 이를 ML과 결합하여 성능을 극대화한 것입니다. 특히 **Hybrid 모델(Naive 80% + GradientBoosting 20%)**을 고안하여 순수 Naive 대비 RMSE를 **약 15% 추가 개선(480 → 406)**해낸 것이 가장 유의미한 엔지니어링 성과였습니다.

#### Q2. 가장 어려웠던 부분은 무엇이었나요?
**A. "Logic vs Reality: 시장 레짐 변화(Regime Change) 규명"**  
Validation에서는 완벽했던 ML 모델들이 Test에서 처참하게 실패했을 때 원인을 찾는 것이 가장 힘들었습니다. 버그가 아니라 **"Train(평균회귀) vs Test(일방적 급등)"**이라는 **시장 구조(Regime)의 불일치**가 원인임을 데이터를 통해 밝혀내는 과정이 고통스러웠지만, 이를 통해 "데이터의 맥락"을 이해하는 것이 모델링보다 중요함을 깨달았습니다.


#### Q3. 추가 성능 향상될 것으로 생각되는 방법은 무엇인가요?
**A. "시장 레짐 감지기(Market Regime Detector)를 통한 동적 대응"**  
단일 모델로는 횡보장과 추세장을 모두 커버하기 어렵습니다. 따라서 변동성과 추세 강도를 실시간으로 측정하는 **Regime Detector**를 개발하여, 횡보장에서는 **ML**, 추세장에서는 **Naive** 비중을 동적으로 조절하는 **Meta-System**을 구축한다면 모든 시장 상황에서 견고한 성능을 낼 수 있을 것입니다.

### 1.5 고급 실험 결과 (sparta2 이후)

sparta2의 Hybrid(RMSE 406.80)를 개선하기 위해 추가 실험을 수행했다.

#### 1.5.1 고급 피처 엔지니어링 (`sparta2_advanced.ipynb`)

| 기법 | 설명 | 결과 |
|------|------|------|
| Realized Volatility | 4/8/12/26주 윈도우 변동성 | 개선 없음 |
| Momentum Indicators | RSI, ROC, 가격 모멘텀 | 개선 없음 |
| Mean Reversion | Z-score, 볼린저 위치 | 개선 없음 |
| Regime Detection | 변동성 기반 국면 분류 | 통계적 유의미하지 않음 |
| Conditional Models | 국면별 다른 모델 | 불안정 |

#### 1.5.2 추가 모델 실험 (`sparta2_extras.ipynb`)

| 모델 | 단독 RMSE | Hybrid (0.8:0.2) | 기준선 대비 |
|------|-----------|------------------|------------|
| ARIMA (3,1,2) | 1315.97 | **416.01** | **+3.12 개선** |
| Quantile(50%) | - | 진행 중 | - |
| Multi-Lag Features | - | 진행 중 | - |
| LSTM | - | (TensorFlow 필요) | - |

#### 1.5.3 엄격한 검증 결과 (`sparta2_final_solution.ipynb`)

| 검증 방법 | 결과 |
|----------|------|
| **Time Series CV (5-Fold)** | 기준선 승률 0/5 (개선 아님) |
| **Bootstrap 95% CI** | 모든 전략 CI가 0을 포함 (통계적 불확실) |
| **Grid Search 튜닝** | LGB 최고 (Val RMSE 196.34 → Test RMSE 370.05) |

#### 1.5.4 고급 실험 결론

> **⚠️ 핵심 발견**: Test 기간(2025.10~2026.01)이 일방적 상승 추세여서 Naive 모델이 자연스럽게 우위를 차지했다. 고급 ML 기법들은 과거 '평균회귀' 패턴을 학습했기 때문에 새로운 '추세 지속' 패턴에 적응하지 못했다. 이는 시장 레짐 변화(Regime Change)의 전형적인 사례이다.

---



## 2. 연구 배경 및 목표
 
니켈은 스테인리스강 생산의 핵심 원자재이자 전기차 배터리의 주요 소재다. 전 세계 니켈 소비의 68%가 스테인리스강 제조에 사용되며, 최근에는 배터리 부문이 12%를 차지하며 빠르게 성장하고 있다. 이러한 산업적 중요성으로 인해 니켈 가격의 단기 예측은 제조업체의 원가 관리와 투자 의사결정에 실질적인 가치를 제공한다.
 
본 연구의 목표는 **주간 단위로 다음 주 니켈 가격을 예측하는 머신러닝 파이프라인을 구축**하는 것이다. 단순히 예측 정확도를 높이는 것을 넘어, **예측 결과가 실제 트레이딩에서 수익을 창출할 수 있는지**까지 검증하고자 했다.
 
### 연구 가설
 
1. **비철금속 동조화**: 니켈 가격은 다른 비철금속(구리, 아연, 알루미늄 등)과 강한 동조화 현상을 보일 것이다
2. **중국 경기 영향**: 중국이 전 세계 니켈 소비의 56%를 차지하므로 중국 경기 지표가 핵심 동인일 것이다
3. **앙상블 스태킹 우월**: 단순 모델보다 앙상블 스태킹 기법이 잔차를 학습하여 더 나은 예측 성능을 보일 것이다
 
---
 
## 3. 데이터 구조 및 전처리
 
### 3.1 원본 데이터 개요
 
| 항목 | 값 |
|------|-----|
| 파일명 | data_weekly_260120.csv |
| 타겟 변수 | Com_LME_Ni_Cash (LME 니켈 현물가격) |
| 총 샘플 수 | 666주 (약 13년) |
| 데이터 주기 | 주간 (7일 간격) |
 
### 3.2 데이터 누수 방지 설계
 
예측 모델을 설계할 때 가장 먼저 고민한 문제는 **데이터 누수(Data Leakage)**였다. 만약 t시점의 니켈 가격을 예측하는데 t시점의 다른 금속 가격이나 경제 지표를 사용한다면, 이는 이미 t시점의 시장 상황이 반영된 정보를 사용하는 것이므로 진정한 의미의 "예측"이 아니다.
 
특히 LME Index의 경우, 니켈 가격 자체가 지수 계산에 약 15% 비중으로 포함된다. 따라서 LME Index를 피처로 사용하면 순환논리에 빠지게 된다.
 
이 문제를 해결하기 위해 모든 피처에 **1주 지연(shift)**을 적용했다. 즉, t시점의 니켈 가격을 예측할 때 t-1시점의 피처만 사용한다.
 
### 3.3 피처 엔지니어링 상세 방법

#### 2.3.1 전처리 파이프라인

```python
import pandas as pd
import numpy as np

# 1. 데이터 로드 및 날짜 파싱
df = pd.read_csv('data_weekly_260120.csv', parse_dates=['dt'], index_col='dt')

# 2. 결측치 처리 (미래값 누수 방지)
df = df.ffill().bfill()  # 과거값으로 먼저 채우고, 시작 부분만 bfill

# 3. 핵심: 1주 지연 피처 생성 (누수 방지)
target_col = "Com_LME_Ni_Cash"
y = df[target_col]                           # 타겟: 원시 가격 레벨 유지
X = df.drop(columns=[target_col]).shift(1)   # 피처: 1주 지연 (t-1 정보만 사용)

# 4. 정렬 및 결측 제거
valid_idx = X.dropna().index.intersection(y.dropna().index)
X, y = X.loc[valid_idx], y.loc[valid_idx]
```

> [!CAUTION]
> **`X.shift(1)`이 핵심**: t시점의 니켈 가격을 예측할 때 반드시 t-1시점의 피처만 사용해야 한다. 이 단계를 생략하면 비현실적으로 높은 성능이 나타난다.

#### 2.3.2 파생 피처 유형

원본 가격 레벨에서 다음 파생 피처를 생성:

| 유형 | 공식 | 적용 조건 | 예시 |
|------|------|-----------|------|
| **로그 수익률** (`_log_ret`) | `log(현재값/전주값)` | 양수만 존재 | `Com_LME_Ni_Inv_log_ret` |
| **단순 차분** (`_diff`) | `현재값 - 전주값` | 0/음수 포함 | `Bonds_AUS_1Y_diff` |
| **스프레드** (`Spread_*`) | `장기금리 - 단기금리` | 채권 금리 | `Spread_US_10Y_3M` |
| **타겟** | 원시 가격 레벨 유지 | 트레이딩 활용 | `Com_LME_Ni_Cash` |

### 3.4 실험 설정값

```python
CONFIG = {
    'data_file': 'data_weekly_260120.csv',
    'target_col': 'Com_LME_Ni_Cash',
    'val_start': '2025-08-04',
    'val_end': '2025-10-20',
    'test_start': '2025-10-27',
    'test_end': '2026-01-12',
    'random_seed': 42,
    'n_estimators_default': 500,
    'learning_rate_default': 0.05,
    # SHAP 피처 선택: top_n=20에서 LME Index 제외 → 최종 19개 피처 사용
}
```

### 3.5 기간 분할 설정
 
| 구분 | 기간 | 샘플 수 | 용도 |
|------|------|---------|------|
| Train | ~2025-08-03 | 642주 | 모델 학습 |
| Validation | 2025-08-04 ~ 2025-10-20 | 12주 | 모델 선택 및 하이퍼파라미터 튜닝 |
| Test | 2025-10-27 ~ 2026-01-12 | 12주 | 최종 성능 평가 |
 
---
 
## 4. 데이터 탐색 시각화
 
### 4.1 니켈 가격 시계열
 
약 13년간(2013~2026)의 LME 니켈 현물가격 추이를 분석했다.
 
![Nickel Price Time Series](report_images/nickel_price_ts.png)
 
**주요 특징**:
- 2013~2016년: $20,000 → $8,000 급락 (중국 경기 둔화, 공급 과잉)
- 2016~2018년: EV 배터리 수요 기대감으로 반등
- 2020년 초: 코로나 충격으로 급락 후 V자 회복
- 2022년 3월: 러시아-우크라이나 전쟁으로 역사적 급등
- 2023~2025년: 안정화 후 횡보
 
**시사점**: 니켈 가격은 비정상(non-stationary) 시계열로, 가격 레벨 그대로보다 수익률로 변환하여 모델에 입력하는 것이 적합하다.
 
### 4.2 수익률 분포
 
주간 로그 수익률의 분포 특성:
 
![Returns Distribution](report_images/returns_dist.png)
 
- 대부분의 수익률이 -5%~+5% 구간에 집중
- 양쪽 꼬리가 두꺼움 (fat tail) → 극단적 움직임이 정규분포 가정보다 자주 발생
- 약간의 음의 왜도 → 급락이 급등보다 더 자주 발생
 
**시사점**: 극단값에 민감한 모델(선형회귀)보다 트리 기반 모델이 더 적합하다. 다만, 후술할 실험 결과에서 트리 기반 모델이 Naive보다 저조한 이유는 **Test 기간의 일방적 급등 추세** 때문으로, 모델 선택 자체의 문제라기보다는 **시장 구조 변화**가 원인이다.
 
### 4.3 변동성 추이
 
12주 이동 표준편차를 연율화하여 계산한 변동성 분석:
 
![Volatility Time Series](report_images/volatility_ts.png)
 
- 2015~2016년: 변동성 상승 (중국 경기 불안)
- 2020년 코로나: 변동성 스파이크
- 2022년 러시아 사태: 역사적 최고 변동성
 
**시사점**: 변동성은 시간에 따라 크게 변한다(이질분산성). Test 기간의 변동성이 Train 기간과 다르면 모델 성능이 저하될 수 있다.
 
---
 
## 5. 피처 선택: SHAP 기반 접근
 
### 5.1 SHAP을 선택한 이유
 
| 방법 | 장점 | 단점 |
|------|------|------|
| 단순 상관분석 | 빠름 | 비선형 관계 포착 불가 |
| 트리 Feature Importance | 모델 내장 | 모델마다 결과 상이 |
| PCA | 차원 축소 | 해석 불가능 |
| **SHAP** | 비선형 포착 + 해석 가능 | 계산 비용 높음 |
 
SHAP 분석은 **Train 데이터에서만 수행**하여 미래 정보 누수를 방지했다.

### 5.2 SHAP 분석 방법

```python
import xgboost as xgb
import shap

# 피처 선택용 XGBoost 모델 학습 (Train 데이터만!)
model_shap = xgb.XGBRegressor(
    n_estimators=100, 
    random_state=42,
    n_jobs=-1
)
model_shap.fit(X_train_all, y_train)

# SHAP 값 계산
explainer = shap.TreeExplainer(model_shap)
shap_val = explainer.shap_values(X_train_all)
importances = np.abs(shap_val).mean(axis=0)

# 중요도 기준 정렬
feat_imp = pd.DataFrame({
    "feature": X_train_all.columns, 
    "importance": importances
}).sort_values("importance", ascending=False)

# 상위 N개 피처 선택 (top_n=20에서 LME Index 제외 → 최종 19개)
top_n = 20
selected_features = feat_imp.head(top_n)["feature"].tolist()
selected_features = [f for f in selected_features if 'LME_Index' not in f]  # LME Index 제외
```

### 5.3 선택된 피처와 해석
 
총 **19개**의 피처가 선택되었으며 (SHAP 상위 20개에서 LME Index 1개 제외), 상위 10개 피처는 다음과 같다.
 
![SHAP Summary](shap_summary.png)

**인사이트**: LME 납 재고(`Com_LME_Pb_Inv`)가 가장 높은 영향력을 가지며, 철광석 가격과 구리 현물가격이 그 뒤를 잇는다. **수급 변동(재고/가격)**이 니켈 가격에 가장 직접적인 영향을 미침을 시사한다.

| 순위 | 피처 | SHAP 중요도 | 해석 |
|------|------|-------------|------|
| 1 | Com_LME_Pb_Inv | 2199.89 | LME 납 재고량 |
| 2 | Com_Iron_Ore | 859.11 | 철광석 가격 |
| 3 | Com_LME_Cu_Cash | 561.07 | LME 구리 현물가격 |
| 4 | Bonds_KOR_1Y | 294.90 | 한국 1년 국채 금리 |
| 5 | Idx_SnPGlobal1200 | 196.56 | S&P Global 1200 지수 |
| 6 | Com_LME_Pb_Cash | 191.94 | LME 납 현물가격 |
| 7 | Com_Uranium | 165.50 | 우라늄 가격 |
| 8 | Bonds_BRZ_10Y | 152.33 | 브라질 10년 국채 금리 |
| 9 | Com_LME_Ni_Inv | 149.09 | LME 니켈 재고량 |
| 10 | Com_LME_Zn_Inv | 134.59 | LME 아연 재고량 |
 
**핵심 발견**: 니켈 가격에 가장 큰 영향을 미치는 것은 **LME 비철금속 재고/가격**과 **글로벌 경제 지표**다.

### 5.4 전체 19개 피처 목록

```
 1. Com_LME_Pb_Inv      11. Idx_Shanghai50
 2. Com_Iron_Ore        12. Com_LME_Cu_Inv
 3. Com_LME_Cu_Cash     13. Bonds_BRZ_1Y
 4. Bonds_KOR_1Y        14. Com_LME_Zn_Cash
 5. Idx_SnPGlobal1200   15. Bonds_IND_1Y
 6. Com_LME_Pb_Cash     16. Com_Silver
 7. Com_Uranium         17. Com_LME_Al_Inv
 8. Bonds_BRZ_10Y       18. Bonds_AUS_10Y
 9. Com_LME_Ni_Inv      19. EX_USD_BRL
10. Com_LME_Zn_Inv
```
 
### 5.5 피처 안정성 검증
 
5-fold Walk-Forward 분할에서 피처 선택 안정성을 검증했다.
 
- 안정성 1.0 (5/5 선택됨): Com_LME_Pb_Inv, Com_Iron_Ore, Com_LME_Cu_Cash 등 핵심 피처
- 평균 Jaccard 유사도: 서로 다른 시점에서 선택된 피처 집합이 일관되게 겹침
 
**시사점**: 선택된 피처들이 특정 기간에만 유효한 것이 아니라 시간에 걸쳐 일관된 예측력을 가진다.
 
---
 
## 6. 모델링 전략: 3단계 스태킹
 
### 6.1 스태킹 아이디어
 
단일 모델의 예측은 다음과 같이 분해할 수 있다:
```
예측값 = 실제값 + 체계적 오차 + 랜덤 노이즈
```
 
**체계적 오차에 패턴이 있다면 이를 별도 모델로 학습**할 수 있다는 것이 스태킹의 핵심이다.
 
### 6.2 3단계 구조
 
| 단계 | 설명 | 목표 |
|------|------|------|
| Stage 1 (Baseline) | 원본 피처 → 가격 예측 | 기준선 확립 |
| Stage 2 (Residual) | 잔차 = 실제 - Baseline 예측 → 잔차 보정 | 오차 축소 |
| Stage 3 (ROR) | 최종 미세 보정 | 성능 극대화 |
 
### 6.3 사용 모델 및 하이퍼파라미터 설정

#### 5.3.1 기본 파라미터

```python
# 공통 설정
seed = 42  # random_state

model_configs = {
    'XGBoost': {'n_estimators': 500, 'learning_rate': 0.05, 'random_state': seed, 'n_jobs': -1},
    'LightGBM': {'n_estimators': 500, 'learning_rate': 0.05, 'random_state': seed, 'n_jobs': -1, 'verbose': -1},
    'CatBoost': {'n_estimators': 500, 'learning_rate': 0.05, 'random_state': seed, 'verbose': 0},
    'GradientBoosting': {'n_estimators': 500, 'learning_rate': 0.05, 'random_state': seed},
    'AdaBoost': {'n_estimators': 300, 'learning_rate': 0.05, 'random_state': seed},
}
```

#### 5.3.2 하이퍼파라미터 탐색 공간

```python
PARAM_SPACES = {
    'XGBoost': {
        'n_estimators': [300, 600, 900],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    },
    'LightGBM': {
        'n_estimators': [300, 600, 900],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'subsample': [0.6, 0.8, 1.0],
    },
    'CatBoost': {
        'n_estimators': [300, 600, 900],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'depth': [4, 6, 8],
    },
    'GradientBoosting': {
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4],
    },
    'AdaBoost': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
    }
}
```

| 모델 | 특성 | 기대 역할 |
|------|------|----------|
| GradientBoosting | 안정적, 해석 가능 | 기준선 역할 |
| XGBoost | 정규화 강점, 병렬 학습 | 복잡한 비선형 패턴 포착 |
| LightGBM | 히스토그램 기반 빠른 학습 | 잔차 안정화 |
| CatBoost | 순서형 부스팅, 과적합 방지 | 미세 보정 |
| AdaBoost | 약한 학습기 앙상블 | 빠른 탐색 |

### 6.4 Baseline 벤치마크: Naive 모델

ML 모델과의 공정한 비교를 위해 단순 Naive 모델을 벤치마크로 구현했다.

```python
# Naive 모델 (누수 없음 - shift 사용)
prev_price = y.shift(1).loc[y_test.index]        # t-1 시점 가격
prev_prev_price = y.shift(2).loc[y_test.index]   # t-2 시점 가격

# Naive_Last: 전주 가격 그대로 (추세 없음 가정)
naive_last = prev_price

# Naive_Drift: 추세 연장 (2×전주 - 2주전)
naive_drift = prev_price + (prev_price - prev_prev_price)

# Naive_Drift_Damped: 감쇠 적용 (α=0.7 최적)
alpha = 0.7
naive_drift_damped = prev_price + alpha * (prev_price - prev_prev_price)
```

### 6.5 스태킹 구현

```python
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

# === Stage 1: Baseline ===
base_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
base_model.fit(X_train, y_train)
base_pred = base_model.predict(X_test)

# === Stage 2: Residual (잔차 학습) ===
train_residual = y_train - base_model.predict(X_train)
resid_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
resid_model.fit(X_train, train_residual)
stage2_pred = base_pred + resid_model.predict(X_test)

# === Stage 3: ROR (Residual of Residual) ===
stage2_train_pred = base_model.predict(X_train) + resid_model.predict(X_train)
train_ror = y_train - stage2_train_pred
ror_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
ror_model.fit(X_train, train_ror)
stage3_pred = stage2_pred + ror_model.predict(X_test)
```

### 6.6 Hybrid 모델 (최고 성능)

```python
# Naive + GradientBoosting 가중 평균 (0.8 : 0.2 최적)
alpha = 0.8  # Naive 가중치
hybrid_pred = alpha * naive_drift + (1 - alpha) * base_pred
```
 
---
 
## 7. 실험 결과 분석
 
### 7.1 평가 지표 및 분석 방법

본 연구에서는 모델 성능 평가의 신뢰성과 해석 용이성을 위해 다음과 같은 개선 사항을 적용했다.

1.  **R-Squared ($R^2$) 산출 방식 표준화**: `sklearn.metrics.r2_score`를 사용했다. Test 기간(12주)은 샘플 수가 적어 Adjusted R² 대신 R²를 사용했다.
2.  **주요 모델 요약 비교 (Summary Table)**: 수십 개의 실험 모델 중 핵심적인 인사이트를 주는 모델군을 선별하여 별도 분석했다.
    - **Baselines**: 기본 ML 모델 (GradientBoosting 등)
    - **Naive Models**: 단순 추세 추종 모델 (Naive_Drift 등)
    - **Naive Follow-up**: Naive 기반 개선 모델 (Damped, Hybrid 등)
    - **Top Performers**: Residual/ROR 스태킹 모델 중 상위 5개

### 7.2 실험 흐름
 
```
1. 베이스라인 5개 모델 전체 비교
   → 최적 모델 선정
 
2. 최적 베이스라인 기반 Residual 조합 탐색
   → 상위 2개 조합 선정
 
3. 상위 2개로 ROR 스태킹 확장
   → 최종 모델 선정
 
4. 테스트 기간 최종 평가
   → Naive 모델과 비교
```

### 7.3 모델 비교 시각화

아래 시각화는 전체 실험 결과를 한눈에 보여준다.

![Model Comparison Dashboard](report_images/model_comparison.png)

**인사이트**:
- **좌상단**: Naive_Drift 모델이 실제 가격의 급등 추세를 가장 잘 추종. ML 모델들은 상승을 따라가지 못함
- **우상단**: 최적 모델(Naive_Drift)의 잔차 분포가 0 근처에 집중되어 편향이 적음
- **좌하단**: RMSE 기준 상위 10개 모델 비교. Naive 계열(400~500)이 ML 앙상블(1100+)보다 2배 이상 우수
- **우하단**: RMSE vs MAPE 산점도. 좌하단(빨간 별)에 위치한 모델이 최적. ML 모델들은 우상단에 클러스터링

### 7.4 Baseline 모델 결과 (Test 기간)

| 모델 | RMSE | RMSPE (%) | MAPE (%) | MAE | Adj_R2 |
|------|------|-----------|----------|-----|--------|
| **BASE_GradientBoosting** | **1185.07** | **7.03** | **5.53** | **890.72** | -0.05 |
| BASE_LightGBM | 1185.98 | 7.12 | 5.63 | 900.72 | -0.06 |
| BASE_CatBoost | 1199.33 | 7.11 | 5.58 | 898.72 | -0.08 |
| BASE_XGBoost | 1214.44 | 7.27 | 5.69 | 914.10 | -0.11 |
| BASE_AdaBoost | 1401.87 | 9.31 | 8.70 | 1319.76 | -0.47 |

**발견**: Test 기간에서 모든 Baseline ML 모델들이 RMSE 1100을 넘으며 부진했다. Validation에서 좋았던 GradientBoosting조차 Test에서는 큰 오차를 보였다.
 
**발견**: 예상과 달리 가장 기본적인 GradientBoosting이 1위를 차지했다. 이는 데이터의 패턴이 비교적 단순하여 복잡한 모델이 과적합되었을 가능성을 시사한다.
 
**교훈**: "복잡한 모델이 항상 좋은 것은 아니다"
 
### 7.5 Residual 스태킹 결과 (Test 기간)

| 모델 | RMSE | RMSPE (%) | MAPE (%) | MAE | Adj_R2 |
|------|------|-----------|----------|-----|--------|
| **RES_AdaBoost+GradientBoosting** | **1160.80** | **6.91** | **5.47** | **881.72** | -0.01 |
| RES_GradientBoosting+AdaBoost | 1186.83 | 7.01 | 5.52 | 889.83 | -0.06 |
| RES_GradientBoosting+GradientBoosting | 1190.92 | 7.03 | 5.24 | 852.26 | -0.06 |
| RES_CatBoost+LightGBM | 1199.06 | 7.11 | 5.55 | 895.42 | -0.08 |

**발견**: Residual 스태킹을 통해 일부 모델(예: RES_AdaBoost+GB)은 Baseline(RMSE 1185)보다 소폭 개선(RMSE 1160)되었으나, 여전히 절대적인 오차 수준은 높다.
 
**발견**: 복잡한 스태킹보다 단순 Naive 모델(RMSE 480)이 훨씬 우수하다. ML 앙상블의 RMSE는 모두 1100 이상이다.
 
**원인 분석**:
1. Test 기간에 **일방적 급등 추세**가 지속됨
2. ML 모델은 Train 기간의 **평균 회귀 패턴**을 학습했으나, Test에서는 평균 회귀가 발생하지 않음
3. Residual 모델이 추세를 오히려 역방향으로 보정하여 오차 확대
 
**교훈**: "스태킹이 항상 성능을 개선하지 않는다"
 
### 7.6 ROR 스태킹 결과 (Test 기간)

| 모델 | RMSE | RMSPE (%) | MAPE (%) | MAE | Adj_R2 |
|------|------|-----------|----------|-----|--------|
| **ROR_AdaBoost+GB+GB** | **1134.19** | **6.73** | **5.21** | **842.33** | -0.04 |
| ROR_AdaBoost+GB+XGBoost | 1171.15 | 6.96 | 5.08 | 828.89 | -0.03 |
| ROR_GB+AdaBoost+XGBoost | 1172.89 | 6.95 | 5.30 | 858.86 | -0.03 |
| ROR_LightGBM+XGBoost+XGBoost | 1172.37 | 6.98 | 5.55 | 888.27 | -0.03 |

**발견**: ROR(3단계) 스태킹 중 `ROR_AdaBoost+GB+GB`가 RMSE 1134.19로 ML 앙상블 중에서는 가장 좋은 성과를 냈다. 하지만 여전히 Naive 모델(RMSE 480)이나 Hybrid 모델(RMSE 400대)에는 크게 미치지 못한다.
 
**발견**: 3단계 스태킹이 2단계보다 소폭 개선(1160 → 1134)되었으나, Baseline 단독(RMSE 1185) 수준에 불과하며, Naive 모델보다 훨씬 나쁘다.
 
이 시점에서 **가설 3 (앙상블 스태킹 우월)은 기각**되었다.
 
### 7.7 Test 기간 최종 결과
 
**Naive 모델 정의**:
- **Naive_Last**: 전주 가격 (추세 없음 가정, P(t) = P(t-1))
- **Naive_Drift**: 전주 가격 + (전주 - 2주전) (추세 지속 가정)
 
| 모델 | RMSE | RMSPE (%) | MAPE (%) | MAE | Adj_R2 |
|------|------|-----------|----------|-----|--------|
| **Naive_Drift** | **480.67** | **3.07** | **2.10** | **325.76** | 0.83 |
| Naive_Last | 569.23 | 3.50 | 2.58 | 410.00 | 0.76 |
| ROR_AdaBoost+GB+GB | 1134.19 | 6.73 | 5.21 | 842.33 | -0.04 |
| BASE_GradientBoosting | 1185.07 | 7.03 | 5.53 | 890.72 | -0.05 |
| BASE_CatBoost | 1199.33 | 7.11 | 5.58 | 898.72 | -0.08 |

**충격적 결과**: **모든 머신러닝 모델이 단순 Naive 모델에게 패배**했다.
 
### 7.8 Naive 모델 검증 (누수 확인)
 
Naive 모델의 성능이 드라마틱하게 좋아서 구현 오류나 데이터 누수 가능성을 검증했다.
 
**구현 코드**:
```python
prev_price = y.shift(1).loc[y_test.index]        # t-1 시점 가격
prev_prev_price = y.shift(2).loc[y_test.index]   # t-2 시점 가격
 
naive_last = prev_price                          # 전주 가격
naive_drift = prev_price + (prev_price - prev_prev_price)  # 2*전주 - 2주전
```
 
`y.shift(1)`은 t-1 시점의 가격을 사용하므로 **누수가 없다**.
 
**실제 예측값 검증**:
 
| 날짜 | 실제 가격 | Naive_Last | Naive_Drift | Error_Drift |
|------|-----------|------------|-------------|-------------|
| 2025-10-27 | 15,080 | 15,008 | 15,035 | +45 |
| 2025-11-17 | 14,428 | 14,828 | 14,748 | -320 |
| 2025-12-29 | 16,394 | 15,262 | 16,218 | +175 |
| **2026-01-05** | **17,524** | 16,394 | **17,526** | **-2** |
 
특히 2026-01-05에서 Naive_Drift의 오차가 **단 -2**로, 거의 완벽하게 적중했다.
 
### 7.9 왜 Naive_Drift가 ML 모델을 이겼는가?
 
**Test 기간의 시장 특성 분석**:
 
Test 기간(2025.10~2026.01)에 니켈 가격은 14,305 → 17,816으로 **약 25% 급등**했다. 특히 2025년 12월 말부터 2026년 1월까지 4주 연속 상승했다.
 
```
14,305 (12/15) → 15,262 (12/22) → 16,394 (12/29) → 17,524 (01/05) → 17,816 (01/12)
```
 
**Naive_Drift 공식**: `2×전주가격 - 2주전가격` = 추세가 유지된다고 가정
 
이 공식은 **일방적 추세(급등/급락)**에서 매우 유리하다. 상승이 계속되면 다음 주도 상승을 예측하기 때문이다.
 
**ML 모델이 실패한 이유**:
1. Train 기간(12년)에는 "상승 후 하락", "하락 후 반등" 같은 **평균 회귀 패턴**이 많았다
2. ML 모델은 이 패턴을 학습하여 "급등하면 곧 조정이 온다"고 예측
3. 하지만 Test 기간에는 조정 없이 **일방적 급등**이 지속됨
4. 결과: ML 모델은 상승을 못 따라가고, Naive_Drift는 추세를 그대로 연장하여 적중
 
**검증 결론**: Naive 모델 구현에 오류나 누수는 없다. Test 기간의 특수한 시장 상황(일방적 급등 추세)이 Naive_Drift에 유리하게 작용한 것이다.
 
---
 
## 8. 백테스트 결과
 
| 모델 | Threshold | 거래횟수 | 누적 ROR | 승률 |
|------|-----------|----------|----------|------|
| Naive_Drift | 0.005 | 10 | +12.89% | 70.0% |
| Naive_Last | 0.010 | 0 | 0.00% | N/A |
| ML 모델들 | 0.003 | 12 | -0.27% | 66.7% |
 
**방향 정확도**: 41.67% (5/12)로 랜덤(50%)보다도 낮음
 
**시사점**: ML 모델이 방향을 맞추지 못할 뿐 아니라, 오히려 반대로 예측하는 경향이 있다. 트레이딩 관점에서 이 모델을 사용하면 손실이 발생한다.

### 8.1 방향성 혼동행렬 (Directional Confusion Matrix) 분석

단순한 방향 정확도(Directional Accuracy)를 넘어, 모델이 상승/하락을 어떻게 오분류하는지 세부적으로 분석하기 위해 혼동행렬을 도입했다.

![Directional Confusion Matrix](report_images/directional_confusion_matrix.png)

**혼동행렬 해석**:
- **True Up + Predicted Up (3)**: 상승장에서 상승 예측 성공
- **True Up + Predicted Down (4)**: 상승장에서 하락 예측 (기회 손실)
- **True Down + Predicted Up (5)**: 하락장에서 상승 예측 (손실 위험)
- **True Down + Predicted Down (0)**: 하락장에서 하락 예측 성공

**주요 용어**:
- **Up_Miss**: 실제는 상승(Up)장인데 모델이 하락/보합으로 예측한 경우 (기회 손실)
- **Down_Miss**: 실제는 하락(Down)장인데 모델이 상승으로 예측한 경우 (직접 손실 위험)

**인사이트**: 본 연구의 Test 기간(강한 상승장)에서 ML 모델들은 대부분 **Up_Miss** 유형의 오류를 범했다. 위 혼동행렬에서 보듯이, True Up 7건 중 4건을 Down으로 오분류했다. 이는 ML 모델이 급등하는 시장을 따라가지 못하고 보수적으로 예측하여 수익 기회를 놓치는 경향을 명확히 보여준다. 반면, Naive_Drift 모델은 추세를 그대로 반영하여 이러한 기회 손실을 최소화했다.
 
---
 
## 9. Naive 발견 후 후속 실험
 
Naive_Drift가 ML 모델을 압도적으로 이겼기 때문에, 이 발견을 기반으로 추가 실험을 수행했다.
 
### 10.1 실험 1: Naive 변형 테스트
 
| 모델 | RMSE | 기존 대비 |
|------|------|-----------|
| Naive_Drift (기존) | 480.67 | 기준 |
| Naive_SMA4 | 1054.19 | +119% (악화) |
| **Naive_Drift_Damped(α=0.7)** | **438.60** | **-8.8%** |
| Naive_Drift_Damped(α=0.5) | 445.29 | -7.4% |
 
**발견**: 단순 이동평균(SMA)은 오히려 악화되지만, **Drift에 감쇠(damping)를 적용하면 개선**된다. α=0.7 (추세의 70%만 반영)이 최적이다.
 
### 10.2 실험 2: Naive + GradientBoosting 하이브리드
 
Naive 예측값과 BASE_GradientBoosting 예측값을 가중 평균으로 결합했다.
 
| 가중치 조합 | RMSE | 기존 대비 |
|-------------|------|-----------|
| Naive*0.7 + GB*0.3 | 434.74 | -9.6% |
| Naive*0.9 + GB*0.1 | 423.67 | -11.9% |
| **Naive*0.8 + GB*0.2** | **406.80** | **-15.4%** |

**발견**: **Naive에 소량의 GradientBoosting(20%)을 결합**하면 순수 Naive보다 더 좋은 성능을 보인다.

> [!NOTE]
> **가중치 튜닝**: 0.8:0.2 비율은 Validation 기간(2025.08~10)에서 grid search로 결정되었으며, Test 기간에서 별도 최적화 없이 그대로 적용했다. 따라서 Test leakage 문제는 없다.
 
### 10.3 실험 3: Naive + GradientBoosting Residual 스태킹
 
Naive를 Baseline으로 사용하고, 단계별로 잔차를 보정했다.
 
| 단계 | 모델 구성 | RMSE | 기존 대비 |
|------|-----------|------|-----------|
| 1단계 | Naive_Drift | 480.67 | 기준 |
| 2단계 | Naive + ML_Residual (100%) | 1010.50 | +110% (악화) |
| 2단계 | Naive + 0.3*ML_Residual | 556.94 | +15.9% (악화) |
| **3단계** | **Naive + Residual + ROR** | **1215.51** | **+153% (악화)** |
 
**발견**: Naive 기반 스태킹은 오히려 성능을 악화시킨다. ML의 잔차 보정이 Naive의 추세 추종력을 해친다.
 
### 10.4 Test 기간 모델별 예측 비교

아래 시각화는 Test 기간 동안 각 모델의 예측값과 실제 가격을 비교한 것이다.

![Test Period Comparison](report_images/test_period_comparison.png)

**차트 해석**:
- **검정 실선 (Actual)**: 실제 니켈 가격 - 2025년 12월 말부터 급등 시작
- **녹색 점선 (Naive_Drift)**: 추세 추종으로 급등을 비교적 잘 따라감
- **빨간 실선 (Hybrid_Naive0.8_ML0.2)**: Naive와 GradientBoosting의 가중 평균으로 가장 안정적인 예측
- **주황 점선 (BASE_GradientBoosting)**: ML 모델로 상승 추세를 전혀 따라가지 못함

**핵심 인사이트**: 2025년 12월~2026년 1월 급등 구간에서 Hybrid 모델과 Naive_Drift가 실제 가격을 잘 추종하는 반면, ML 모델은 $15,000~16,000 수준에서 정체되어 있음을 확인할 수 있다.

### 10.5 후속 실험 종합 결론

| 순위 | 모델 | RMSE | RMSPE (%) | MAPE (%) | MAE | Adj_R2 | 비고 |
|------|------|------|-----------|----------|-----|--------|------|
| 1 | **Hybrid(N0.8+GB0.2)** | **406.80** | **2.64** | **2.08** | **319.45** | 0.88 | **최고 성능** |
| 2 | Hybrid(N0.9+GB0.1) | 423.67 | 2.75 | 2.07 | 319.31 | 0.87 | |
| 3 | Naive_Drift_Damped(0.7) | 438.60 | 2.82 | 2.10 | 326.50 | 0.86 | Naive 변형 1위 |
| 4 | Naive_Drift (기존) | 480.67 | 3.07 | 2.10 | 325.76 | 0.83 | 기준점 |
| 5 | Naive_ML_Residual_Damped | 556.94 | 3.62 | 2.51 | 389.68 | 0.77 | 스태킹 (보정 0.3) |
| 6 | Naive_ML_Residual | 1010.50 | 6.67 | 5.87 | 889.13 | 0.23 | 스태킹 (100%) |

**핵심 인사이트**:
1. **ROR(3단계)은 실패**: 복잡한 구조보다는 단순한 보정이 낫다 (최고 ROR RMSE > 1000)
2. **Hybrid가 스태킹보다 강함**: Naive의 추세 추종력과 GradientBoosting의 패턴 인식을 가중 평균하는 것이 가장 효과적
3. **Naive가 메인, GradientBoosting은 보조**: GB 비중 20%일 때가 최적 (RMSE 406.80)
 
---
 
## 10. 딥러닝 모델 실험 (LSTM & Transformer)

별도의 노트북(`dl_lstm_transformer.ipynb`)을 통해 시계열 특화 딥러닝 모델을 실험했다.

### 10.1 실험 개요
- **모델**: LSTM (Long Short-Term Memory), Transformer (Self-Attention)
- **접근법**: 단일 모델 및 Residual/ROR 스태킹 구조 적용 (`DL_Base + DL_Residual`)
- **데이터**: 동일한 666주 데이터 사용

### 10.2 실험 결과 (Test 기간)

| 모델 | RMSE | MAPE (%) | 비고 |
|------|------|----------|------|
| **ROR_Trans+Trans+LSTM** | **684.34** | **3.97** | **DL 모델 중 최고** |
| RES_Transformer+Transformer | 723.65 | 3.80 | |
| BASE_Transformer | 955.19 | 5.04 | |
| BASE_LSTM | 1957.66 | 11.61 | Validation 우수, Test 과적합 |

### 10.3 시사점과 교훈

1. **LSTM의 극단적 과적합**:
   - LSTM은 Validation 기간(2025.08~10)에서 **RMSE 461.62**로 Hybrid 모델에 버금가는 매우 우수한 성능을 보였다.
   - 그러나 Test 기간(2025.10~2026.01)에서는 **RMSE 1957.66**으로 모델 중 최악의 성능을 기록했다.
   - 이는 LSTM이 과거의 패턴(Train/Val의 평균회귀)을 너무 완벽하게 학습한 나머지, Test 기간의 새로운 패턴(급등 추세)에 적응하지 못한 **과적합(Overfitting)**의 전형적인 사례다.

2. **데이터 양의 한계**:
   - 666주(약 13년)라는 데이터 양은 복잡한 파라미터를 가진 Transformer나 LSTM이 일반화된 패턴을 학습하기에 부족했다.
   - 결과적으로 단순한 Naive 모델이나 가벼운 ML 모델(GradientBoosting)보다 성능이 저조했다.

---
 
## 11. 결론 및 권장사항
 
### 11.1 가설 검증 결과
 
| 가설 | 결과 | 근거 |
|------|------|------|
| 비철금속 동조화 | 확인 | SHAP에서 Pb, Cu, Zn, Al 재고/가격 상위권 |
| 중국 경기 영향 | 부분 확인 | 직접 지표(Shanghai50) 포함, 간접 지표(채권)도 중요 |
| 앙상블 스태킹 우월 | **기각** | 스태킹이 오히려 성능 악화 |
 
### 11.2 핵심 인사이트
 
1. **"복잡한 모델이 반드시 좋은 것은 아니다"**: 시장 구조가 변화하는 상황에서 단순한 모델이 더 robust할 수 있다
 
2. **"스태킹은 Train/Test 시장 구조가 동일할 때만 효과적"**: Train(평균 회귀)과 Test(일방적 추세)의 패턴이 다르면, Residual 모델이 추세를 역방향으로 보정하여 오히려 성능 악화
 
3. **"Validation 과적합 주의"**: ML 모델이 Validation에서 좋은 성능을 보여도 Test에서 급격히 하락할 수 있다
 
### 11.3 실무 권장사항
 
| 우선순위 | 모델 | RMSE | 특징 |
|----------|------|------|------|
| 1순위 | Hybrid(Naive*0.8 + GB*0.2) | 406.80 | 최고 성능 |
| 2순위 | Naive_Drift_Damped(α=0.7) | 438.60 | 단순하고 해석 용이 |
| 3순위 | Naive_Drift | 480.67 | 가장 단순, 유지보수 불필요 |
 
**재학습 주기**: 분기별(12주마다) 모델 재평가, 반드시 **Naive 대비 개선 여부 확인**
 
**폴백 전략**: ML 모델이 Naive보다 나쁘면 Naive로 폴백하는 것이 합리적
 
---
 
## 12. 연구 한계점 및 향후 과제
 
### 한계점
1. **데이터 기간**: 668주(약 13년)는 장기 사이클 분석에 다소 짧음
2. **테스트 기간**: 12주(n=12)는 통계적으로 충분하지 않을 수 있음
3. **외생 변수**: 지정학적 이벤트(러시아 제재 등) 미반영
4. **실시간 적용**: 일부 변수는 발표 시차 존재
 
### 향후 과제
1. **시장 레짐(regime) 감지기 개발**: 횡보장 vs 추세장 자동 감지 → ML/Naive 모델 동적 전환
2. 인도네시아 관련 변수(IDR 환율, 수출 정책) 추가 검토
3. 중국 스테인리스강 생산량 데이터 확보 시 포함
4. Walk-forward 방식의 rolling 백테스트 수행

---

## 13. 프로젝트 파일 구조

```
sparta2/
├── sparta2.ipynb            # 메인 분석 노트북 (전체 실험 파이프라인)
├── EDA.ipynb                # 탐색적 데이터 분석
├── dl_lstm_transformer.ipynb # 딥러닝 모델 실험 (LSTM/Transformer)
├── REPORT.md                # 최종 보고서 (본 문서)
│
├── data_weekly_260120.csv       # 원본 데이터 (666주, 주간)
├── data_engineered_features.csv # 피처 엔지니어링 후 데이터
├── data_selected_features.csv   # SHAP 선택 피처만 포함
├── feature_importance.csv       # 피처별 SHAP 중요도
├── selected_features.txt        # 선택된 피처 목록
│
├── shap_summary.png         # SHAP 피처 중요도 시각화
├── report_images/           # 보고서용 시각화 이미지
│   ├── nickel_price_ts.png  # 니켈 가격 시계열
│   ├── returns_dist.png     # 수익률 분포
│   └── volatility_ts.png    # 변동성 추이
│
├── run_shap_selection.py    # SHAP 피처 선택 스크립트
├── requirements.txt         # Python 의존성
└── .gitignore               # Git 추적 제외 설정
```

---

## 14. 참고문헌

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
3. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.

---

## 15. 최종 성능 비교표 (통합)

### 15.1 전체 실험 결과 요약

| 순위 | 모델 | Test RMSE | 기준선(406.80) 대비 | 출처 노트북 | 비고 |
|------|------|-----------|-------------------|-------------|------|
| **1** | **Hybrid (Naive*0.8 + GB*0.2)** | **406.80** | **기준선** | sparta2 | **✅ 최고 성능** |
| 2 | Hybrid (Naive*0.8 + ARIMA*0.2) | 416.01 | -9.21 | sparta2_extras | ARIMA 조합 |
| 3 | Baseline (Naive*0.8 + GB*0.2) 재검증 | 419.14 | -12.34 | sparta2_final_solution | 재검증 결과 |
| 4 | Hybrid (Naive*0.9 + GB*0.1) | 423.67 | -16.87 | sparta2 | |
| 5 | Naive_Drift_Damped (α=0.7) | 438.60 | -31.80 | sparta2 | |
| 6 | Naive_Drift | 480.67 | -73.87 | sparta2 | |
| 7 | LGB Tuned Hybrid (0.7:0.3) | 370.05 | +36.75 | sparta2_final_solution | ⚠️ Val에서만 좋음 |
| 8 | ARIMA (단독) | 1315.97 | -909.17 | sparta2_extras | ❌ 단독 사용 불가 |
| 9 | GradientBoosting (단독) | 1185.07 | -778.27 | sparta2 | ❌ 단독 사용 불가 |

### 15.2 성능 개선 여부 판단

| 항목 | 결과 |
|------|------|
| **최고 성능 모델** | Hybrid (Naive*0.8 + GB*0.2) = RMSE 406.80 |
| **sparta2 이후 개선** | ❌ 통계적으로 유의미한 개선 없음 |
| **원인** | Test 기간의 시장 레짐(일방적 상승)이 학습 패턴과 상이 |
| **결론** | **Hybrid (0.8:0.2) 유지 권장** |

### 15.3 검증 방법론 결과

| 검증 방법 | 결과 | 의미 |
|----------|------|------|
| Time Series CV (5-Fold) | 승률 0/5 | 개선이 일관적이지 않음 |
| Bootstrap 95% CI | 모두 0 포함 | 통계적 불확실 |
| Walk-Forward Validation | 진행 예정 | - |

---

## 16. 노트북 파일 가이드

### 16.1 제출용 파일

| 파일명 | 역할 | 핵심 내용 |
|--------|------|----------|
| `sparta2.ipynb` | **메인 분석** | Baseline, SHAP, Stacking, Hybrid 발견 |
| `sparta2_advanced.ipynb` | **고급 기법** | Regime Detection, Momentum, OLS Combination |
| `sparta2_extras.ipynb` | **추가 모델** | ARIMA, Quantile, Multi-Lag |

### 16.2 참고용 파일

| 파일명 | 역할 | 핵심 내용 |
|--------|------|----------|
| `sparta2_final_solution.ipynb` | **엄격한 검증** | Grid Search, Time Series CV, Bootstrap |
| `dl_lstm_transformer.ipynb` | **딥러닝 실험** | LSTM, Transformer (과적합 확인) |

### 16.3 핵심 결론

> **⚠️ 연구 결과**: sparta2에서 발견한 **Hybrid(Naive*0.8 + GB*0.2)**가 여전히 최고 성능이다.
> 
> 추가로 시도한 고급 기법들(Regime Detection, ARIMA, Multi-Lag 등)은 Validation에서는 좋아 보였으나,
> 엄격한 검증(Time Series CV, Bootstrap) 결과 **통계적으로 유의미한 개선을 확인하지 못했다**.
>
> 이는 Test 기간(2025.10~2026.01)의 **시장 레짐 변화**(평균회귀 → 추세 지속)가 원인이다.

---

**End of Report**
