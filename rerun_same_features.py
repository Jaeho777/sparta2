#!/usr/bin/env python3
"""
동일 피처로 LightGBM 실험 재현
- sparta2와 동일한 19개 SHAP 피처 사용
- LightGBM + GridSearchCV 튜닝
- Hybrid 모델 (Naive 0.8 + LGB 0.2) 평가
"""

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
df = pd.read_csv('/Users/jaeholee/Desktop/sparta_2/data_weekly_260120.csv')
df['dt'] = pd.to_datetime(df['dt'])
df.set_index('dt', inplace=True)

target_col = 'Com_LME_Ni_Cash'

# sparta2에서 사용한 동일한 19개 SHAP 피처
selected_features_19 = [
    'Com_LME_Pb_Inv',
    'Com_Iron_Ore',
    'Com_LME_Cu_Cash',
    'Bonds_KOR_1Y',
    'Idx_SnPGlobal1200',
    'Com_LME_Pb_Cash',
    'Com_Uranium',
    'Bonds_BRZ_10Y',
    'Com_LME_Ni_Inv',
    'Com_LME_Zn_Inv',
    'Idx_Shanghai50',
    'Com_LME_Cu_Inv',
    'Bonds_BRZ_1Y',
    'Com_LME_Zn_Cash',
    'Bonds_IND_1Y',
    'Com_Silver',
    'Com_LME_Al_Inv',
    'Bonds_AUS_10Y',
    'EX_USD_BRL'
]

print(f"사용 피처 수: {len(selected_features_19)}개")
print("피처 목록:", selected_features_19[:5], "...")

# 기간 분할 (sparta2와 동일)
train = df[:'2025-08-03']
val = df['2025-08-04':'2025-10-20']
test = df['2025-10-27':'2026-01-12']

print(f"\n기간 분할:")
print(f"  Train: {len(train)} samples ({train.index[0]} ~ {train.index[-1]})")
print(f"  Val: {len(val)} samples ({val.index[0]} ~ {val.index[-1]})")
print(f"  Test: {len(test)} samples ({test.index[0]} ~ {test.index[-1]})")

# shift(1) 적용 (Data Leakage 방지)
X_shifted = df[selected_features_19].shift(1)
X_shifted = X_shifted.fillna(method='ffill').fillna(method='bfill').fillna(0)

X_train = X_shifted.loc[train.index]
X_val = X_shifted.loc[val.index]
X_test = X_shifted.loc[test.index]

y_train = train[target_col]
y_val = val[target_col]
y_test = test[target_col]

print(f"\n데이터 shape:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

# Naive Drift 계산 함수
def calc_naive_drift(idx):
    """Naive Drift: prev_price + (prev_price - prev_prev_price)"""
    naive_preds = []
    for date in idx:
        pos = df.index.get_loc(date)
        if pos >= 2:
            prev = df[target_col].iloc[pos-1]
            prev_prev = df[target_col].iloc[pos-2]
            drift = prev + (prev - prev_prev)
        else:
            drift = df[target_col].iloc[pos-1] if pos >= 1 else df[target_col].iloc[0]
        naive_preds.append(drift)
    return np.array(naive_preds)

# GridSearchCV for LightGBM
print("\n" + "="*60)
print("LightGBM GridSearchCV (19개 피처, Hybrid 기반 평가)")
print("="*60)

best_rmse = float('inf')
best_params = None
results = []

naive_val = calc_naive_drift(val.index)

for n_est in [50, 100, 200]:
    for depth in [2, 3, 5]:
        for lr in [0.05, 0.1]:
            model = lgb.LGBMRegressor(
                n_estimators=n_est, 
                max_depth=depth, 
                learning_rate=lr,
                random_state=42, 
                verbose=-1
            )
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            
            # Hybrid 기반 평가 (Naive 0.8 + LGB 0.2)
            hybrid_val = 0.8 * naive_val + 0.2 * val_pred
            val_rmse = sqrt(mean_squared_error(y_val, hybrid_val))
            
            results.append({
                'n_estimators': n_est,
                'max_depth': depth,
                'learning_rate': lr,
                'val_rmse': val_rmse
            })
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_params = {'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr}

print(f"\n최적 파라미터: {best_params}")
print(f"최적 Validation RMSE (Hybrid 0.8:0.2): {best_rmse:.2f}")

# 최적 모델로 Test 평가
lgb_best = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
lgb_best.fit(X_train, y_train)

test_pred_lgb = lgb_best.predict(X_test)
naive_test = calc_naive_drift(test.index)

# Hybrid 예측 (Naive 0.8 + LGB 0.2)
hybrid_test = 0.8 * naive_test + 0.2 * test_pred_lgb

# 최종 성능 평가
test_rmse_hybrid = sqrt(mean_squared_error(y_test, hybrid_test))
test_mape_hybrid = mean_absolute_percentage_error(y_test, hybrid_test) * 100

test_rmse_naive = sqrt(mean_squared_error(y_test, naive_test))
test_rmse_lgb_only = sqrt(mean_squared_error(y_test, test_pred_lgb))

print("\n" + "="*60)
print("최종 결과 (Test 기간)")
print("="*60)
print(f"\n{'모델':<35} {'Test RMSE':>12} {'비고'}")
print("-"*60)
print(f"{'Naive_Drift (단독)':<35} {test_rmse_naive:>12.2f}")
print(f"{'LightGBM (단독, 19 피처)':<35} {test_rmse_lgb_only:>12.2f}")
print(f"{'Hybrid_Naive0.8_LGB0.2 (19 피처)':<35} {test_rmse_hybrid:>12.2f} {'← 동일 피처 비교'}")
print("-"*60)

print(f"\n기존 sparta2 기준선 비교:")
print(f"  - sparta2 Hybrid (Naive0.8+GB0.2, 19 피처): RMSE 406.80")
print(f"  - sparta2_advanced (동일 19 피처 + LGB): RMSE {test_rmse_hybrid:.2f}")
print(f"  - 개선량: {406.80 - test_rmse_hybrid:.2f} (GB → LGB 순수 효과)")

# 모델 파라미터 및 Hybrid MAPE 출력
print(f"\nHybrid MAPE: {test_mape_hybrid:.2f}%")
print(f"파라미터: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, lr={best_params['learning_rate']}")
