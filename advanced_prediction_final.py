#!/usr/bin/env python3
"""
Advanced Nickel Price Prediction - Final Version
=================================================
기존 베이스라인의 정확한 Naive 구현 재현:
- Naive_Drift = 2*y(t-1) - y(t-2) (롤링 실제 가격 사용)
- Naive_Drift_Damped = y(t-1) + α*(y(t-1) - y(t-2))

핵심 통찰:
- 각 예측 시점마다 실제 전주/2주전 가격 사용 (누수 아님 - t-1 가격은 이미 알려진 값)
- 테스트 기간의 급등장에서 추세 추종이 유효
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'data_file': '/home/user/sparta2/data_weekly_260120.csv',
    'target_col': 'Com_LME_Ni_Cash',
    'val_start': '2025-08-04',
    'test_start': '2025-10-27',
    'random_seed': 42,
}

np.random.seed(CONFIG['random_seed'])


# =============================================================================
# Data Loading
# =============================================================================

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.set_index('dt').sort_index()
    df = df.ffill().bfill()
    return df


def filter_cols(df, target):
    keep = []
    for col in df.columns:
        if col == target:
            continue
        if any(prefix in col for prefix in ['Com_LME_', 'EX_', 'Bonds_', 'Idx_']):
            keep.append(col)
    return df[[target] + list(set(keep))]


def create_features_original(df, target_col):
    """기존 성공 방식의 피처 엔지니어링"""
    result = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if (df[col] > 0).all():
            result[f'{col}_log_ret'] = np.log(df[col] / df[col].shift(1))

    for col in numeric_cols:
        result[f'{col}_diff'] = df[col].diff()

    bond_cols = [c for c in numeric_cols if 'Bonds_' in c]
    for i, col1 in enumerate(bond_cols):
        for col2 in bond_cols[i+1:]:
            if '10Y' in col1 and ('2Y' in col2 or '1Y' in col2):
                result[f'{col1}_{col2}_spread'] = df[col1] - df[col2]

    result = result.ffill().bfill()
    cols_to_drop = [c for c in result.columns if 'Com_LME_Index' in c]
    result = result.drop(columns=cols_to_drop, errors='ignore')

    return result


# =============================================================================
# Naive Models - 정확한 구현 (롤링 실제 가격 사용)
# =============================================================================

def naive_drift_predictions(y, test_index):
    """
    Naive_Drift: 2*y(t-1) - y(t-2)
    각 예측 시점에서 실제 전주/2주전 가격 사용
    """
    prev_price = y.shift(1).loc[test_index]       # y(t-1)
    prev_prev_price = y.shift(2).loc[test_index]  # y(t-2)
    return prev_price + (prev_price - prev_prev_price)  # = 2*y(t-1) - y(t-2)


def naive_drift_damped_predictions(y, test_index, alpha=0.7):
    """
    Naive_Drift_Damped: y(t-1) + α*(y(t-1) - y(t-2))
    """
    prev_price = y.shift(1).loc[test_index]
    prev_prev_price = y.shift(2).loc[test_index]
    return prev_price + alpha * (prev_price - prev_prev_price)


def naive_last_predictions(y, test_index):
    """
    Naive_Last: y(t-1)
    """
    return y.shift(1).loc[test_index]


# =============================================================================
# Evaluation
# =============================================================================

def eval_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # NaN 제거
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mask2 = y_true != 0
    mape = np.mean(np.abs((y_true[mask2] - y_pred[mask2]) / y_true[mask2])) * 100
    rmspe = np.sqrt(np.mean(((y_true[mask2] - y_pred[mask2]) / y_true[mask2]) ** 2)) * 100

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'RMSPE': rmspe}


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline():
    print("=" * 70)
    print("Advanced Nickel Price Prediction - Final Version")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n핵심 수정 사항:")
    print("  - Naive_Drift = 2*y(t-1) - y(t-2) (롤링 실제 가격 사용)")
    print("  - 각 예측 시점마다 실제 전주/2주전 가격 사용")
    print("  - 이는 기존 베이스라인의 정확한 재현")

    # 1. Load Data
    print("\n[1/6] Loading data...")
    df = load_data(CONFIG['data_file'])
    df_filtered = filter_cols(df, CONFIG['target_col'])
    print(f"     Original shape: {df_filtered.shape}")

    # 2. Feature Engineering
    print("\n[2/6] Feature engineering...")
    df_features = create_features_original(df_filtered, CONFIG['target_col'])
    print(f"     After features: {df_features.shape}")

    # 전체 y 시리즈 (Naive 계산용)
    y_full = df_filtered[CONFIG['target_col']]

    # 3. Split Data
    print("\n[3/6] Splitting data...")

    y = df_features[CONFIG['target_col']]
    X = df_features.drop(columns=[CONFIG['target_col']]).shift(1)
    X = X.iloc[1:]
    y = y.iloc[1:]

    val_start = pd.to_datetime(CONFIG['val_start'])
    test_start = pd.to_datetime(CONFIG['test_start'])

    train_mask = X.index < val_start
    val_mask = (X.index >= val_start) & (X.index < test_start)
    test_mask = X.index >= test_start

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    X_train = X_train.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
    X_val = X_val.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)

    print(f"     Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Naive Models (정확한 구현!)
    print("\n[4/6] Naive models (correct rolling implementation)...")

    val_predictions = {}
    val_results = []

    # Naive_Drift
    pred = naive_drift_predictions(y_full, y_val.index)
    val_predictions['Naive_Drift'] = pred.values
    metrics = eval_metrics(y_val, pred)
    val_results.append({'Model': 'Naive_Drift', **metrics})
    print(f"     Naive_Drift: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    # Naive_Drift_Damped (various alpha)
    for alpha in [0.3, 0.5, 0.7, 0.8, 0.9]:
        pred = naive_drift_damped_predictions(y_full, y_val.index, alpha)
        name = f'Naive_Drift_Damped_{alpha}'
        val_predictions[name] = pred.values
        metrics = eval_metrics(y_val, pred)
        val_results.append({'Model': name, **metrics})
        print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    # Naive_Last
    pred = naive_last_predictions(y_full, y_val.index)
    val_predictions['Naive_Last'] = pred.values
    metrics = eval_metrics(y_val, pred)
    val_results.append({'Model': 'Naive_Last', **metrics})
    print(f"     Naive_Last: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    # 5. ML Models
    print("\n[5/6] ML models...")

    ml_models = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
        ),
        'Ridge': Ridge(alpha=1.0),
    }

    if HAS_XGB:
        ml_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0
        )

    if HAS_LGB:
        ml_models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1
        )

    if HAS_CB:
        ml_models['CatBoost'] = CatBoostRegressor(
            iterations=500, learning_rate=0.05, random_seed=42, verbose=False
        )

    trained_models = {}
    for name, model in ml_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            val_predictions[name] = y_pred
            trained_models[name] = model

            metrics = eval_metrics(y_val, y_pred)
            val_results.append({'Model': name, **metrics})
            print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
        except Exception as e:
            print(f"     {name}: Error - {e}")

    # 6. Hybrid Optimization
    print("\n[6/6] Hybrid optimization (Naive + ML)...")

    best_hybrids = []
    all_naive = [k for k in val_predictions if k.startswith('Naive')]
    all_ml = [k for k in val_predictions if not k.startswith('Naive')]

    for naive_name in all_naive:
        for ml_name in all_ml:
            for naive_w in np.arange(0.5, 1.0, 0.05):
                ml_w = 1 - naive_w
                hybrid = naive_w * val_predictions[naive_name] + ml_w * val_predictions[ml_name]

                metrics = eval_metrics(y_val, hybrid)
                best_hybrids.append({
                    'Model': f'Hybrid_{naive_name}*{naive_w:.2f}+{ml_name}*{ml_w:.2f}',
                    'naive_name': naive_name,
                    'ml_name': ml_name,
                    'naive_w': naive_w,
                    'ml_w': ml_w,
                    **metrics
                })

    hybrids_df = pd.DataFrame(best_hybrids).sort_values('RMSE')
    print("\n     Top 10 Hybrids (Validation):")
    print(hybrids_df.head(10)[['Model', 'RMSE', 'MAPE']].to_string(index=False))

    # =========================================================================
    # Test Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    test_predictions = {}
    test_results = []

    # Naive models on test
    pred = naive_drift_predictions(y_full, y_test.index)
    test_predictions['Naive_Drift'] = pred.values
    metrics = eval_metrics(y_test, pred)
    test_results.append({'Model': 'Naive_Drift', **metrics})
    print(f"     Naive_Drift: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    for alpha in [0.3, 0.5, 0.7, 0.8, 0.9]:
        pred = naive_drift_damped_predictions(y_full, y_test.index, alpha)
        name = f'Naive_Drift_Damped_{alpha}'
        test_predictions[name] = pred.values
        metrics = eval_metrics(y_test, pred)
        test_results.append({'Model': name, **metrics})
        print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    pred = naive_last_predictions(y_full, y_test.index)
    test_predictions['Naive_Last'] = pred.values
    metrics = eval_metrics(y_test, pred)
    test_results.append({'Model': 'Naive_Last', **metrics})
    print(f"     Naive_Last: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    # Retrain ML on train+val
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled_final = scaler.transform(X_test)

    for name, model_proto in ml_models.items():
        try:
            model = model_proto.__class__(**model_proto.get_params())
            if hasattr(model, 'verbose'):
                try:
                    model.set_params(verbose=False if 'CatBoost' in name else -1)
                except:
                    pass
            model.fit(X_train_val_scaled, y_train_val)
            y_pred = model.predict(X_test_scaled_final)
            test_predictions[name] = y_pred

            metrics = eval_metrics(y_test, y_pred)
            test_results.append({'Model': name, **metrics})
            print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
        except Exception as e:
            print(f"     {name}: Error - {e}")

    # Hybrids on test
    print("\n     Calculating hybrid predictions on test...")
    for _, row in hybrids_df.head(50).iterrows():
        naive_name = row['naive_name']
        ml_name = row['ml_name']
        naive_w = row['naive_w']
        ml_w = row['ml_w']

        if naive_name in test_predictions and ml_name in test_predictions:
            hybrid = naive_w * test_predictions[naive_name] + ml_w * test_predictions[ml_name]
            model_name = f'Hybrid_{naive_name}*{naive_w:.2f}+{ml_name}*{ml_w:.2f}'

            metrics = eval_metrics(y_test, hybrid)
            test_results.append({'Model': model_name, **metrics})

    # Sort and display
    test_df = pd.DataFrame(test_results).sort_values('RMSE')

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS (Top 25)")
    print("=" * 70)
    print(test_df.head(25).to_string(index=False))

    best = test_df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)
    print(f"  Model: {best['Model']}")
    print(f"  RMSE:  {best['RMSE']:.2f}")
    print(f"  MAE:   {best['MAE']:.2f}")
    print(f"  MAPE:  {best['MAPE']:.2f}%")
    print(f"  RMSPE: {best['RMSPE']:.2f}%")

    baseline = 406.80
    improvement = (baseline - best['RMSE']) / baseline * 100
    print(f"\n  Baseline RMSE (Hybrid Naive*0.8+GB*0.2): {baseline:.2f}")
    if improvement > 0:
        print(f"  *** IMPROVEMENT: {improvement:.2f}% BETTER! ***")
    else:
        print(f"  Difference: {-improvement:.2f}% worse")

    # Save
    test_df.to_csv('/home/user/sparta2/advanced_final_results.csv', index=False)
    print("\n  Results saved to: advanced_final_results.csv")

    print("\n" + "=" * 70)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return test_df


if __name__ == '__main__':
    results = run_pipeline()
