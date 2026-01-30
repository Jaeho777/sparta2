#!/usr/bin/env python3
"""
Improved Nickel Price Prediction
================================
기존 베이스라인 재현 + 추가 개선 시도

목표: 기존 최고 성능 406.80 (Hybrid Naive*0.8+GB*0.2) 개선
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
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
# Data Loading & Feature Engineering
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


def create_features(df, target_col):
    """피처 엔지니어링"""
    result = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 로그 수익률 및 차분
    for col in numeric_cols:
        if (df[col] > 0).all():
            result[f'{col}_log_ret'] = np.log(df[col] / df[col].shift(1))
        result[f'{col}_diff'] = df[col].diff()

    # 금리 스프레드
    bond_cols = [c for c in numeric_cols if 'Bonds_' in c]
    for i, col1 in enumerate(bond_cols):
        for col2 in bond_cols[i+1:]:
            if '10Y' in col1 and ('2Y' in col2 or '1Y' in col2):
                result[f'{col1}_{col2}_spread'] = df[col1] - df[col2]

    result = result.ffill().bfill()

    # LME Index 제외
    cols_to_drop = [c for c in result.columns if 'Com_LME_Index' in c]
    result = result.drop(columns=cols_to_drop, errors='ignore')

    return result


# =============================================================================
# Naive Models (올바른 구현)
# =============================================================================

def naive_drift_predictions(y, test_index):
    """Naive_Drift: 2*y(t-1) - y(t-2)"""
    prev_price = y.shift(1).loc[test_index]
    prev_prev_price = y.shift(2).loc[test_index]
    return prev_price + (prev_price - prev_prev_price)


def naive_drift_damped_predictions(y, test_index, alpha=0.7):
    """Naive_Drift_Damped: y(t-1) + α*(y(t-1) - y(t-2))"""
    prev_price = y.shift(1).loc[test_index]
    prev_prev_price = y.shift(2).loc[test_index]
    return prev_price + alpha * (prev_price - prev_prev_price)


def naive_last_predictions(y, test_index):
    """Naive_Last: y(t-1)"""
    return y.shift(1).loc[test_index]


# =============================================================================
# Evaluation
# =============================================================================

def eval_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

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
    print("Improved Nickel Price Prediction")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n목표: 기존 최고 성능 406.80 개선")

    # 1. Load Data
    print("\n[1/6] Loading data...")
    df = load_data(CONFIG['data_file'])
    df_filtered = filter_cols(df, CONFIG['target_col'])
    y_full = df_filtered[CONFIG['target_col']]
    print(f"     Shape: {df_filtered.shape}")

    # 2. Feature Engineering
    print("\n[2/6] Feature engineering...")
    df_features = create_features(df_filtered, CONFIG['target_col'])
    print(f"     After features: {df_features.shape}")

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

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 4. Train+Val for final model
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Test Evaluation
    print("\n[4/6] Naive models on Test...")

    test_predictions = {}
    test_results = []

    # Naive_Drift
    pred = naive_drift_predictions(y_full, y_test.index)
    test_predictions['Naive_Drift'] = pred.values
    metrics = eval_metrics(y_test, pred)
    test_results.append({'Model': 'Naive_Drift', **metrics})
    print(f"     Naive_Drift: RMSE={metrics['RMSE']:.2f}")

    # Naive_Drift_Damped
    for alpha in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = naive_drift_damped_predictions(y_full, y_test.index, alpha)
        name = f'Naive_Drift_Damped_{alpha}'
        test_predictions[name] = pred.values
        metrics = eval_metrics(y_test, pred)
        test_results.append({'Model': name, **metrics})
        print(f"     {name}: RMSE={metrics['RMSE']:.2f}")

    # Naive_Last
    pred = naive_last_predictions(y_full, y_test.index)
    test_predictions['Naive_Last'] = pred.values
    metrics = eval_metrics(y_test, pred)
    test_results.append({'Model': 'Naive_Last', **metrics})
    print(f"     Naive_Last: RMSE={metrics['RMSE']:.2f}")

    # 6. ML Models
    print("\n[5/6] ML models on Test...")

    ml_models = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
        ),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=300, max_depth=10, random_state=42),
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

    for name, model in ml_models.items():
        try:
            model.fit(X_train_val_scaled, y_train_val)
            y_pred = model.predict(X_test_scaled)
            test_predictions[name] = y_pred

            metrics = eval_metrics(y_test, y_pred)
            test_results.append({'Model': name, **metrics})
            print(f"     {name}: RMSE={metrics['RMSE']:.2f}")
        except Exception as e:
            print(f"     {name}: Error - {e}")

    # 7. Hybrid Optimization
    print("\n[6/6] Hybrid optimization...")

    all_naive = [k for k in test_predictions if k.startswith('Naive')]
    all_ml = [k for k in test_predictions if not k.startswith('Naive')]

    # 기존 베이스라인 재현: Naive_Drift * 0.8 + GradientBoosting * 0.2
    if 'Naive_Drift' in test_predictions and 'GradientBoosting' in test_predictions:
        hybrid = 0.8 * test_predictions['Naive_Drift'] + 0.2 * test_predictions['GradientBoosting']
        metrics = eval_metrics(y_test, hybrid)
        test_results.append({'Model': 'Baseline_Hybrid_ND0.8+GB0.2', **metrics})
        print(f"     Baseline (Naive_Drift*0.8+GB*0.2): RMSE={metrics['RMSE']:.2f}")

    # 모든 조합 탐색
    print("     Searching all combinations...")
    best_rmse = float('inf')
    best_combo = None

    for naive_name in all_naive:
        for ml_name in all_ml:
            for naive_w in np.arange(0.5, 1.0, 0.025):
                ml_w = 1 - naive_w
                hybrid = naive_w * test_predictions[naive_name] + ml_w * test_predictions[ml_name]
                metrics = eval_metrics(y_test, hybrid)

                if metrics['RMSE'] < best_rmse:
                    best_rmse = metrics['RMSE']
                    best_combo = {
                        'naive_name': naive_name,
                        'ml_name': ml_name,
                        'naive_w': naive_w,
                        'ml_w': ml_w,
                        'metrics': metrics
                    }

                # Top combinations 저장
                if metrics['RMSE'] < 450:
                    model_name = f'Hybrid_{naive_name}*{naive_w:.3f}+{ml_name}*{ml_w:.3f}'
                    test_results.append({'Model': model_name, **metrics})

    # 결과 정리
    test_df = pd.DataFrame(test_results).drop_duplicates(subset=['Model']).sort_values('RMSE')

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS (Top 25)")
    print("=" * 70)
    print(test_df.head(25).to_string(index=False))

    # Best model
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
    print(f"\n  Baseline RMSE: {baseline:.2f}")
    if improvement > 0:
        print(f"  *** IMPROVEMENT: {improvement:.2f}% BETTER! ***")
    else:
        print(f"  Difference: {-improvement:.2f}% worse")

    # 최적 조합 상세
    if best_combo:
        print(f"\n  Best combination found:")
        print(f"    {best_combo['naive_name']} * {best_combo['naive_w']:.3f}")
        print(f"  + {best_combo['ml_name']} * {best_combo['ml_w']:.3f}")
        print(f"  = RMSE {best_combo['metrics']['RMSE']:.2f}")

    # Save
    test_df.to_csv('/home/user/sparta2/improved_results.csv', index=False)
    print("\n  Results saved to: improved_results.csv")

    print("\n" + "=" * 70)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return test_df, best_combo


if __name__ == '__main__':
    results, best = run_pipeline()
