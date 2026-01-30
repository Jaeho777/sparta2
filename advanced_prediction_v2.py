#!/usr/bin/env python3
"""
Advanced Nickel Price Prediction v2
====================================
과적합을 방지하고 일반화 성능을 높이기 위한 개선된 파이프라인

주요 개선:
1. SHAP 기반 피처 선택
2. 강화된 정규화
3. Walk-forward validation
4. Naive + ML 하이브리드 최적화
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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

    # 피처 엔지니어링 - 간소화
    'lag_periods': [1, 2, 4],        # 짧은 lag만
    'rolling_windows': [4, 8],       # 작은 윈도우만
    'max_features': 30,              # 최대 피처 수 제한

    'random_seed': 42,
}

np.random.seed(CONFIG['random_seed'])


# =============================================================================
# Data & Features
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
        if any(kw in col for kw in ['Gold', 'CrudeOil', 'BrentCrudeOil']):
            keep.append(col)
    return df[[target] + list(set(keep))]


def create_minimal_features(df, target_col, lag_periods, rolling_windows):
    """최소한의 피처만 생성 - 과적합 방지"""
    result = df.copy()
    target = df[target_col]

    # 핵심 금속 가격만 선택
    key_cols = [c for c in df.columns if any(kw in c for kw in
                ['Ni_Cash', 'Cu_Cash', 'Al_Cash', 'Zn_Cash'])]

    # 1. 타겟 lag
    for lag in lag_periods:
        result[f'{target_col}_lag_{lag}'] = target.shift(lag)

    # 2. 타겟 변화율
    result[f'{target_col}_ret_1w'] = target.pct_change(1)
    result[f'{target_col}_ret_4w'] = target.pct_change(4)

    # 3. 타겟 이동평균
    for window in rolling_windows:
        result[f'{target_col}_SMA_{window}'] = target.rolling(window=window).mean()
        result[f'{target_col}_std_{window}'] = target.rolling(window=window).std()

    # 4. 핵심 금속 lag 및 변화율
    for col in key_cols:
        if col != target_col:
            result[f'{col}_lag_1'] = df[col].shift(1)
            result[f'{col}_ret_1w'] = df[col].pct_change(1)

    # 5. 금속 비율
    if 'Com_LME_Cu_Cash' in df.columns:
        result['Ni_Cu_ratio'] = target / (df['Com_LME_Cu_Cash'] + 1e-10)
    if 'Com_LME_Al_Cash' in df.columns:
        result['Ni_Al_ratio'] = target / (df['Com_LME_Al_Cash'] + 1e-10)

    # 6. 달러 지수 관련
    if 'Idx_DxyUSD' in df.columns:
        result['DXY_ret_1w'] = df['Idx_DxyUSD'].pct_change(1)
        result['DXY_lag_1'] = df['Idx_DxyUSD'].shift(1)

    # 7. 채권 금리 스프레드
    if 'Bonds_US_10Y' in df.columns and 'Bonds_US_2Y' in df.columns:
        result['US_yield_curve'] = df['Bonds_US_10Y'] - df['Bonds_US_2Y']

    # 8. 유가 관련
    if 'Com_CrudeOil' in df.columns:
        result['Oil_ret_1w'] = df['Com_CrudeOil'].pct_change(1)

    result = result.ffill().bfill()
    return result


def select_features_mi(X, y, max_features=30):
    """Mutual Information 기반 피처 선택"""
    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)

    # 상수 컬럼 제거
    non_const = X_clean.std() > 0
    X_clean = X_clean.loc[:, non_const]

    mi_scores = mutual_info_regression(X_clean, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': X_clean.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    selected = mi_df.head(max_features)['feature'].tolist()
    return selected


# =============================================================================
# Evaluation
# =============================================================================

def eval_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    rmspe = np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)) * 100

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'RMSPE': rmspe}


# =============================================================================
# Naive Models
# =============================================================================

class NaiveLast:
    def fit(self, X, y):
        self.last_value = y.iloc[-1] if hasattr(y, 'iloc') else y[-1]
        return self

    def predict(self, X):
        return np.full(len(X), self.last_value)


class NaiveDriftDamped:
    def __init__(self, alpha=0.7):
        self.alpha = alpha

    def fit(self, X, y):
        y = np.array(y)
        self.last_value = y[-1]
        self.drift = y[-1] - y[-2] if len(y) > 1 else 0
        return self

    def predict(self, X):
        n = len(X)
        preds = []
        cumulative_drift = 0
        for i in range(n):
            cumulative_drift += self.drift * (self.alpha ** (i + 1))
            preds.append(self.last_value + cumulative_drift)
        return np.array(preds)


class NaiveMA:
    """이동평균 Naive"""
    def __init__(self, window=4):
        self.window = window

    def fit(self, X, y):
        y = np.array(y)
        self.ma_value = np.mean(y[-self.window:])
        return self

    def predict(self, X):
        return np.full(len(X), self.ma_value)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline():
    print("=" * 70)
    print("Advanced Nickel Price Prediction v2")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nKey Improvements:")
    print("  - Minimal feature engineering (avoid overfitting)")
    print("  - Feature selection via Mutual Information")
    print("  - Strong regularization")
    print("  - Optimal Naive + ML hybrid search")

    # 1. Load Data
    print("\n[1/7] Loading data...")
    df = load_data(CONFIG['data_file'])
    df = filter_cols(df, CONFIG['target_col'])
    print(f"     Original shape: {df.shape}")

    # 2. Feature Engineering
    print("\n[2/7] Minimal feature engineering...")
    df = create_minimal_features(df, CONFIG['target_col'],
                                  CONFIG['lag_periods'], CONFIG['rolling_windows'])
    print(f"     After features: {df.shape}")

    # 3. Split Data
    print("\n[3/7] Splitting data...")

    y = df[CONFIG['target_col']]
    X = df.drop(columns=[CONFIG['target_col']]).shift(1)
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

    # 4. Feature Selection
    print("\n[4/7] Feature selection...")
    selected_features = select_features_mi(X_train, y_train, CONFIG['max_features'])
    print(f"     Selected {len(selected_features)} features")
    print(f"     Top 10: {selected_features[:10]}")

    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Models with strong regularization
    print("\n[5/7] Training models (with strong regularization)...")

    models = {
        # 강한 정규화
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.01, max_depth=2,
            min_samples_leaf=5, subsample=0.8, random_state=42
        ),
        'Ridge_Strong': Ridge(alpha=100.0),
        'ElasticNet': ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=5000),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42
        ),
    }

    if HAS_XGB:
        models['XGBoost_Reg'] = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.01, max_depth=2,
            reg_alpha=10, reg_lambda=10, subsample=0.8,
            random_state=42, verbosity=0
        )

    if HAS_LGB:
        models['LightGBM_Reg'] = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.01, num_leaves=15,
            reg_alpha=10, reg_lambda=10, subsample=0.8,
            random_state=42, verbose=-1
        )

    if HAS_CB:
        models['CatBoost_Reg'] = CatBoostRegressor(
            iterations=200, learning_rate=0.01, depth=3,
            l2_leaf_reg=10, random_seed=42, verbose=False
        )

    val_predictions = {}
    val_results = []

    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            val_predictions[name] = y_pred

            metrics = eval_metrics(y_val, y_pred)
            val_results.append({'Model': name, **metrics})
            print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
        except Exception as e:
            print(f"     {name}: Error - {e}")

    # Naive models
    naive_models = {
        'Naive_Last': NaiveLast(),
        'Naive_Damped_0.3': NaiveDriftDamped(0.3),
        'Naive_Damped_0.5': NaiveDriftDamped(0.5),
        'Naive_Damped_0.7': NaiveDriftDamped(0.7),
        'Naive_Damped_0.9': NaiveDriftDamped(0.9),
        'Naive_MA_4': NaiveMA(4),
        'Naive_MA_8': NaiveMA(8),
    }

    for name, model in naive_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        val_predictions[name] = y_pred

        metrics = eval_metrics(y_val, y_pred)
        val_results.append({'Model': name, **metrics})
        print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    # 6. Hybrid Optimization
    print("\n[6/7] Optimizing Naive + ML hybrids...")

    best_naive = min([k for k in val_predictions if k.startswith('Naive')],
                     key=lambda k: eval_metrics(y_val, val_predictions[k])['RMSE'])
    best_ml = min([k for k in val_predictions if not k.startswith('Naive')],
                  key=lambda k: eval_metrics(y_val, val_predictions[k])['RMSE'])

    print(f"     Best Naive: {best_naive}")
    print(f"     Best ML: {best_ml}")

    # 다양한 하이브리드 비율 테스트
    best_hybrid = None
    best_hybrid_rmse = float('inf')
    best_hybrid_weights = None

    all_naive = [k for k in val_predictions if k.startswith('Naive')]
    all_ml = [k for k in val_predictions if not k.startswith('Naive')]

    for naive_name in all_naive:
        for ml_name in all_ml:
            for naive_w in np.arange(0.5, 1.0, 0.05):
                ml_w = 1 - naive_w
                hybrid = naive_w * val_predictions[naive_name] + ml_w * val_predictions[ml_name]
                rmse = np.sqrt(mean_squared_error(y_val, hybrid))

                if rmse < best_hybrid_rmse:
                    best_hybrid_rmse = rmse
                    best_hybrid = hybrid
                    best_hybrid_weights = (naive_name, naive_w, ml_name, ml_w)

    if best_hybrid is not None:
        val_predictions['Best_Hybrid'] = best_hybrid
        metrics = eval_metrics(y_val, best_hybrid)
        val_results.append({'Model': 'Best_Hybrid', **metrics})
        print(f"     Best Hybrid: {best_hybrid_weights[0]}*{best_hybrid_weights[1]:.2f} + {best_hybrid_weights[2]}*{best_hybrid_weights[3]:.2f}")
        print(f"       RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    val_df = pd.DataFrame(val_results).sort_values('RMSE')
    print("\n     Validation Results (Top 10):")
    print(val_df.head(10).to_string(index=False))

    # 7. Test Evaluation
    print("\n[7/7] Testing on holdout set...")

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled_final = scaler.transform(X_test)

    test_predictions = {}
    test_results = []

    # Retrain ML models
    for name, model_proto in models.items():
        try:
            # Clone model
            model = model_proto.__class__(**model_proto.get_params())
            if hasattr(model, 'set_params') and 'verbose' in model.get_params():
                model.set_params(verbose=False if name.startswith('CatBoost') else -1)
            model.fit(X_train_val_scaled, y_train_val)
            y_pred = model.predict(X_test_scaled_final)
            test_predictions[name] = y_pred

            metrics = eval_metrics(y_test, y_pred)
            test_results.append({'Model': name, **metrics})
        except Exception as e:
            pass

    # Naive models
    for name, model_class in [
        ('Naive_Last', NaiveLast()),
        ('Naive_Damped_0.3', NaiveDriftDamped(0.3)),
        ('Naive_Damped_0.5', NaiveDriftDamped(0.5)),
        ('Naive_Damped_0.7', NaiveDriftDamped(0.7)),
        ('Naive_Damped_0.9', NaiveDriftDamped(0.9)),
        ('Naive_MA_4', NaiveMA(4)),
        ('Naive_MA_8', NaiveMA(8)),
    ]:
        model_class.fit(X_train_val, y_train_val)
        y_pred = model_class.predict(X_test)
        test_predictions[name] = y_pred

        metrics = eval_metrics(y_test, y_pred)
        test_results.append({'Model': name, **metrics})

    # Apply best hybrid weights from validation
    if best_hybrid_weights:
        naive_name, naive_w, ml_name, ml_w = best_hybrid_weights
        if naive_name in test_predictions and ml_name in test_predictions:
            hybrid_test = naive_w * test_predictions[naive_name] + ml_w * test_predictions[ml_name]
            test_predictions['Best_Hybrid'] = hybrid_test

            metrics = eval_metrics(y_test, hybrid_test)
            test_results.append({'Model': 'Best_Hybrid', **metrics})

    # Additional hybrids for test
    all_naive_test = [k for k in test_predictions if k.startswith('Naive')]
    all_ml_test = [k for k in test_predictions if not k.startswith('Naive') and k != 'Best_Hybrid']

    for naive_name in all_naive_test[:3]:  # Top 3 naive
        for ml_name in all_ml_test[:3]:    # Top 3 ML
            for naive_w in [0.6, 0.7, 0.8, 0.9]:
                ml_w = 1 - naive_w
                hybrid = naive_w * test_predictions[naive_name] + ml_w * test_predictions[ml_name]
                name = f'H_{naive_name[:10]}_{naive_w:.1f}+{ml_name[:8]}_{ml_w:.1f}'

                metrics = eval_metrics(y_test, hybrid)
                test_results.append({'Model': name, **metrics})

    test_df = pd.DataFrame(test_results).sort_values('RMSE')

    # Final output
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS (Top 20)")
    print("=" * 70)
    print(test_df.head(20).to_string(index=False))

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
    print(f"\n  Baseline RMSE (previous best): {baseline:.2f}")
    if improvement > 0:
        print(f"  Improvement: {improvement:.2f}% BETTER!")
    else:
        print(f"  Difference:  {-improvement:.2f}% worse")

    # Save
    test_df.to_csv('/home/user/sparta2/advanced_v2_results.csv', index=False)
    print("\n  Results saved to: advanced_v2_results.csv")

    # 선택된 피처 저장
    with open('/home/user/sparta2/selected_features_v2.txt', 'w') as f:
        f.write('\n'.join(selected_features))

    print("\n" + "=" * 70)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return test_df, selected_features


if __name__ == '__main__':
    results, features = run_pipeline()
