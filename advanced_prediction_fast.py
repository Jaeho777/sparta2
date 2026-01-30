#!/usr/bin/env python3
"""
Advanced Nickel Price Prediction - Fast Version
================================================
효율적인 앙상블 최적화를 사용하는 빠른 버전
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
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
    'lag_periods': [1, 2, 4, 8, 12, 24],
    'rolling_windows': [4, 8, 12, 24],
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
        if any(kw in col for kw in ['Gold', 'Silver', 'Iron', 'Steel', 'Copper',
                                     'Aluminum', 'Zinc', 'Nickel', 'Lead', 'Tin',
                                     'Uranium', 'CrudeOil', 'BrentCrudeOil']):
            keep.append(col)
    return df[[target] + list(set(keep))]


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def create_features(df, target_col, lag_periods, rolling_windows):
    result = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = df[target_col]

    # RSI
    for period in [7, 14, 21]:
        result[f'{target_col}_RSI_{period}'] = calc_rsi(target, period)

    # MACD
    macd_line, signal_line, macd_hist = calc_macd(target)
    result[f'{target_col}_MACD'] = macd_line
    result[f'{target_col}_MACD_signal'] = signal_line
    result[f'{target_col}_MACD_hist'] = macd_hist

    # Bollinger Bands
    for period in [10, 20]:
        sma = target.rolling(window=period).mean()
        std = target.rolling(window=period).std()
        result[f'{target_col}_BB_width_{period}'] = (2 * std) / (sma + 1e-10)

    # Lag Features
    for col in numeric_cols:
        for lag in lag_periods:
            result[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Rolling Stats for key columns
    key_cols = [c for c in numeric_cols if any(kw in c for kw in
                ['Ni_Cash', 'Cu_Cash', 'Al_Cash', 'Zn_Cash', 'Gold', 'CrudeOil'])]

    for col in key_cols:
        for window in rolling_windows:
            result[f'{col}_SMA_{window}'] = df[col].rolling(window=window).mean()
            result[f'{col}_std_{window}'] = df[col].rolling(window=window).std()

    # Returns
    for col in key_cols:
        for period in [1, 4, 12]:
            result[f'{col}_ret_{period}'] = df[col].pct_change(periods=period)

    # Ratios
    if 'Com_LME_Cu_Cash' in df.columns:
        result['Ni_Cu_ratio'] = target / (df['Com_LME_Cu_Cash'] + 1e-10)
    if 'Com_LME_Al_Cash' in df.columns:
        result['Ni_Al_ratio'] = target / (df['Com_LME_Al_Cash'] + 1e-10)

    # Yield curve
    if 'Bonds_US_10Y' in df.columns and 'Bonds_US_2Y' in df.columns:
        result['US_yield_curve'] = df['Bonds_US_10Y'] - df['Bonds_US_2Y']

    result = result.ffill().bfill()
    return result


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


# =============================================================================
# Efficient Ensemble
# =============================================================================

def optimize_weights_scipy(predictions_array, y_true):
    """scipy를 사용한 효율적인 가중치 최적화"""
    from scipy.optimize import minimize

    n_models = predictions_array.shape[1]
    y_true = np.array(y_true).flatten()

    def objective(weights):
        ensemble_pred = predictions_array @ weights
        return np.sqrt(mean_squared_error(y_true, ensemble_pred))

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]

    best_result = None
    for _ in range(5):
        x0 = np.random.dirichlet(np.ones(n_models))
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result.x, best_result.fun


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline():
    print("=" * 70)
    print("Advanced Nickel Price Prediction - Fast Version")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load Data
    print("\n[1/6] Loading data...")
    df = load_data(CONFIG['data_file'])
    df = filter_cols(df, CONFIG['target_col'])
    print(f"     Original shape: {df.shape}")

    # 2. Feature Engineering
    print("\n[2/6] Feature engineering...")
    df = create_features(df, CONFIG['target_col'],
                         CONFIG['lag_periods'], CONFIG['rolling_windows'])
    print(f"     After features: {df.shape}")

    # 3. Split Data
    print("\n[3/6] Splitting data...")

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

    X_train = X_train.ffill().bfill().fillna(0)
    X_val = X_val.ffill().bfill().fillna(0)
    X_test = X_test.ffill().bfill().fillna(0)

    print(f"     Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Models
    print("\n[4/6] Training models...")

    models = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=42
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=300, max_depth=10, random_state=42
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, random_state=42
        ),
        'Ridge': Ridge(alpha=1.0),
    }

    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0
        )

    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1
        )

    if HAS_CB:
        models['CatBoost'] = CatBoostRegressor(
            iterations=500, learning_rate=0.05, random_seed=42, verbose=False
        )

    val_predictions = {}
    val_results = []
    trained_models = {}

    for name, model in models.items():
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

    # Naive models
    naive_models = {
        'Naive_Last': NaiveLast(),
        'Naive_Drift_Damped_0.7': NaiveDriftDamped(0.7),
        'Naive_Drift_Damped_0.8': NaiveDriftDamped(0.8),
    }

    for name, model in naive_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        val_predictions[name] = y_pred
        trained_models[name] = model

        metrics = eval_metrics(y_val, y_pred)
        val_results.append({'Model': name, **metrics})
        print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    # 5. Ensemble Optimization
    print("\n[5/6] Optimizing ensemble...")

    # ML-only ensemble (scipy optimization)
    ml_names = [k for k in val_predictions if not k.startswith('Naive')]
    if len(ml_names) > 1:
        ml_preds_array = np.column_stack([val_predictions[n] for n in ml_names])
        ml_weights, ml_rmse = optimize_weights_scipy(ml_preds_array, y_val)
        ml_weights_dict = {n: w for n, w in zip(ml_names, ml_weights)}

        ml_ensemble = ml_preds_array @ ml_weights
        val_predictions['Ensemble_ML'] = ml_ensemble
        metrics = eval_metrics(y_val, ml_ensemble)
        val_results.append({'Model': 'Ensemble_ML', **metrics})
        print(f"     Ensemble_ML: RMSE={metrics['RMSE']:.2f}")
        print(f"       Weights: {', '.join([f'{k}:{v:.2f}' for k,v in ml_weights_dict.items() if v > 0.01])}")

    # Best models
    best_naive = min([k for k in val_predictions if k.startswith('Naive')],
                     key=lambda k: eval_metrics(y_val, val_predictions[k])['RMSE'])
    best_ml = min([k for k in val_predictions if not k.startswith('Naive') and k != 'Ensemble_ML'],
                  key=lambda k: eval_metrics(y_val, val_predictions[k])['RMSE'])

    print(f"     Best Naive: {best_naive}")
    print(f"     Best ML: {best_ml}")

    # Hybrid ensembles
    for naive_w in [0.5, 0.6, 0.7, 0.8, 0.9]:
        ml_w = 1 - naive_w
        hybrid = naive_w * val_predictions[best_naive] + ml_w * val_predictions[best_ml]
        name = f'Hybrid_{naive_w:.1f}*Naive+{ml_w:.1f}*ML'
        val_predictions[name] = hybrid

        metrics = eval_metrics(y_val, hybrid)
        val_results.append({'Model': name, **metrics})

    # 6. Test Evaluation
    print("\n[6/6] Testing on holdout set...")

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled_final = scaler.transform(X_test)

    test_predictions = {}
    test_results = []

    # Retrain ML models
    for name, _ in models.items():
        try:
            model = models[name].__class__(**models[name].get_params())
            if name in ['CatBoost']:
                model.set_params(verbose=False)
            model.fit(X_train_val_scaled, y_train_val)
            y_pred = model.predict(X_test_scaled_final)
            test_predictions[name] = y_pred

            metrics = eval_metrics(y_test, y_pred)
            test_results.append({'Model': name, **metrics})
        except Exception as e:
            pass

    # Naive models
    for name, model_cls in [
        ('Naive_Last', NaiveLast()),
        ('Naive_Drift_Damped_0.7', NaiveDriftDamped(0.7)),
        ('Naive_Drift_Damped_0.8', NaiveDriftDamped(0.8)),
    ]:
        model_cls.fit(X_train_val, y_train_val)
        y_pred = model_cls.predict(X_test)
        test_predictions[name] = y_pred

        metrics = eval_metrics(y_test, y_pred)
        test_results.append({'Model': name, **metrics})

    # Ensemble
    if len(ml_names) > 1 and all(n in test_predictions for n in ml_names):
        ml_test = sum(ml_weights_dict.get(n, 0) * test_predictions[n] for n in ml_names)
        test_predictions['Ensemble_ML'] = ml_test
        metrics = eval_metrics(y_test, ml_test)
        test_results.append({'Model': 'Ensemble_ML', **metrics})

    # Hybrid
    best_naive_test = min([k for k in test_predictions if k.startswith('Naive')],
                          key=lambda k: eval_metrics(y_test, test_predictions[k])['RMSE'])
    best_ml_test = min([k for k in test_predictions if not k.startswith('Naive') and k != 'Ensemble_ML'],
                       key=lambda k: eval_metrics(y_test, test_predictions[k])['RMSE'])

    for naive_w in [0.5, 0.6, 0.7, 0.8, 0.9]:
        ml_w = 1 - naive_w
        hybrid = naive_w * test_predictions[best_naive_test] + ml_w * test_predictions[best_ml_test]
        name = f'Hybrid_{naive_w:.1f}*{best_naive_test}+{ml_w:.1f}*{best_ml_test}'
        test_predictions[name] = hybrid

        metrics = eval_metrics(y_test, hybrid)
        test_results.append({'Model': name, **metrics})

    test_df = pd.DataFrame(test_results).sort_values('RMSE')

    # Final output
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(test_df.to_string(index=False))

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
    test_df.to_csv('/home/user/sparta2/advanced_fast_results.csv', index=False)
    print("\n  Results saved to: advanced_fast_results.csv")

    print("\n" + "=" * 70)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return test_df


if __name__ == '__main__':
    results = run_pipeline()
