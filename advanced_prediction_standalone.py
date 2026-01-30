#!/usr/bin/env python3
"""
Advanced Nickel Price Prediction - Standalone Script
=====================================================
필요한 패키지가 없어도 기본 버전으로 동작하도록 설계됨
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Optional imports
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_HIST_GB = True
except ImportError:
    HAS_HIST_GB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available")

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("CatBoost not available")

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not available - using random search")

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'data_file': '/home/user/sparta2/data_weekly_260120.csv',
    'target_col': 'Com_LME_Ni_Cash',

    'val_start': '2025-08-04',
    'val_end': '2025-10-20',
    'test_start': '2025-10-27',
    'test_end': '2026-01-12',

    'lag_periods': [1, 2, 4, 8, 12, 24],
    'rolling_windows': [4, 8, 12, 24],

    'optuna_trials': 50,
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
        if any(kw in col for kw in ['Gold', 'Silver', 'Iron', 'Steel', 'Copper',
                                     'Aluminum', 'Zinc', 'Nickel', 'Lead', 'Tin',
                                     'Uranium', 'CrudeOil', 'BrentCrudeOil']):
            keep.append(col)
    return df[[target] + list(set(keep))]


# =============================================================================
# Feature Engineering
# =============================================================================

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
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calc_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bb_width = (upper - lower) / (sma + 1e-10)
    bb_position = (series - lower) / (upper - lower + 1e-10)
    return bb_width, bb_position


def create_advanced_features(df, target_col, lag_periods, rolling_windows):
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
        bb_width, bb_pos = calc_bollinger_bands(target, period)
        result[f'{target_col}_BB_width_{period}'] = bb_width
        result[f'{target_col}_BB_pos_{period}'] = bb_pos

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
            result[f'{col}_EMA_{window}'] = df[col].ewm(span=window, adjust=False).mean()
            result[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
            result[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
            result[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
            result[f'{col}_range_{window}'] = (
                result[f'{col}_max_{window}'] - result[f'{col}_min_{window}']
            )

    # Returns and Momentum
    for col in key_cols:
        for period in [1, 4, 12]:
            result[f'{col}_ret_{period}'] = df[col].pct_change(periods=period)
            result[f'{col}_mom_{period}'] = df[col] - df[col].shift(period)

    # Metal ratios
    if 'Com_LME_Cu_Cash' in df.columns:
        result['Ni_Cu_ratio'] = target / (df['Com_LME_Cu_Cash'] + 1e-10)
    if 'Com_LME_Al_Cash' in df.columns:
        result['Ni_Al_ratio'] = target / (df['Com_LME_Al_Cash'] + 1e-10)
    if 'Com_LME_Zn_Cash' in df.columns:
        result['Ni_Zn_ratio'] = target / (df['Com_LME_Zn_Cash'] + 1e-10)

    # Yield curve
    if 'Bonds_US_10Y' in df.columns and 'Bonds_US_2Y' in df.columns:
        result['US_yield_curve'] = df['Bonds_US_10Y'] - df['Bonds_US_2Y']

    # Dollar index change
    if 'Idx_DxyUSD' in df.columns:
        result['DXY_change_4w'] = df['Idx_DxyUSD'].pct_change(4)
        result['DXY_change_12w'] = df['Idx_DxyUSD'].pct_change(12)

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


class NaiveDrift:
    def fit(self, X, y):
        y = np.array(y)
        self.last_value = y[-1]
        self.drift = y[-1] - y[-2] if len(y) > 1 else 0
        return self

    def predict(self, X):
        n = len(X)
        return np.array([self.last_value + self.drift * (i + 1) for i in range(n)])


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
# Model Building
# =============================================================================

def get_models(seed=42):
    models = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=seed
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=seed
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=300, max_depth=10, random_state=seed
        ),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'BayesianRidge': BayesianRidge(),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=seed),
    }

    if HAS_HIST_GB:
        models['HistGradientBoosting'] = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, random_state=seed
        )

    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            random_state=seed, verbosity=0
        )

    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            random_state=seed, verbose=-1
        )

    if HAS_CB:
        models['CatBoost'] = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=seed, verbose=False
        )

    return models


# =============================================================================
# Ensemble Optimization
# =============================================================================

def optimize_ensemble_weights(predictions_dict, y_true):
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)

    if n_models == 0:
        return {}, None
    if n_models == 1:
        return {model_names[0]: 1.0}, predictions_dict[model_names[0]]

    predictions_array = np.column_stack([predictions_dict[name] for name in model_names])
    y_true = np.array(y_true).flatten()

    best_rmse = float('inf')
    best_weights = None

    # Grid search
    step = 0.05
    from itertools import product
    weight_range = np.arange(0, 1.01, step)

    for weights in product(weight_range, repeat=n_models):
        if abs(sum(weights) - 1.0) > 0.01:
            continue

        weights = np.array(weights)
        ensemble_pred = predictions_array @ weights
        rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights

    if best_weights is None:
        best_weights = np.ones(n_models) / n_models

    weight_dict = {name: w for name, w in zip(model_names, best_weights)}
    ensemble_pred = predictions_array @ best_weights

    return weight_dict, ensemble_pred


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(verbose=True):
    if verbose:
        print("=" * 70)
        print("Advanced Nickel Price Prediction Pipeline")
        print("=" * 70)
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load Data
    if verbose:
        print("\n[1/6] Loading data...")
    df = load_data(CONFIG['data_file'])
    df = filter_cols(df, CONFIG['target_col'])
    if verbose:
        print(f"     Original shape: {df.shape}")

    # 2. Feature Engineering
    if verbose:
        print("\n[2/6] Feature engineering...")
    df = create_advanced_features(df, CONFIG['target_col'],
                                   CONFIG['lag_periods'], CONFIG['rolling_windows'])
    if verbose:
        print(f"     After features: {df.shape}")

    # 3. Split Data
    if verbose:
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

    if verbose:
        print(f"     Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Models
    if verbose:
        print("\n[4/6] Training models...")

    models = get_models(CONFIG['random_seed'])
    val_predictions = {}
    val_results = []

    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            val_predictions[name] = y_pred

            metrics = eval_metrics(y_val, y_pred)
            val_results.append({'Model': name, **metrics})

            if verbose:
                print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
        except Exception as e:
            if verbose:
                print(f"     {name}: Error - {e}")

    # Naive models
    naive_models = {
        'Naive_Last': NaiveLast(),
        'Naive_Drift': NaiveDrift(),
        'Naive_Drift_Damped_0.7': NaiveDriftDamped(0.7),
        'Naive_Drift_Damped_0.8': NaiveDriftDamped(0.8),
        'Naive_Drift_Damped_0.9': NaiveDriftDamped(0.9),
    }

    for name, model in naive_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        val_predictions[name] = y_pred

        metrics = eval_metrics(y_val, y_pred)
        val_results.append({'Model': name, **metrics})

        if verbose:
            print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    # 5. Ensemble Optimization
    if verbose:
        print("\n[5/6] Optimizing ensemble...")

    # ML ensemble
    ml_preds = {k: v for k, v in val_predictions.items() if not k.startswith('Naive')}
    ml_weights, ml_ensemble = optimize_ensemble_weights(ml_preds, y_val)

    if ml_ensemble is not None:
        val_predictions['Ensemble_ML'] = ml_ensemble
        metrics = eval_metrics(y_val, ml_ensemble)
        val_results.append({'Model': 'Ensemble_ML', **metrics})
        if verbose:
            print(f"     Ensemble_ML: RMSE={metrics['RMSE']:.2f}")

    # Hybrid ensembles
    best_naive = min([k for k in val_predictions if k.startswith('Naive')],
                     key=lambda k: eval_metrics(y_val, val_predictions[k])['RMSE'])
    best_ml = min([k for k in val_predictions if not k.startswith('Naive') and k != 'Ensemble_ML'],
                  key=lambda k: eval_metrics(y_val, val_predictions[k])['RMSE'])

    for naive_w in [0.6, 0.7, 0.8, 0.9]:
        ml_w = 1 - naive_w
        hybrid = naive_w * val_predictions[best_naive] + ml_w * val_predictions[best_ml]
        name = f'Hybrid_{best_naive}*{naive_w:.1f}+{best_ml}*{ml_w:.1f}'
        val_predictions[name] = hybrid

        metrics = eval_metrics(y_val, hybrid)
        val_results.append({'Model': name, **metrics})

    val_df = pd.DataFrame(val_results).sort_values('RMSE')

    # 6. Test Evaluation
    if verbose:
        print("\n[6/6] Testing...")

    # Retrain on train+val
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled_final = scaler.transform(X_test)

    test_predictions = {}
    test_results = []

    # ML models
    models = get_models(CONFIG['random_seed'])
    for name, model in models.items():
        try:
            model.fit(X_train_val_scaled, y_train_val)
            y_pred = model.predict(X_test_scaled_final)
            test_predictions[name] = y_pred

            metrics = eval_metrics(y_test, y_pred)
            test_results.append({'Model': name, **metrics})
        except:
            pass

    # Naive models
    for name, model_class in [
        ('Naive_Last', NaiveLast()),
        ('Naive_Drift', NaiveDrift()),
        ('Naive_Drift_Damped_0.7', NaiveDriftDamped(0.7)),
        ('Naive_Drift_Damped_0.8', NaiveDriftDamped(0.8)),
        ('Naive_Drift_Damped_0.9', NaiveDriftDamped(0.9)),
    ]:
        model_class.fit(X_train_val, y_train_val)
        y_pred = model_class.predict(X_test)
        test_predictions[name] = y_pred

        metrics = eval_metrics(y_test, y_pred)
        test_results.append({'Model': name, **metrics})

    # Ensemble
    if ml_weights:
        ml_test = sum(w * test_predictions[n] for n, w in ml_weights.items() if n in test_predictions)
        test_predictions['Ensemble_ML'] = ml_test
        metrics = eval_metrics(y_test, ml_test)
        test_results.append({'Model': 'Ensemble_ML', **metrics})

    # Hybrids
    if best_naive in test_predictions and best_ml in test_predictions:
        for naive_w in [0.6, 0.7, 0.8, 0.9]:
            ml_w = 1 - naive_w
            hybrid = naive_w * test_predictions[best_naive] + ml_w * test_predictions[best_ml]
            name = f'Hybrid_{best_naive}*{naive_w:.1f}+{best_ml}*{ml_w:.1f}'
            test_predictions[name] = hybrid

            metrics = eval_metrics(y_test, hybrid)
            test_results.append({'Model': name, **metrics})

    test_df = pd.DataFrame(test_results).sort_values('RMSE')

    # Final output
    if verbose:
        print("\n" + "=" * 70)
        print("FINAL TEST RESULTS (Top 15)")
        print("=" * 70)
        print(test_df.head(15).to_string(index=False))

        best = test_df.iloc[0]
        print("\n" + "=" * 70)
        print("BEST MODEL")
        print("=" * 70)
        print(f"  Model: {best['Model']}")
        print(f"  RMSE:  {best['RMSE']:.2f}")
        print(f"  MAE:   {best['MAE']:.2f}")
        print(f"  MAPE:  {best['MAPE']:.2f}%")

        baseline = 406.80
        improvement = (baseline - best['RMSE']) / baseline * 100
        print(f"\n  Baseline RMSE: {baseline:.2f}")
        if improvement > 0:
            print(f"  Improvement:   {improvement:.2f}% better")
        else:
            print(f"  Difference:    {-improvement:.2f}% worse")

    return {
        'val_results': val_df,
        'test_results': test_df,
        'test_predictions': test_predictions,
        'y_test': y_test,
        'ml_weights': ml_weights
    }


if __name__ == '__main__':
    results = run_pipeline(verbose=True)

    # Save results
    results['test_results'].to_csv('/home/user/sparta2/advanced_standalone_results.csv', index=False)
    print("\nResults saved to advanced_standalone_results.csv")
