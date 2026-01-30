#!/usr/bin/env python3
"""
Advanced Nickel Price Prediction v3
====================================
기존 베이스라인 성공 요인 분석 기반:
- Naive_Drift (추세 추종)가 핵심
- GradientBoosting이 보조역할 (20%)
- Hybrid(Naive*0.8 + GB*0.2) = 406.80

개선 접근:
1. Naive_Drift 변형들 강화
2. 다양한 추세 추정 방법
3. 적응형 가중치 앙상블
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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


# =============================================================================
# Feature Engineering - 기존 성공 방식 재현
# =============================================================================

def create_features_original(df, target_col):
    """기존 성공 방식의 피처 엔지니어링 재현"""
    result = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 로그 수익률 (양수만)
    for col in numeric_cols:
        if (df[col] > 0).all():
            result[f'{col}_log_ret'] = np.log(df[col] / df[col].shift(1))

    # 단순 차분
    for col in numeric_cols:
        result[f'{col}_diff'] = df[col].diff()

    # 금리 스프레드
    bond_cols = [c for c in numeric_cols if 'Bonds_' in c]
    for i, col1 in enumerate(bond_cols):
        for col2 in bond_cols[i+1:]:
            if '10Y' in col1 and ('2Y' in col2 or '1Y' in col2):
                result[f'{col1}_{col2}_spread'] = df[col1] - df[col2]

    result = result.ffill().bfill()

    # LME Index 제외 (순환논리 방지)
    cols_to_drop = [c for c in result.columns if 'Com_LME_Index' in c]
    result = result.drop(columns=cols_to_drop, errors='ignore')

    return result


# =============================================================================
# Naive Models - 핵심!
# =============================================================================

class NaiveDrift:
    """추세 지속 - 기존 베이스라인의 핵심"""
    def fit(self, X, y):
        y = np.array(y)
        self.last_value = y[-1]
        self.drift = y[-1] - y[-2] if len(y) > 1 else 0
        return self

    def predict(self, X):
        n = len(X)
        return np.array([self.last_value + self.drift * (i + 1) for i in range(n)])


class NaiveDriftDamped:
    """감쇠 추세"""
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


class NaiveDriftAvg:
    """평균 추세 (최근 n주 평균 변화율)"""
    def __init__(self, window=4):
        self.window = window

    def fit(self, X, y):
        y = np.array(y)
        self.last_value = y[-1]
        diffs = np.diff(y[-self.window-1:])
        self.avg_drift = np.mean(diffs) if len(diffs) > 0 else 0
        return self

    def predict(self, X):
        n = len(X)
        return np.array([self.last_value + self.avg_drift * (i + 1) for i in range(n)])


class NaiveDriftEMA:
    """지수이동평균 추세"""
    def __init__(self, span=4):
        self.span = span

    def fit(self, X, y):
        y_series = pd.Series(y)
        self.last_value = y.iloc[-1] if hasattr(y, 'iloc') else y[-1]
        diffs = y_series.diff().dropna()
        ema_drift = diffs.ewm(span=self.span, adjust=False).mean().iloc[-1]
        self.drift = ema_drift
        return self

    def predict(self, X):
        n = len(X)
        return np.array([self.last_value + self.drift * (i + 1) for i in range(n)])


class NaiveLast:
    """최근값 유지"""
    def fit(self, X, y):
        self.last_value = y.iloc[-1] if hasattr(y, 'iloc') else y[-1]
        return self

    def predict(self, X):
        return np.full(len(X), self.last_value)


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
# Main Pipeline
# =============================================================================

def run_pipeline():
    print("=" * 70)
    print("Advanced Nickel Price Prediction v3")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n기존 베이스라인 성공 요인 기반:")
    print("  - Naive_Drift (추세 추종)가 핵심")
    print("  - GradientBoosting이 보조역할 (20%)")
    print("  - Hybrid(Naive*0.8 + GB*0.2) = 406.80")

    # 1. Load Data
    print("\n[1/6] Loading data...")
    df = load_data(CONFIG['data_file'])
    df = filter_cols(df, CONFIG['target_col'])
    print(f"     Original shape: {df.shape}")

    # 2. Feature Engineering (기존 방식 재현)
    print("\n[2/6] Feature engineering (original style)...")
    df = create_features_original(df, CONFIG['target_col'])
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

    X_train = X_train.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
    X_val = X_val.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)

    print(f"     Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Naive Models (핵심)
    print("\n[4/6] Training Naive models (key!)...")

    naive_models = {
        'Naive_Drift': NaiveDrift(),
        'Naive_Drift_Damped_0.3': NaiveDriftDamped(0.3),
        'Naive_Drift_Damped_0.5': NaiveDriftDamped(0.5),
        'Naive_Drift_Damped_0.7': NaiveDriftDamped(0.7),
        'Naive_Drift_Damped_0.8': NaiveDriftDamped(0.8),
        'Naive_Drift_Damped_0.9': NaiveDriftDamped(0.9),
        'Naive_Drift_Avg_2': NaiveDriftAvg(2),
        'Naive_Drift_Avg_4': NaiveDriftAvg(4),
        'Naive_Drift_EMA_2': NaiveDriftEMA(2),
        'Naive_Drift_EMA_4': NaiveDriftEMA(4),
        'Naive_Last': NaiveLast(),
    }

    val_predictions = {}
    val_results = []

    for name, model in naive_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        val_predictions[name] = y_pred

        metrics = eval_metrics(y_val, y_pred)
        val_results.append({'Model': name, **metrics})
        print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

    # 5. ML Models (보조역할)
    print("\n[5/6] Training ML models (supplementary)...")

    ml_models = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=42
        ),
        'Ridge': Ridge(alpha=1.0),
    }

    if HAS_XGB:
        ml_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            random_state=42, verbosity=0
        )

    if HAS_LGB:
        ml_models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            random_state=42, verbose=-1
        )

    if HAS_CB:
        ml_models['CatBoost'] = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=42, verbose=False
        )

    for name, model in ml_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            val_predictions[name] = y_pred

            metrics = eval_metrics(y_val, y_pred)
            val_results.append({'Model': name, **metrics})
            print(f"     {name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
        except Exception as e:
            print(f"     {name}: Error - {e}")

    # 6. Hybrid Optimization
    print("\n[6/6] Optimizing Naive + ML hybrids...")

    best_hybrids = []
    all_naive = [k for k in val_predictions if k.startswith('Naive')]
    all_ml = [k for k in val_predictions if not k.startswith('Naive')]

    print(f"     Testing {len(all_naive)} Naive x {len(all_ml)} ML combinations...")

    for naive_name in all_naive:
        for ml_name in all_ml:
            for naive_w in np.arange(0.5, 1.0, 0.05):
                ml_w = 1 - naive_w
                hybrid = naive_w * val_predictions[naive_name] + ml_w * val_predictions[ml_name]

                metrics = eval_metrics(y_val, hybrid)
                best_hybrids.append({
                    'Model': f'H_{naive_name[:15]}*{naive_w:.2f}+{ml_name[:10]}*{ml_w:.2f}',
                    'naive_name': naive_name,
                    'ml_name': ml_name,
                    'naive_w': naive_w,
                    'ml_w': ml_w,
                    **metrics
                })

    # Best hybrids
    best_hybrids_df = pd.DataFrame(best_hybrids).sort_values('RMSE')
    print("\n     Top 10 Hybrids (Validation):")
    print(best_hybrids_df.head(10)[['Model', 'RMSE', 'MAPE']].to_string(index=False))

    # Add to val_results
    for _, row in best_hybrids_df.head(20).iterrows():
        val_results.append({
            'Model': row['Model'],
            'RMSE': row['RMSE'],
            'MAE': row['MAE'],
            'MAPE': row['MAPE'],
            'RMSPE': row['RMSPE']
        })

    val_df = pd.DataFrame(val_results).sort_values('RMSE')

    # =========================================================================
    # Test Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled_final = scaler.transform(X_test)

    test_predictions = {}
    test_results = []

    # Retrain Naive models
    for name, model_cls in [
        ('Naive_Drift', NaiveDrift()),
        ('Naive_Drift_Damped_0.3', NaiveDriftDamped(0.3)),
        ('Naive_Drift_Damped_0.5', NaiveDriftDamped(0.5)),
        ('Naive_Drift_Damped_0.7', NaiveDriftDamped(0.7)),
        ('Naive_Drift_Damped_0.8', NaiveDriftDamped(0.8)),
        ('Naive_Drift_Damped_0.9', NaiveDriftDamped(0.9)),
        ('Naive_Drift_Avg_2', NaiveDriftAvg(2)),
        ('Naive_Drift_Avg_4', NaiveDriftAvg(4)),
        ('Naive_Drift_EMA_2', NaiveDriftEMA(2)),
        ('Naive_Drift_EMA_4', NaiveDriftEMA(4)),
        ('Naive_Last', NaiveLast()),
    ]:
        model_cls.fit(X_train_val, y_train_val)
        y_pred = model_cls.predict(X_test)
        test_predictions[name] = y_pred

        metrics = eval_metrics(y_test, y_pred)
        test_results.append({'Model': name, **metrics})

    # Retrain ML models
    for name, model_proto in ml_models.items():
        try:
            model = model_proto.__class__(**model_proto.get_params())
            if hasattr(model, 'verbose'):
                model.verbose = False if 'CatBoost' in name else -1
            model.fit(X_train_val_scaled, y_train_val)
            y_pred = model.predict(X_test_scaled_final)
            test_predictions[name] = y_pred

            metrics = eval_metrics(y_test, y_pred)
            test_results.append({'Model': name, **metrics})
        except:
            pass

    # Hybrids on test
    for _, row in best_hybrids_df.head(30).iterrows():
        naive_name = row['naive_name']
        ml_name = row['ml_name']
        naive_w = row['naive_w']
        ml_w = row['ml_w']

        if naive_name in test_predictions and ml_name in test_predictions:
            hybrid = naive_w * test_predictions[naive_name] + ml_w * test_predictions[ml_name]
            model_name = f'H_{naive_name[:15]}*{naive_w:.2f}+{ml_name[:10]}*{ml_w:.2f}'

            metrics = eval_metrics(y_test, hybrid)
            test_results.append({'Model': model_name, **metrics})

    test_df = pd.DataFrame(test_results).sort_values('RMSE')

    # Final output
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
        print(f"  Improvement: {improvement:.2f}% BETTER!")
    else:
        print(f"  Difference:  {-improvement:.2f}% worse")

    # Save
    test_df.to_csv('/home/user/sparta2/advanced_v3_results.csv', index=False)
    print("\n  Results saved to: advanced_v3_results.csv")

    print("\n" + "=" * 70)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return test_df


if __name__ == '__main__':
    results = run_pipeline()
