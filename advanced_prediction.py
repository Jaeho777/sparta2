"""
Advanced Nickel Price Prediction Pipeline
==========================================
고급 피처 엔지니어링과 Optuna 기반 하이퍼파라미터 최적화를 적용한
니켈 가격 예측 파이프라인

주요 개선사항:
1. 기술적 지표 (RSI, MACD, Bollinger Bands, ATR)
2. 다중 시점 lag 피처 (1, 2, 4, 8, 12, 24주)
3. 롤링 통계량 (다양한 윈도우 크기)
4. Optuna 기반 하이퍼파라미터 최적화
5. 최적 앙상블 가중치 탐색
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Using default hyperparameters.")

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'data_file': 'data_weekly_260120.csv',
    'target_col': 'Com_LME_Ni_Cash',

    # 기간 분할
    'val_start': '2025-08-04',
    'val_end': '2025-10-20',
    'test_start': '2025-10-27',
    'test_end': '2026-01-12',

    # 피처 엔지니어링
    'lag_periods': [1, 2, 4, 8, 12, 24],  # 다중 시점 lag
    'rolling_windows': [4, 8, 12, 24],    # 롤링 윈도우 크기

    # Optuna 설정
    'optuna_trials': 100,
    'optuna_timeout': 600,  # 10분

    'random_seed': 42,
}

np.random.seed(CONFIG['random_seed'])


# =============================================================================
# 1. 데이터 로드 및 기본 전처리
# =============================================================================

def load_data(file_path):
    """데이터 로드 및 기본 전처리"""
    df = pd.read_csv(file_path)
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.set_index('dt').sort_index()
    df = df.ffill().bfill()
    return df


def filter_cols(df, target):
    """관련 컬럼 필터링"""
    keep = []
    for col in df.columns:
        if col == target:
            continue
        # LME 금속, 환율, 채권, 주요 지수만 사용
        if any(prefix in col for prefix in ['Com_LME_', 'EX_', 'Bonds_', 'Idx_']):
            keep.append(col)
        # 주요 원자재
        if any(kw in col for kw in ['Gold', 'Silver', 'Iron', 'Steel', 'Copper',
                                     'Aluminum', 'Zinc', 'Nickel', 'Lead', 'Tin',
                                     'Uranium', 'CrudeOil', 'BrentCrudeOil']):
            keep.append(col)
    return df[[target] + list(set(keep))]


# =============================================================================
# 2. 고급 피처 엔지니어링
# =============================================================================

def calc_rsi(series, period=14):
    """RSI (Relative Strength Index) 계산"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calc_bollinger_bands(series, period=20, std_dev=2):
    """Bollinger Bands 계산"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bb_width = (upper - lower) / (sma + 1e-10)
    bb_position = (series - lower) / (upper - lower + 1e-10)
    return bb_width, bb_position


def calc_atr(high, low, close, period=14):
    """ATR (Average True Range) - 가격 변동성 지표"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def create_advanced_features(df, target_col, lag_periods, rolling_windows):
    """고급 피처 엔지니어링"""
    result = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 타겟 변수에 대한 기술적 지표
    target = df[target_col]

    # 1. RSI
    for period in [7, 14, 21]:
        result[f'{target_col}_RSI_{period}'] = calc_rsi(target, period)

    # 2. MACD
    macd_line, signal_line, macd_hist = calc_macd(target)
    result[f'{target_col}_MACD'] = macd_line
    result[f'{target_col}_MACD_signal'] = signal_line
    result[f'{target_col}_MACD_hist'] = macd_hist

    # 3. Bollinger Bands
    for period in [10, 20]:
        bb_width, bb_pos = calc_bollinger_bands(target, period)
        result[f'{target_col}_BB_width_{period}'] = bb_width
        result[f'{target_col}_BB_pos_{period}'] = bb_pos

    # 4. 다중 시점 Lag 피처
    for col in numeric_cols:
        for lag in lag_periods:
            result[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # 5. 롤링 통계량
    key_cols = [c for c in numeric_cols if any(kw in c for kw in
                ['Ni_Cash', 'Cu_Cash', 'Al_Cash', 'Zn_Cash', 'Gold', 'CrudeOil'])]

    for col in key_cols:
        for window in rolling_windows:
            # 이동평균
            result[f'{col}_SMA_{window}'] = df[col].rolling(window=window).mean()
            result[f'{col}_EMA_{window}'] = df[col].ewm(span=window, adjust=False).mean()

            # 변동성
            result[f'{col}_std_{window}'] = df[col].rolling(window=window).std()

            # 최대/최소
            result[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
            result[f'{col}_min_{window}'] = df[col].rolling(window=window).min()

            # 범위 (max - min)
            result[f'{col}_range_{window}'] = (
                result[f'{col}_max_{window}'] - result[f'{col}_min_{window}']
            )

    # 6. 변화율/모멘텀
    for col in key_cols:
        for period in [1, 4, 12]:
            # 수익률
            result[f'{col}_ret_{period}'] = df[col].pct_change(periods=period)

            # 모멘텀 (현재 - N기간 전)
            result[f'{col}_mom_{period}'] = df[col] - df[col].shift(period)

    # 7. 상대 강도 (니켈 vs 다른 금속)
    if 'Com_LME_Cu_Cash' in df.columns:
        result['Ni_Cu_ratio'] = target / (df['Com_LME_Cu_Cash'] + 1e-10)
    if 'Com_LME_Al_Cash' in df.columns:
        result['Ni_Al_ratio'] = target / (df['Com_LME_Al_Cash'] + 1e-10)
    if 'Com_LME_Zn_Cash' in df.columns:
        result['Ni_Zn_ratio'] = target / (df['Com_LME_Zn_Cash'] + 1e-10)

    # 8. 금리 스프레드
    if 'Bonds_US_10Y' in df.columns and 'Bonds_US_2Y' in df.columns:
        result['US_yield_curve'] = df['Bonds_US_10Y'] - df['Bonds_US_2Y']
    if 'Bonds_US_10Y' in df.columns and 'Bonds_US_1Y' in df.columns:
        result['US_10Y_1Y_spread'] = df['Bonds_US_10Y'] - df['Bonds_US_1Y']

    # 9. 달러 지수 변화
    if 'Idx_DxyUSD' in df.columns:
        result['DXY_change_4w'] = df['Idx_DxyUSD'].pct_change(4)
        result['DXY_change_12w'] = df['Idx_DxyUSD'].pct_change(12)

    # 결측치 처리
    result = result.ffill().bfill()

    return result


# =============================================================================
# 3. 평가 메트릭
# =============================================================================

def eval_metrics(y_true, y_pred):
    """평가 메트릭 계산"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # RMSPE
    rmspe = np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'RMSPE': rmspe
    }


# =============================================================================
# 4. Naive 모델
# =============================================================================

class NaiveLast:
    """마지막 값 사용"""
    def fit(self, X, y):
        self.last_value = y.iloc[-1] if hasattr(y, 'iloc') else y[-1]
        return self

    def predict(self, X):
        return np.full(len(X), self.last_value)


class NaiveDrift:
    """추세 지속"""
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


# =============================================================================
# 5. Optuna 기반 하이퍼파라미터 최적화
# =============================================================================

def create_optuna_objective(model_name, X_train, y_train, X_val, y_val):
    """Optuna objective 함수 생성"""

    def objective(trial):
        if model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': CONFIG['random_seed'],
                'verbosity': 0
            }
            model = xgb.XGBRegressor(**params)

        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'random_state': CONFIG['random_seed'],
                'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)

        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_seed': CONFIG['random_seed'],
                'verbose': False
            }
            model = CatBoostRegressor(**params)

        elif model_name == 'GradientBoosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': CONFIG['random_seed']
            }
            model = GradientBoostingRegressor(**params)

        elif model_name == 'HistGradientBoosting':
            params = {
                'max_iter': trial.suggest_int('max_iter', 100, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
                'random_state': CONFIG['random_seed']
            }
            model = HistGradientBoostingRegressor(**params)

        elif model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': CONFIG['random_seed']
            }
            model = RandomForestRegressor(**params)

        elif model_name == 'ExtraTrees':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': CONFIG['random_seed']
            }
            model = ExtraTreesRegressor(**params)

        elif model_name == 'Ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-3, 1000, log=True),
            }
            model = Ridge(**params)

        elif model_name == 'ElasticNet':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-3, 100, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            }
            model = ElasticNet(**params)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        return rmse

    return objective


def optimize_model(model_name, X_train, y_train, X_val, y_val, n_trials=50, timeout=300):
    """Optuna로 모델 최적화"""
    if not OPTUNA_AVAILABLE:
        return get_default_model(model_name)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=CONFIG['random_seed'])
    )

    objective = create_optuna_objective(model_name, X_train, y_train, X_val, y_val)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params

    # 최적 파라미터로 모델 생성
    if model_name == 'XGBoost':
        best_params['random_state'] = CONFIG['random_seed']
        best_params['verbosity'] = 0
        return xgb.XGBRegressor(**best_params)
    elif model_name == 'LightGBM':
        best_params['random_state'] = CONFIG['random_seed']
        best_params['verbose'] = -1
        return lgb.LGBMRegressor(**best_params)
    elif model_name == 'CatBoost':
        if 'iterations' in best_params:
            best_params['iterations'] = best_params.pop('iterations')
        best_params['random_seed'] = CONFIG['random_seed']
        best_params['verbose'] = False
        return CatBoostRegressor(**best_params)
    elif model_name == 'GradientBoosting':
        best_params['random_state'] = CONFIG['random_seed']
        return GradientBoostingRegressor(**best_params)
    elif model_name == 'HistGradientBoosting':
        best_params['random_state'] = CONFIG['random_seed']
        return HistGradientBoostingRegressor(**best_params)
    elif model_name == 'RandomForest':
        best_params['random_state'] = CONFIG['random_seed']
        return RandomForestRegressor(**best_params)
    elif model_name == 'ExtraTrees':
        best_params['random_state'] = CONFIG['random_seed']
        return ExtraTreesRegressor(**best_params)
    elif model_name == 'Ridge':
        return Ridge(**best_params)
    elif model_name == 'ElasticNet':
        return ElasticNet(**best_params)


def get_default_model(model_name):
    """기본 하이퍼파라미터 모델"""
    seed = CONFIG['random_seed']

    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            random_state=seed, verbosity=0
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            random_state=seed, verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=seed, verbose=False
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            random_state=seed
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=6,
            random_state=seed
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=seed
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=300, max_depth=10, random_state=seed
        ),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    }
    return models.get(model_name)


# =============================================================================
# 6. 앙상블 가중치 최적화
# =============================================================================

def optimize_ensemble_weights(predictions_dict, y_true, method='grid'):
    """앙상블 가중치 최적화"""
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

    if method == 'grid':
        # 그리드 서치
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

    elif method == 'scipy':
        # scipy 최적화
        from scipy.optimize import minimize

        def objective(weights):
            ensemble_pred = predictions_array @ weights
            return np.sqrt(mean_squared_error(y_true, ensemble_pred))

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        # 여러 초기값으로 시도
        best_result = None
        for _ in range(10):
            x0 = np.random.dirichlet(np.ones(n_models))
            result = minimize(objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            if best_result is None or result.fun < best_result.fun:
                best_result = result

        best_weights = best_result.x
        best_rmse = best_result.fun

    weight_dict = {name: w for name, w in zip(model_names, best_weights)}
    ensemble_pred = predictions_array @ best_weights

    return weight_dict, ensemble_pred


# =============================================================================
# 7. 메인 파이프라인
# =============================================================================

def run_pipeline(use_optuna=True, verbose=True):
    """메인 파이프라인 실행"""

    if verbose:
        print("=" * 70)
        print("Advanced Nickel Price Prediction Pipeline")
        print("=" * 70)

    # 1. 데이터 로드
    if verbose:
        print("\n[1/7] 데이터 로드 중...")
    df = load_data(CONFIG['data_file'])
    df = filter_cols(df, CONFIG['target_col'])
    if verbose:
        print(f"     원본 데이터: {df.shape}")

    # 2. 고급 피처 엔지니어링
    if verbose:
        print("\n[2/7] 고급 피처 엔지니어링 적용 중...")
    df = create_advanced_features(
        df,
        CONFIG['target_col'],
        CONFIG['lag_periods'],
        CONFIG['rolling_windows']
    )
    if verbose:
        print(f"     피처 엔지니어링 후: {df.shape}")

    # 3. Train/Val/Test 분할
    if verbose:
        print("\n[3/7] 데이터 분할 중...")

    y = df[CONFIG['target_col']]
    X = df.drop(columns=[CONFIG['target_col']]).shift(1)  # 1주 지연

    # 첫 번째 행 제거 (shift로 인한 NaN)
    X = X.iloc[1:]
    y = y.iloc[1:]

    # 기간별 분할
    val_start = pd.to_datetime(CONFIG['val_start'])
    test_start = pd.to_datetime(CONFIG['test_start'])

    train_mask = X.index < val_start
    val_mask = (X.index >= val_start) & (X.index < test_start)
    test_mask = X.index >= test_start

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if verbose:
        print(f"     Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 결측치 확인 및 처리
    X_train = X_train.ffill().bfill().fillna(0)
    X_val = X_val.ffill().bfill().fillna(0)
    X_test = X_test.ffill().bfill().fillna(0)

    # 스케일링
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. 모델 학습 및 최적화
    if verbose:
        print("\n[4/7] 모델 학습 및 최적화 중...")

    model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting',
                   'HistGradientBoosting', 'RandomForest', 'ExtraTrees']

    models = {}
    val_predictions = {}
    val_results = []

    for name in model_names:
        if verbose:
            print(f"     - {name} 최적화 중...")

        try:
            if use_optuna and OPTUNA_AVAILABLE:
                model = optimize_model(
                    name, X_train_scaled, y_train, X_val_scaled, y_val,
                    n_trials=CONFIG['optuna_trials'],
                    timeout=CONFIG['optuna_timeout'] // len(model_names)
                )
            else:
                model = get_default_model(name)

            model.fit(X_train_scaled, y_train)
            y_pred_val = model.predict(X_val_scaled)

            models[name] = model
            val_predictions[name] = y_pred_val

            metrics = eval_metrics(y_val, y_pred_val)
            val_results.append({
                'Model': name,
                **metrics
            })

            if verbose:
                print(f"       RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

        except Exception as e:
            if verbose:
                print(f"       오류 발생: {e}")

    # Naive 모델 추가
    naive_models = {
        'Naive_Last': NaiveLast(),
        'Naive_Drift': NaiveDrift(),
        'Naive_Drift_Damped_0.7': NaiveDriftDamped(0.7),
        'Naive_Drift_Damped_0.8': NaiveDriftDamped(0.8),
        'Naive_Drift_Damped_0.9': NaiveDriftDamped(0.9),
    }

    for name, model in naive_models.items():
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)

        models[name] = model
        val_predictions[name] = y_pred_val

        metrics = eval_metrics(y_val, y_pred_val)
        val_results.append({
            'Model': name,
            **metrics
        })

        if verbose:
            print(f"     - {name}: RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

    # 5. 앙상블 가중치 최적화
    if verbose:
        print("\n[5/7] 앙상블 가중치 최적화 중...")

    # ML 모델만 앙상블
    ml_predictions = {k: v for k, v in val_predictions.items()
                      if not k.startswith('Naive')}
    ml_weights, ml_ensemble_val = optimize_ensemble_weights(ml_predictions, y_val)

    if ml_ensemble_val is not None:
        val_predictions['Ensemble_ML'] = ml_ensemble_val
        metrics = eval_metrics(y_val, ml_ensemble_val)
        val_results.append({'Model': 'Ensemble_ML', **metrics})
        if verbose:
            print(f"     - Ensemble_ML: RMSE: {metrics['RMSE']:.2f}")
            print(f"       가중치: {', '.join([f'{k}:{v:.2f}' for k,v in ml_weights.items() if v > 0.01])}")

    # 전체 모델 앙상블
    all_weights, all_ensemble_val = optimize_ensemble_weights(val_predictions, y_val)

    if all_ensemble_val is not None:
        val_predictions['Ensemble_All'] = all_ensemble_val
        metrics = eval_metrics(y_val, all_ensemble_val)
        val_results.append({'Model': 'Ensemble_All', **metrics})
        if verbose:
            print(f"     - Ensemble_All: RMSE: {metrics['RMSE']:.2f}")

    # Naive + ML 하이브리드 최적화
    best_naive_name = min(
        [k for k in val_predictions.keys() if k.startswith('Naive')],
        key=lambda k: eval_metrics(y_val, val_predictions[k])['RMSE']
    )
    best_ml_name = min(
        [k for k in val_predictions.keys() if not k.startswith('Naive') and not k.startswith('Ensemble')],
        key=lambda k: eval_metrics(y_val, val_predictions[k])['RMSE']
    )

    # 다양한 비율로 하이브리드 테스트
    for naive_weight in np.arange(0.5, 1.0, 0.05):
        ml_weight = 1 - naive_weight
        hybrid_pred = (
            naive_weight * val_predictions[best_naive_name] +
            ml_weight * val_predictions[best_ml_name]
        )
        hybrid_name = f'Hybrid_{best_naive_name}*{naive_weight:.2f}+{best_ml_name}*{ml_weight:.2f}'
        val_predictions[hybrid_name] = hybrid_pred

        metrics = eval_metrics(y_val, hybrid_pred)
        val_results.append({'Model': hybrid_name, **metrics})

    # 6. Validation 결과 정리
    if verbose:
        print("\n[6/7] Validation 결과 정리...")

    val_df = pd.DataFrame(val_results).sort_values('RMSE')

    if verbose:
        print("\n     Top 10 모델 (Validation RMSE 기준):")
        print(val_df.head(10).to_string(index=False))

    # 7. Test 세트 평가
    if verbose:
        print("\n[7/7] Test 세트 평가 중...")

    # Train + Val로 재학습
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled_final = scaler.transform(X_test)

    test_predictions = {}
    test_results = []

    # ML 모델 재학습 및 테스트
    for name in model_names:
        if name in models:
            try:
                model = get_default_model(name) if not use_optuna else models[name].__class__(
                    **{k: v for k, v in models[name].get_params().items()
                       if k not in ['verbose', 'verbosity']}
                )
                model.fit(X_train_val_scaled, y_train_val)
                y_pred_test = model.predict(X_test_scaled_final)

                test_predictions[name] = y_pred_test
                metrics = eval_metrics(y_test, y_pred_test)
                test_results.append({'Model': name, **metrics})
            except:
                # Fallback: 원래 모델 파라미터 사용
                model = get_default_model(name)
                model.fit(X_train_val_scaled, y_train_val)
                y_pred_test = model.predict(X_test_scaled_final)

                test_predictions[name] = y_pred_test
                metrics = eval_metrics(y_test, y_pred_test)
                test_results.append({'Model': name, **metrics})

    # Naive 모델 테스트
    for name, model_class in [
        ('Naive_Last', NaiveLast()),
        ('Naive_Drift', NaiveDrift()),
        ('Naive_Drift_Damped_0.7', NaiveDriftDamped(0.7)),
        ('Naive_Drift_Damped_0.8', NaiveDriftDamped(0.8)),
        ('Naive_Drift_Damped_0.9', NaiveDriftDamped(0.9)),
    ]:
        model_class.fit(X_train_val, y_train_val)
        y_pred_test = model_class.predict(X_test)

        test_predictions[name] = y_pred_test
        metrics = eval_metrics(y_test, y_pred_test)
        test_results.append({'Model': name, **metrics})

    # 앙상블 (Validation에서 찾은 가중치 사용)
    if ml_weights:
        ml_test_pred = sum(
            w * test_predictions[name]
            for name, w in ml_weights.items()
            if name in test_predictions
        )
        test_predictions['Ensemble_ML'] = ml_test_pred
        metrics = eval_metrics(y_test, ml_test_pred)
        test_results.append({'Model': 'Ensemble_ML', **metrics})

    # Hybrid 앙상블
    if best_naive_name in test_predictions and best_ml_name in test_predictions:
        for naive_weight in [0.6, 0.7, 0.8, 0.9]:
            ml_weight = 1 - naive_weight
            hybrid_pred = (
                naive_weight * test_predictions[best_naive_name] +
                ml_weight * test_predictions[best_ml_name]
            )
            hybrid_name = f'Hybrid_{best_naive_name}*{naive_weight:.1f}+{best_ml_name}*{ml_weight:.1f}'
            test_predictions[hybrid_name] = hybrid_pred

            metrics = eval_metrics(y_test, hybrid_pred)
            test_results.append({'Model': hybrid_name, **metrics})

    # Test 결과 정리
    test_df = pd.DataFrame(test_results).sort_values('RMSE')

    if verbose:
        print("\n" + "=" * 70)
        print("Final Test Results (Top 15)")
        print("=" * 70)
        print(test_df.head(15).to_string(index=False))

        print("\n" + "=" * 70)
        print("Best Model:")
        print("=" * 70)
        best = test_df.iloc[0]
        print(f"  Model: {best['Model']}")
        print(f"  RMSE:  {best['RMSE']:.2f}")
        print(f"  MAE:   {best['MAE']:.2f}")
        print(f"  MAPE:  {best['MAPE']:.2f}%")
        print(f"  RMSPE: {best['RMSPE']:.2f}%")

    return {
        'val_results': val_df,
        'test_results': test_df,
        'models': models,
        'test_predictions': test_predictions,
        'y_test': y_test,
        'ml_weights': ml_weights,
        'config': CONFIG
    }


# =============================================================================
# 8. 실행
# =============================================================================

if __name__ == '__main__':
    results = run_pipeline(use_optuna=OPTUNA_AVAILABLE, verbose=True)

    # 결과 저장
    results['test_results'].to_csv('advanced_prediction_results.csv', index=False)
    print("\n결과가 'advanced_prediction_results.csv'에 저장되었습니다.")
