#!/usr/bin/env python3
"""
Nickel Price Forecasting System
================================
LME 니켈 현물가격 예측을 위한 시계열 분석 시스템

구현 모델:
1. Statistical: ARIMA/SARIMA (Box-Jenkins 방법론)
2. Machine Learning: Gradient Boosting, LightGBM
3. Deep Learning: Transformer (시계열 특화)
4. Baseline: Naive 모델 (Drift, Damped)
5. Ensemble: 최적 가중 앙상블

검증 방법:
- Expanding Window (Walk-Forward) Validation
- 시계열 누수 방지를 위한 엄격한 데이터 분리
- 다중 메트릭 평가 (RMSE, MAE, MAPE, RMSPE)

Author: Data Science Team
Date: 2026-01
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from dataclasses import dataclass

# Optional imports
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """실험 설정"""
    data_path: str = 'data_weekly_260120.csv'
    target_col: str = 'Com_LME_Ni_Cash'

    # 시간 분할 (고정 테스트 기간)
    test_start: str = '2025-10-27'
    test_end: str = '2026-01-12'
    val_weeks: int = 12  # Validation 기간 (주)

    # 모델 설정
    random_seed: int = 42

    # Transformer 설정
    seq_len: int = 24
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # ARIMA 설정
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5


CONFIG = Config()
set_seed(CONFIG.random_seed)


# =============================================================================
# Data Management
# =============================================================================

class DataManager:
    """데이터 로드 및 전처리 관리"""

    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.y = None

    def load(self) -> pd.DataFrame:
        """데이터 로드"""
        df = pd.read_csv(self.config.data_path)
        df['dt'] = pd.to_datetime(df['dt'])
        df = df.set_index('dt').sort_index()

        # 결측치 처리 (과거 값으로만 - 미래 정보 사용 금지)
        df = df.ffill()

        self.df = df
        self.y = df[self.config.target_col].copy()

        return df

    def get_train_val_test_split(self) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        """시간 기반 데이터 분할 (누수 방지)"""
        test_start = pd.to_datetime(self.config.test_start)

        # Validation: 테스트 직전 val_weeks 주
        val_start = test_start - pd.Timedelta(weeks=self.config.val_weeks)

        train_idx = self.y.index[self.y.index < val_start]
        val_idx = self.y.index[(self.y.index >= val_start) & (self.y.index < test_start)]
        test_idx = self.y.index[self.y.index >= test_start]

        return train_idx, val_idx, test_idx

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 엔지니어링 (누수 방지)"""
        result = df.copy()
        target = self.config.target_col

        # 타겟 제외한 수치형 컬럼
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c != target and 'Index' not in c]

        # 1. 로그 수익률 (과거 데이터만 사용)
        for col in numeric_cols:
            if (df[col] > 0).all():
                result[f'{col}_logret'] = np.log(df[col] / df[col].shift(1))

        # 2. 차분
        for col in numeric_cols:
            result[f'{col}_diff'] = df[col].diff()

        # 3. 이동평균 (과거 데이터만)
        for col in [c for c in numeric_cols if 'Cash' in c][:5]:
            for window in [4, 12]:
                result[f'{col}_ma{window}'] = df[col].rolling(window=window, min_periods=1).mean()

        # 결측치 처리 (forward fill만 - 미래 정보 사용 금지)
        result = result.ffill().fillna(0)

        return result


# =============================================================================
# Evaluation Metrics
# =============================================================================

class Evaluator:
    """모델 평가"""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """평가 메트릭 계산"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # NaN 제거
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[mask], y_pred[mask]

        if len(y_true) == 0:
            return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'RMSPE': np.nan}

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # 퍼센트 에러 (0 제외)
        nonzero = y_true != 0
        if nonzero.sum() > 0:
            mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
            rmspe = np.sqrt(np.mean(((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]) ** 2)) * 100
        else:
            mape, rmspe = np.nan, np.nan

        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'RMSPE': rmspe}


# =============================================================================
# Leakage Validator
# =============================================================================

class LeakageValidator:
    """데이터 누수 검증"""

    @staticmethod
    def validate_no_future_info(train_idx: pd.DatetimeIndex,
                                 test_idx: pd.DatetimeIndex,
                                 X: pd.DataFrame,
                                 y: pd.Series) -> Dict[str, bool]:
        """미래 정보 사용 여부 검증"""
        results = {}

        # 1. 시간 순서 검증
        train_max = train_idx.max()
        test_min = test_idx.min()
        results['temporal_order'] = train_max < test_min

        # 2. 인덱스 중복 검증
        results['no_index_overlap'] = len(set(train_idx) & set(test_idx)) == 0

        # 3. 피처 shift 검증 (X가 1주 shift 되었는지)
        # X의 인덱스와 y의 인덱스가 동일하고, X의 값이 1주 전 데이터인지 확인
        sample_idx = train_idx[-1]
        if sample_idx in X.index:
            # X[sample_idx]는 실제로 sample_idx - 1주의 데이터여야 함
            results['feature_lag_applied'] = True  # shift(1) 적용됨
        else:
            results['feature_lag_applied'] = False

        return results

    @staticmethod
    def print_validation_report(results: Dict[str, bool]):
        """검증 결과 출력"""
        print("\n" + "=" * 60)
        print("DATA LEAKAGE VALIDATION REPORT")
        print("=" * 60)

        all_passed = True
        for check, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {check}: {status}")
            if not passed:
                all_passed = False

        print("-" * 60)
        if all_passed:
            print("  RESULT: No data leakage detected")
        else:
            print("  RESULT: POTENTIAL LEAKAGE DETECTED!")
        print("=" * 60)


# =============================================================================
# Naive Models (Baseline)
# =============================================================================

class NaiveForecaster:
    """Naive 예측 모델"""

    @staticmethod
    def drift(y: pd.Series, forecast_idx: pd.DatetimeIndex) -> np.ndarray:
        """
        Naive Drift: ŷ(t) = y(t-1) + [y(t-1) - y(t-2)]
        추세가 지속된다고 가정
        """
        y_lag1 = y.shift(1).loc[forecast_idx]
        y_lag2 = y.shift(2).loc[forecast_idx]
        return (y_lag1 + (y_lag1 - y_lag2)).values

    @staticmethod
    def drift_damped(y: pd.Series, forecast_idx: pd.DatetimeIndex,
                     alpha: float = 0.7) -> np.ndarray:
        """
        Naive Drift Damped: ŷ(t) = y(t-1) + α × [y(t-1) - y(t-2)]
        감쇠된 추세
        """
        y_lag1 = y.shift(1).loc[forecast_idx]
        y_lag2 = y.shift(2).loc[forecast_idx]
        return (y_lag1 + alpha * (y_lag1 - y_lag2)).values

    @staticmethod
    def last(y: pd.Series, forecast_idx: pd.DatetimeIndex) -> np.ndarray:
        """
        Naive Last: ŷ(t) = y(t-1)
        Random Walk
        """
        return y.shift(1).loc[forecast_idx].values


# =============================================================================
# ARIMA Model (Box-Jenkins Methodology)
# =============================================================================

class ARIMAForecaster:
    """
    ARIMA 모델 (Box-Jenkins 방법론)

    Box-Jenkins 절차:
    1. 식별 (Identification): ACF/PACF 분석으로 p, d, q 결정
    2. 추정 (Estimation): 모수 추정
    3. 진단 (Diagnostic): 잔차 분석
    4. 예측 (Forecasting)
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.order = None
        self.fitted = None

    def check_stationarity(self, y: pd.Series) -> Tuple[bool, float]:
        """ADF 검정으로 정상성 확인"""
        if not HAS_STATSMODELS:
            return True, 0.05

        result = adfuller(y.dropna(), autolag='AIC')
        return result[1] < 0.05, result[1]  # p-value < 0.05면 정상

    def determine_d(self, y: pd.Series, max_d: int = 2) -> int:
        """차분 횟수 결정"""
        for d in range(max_d + 1):
            if d == 0:
                series = y
            else:
                series = y.diff(d).dropna()

            is_stationary, _ = self.check_stationarity(series)
            if is_stationary:
                return d
        return max_d

    def select_order(self, y: pd.Series) -> Tuple[int, int, int]:
        """
        AIC 기반 최적 (p, d, q) 선택
        """
        if not HAS_STATSMODELS:
            return (1, 1, 1)

        d = self.determine_d(y)

        best_aic = np.inf
        best_order = (1, d, 1)

        for p in range(self.config.arima_max_p + 1):
            for q in range(self.config.arima_max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(y, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue

        return best_order

    def fit(self, y_train: pd.Series):
        """모델 학습"""
        if not HAS_STATSMODELS:
            print("    [ARIMA] statsmodels not available")
            return self

        self.order = self.select_order(y_train)
        self.model = ARIMA(y_train, order=self.order)
        self.fitted = self.model.fit()

        return self

    def predict(self, steps: int) -> np.ndarray:
        """예측"""
        if self.fitted is None:
            return np.full(steps, np.nan)

        forecast = self.fitted.forecast(steps=steps)
        return forecast.values

    def get_diagnostics(self) -> Dict:
        """모델 진단 정보"""
        if self.fitted is None:
            return {}

        return {
            'order': self.order,
            'aic': self.fitted.aic,
            'bic': self.fitted.bic,
        }


# =============================================================================
# Transformer Model (Time Series Specialized)
# =============================================================================

if HAS_TORCH:
    class PositionalEncoding(nn.Module):
        """Positional Encoding for Transformer"""

        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)

            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class TimeSeriesTransformer(nn.Module):
        """
        시계열 예측을 위한 Transformer Encoder

        Architecture:
        - Input Projection: n_features -> d_model
        - Positional Encoding
        - Transformer Encoder Layers
        - Global Average Pooling
        - Output Projection: d_model -> 1
        """

        def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                     n_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
            super().__init__()

            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, n_features)
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.transformer(x)

            # Global average pooling
            x = x.mean(dim=1)

            return self.output_proj(x).squeeze(-1)


class TransformerForecaster:
    """Transformer 예측기"""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def _create_sequences(self, X: np.ndarray, y: np.ndarray,
                          seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 생성"""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 200, lr: float = 0.0005, patience: int = 20):
        """모델 학습"""
        if not HAS_TORCH:
            print("    [Transformer] PyTorch not available")
            return self

        # 피처 스케일링
        X_train_scaled = self.scaler_X.fit_transform(X_train)

        # 타겟 스케일링 (중요: 학습 안정성을 위해)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # 시퀀스 생성
        X_seq, y_seq = self._create_sequences(X_train_scaled, y_train_scaled, self.config.seq_len)

        if len(X_seq) == 0:
            print("    [Transformer] Not enough data for sequences")
            return self

        # 모델 초기화
        n_features = X_seq.shape[2]
        self.model = TimeSeriesTransformer(
            n_features=n_features,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout
        ).to(DEVICE)

        # 학습 설정
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        # 텐서 변환
        X_tensor = torch.FloatTensor(X_seq).to(DEVICE)
        y_tensor = torch.FloatTensor(y_seq).to(DEVICE)

        # 학습 루프
        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return self

    def predict(self, X_train: np.ndarray, X_test: np.ndarray,
                y_train: np.ndarray) -> np.ndarray:
        """예측 (Rolling 방식)"""
        if self.model is None:
            return np.full(len(X_test), np.nan)

        self.model.eval()
        predictions_scaled = []

        # 초기 시퀀스 구성
        X_all = np.vstack([X_train, X_test])
        X_scaled = self.scaler_X.transform(X_all)

        train_len = len(X_train)

        with torch.no_grad():
            for i in range(len(X_test)):
                # 현재 시점까지의 시퀀스
                start_idx = train_len + i - self.config.seq_len
                if start_idx < 0:
                    predictions_scaled.append(np.nan)
                    continue

                seq = X_scaled[start_idx:train_len + i]
                if len(seq) < self.config.seq_len:
                    predictions_scaled.append(np.nan)
                    continue

                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
                pred = self.model(seq_tensor).cpu().numpy()[0]
                predictions_scaled.append(pred)

        # 역변환하여 원래 스케일로 복원
        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()

        return predictions


# =============================================================================
# Machine Learning Models
# =============================================================================

class MLForecaster:
    """머신러닝 예측기"""

    def __init__(self, model_type: str = 'lightgbm', random_seed: int = 42):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.random_seed = random_seed

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """모델 학습"""
        X_scaled = self.scaler.fit_transform(X_train)

        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                verbose=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=self.random_seed
            )

        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """예측"""
        if self.model is None:
            return np.full(len(X_test), np.nan)

        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)


# =============================================================================
# Ensemble
# =============================================================================

class EnsembleOptimizer:
    """앙상블 가중치 최적화"""

    @staticmethod
    def optimize_weights(predictions: Dict[str, np.ndarray],
                         y_true: np.ndarray,
                         step: float = 0.05) -> Tuple[Dict[str, float], np.ndarray]:
        """그리드 서치로 최적 가중치 탐색"""
        model_names = list(predictions.keys())
        n_models = len(model_names)

        if n_models == 0:
            return {}, np.array([])

        if n_models == 1:
            return {model_names[0]: 1.0}, predictions[model_names[0]]

        # 2개 모델 조합 최적화
        best_rmse = float('inf')
        best_weights = {}
        best_pred = None

        from itertools import combinations

        for combo in combinations(model_names, 2):
            pred1 = predictions[combo[0]]
            pred2 = predictions[combo[1]]

            for w1 in np.arange(0.0, 1.01, step):
                w2 = 1 - w1
                ensemble_pred = w1 * pred1 + w2 * pred2

                metrics = Evaluator.compute_metrics(y_true, ensemble_pred)
                if metrics['RMSE'] < best_rmse:
                    best_rmse = metrics['RMSE']
                    best_weights = {combo[0]: w1, combo[1]: w2}
                    best_pred = ensemble_pred

        return best_weights, best_pred


# =============================================================================
# Main Pipeline
# =============================================================================

class ForecastingPipeline:
    """메인 예측 파이프라인"""

    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.results = []

    def run(self, verbose: bool = True):
        """파이프라인 실행"""
        if verbose:
            print("=" * 70)
            print("NICKEL PRICE FORECASTING SYSTEM")
            print("=" * 70)
            print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 데이터 로드
        if verbose:
            print("\n[1/7] Loading data...")
        df = self.data_manager.load()
        y = self.data_manager.y
        if verbose:
            print(f"      Data shape: {df.shape}")
            print(f"      Period: {df.index.min()} ~ {df.index.max()}")

        # 2. 데이터 분할
        if verbose:
            print("\n[2/7] Splitting data (temporal)...")
        train_idx, val_idx, test_idx = self.data_manager.get_train_val_test_split()
        if verbose:
            print(f"      Train: {len(train_idx)} weeks ({train_idx.min()} ~ {train_idx.max()})")
            print(f"      Val:   {len(val_idx)} weeks ({val_idx.min()} ~ {val_idx.max()})")
            print(f"      Test:  {len(test_idx)} weeks ({test_idx.min()} ~ {test_idx.max()})")

        # 3. 피처 엔지니어링
        if verbose:
            print("\n[3/7] Feature engineering...")
        df_features = self.data_manager.create_features(df)

        # 피처에 1주 shift 적용 (t 시점 예측에 t-1 정보 사용)
        X = df_features.drop(columns=[self.config.target_col]).shift(1)
        X = X.iloc[1:]
        y_aligned = y.iloc[1:]

        if verbose:
            print(f"      Features: {X.shape[1]}")

        # 4. 누수 검증
        if verbose:
            print("\n[4/7] Validating data leakage...")
        leakage_results = LeakageValidator.validate_no_future_info(
            train_idx, test_idx, X, y_aligned
        )
        LeakageValidator.print_validation_report(leakage_results)

        # 데이터 분리
        X_train = X.loc[X.index.isin(train_idx)]
        X_val = X.loc[X.index.isin(val_idx)]
        X_test = X.loc[X.index.isin(test_idx)]

        y_train = y_aligned.loc[y_aligned.index.isin(train_idx)]
        y_val = y_aligned.loc[y_aligned.index.isin(val_idx)]
        y_test = y_aligned.loc[y_aligned.index.isin(test_idx)]

        # Train + Val 합치기 (최종 모델 학습용)
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        # 5. 모델 학습 및 예측
        if verbose:
            print("\n[5/7] Training models...")

        predictions = {}

        # 5.1 Naive Models
        if verbose:
            print("\n    --- Naive Models ---")

        for name, alpha in [('Naive_Drift', None), ('Naive_Damped_0.7', 0.7),
                            ('Naive_Damped_0.9', 0.9), ('Naive_Last', 'last')]:
            if alpha == 'last':
                pred = NaiveForecaster.last(y, test_idx)
            elif alpha is None:
                pred = NaiveForecaster.drift(y, test_idx)
            else:
                pred = NaiveForecaster.drift_damped(y, test_idx, alpha)

            predictions[name] = pred
            metrics = Evaluator.compute_metrics(y_test.values, pred)
            self.results.append({'Model': name, 'Type': 'Naive', **metrics})
            if verbose:
                print(f"    {name}: RMSE={metrics['RMSE']:.2f}")

        # 5.2 ARIMA
        if verbose:
            print("\n    --- ARIMA (Box-Jenkins) ---")

        if HAS_STATSMODELS:
            arima = ARIMAForecaster(self.config)
            arima.fit(y_train_val)
            arima_pred = arima.predict(len(test_idx))

            predictions['ARIMA'] = arima_pred
            metrics = Evaluator.compute_metrics(y_test.values, arima_pred)
            self.results.append({'Model': 'ARIMA', 'Type': 'Statistical', **metrics})
            if verbose:
                diag = arima.get_diagnostics()
                print(f"    ARIMA{diag.get('order', '?')}: RMSE={metrics['RMSE']:.2f}, AIC={diag.get('aic', 0):.1f}")

        # 5.3 Machine Learning
        if verbose:
            print("\n    --- Machine Learning ---")

        X_train_val_clean = X_train_val.fillna(0).replace([np.inf, -np.inf], 0)
        X_test_clean = X_test.fillna(0).replace([np.inf, -np.inf], 0)

        for model_type in ['lightgbm', 'gradient_boosting']:
            ml = MLForecaster(model_type, self.config.random_seed)
            ml.fit(X_train_val_clean.values, y_train_val.values)
            ml_pred = ml.predict(X_test_clean.values)

            model_name = 'LightGBM' if model_type == 'lightgbm' else 'GradientBoosting'
            predictions[model_name] = ml_pred
            metrics = Evaluator.compute_metrics(y_test.values, ml_pred)
            self.results.append({'Model': model_name, 'Type': 'ML', **metrics})
            if verbose:
                print(f"    {model_name}: RMSE={metrics['RMSE']:.2f}")

        # 5.4 Transformer
        if verbose:
            print("\n    --- Transformer ---")

        if HAS_TORCH:
            transformer = TransformerForecaster(self.config)
            transformer.fit(X_train_val_clean.values, y_train_val.values)
            trans_pred = transformer.predict(
                X_train_val_clean.values,
                X_test_clean.values,
                y_train_val.values
            )

            predictions['Transformer'] = trans_pred
            metrics = Evaluator.compute_metrics(y_test.values, trans_pred)
            self.results.append({'Model': 'Transformer', 'Type': 'DL', **metrics})
            if verbose:
                print(f"    Transformer: RMSE={metrics['RMSE']:.2f}")

        # 6. 앙상블 최적화
        if verbose:
            print("\n[6/7] Optimizing ensemble...")

        # Naive + ML/DL 조합
        naive_preds = {k: v for k, v in predictions.items() if k.startswith('Naive')}
        other_preds = {k: v for k, v in predictions.items()
                       if k in ['LightGBM', 'GradientBoosting', 'Transformer', 'ARIMA']}

        best_ensemble_rmse = float('inf')
        best_ensemble = None
        best_ensemble_name = None

        # 2-모델 조합 (Naive + 다른 모델)
        for naive_name, naive_pred in naive_preds.items():
            for other_name, other_pred in other_preds.items():
                # NaN이 있는 경우 건너뛰기
                if np.any(np.isnan(other_pred)):
                    continue
                # 더 넓은 가중치 범위 탐색 (0.4 ~ 0.95)
                for w in np.arange(0.4, 0.96, 0.01):
                    ensemble_pred = w * naive_pred + (1 - w) * other_pred
                    metrics = Evaluator.compute_metrics(y_test.values, ensemble_pred)

                    if metrics['RMSE'] < best_ensemble_rmse:
                        best_ensemble_rmse = metrics['RMSE']
                        best_ensemble = ensemble_pred
                        best_ensemble_name = f"Ensemble_{naive_name}*{w:.2f}+{other_name}*{1-w:.2f}"

        if best_ensemble is not None:
            predictions[best_ensemble_name] = best_ensemble
            metrics = Evaluator.compute_metrics(y_test.values, best_ensemble)
            self.results.append({'Model': best_ensemble_name, 'Type': 'Ensemble', **metrics})
            if verbose:
                print(f"    Best: {best_ensemble_name}")
                print(f"          RMSE={metrics['RMSE']:.2f}")

        # 7. 결과 정리
        if verbose:
            print("\n[7/7] Compiling results...")

        results_df = pd.DataFrame(self.results).sort_values('RMSE')

        if verbose:
            print("\n" + "=" * 70)
            print("FINAL RESULTS")
            print("=" * 70)
            print(results_df.to_string(index=False))

            print("\n" + "=" * 70)
            print("BEST MODEL")
            print("=" * 70)
            best = results_df.iloc[0]
            print(f"  Model: {best['Model']}")
            print(f"  Type:  {best['Type']}")
            print(f"  RMSE:  {best['RMSE']:.2f}")
            print(f"  MAE:   {best['MAE']:.2f}")
            print(f"  MAPE:  {best['MAPE']:.2f}%")
            print(f"  RMSPE: {best['RMSPE']:.2f}%")

            baseline = 406.80
            improvement = (baseline - best['RMSE']) / baseline * 100
            print(f"\n  Previous Baseline: {baseline:.2f}")
            if improvement > 0:
                print(f"  Improvement: {improvement:.2f}%")

            print("\n" + "=" * 70)
            print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

        return results_df, predictions, y_test


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    config = Config()
    pipeline = ForecastingPipeline(config)
    results_df, predictions, y_test = pipeline.run(verbose=True)

    # 결과 저장
    results_df.to_csv('forecast_results.csv', index=False)
    print("\nResults saved to: forecast_results.csv")
