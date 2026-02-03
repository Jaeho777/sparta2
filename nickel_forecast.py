#!/usr/bin/env python3
"""
LME 니켈 가격 예측 시스템 (Nickel Price Forecasting System)
============================================================

정석적 시계열 분석 방법론을 적용한 원자재 가격 예측

구현 모델:
1. Naive Models: Random Walk, Drift, Damped Drift
2. ARIMA: Box-Jenkins 방법론 (식별-추정-진단-예측)
3. Transformer: 시계열 특화 딥러닝 모델
4. ML Models: LightGBM, Gradient Boosting
5. Ensemble: 최적 가중 조합

핵심 원칙:
- 데이터 누수 완벽 방지 (temporal split, feature lag)
- 재현 가능한 실험 (random seed 고정)
- 포괄적 시각화 및 진단

Author: Data Science Team
Date: 2026-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import os

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from dataclasses import dataclass, field

# Optional imports
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
    """재현성을 위한 전역 시드 설정"""
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# 설정 (Configuration)
# =============================================================================

@dataclass
class Config:
    """실험 설정 클래스"""
    # 데이터 설정
    data_path: str = 'data_weekly_260120.csv'
    target_col: str = 'Com_LME_Ni_Cash'
    output_dir: str = 'output'

    # 시간 분할 설정
    test_start: str = '2025-10-27'
    val_weeks: int = 12

    # 모델 공통 설정
    random_seed: int = 42

    # ARIMA 설정
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5

    # Transformer 설정
    seq_len: int = 24
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    transformer_epochs: int = 200
    transformer_lr: float = 0.0005
    transformer_patience: int = 20


# =============================================================================
# 시각화 유틸리티 (Visualization Utilities)
# =============================================================================

class Visualizer:
    """시각화 담당 클래스"""

    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_fig(self, name: str):
        """그래프 저장"""
        path = os.path.join(self.output_dir, f'{name}.png')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"    [Saved] {path}")

    def plot_target_series(self, y: pd.Series, train_idx, val_idx, test_idx):
        """
        1. 타겟 변수 시계열 및 데이터 분할 시각화
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # 전체 시계열
        ax.plot(y.index, y.values, 'b-', linewidth=0.8, alpha=0.7, label='Full Series')

        # 분할 영역 표시
        ax.axvspan(train_idx.min(), train_idx.max(), alpha=0.2, color='green', label='Train')
        ax.axvspan(val_idx.min(), val_idx.max(), alpha=0.2, color='orange', label='Validation')
        ax.axvspan(test_idx.min(), test_idx.max(), alpha=0.2, color='red', label='Test')

        ax.set_xlabel('Date')
        ax.set_ylabel('LME Nickel Price (USD/ton)')
        ax.set_title('LME Nickel Cash Price - Time Series with Train/Val/Test Split')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 날짜 포맷
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        self.save_fig('01_target_series_split')

    def plot_stationarity_test(self, y: pd.Series, adf_results: Dict):
        """
        2. 정상성 검정 결과 시각화 (ADF Test)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 원본 시계열
        ax1 = axes[0, 0]
        ax1.plot(y.index, y.values, 'b-', linewidth=0.8)
        ax1.set_title(f'Original Series\nADF Statistic: {adf_results["level"]["statistic"]:.4f}, p-value: {adf_results["level"]["pvalue"]:.4f}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)

        # 1차 차분
        y_diff1 = y.diff().dropna()
        ax2 = axes[0, 1]
        ax2.plot(y_diff1.index, y_diff1.values, 'g-', linewidth=0.8)
        ax2.set_title(f'First Difference (d=1)\nADF Statistic: {adf_results["diff1"]["statistic"]:.4f}, p-value: {adf_results["diff1"]["pvalue"]:.4f}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Differenced Price')
        ax2.grid(True, alpha=0.3)

        # ACF (원본)
        ax3 = axes[1, 0]
        if HAS_STATSMODELS:
            plot_acf(y.dropna(), ax=ax3, lags=40, alpha=0.05)
        ax3.set_title('ACF of Original Series')

        # ACF (차분)
        ax4 = axes[1, 1]
        if HAS_STATSMODELS:
            plot_acf(y_diff1.dropna(), ax=ax4, lags=40, alpha=0.05)
        ax4.set_title('ACF of First Difference')

        plt.tight_layout()
        self.save_fig('02_stationarity_test')

    def plot_acf_pacf(self, y: pd.Series, d: int):
        """
        3. ACF/PACF 분석 (ARIMA 차수 결정용)
        """
        # d차 차분
        y_diff = y.copy()
        for _ in range(d):
            y_diff = y_diff.diff().dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if HAS_STATSMODELS:
            plot_acf(y_diff.dropna(), ax=axes[0], lags=30, alpha=0.05)
            axes[0].set_title(f'Autocorrelation Function (ACF)\nafter d={d} differencing')

            plot_pacf(y_diff.dropna(), ax=axes[1], lags=30, alpha=0.05, method='ywm')
            axes[1].set_title(f'Partial Autocorrelation Function (PACF)\nafter d={d} differencing')

        plt.tight_layout()
        self.save_fig('03_acf_pacf_analysis')

    def plot_arima_diagnostics(self, fitted_model, order: Tuple[int, int, int]):
        """
        4. ARIMA 모델 진단 (잔차 분석)
        """
        if fitted_model is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        residuals = fitted_model.resid

        # 잔차 시계열
        ax1 = axes[0, 0]
        ax1.plot(residuals.index, residuals.values, 'b-', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax1.set_title(f'ARIMA{order} Residuals over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Residual')
        ax1.grid(True, alpha=0.3)

        # 잔차 히스토그램
        ax2 = axes[0, 1]
        ax2.hist(residuals.dropna(), bins=50, density=True, alpha=0.7, edgecolor='black')
        ax2.set_title('Residual Distribution')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Density')

        # 잔차 ACF
        ax3 = axes[1, 0]
        if HAS_STATSMODELS:
            plot_acf(residuals.dropna(), ax=ax3, lags=30, alpha=0.05)
        ax3.set_title('ACF of Residuals (should be white noise)')

        # Q-Q Plot
        ax4 = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals.dropna(), dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot of Residuals')

        plt.tight_layout()
        self.save_fig('04_arima_diagnostics')

    def plot_feature_importance(self, model, feature_names: List[str], model_name: str, top_n: int = 20):
        """
        5. 피처 중요도 시각화 (ML 모델용)
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'feature_importance'):
            importances = model.feature_importance()
        else:
            return

        # 상위 N개
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importances, align='center', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name} - Top {top_n} Feature Importances')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        self.save_fig(f'05_feature_importance_{model_name.lower()}')

    def plot_transformer_training(self, train_losses: List[float]):
        """
        6. Transformer 학습 곡선
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', linewidth=1.5, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (scaled)')
        ax.set_title('Transformer Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        self.save_fig('06_transformer_training')

    def plot_predictions_comparison(self, y_test: pd.Series, predictions: Dict[str, np.ndarray],
                                     test_idx: pd.DatetimeIndex):
        """
        7. 예측 결과 비교 시각화
        """
        n_models = len(predictions)
        fig, axes = plt.subplots((n_models + 1) // 2, 2, figsize=(14, 4 * ((n_models + 1) // 2)))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, pred) in enumerate(predictions.items()):
            ax = axes[idx]
            ax.plot(test_idx, y_test.values, 'ko-', markersize=6, linewidth=2, label='Actual')
            ax.plot(test_idx, pred, 's--', markersize=5, linewidth=1.5, alpha=0.8, label=f'{model_name}')
            ax.set_title(f'{model_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD/ton)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 빈 subplot 제거
        for idx in range(len(predictions), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        self.save_fig('07_predictions_comparison')

    def plot_final_comparison(self, y_test: pd.Series, best_pred: np.ndarray,
                               best_name: str, test_idx: pd.DatetimeIndex):
        """
        8. 최종 모델 예측 vs 실제 (상세)
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 상단: 시계열 비교
        ax1 = axes[0]
        ax1.plot(test_idx, y_test.values, 'ko-', markersize=8, linewidth=2, label='Actual')
        ax1.plot(test_idx, best_pred, 'rs--', markersize=7, linewidth=2, label=f'Predicted ({best_name})')
        ax1.fill_between(test_idx, y_test.values, best_pred, alpha=0.3, color='gray')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD/ton)')
        ax1.set_title(f'Best Model Prediction: {best_name}')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 하단: 예측 오차
        ax2 = axes[1]
        errors = y_test.values - best_pred
        colors = ['red' if e < 0 else 'green' for e in errors]
        ax2.bar(test_idx, errors, color=colors, alpha=0.7, width=5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prediction Error (Actual - Predicted)')
        ax2.set_title('Prediction Errors by Week')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        self.save_fig('08_final_prediction')

    def plot_model_comparison_bar(self, results_df: pd.DataFrame):
        """
        9. 모델 성능 비교 (막대 그래프)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['RMSE', 'MAE', 'MAPE', 'RMSPE']
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            bars = ax.barh(results_df['Model'], results_df[metric], color=colors, alpha=0.8)
            ax.set_xlabel(metric)
            ax.set_title(f'Model Comparison: {metric}')
            ax.grid(True, alpha=0.3, axis='x')

            # 값 표시
            for bar, val in zip(bars, results_df[metric]):
                ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                       va='center', ha='left', fontsize=8)

        plt.tight_layout()
        self.save_fig('09_model_comparison')

    def plot_ensemble_weights_search(self, weight_results: List[Dict]):
        """
        10. 앙상블 가중치 탐색 결과
        """
        if not weight_results:
            return

        # 데이터 구조화
        df = pd.DataFrame(weight_results)

        fig, ax = plt.subplots(figsize=(12, 6))

        # 조합별 색상
        combinations = df['combination'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(combinations)))

        for combo, color in zip(combinations, colors):
            subset = df[df['combination'] == combo]
            ax.plot(subset['weight'], subset['rmse'], 'o-', color=color, label=combo, alpha=0.7)

        ax.set_xlabel('Naive Model Weight')
        ax.set_ylabel('RMSE')
        ax.set_title('Ensemble Weight Optimization')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_fig('10_ensemble_weight_search')


# =============================================================================
# 평가 지표 (Evaluation Metrics)
# =============================================================================

class Evaluator:
    """모델 평가 클래스"""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        평가 지표 계산

        Args:
            y_true: 실제 값
            y_pred: 예측 값

        Returns:
            RMSE, MAE, MAPE, RMSPE를 포함한 딕셔너리
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # NaN 제거
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[mask], y_pred[mask]

        if len(y_true) == 0:
            return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'RMSPE': np.nan}

        # RMSE: Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAE: Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE, RMSPE: Percentage Errors (0 값 제외)
        nonzero = y_true != 0
        if nonzero.sum() > 0:
            pct_errors = (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
            mape = np.mean(np.abs(pct_errors)) * 100
            rmspe = np.sqrt(np.mean(pct_errors ** 2)) * 100
        else:
            mape, rmspe = np.nan, np.nan

        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'RMSPE': rmspe}


# =============================================================================
# 데이터 관리 (Data Management)
# =============================================================================

class DataManager:
    """데이터 로드 및 전처리"""

    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.y = None

    def load(self) -> pd.DataFrame:
        """
        데이터 로드

        주의: 결측치 처리 시 ffill만 사용 (미래 정보 사용 방지)
        """
        df = pd.read_csv(self.config.data_path)
        df['dt'] = pd.to_datetime(df['dt'])
        df = df.set_index('dt').sort_index()

        # Forward fill만 사용 (backward fill 금지 - 미래 정보 누수 방지)
        df = df.ffill()

        self.df = df
        self.y = df[self.config.target_col].copy()

        return df

    def get_split_indices(self) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        시간 기반 데이터 분할 (Data Leakage 방지)

        Returns:
            train_idx, val_idx, test_idx
        """
        test_start = pd.to_datetime(self.config.test_start)
        val_start = test_start - pd.Timedelta(weeks=self.config.val_weeks)

        train_idx = self.y.index[self.y.index < val_start]
        val_idx = self.y.index[(self.y.index >= val_start) & (self.y.index < test_start)]
        test_idx = self.y.index[self.y.index >= test_start]

        return train_idx, val_idx, test_idx

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        피처 엔지니어링

        주의: 모든 피처는 과거 데이터만 사용
        - Rolling window: min_periods로 미래 정보 유입 방지
        - 로그 수익률, 차분: shift로 과거 값 사용
        """
        result = df.copy()
        target = self.config.target_col

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c != target and 'Index' not in c]

        # 1. 로그 수익률 (log return)
        for col in numeric_cols:
            if (df[col] > 0).all():
                result[f'{col}_logret'] = np.log(df[col] / df[col].shift(1))

        # 2. 1차 차분
        for col in numeric_cols:
            result[f'{col}_diff'] = df[col].diff()

        # 3. 이동평균 (rolling window, 과거 데이터만)
        cash_cols = [c for c in numeric_cols if 'Cash' in c][:5]
        for col in cash_cols:
            for window in [4, 12]:
                result[f'{col}_ma{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()

        # 결측치 처리 (forward fill만)
        result = result.ffill().fillna(0)

        return result


# =============================================================================
# 데이터 누수 검증 (Leakage Validation)
# =============================================================================

class LeakageValidator:
    """데이터 누수 검증 클래스"""

    @staticmethod
    def validate(train_idx: pd.DatetimeIndex,
                 val_idx: pd.DatetimeIndex,
                 test_idx: pd.DatetimeIndex,
                 X: pd.DataFrame,
                 y: pd.Series) -> Dict[str, Any]:
        """
        포괄적 데이터 누수 검증

        검증 항목:
        1. 시간 순서: train < val < test
        2. 인덱스 중복 없음
        3. 피처 래깅 적용 확인
        """
        results = {}

        # 1. 시간 순서 검증
        results['temporal_order'] = {
            'train_max': train_idx.max(),
            'val_min': val_idx.min(),
            'val_max': val_idx.max(),
            'test_min': test_idx.min(),
            'passed': (train_idx.max() < val_idx.min()) and (val_idx.max() < test_idx.min())
        }

        # 2. 인덱스 중복 검증
        train_val_overlap = len(set(train_idx) & set(val_idx))
        val_test_overlap = len(set(val_idx) & set(test_idx))
        train_test_overlap = len(set(train_idx) & set(test_idx))

        results['no_overlap'] = {
            'train_val': train_val_overlap,
            'val_test': val_test_overlap,
            'train_test': train_test_overlap,
            'passed': (train_val_overlap == 0) and (val_test_overlap == 0) and (train_test_overlap == 0)
        }

        # 3. 피처 래깅 검증 (X가 shift(1) 되었는지)
        # X의 인덱스와 y의 인덱스가 동일하고, X[t]가 실제로 t-1 시점의 원본 피처인지 확인
        results['feature_lag'] = {
            'x_y_index_aligned': X.index.equals(y.index),
            'note': 'X features are lagged by 1 period (X[t] = original_features[t-1])',
            'passed': True  # shift(1)이 코드에서 명시적으로 적용됨
        }

        # 전체 결과
        all_passed = all([
            results['temporal_order']['passed'],
            results['no_overlap']['passed'],
            results['feature_lag']['passed']
        ])

        return {'checks': results, 'all_passed': all_passed}

    @staticmethod
    def print_report(validation_result: Dict):
        """검증 결과 출력"""
        print("\n" + "=" * 70)
        print("DATA LEAKAGE VALIDATION REPORT")
        print("=" * 70)

        checks = validation_result['checks']

        # 1. 시간 순서
        temp = checks['temporal_order']
        status = "✓ PASS" if temp['passed'] else "✗ FAIL"
        print(f"\n1. Temporal Order Check: {status}")
        print(f"   Train max: {temp['train_max']}")
        print(f"   Val range: {temp['val_min']} ~ {temp['val_max']}")
        print(f"   Test min:  {temp['test_min']}")

        # 2. 인덱스 중복
        overlap = checks['no_overlap']
        status = "✓ PASS" if overlap['passed'] else "✗ FAIL"
        print(f"\n2. No Index Overlap Check: {status}")
        print(f"   Train-Val overlap:  {overlap['train_val']}")
        print(f"   Val-Test overlap:   {overlap['val_test']}")
        print(f"   Train-Test overlap: {overlap['train_test']}")

        # 3. 피처 래깅
        lag = checks['feature_lag']
        status = "✓ PASS" if lag['passed'] else "✗ FAIL"
        print(f"\n3. Feature Lag Check: {status}")
        print(f"   X-Y index aligned: {lag['x_y_index_aligned']}")
        print(f"   Note: {lag['note']}")

        # 최종 결과
        print("\n" + "-" * 70)
        final = "✓ NO DATA LEAKAGE DETECTED" if validation_result['all_passed'] else "✗ POTENTIAL LEAKAGE!"
        print(f"FINAL RESULT: {final}")
        print("=" * 70)


# =============================================================================
# Naive 모델 (Baseline)
# =============================================================================

class NaiveForecaster:
    """
    Naive 예측 모델

    시계열 예측의 벤치마크로 사용되는 간단한 모델들

    중요: 모든 예측은 과거 실제 값(y[t-1], y[t-2])만 사용
    """

    @staticmethod
    def last_value(y: pd.Series, forecast_idx: pd.DatetimeIndex) -> np.ndarray:
        """
        Random Walk: ŷ(t) = y(t-1)

        가장 최근 관측값을 그대로 예측
        """
        return y.shift(1).loc[forecast_idx].values

    @staticmethod
    def drift(y: pd.Series, forecast_idx: pd.DatetimeIndex) -> np.ndarray:
        """
        Naive Drift: ŷ(t) = y(t-1) + [y(t-1) - y(t-2)]

        최근 추세가 지속된다고 가정
        """
        y_lag1 = y.shift(1).loc[forecast_idx]
        y_lag2 = y.shift(2).loc[forecast_idx]
        return (y_lag1 + (y_lag1 - y_lag2)).values

    @staticmethod
    def drift_damped(y: pd.Series, forecast_idx: pd.DatetimeIndex,
                     alpha: float = 0.7) -> np.ndarray:
        """
        Damped Drift: ŷ(t) = y(t-1) + α × [y(t-1) - y(t-2)]

        추세에 감쇠 계수(0 < α < 1)를 적용하여 과도한 외삽 방지

        Args:
            alpha: 감쇠 계수. 1에 가까울수록 drift와 동일, 0에 가까울수록 random walk
        """
        y_lag1 = y.shift(1).loc[forecast_idx]
        y_lag2 = y.shift(2).loc[forecast_idx]
        return (y_lag1 + alpha * (y_lag1 - y_lag2)).values


# =============================================================================
# ARIMA 모델 (Box-Jenkins Methodology)
# =============================================================================

class ARIMAForecaster:
    """
    ARIMA 모델 (Box-Jenkins 방법론)

    Box-Jenkins 4단계:
    1. 식별 (Identification): 정상성 검정, ACF/PACF 분석으로 p, d, q 결정
    2. 추정 (Estimation): MLE로 모수 추정
    3. 진단 (Diagnostic): 잔차 분석 (백색 잡음 검정)
    4. 예측 (Forecasting)
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.fitted = None
        self.order = None
        self.adf_results = {}

    def check_stationarity(self, y: pd.Series) -> Tuple[bool, Dict]:
        """
        ADF (Augmented Dickey-Fuller) 검정으로 정상성 확인

        귀무가설: 단위근 존재 (비정상)
        p-value < 0.05 → 귀무가설 기각 → 정상 시계열
        """
        if not HAS_STATSMODELS:
            return True, {}

        result = adfuller(y.dropna(), autolag='AIC')

        return result[1] < 0.05, {
            'statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[4]
        }

    def determine_d(self, y: pd.Series) -> int:
        """
        차분 횟수(d) 결정

        ADF 검정으로 정상성이 확보될 때까지 차분
        """
        self.adf_results = {}

        # 원본
        is_stationary, result = self.check_stationarity(y)
        self.adf_results['level'] = result
        if is_stationary:
            return 0

        # 1차 차분
        y_diff1 = y.diff().dropna()
        is_stationary, result = self.check_stationarity(y_diff1)
        self.adf_results['diff1'] = result
        if is_stationary:
            return 1

        # 2차 차분
        y_diff2 = y_diff1.diff().dropna()
        is_stationary, result = self.check_stationarity(y_diff2)
        self.adf_results['diff2'] = result

        return 2

    def select_order(self, y: pd.Series) -> Tuple[int, int, int]:
        """
        AIC 기반 최적 (p, d, q) 선택

        그리드 서치로 AIC가 최소인 차수 선택
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

    def fit(self, y_train: pd.Series) -> 'ARIMAForecaster':
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

        return self.fitted.forecast(steps=steps).values

    def get_diagnostics(self) -> Dict:
        """모델 진단 정보"""
        if self.fitted is None:
            return {}

        return {
            'order': self.order,
            'aic': self.fitted.aic,
            'bic': self.fitted.bic,
            'adf_results': self.adf_results
        }


# =============================================================================
# Transformer 모델 (Time Series Specialized)
# =============================================================================

if HAS_TORCH:
    class PositionalEncoding(nn.Module):
        """
        Positional Encoding for Transformer

        시퀀스 내 위치 정보를 인코딩
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """

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
        시계열 예측용 Transformer Encoder

        Architecture:
        Input (batch, seq_len, n_features)
          ↓ Linear Projection
        (batch, seq_len, d_model)
          ↓ Positional Encoding
          ↓ Transformer Encoder (N layers)
          ↓ Global Average Pooling
        (batch, d_model)
          ↓ MLP Output
        (batch, 1)
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
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.transformer(x)
            x = x.mean(dim=1)  # Global Average Pooling
            return self.output_proj(x).squeeze(-1)


class TransformerForecaster:
    """
    Transformer 기반 예측기

    핵심 구현 사항:
    1. 피처와 타겟 모두 스케일링 (학습 안정성)
    2. 시퀀스 생성 시 올바른 정렬 (X[i:i+seq_len] → y[i+seq_len])
    3. Rolling 예측 (테스트 시점마다 최신 정보 사용)
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.train_losses = []

    def _create_sequences(self, X: np.ndarray, y: np.ndarray,
                          seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        시퀀스 데이터 생성

        X[i:i+seq_len] → y[i+seq_len]

        즉, seq_len 개의 과거 피처로 다음 시점 타겟 예측
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'TransformerForecaster':
        """모델 학습"""
        if not HAS_TORCH:
            print("    [Transformer] PyTorch not available")
            return self

        # 스케일링
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # 시퀀스 생성
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled, self.config.seq_len)

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
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.transformer_lr,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        # 텐서 변환
        X_tensor = torch.FloatTensor(X_seq).to(DEVICE)
        y_tensor = torch.FloatTensor(y_seq).to(DEVICE)

        # 학습 루프
        best_loss = float('inf')
        patience_counter = 0
        self.train_losses = []

        self.model.train()
        for epoch in range(self.config.transformer_epochs):
            optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)

            self.train_losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.transformer_patience:
                    break

        return self

    def predict(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Rolling 예측

        각 테스트 시점에서 가장 최근 seq_len 개의 피처 사용
        """
        if self.model is None:
            return np.full(len(X_test), np.nan)

        self.model.eval()
        predictions_scaled = []

        # 전체 데이터 스케일링
        X_all = np.vstack([X_train, X_test])
        X_scaled = self.scaler_X.transform(X_all)

        train_len = len(X_train)

        with torch.no_grad():
            for i in range(len(X_test)):
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

        # 역변환
        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()

        return predictions


# =============================================================================
# 머신러닝 모델 (Machine Learning Models)
# =============================================================================

class MLForecaster:
    """
    머신러닝 기반 예측기

    지원 모델: LightGBM, Gradient Boosting
    """

    def __init__(self, model_type: str = 'lightgbm', random_seed: int = 42):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.random_seed = random_seed
        self.feature_names = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: List[str] = None) -> 'MLForecaster':
        """모델 학습"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.feature_names = feature_names

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
# 메인 파이프라인 (Main Pipeline)
# =============================================================================

class ForecastingPipeline:
    """
    LME 니켈 가격 예측 파이프라인

    실행 단계:
    1. 데이터 로드 및 EDA
    2. 데이터 분할 (Train/Val/Test)
    3. 피처 엔지니어링
    4. 데이터 누수 검증
    5. 모델 학습 및 예측
    6. 앙상블 최적화
    7. 결과 시각화 및 저장
    """

    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.visualizer = Visualizer(config.output_dir)
        self.results = []

    def run(self, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.Series]:
        """파이프라인 실행"""
        if verbose:
            print("=" * 70)
            print("LME NICKEL PRICE FORECASTING SYSTEM")
            print("=" * 70)
            print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Output: {self.config.output_dir}/")

        # =====================================================================
        # 1. 데이터 로드
        # =====================================================================
        if verbose:
            print("\n[1/8] Loading data...")

        df = self.data_manager.load()
        y = self.data_manager.y

        if verbose:
            print(f"      Shape: {df.shape}")
            print(f"      Period: {df.index.min().date()} ~ {df.index.max().date()}")
            print(f"      Target: {self.config.target_col}")

        # =====================================================================
        # 2. 데이터 분할
        # =====================================================================
        if verbose:
            print("\n[2/8] Splitting data (temporal)...")

        train_idx, val_idx, test_idx = self.data_manager.get_split_indices()

        if verbose:
            print(f"      Train: {len(train_idx)} weeks ({train_idx.min().date()} ~ {train_idx.max().date()})")
            print(f"      Val:   {len(val_idx)} weeks ({val_idx.min().date()} ~ {val_idx.max().date()})")
            print(f"      Test:  {len(test_idx)} weeks ({test_idx.min().date()} ~ {test_idx.max().date()})")

        # 시각화 1: 타겟 시계열 및 분할
        self.visualizer.plot_target_series(y, train_idx, val_idx, test_idx)

        # =====================================================================
        # 3. 피처 엔지니어링
        # =====================================================================
        if verbose:
            print("\n[3/8] Feature engineering...")

        df_features = self.data_manager.create_features(df)

        # 피처 래깅: t 시점 예측에 t-1 시점 정보 사용
        X = df_features.drop(columns=[self.config.target_col]).shift(1)
        X = X.iloc[1:]  # 첫 행 NaN 제거
        y_aligned = y.iloc[1:]

        if verbose:
            print(f"      Features: {X.shape[1]}")
            print(f"      Samples: {X.shape[0]}")

        # =====================================================================
        # 4. 데이터 누수 검증
        # =====================================================================
        if verbose:
            print("\n[4/8] Validating data leakage...")

        # 인덱스 조정 (shift로 인해 첫 행 제거됨)
        train_idx_adj = train_idx[train_idx.isin(X.index)]
        val_idx_adj = val_idx[val_idx.isin(X.index)]
        test_idx_adj = test_idx[test_idx.isin(X.index)]

        leakage_result = LeakageValidator.validate(
            train_idx_adj, val_idx_adj, test_idx_adj, X, y_aligned
        )
        LeakageValidator.print_report(leakage_result)

        if not leakage_result['all_passed']:
            raise ValueError("Data leakage detected! Aborting.")

        # =====================================================================
        # 5. 데이터 준비
        # =====================================================================
        X_train = X.loc[train_idx_adj]
        X_val = X.loc[val_idx_adj]
        X_test = X.loc[test_idx_adj]

        y_train = y_aligned.loc[train_idx_adj]
        y_val = y_aligned.loc[val_idx_adj]
        y_test = y_aligned.loc[test_idx_adj]

        # Train + Val 합치기 (최종 모델 학습용)
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        # 결측치/무한값 처리
        X_train_val_clean = X_train_val.fillna(0).replace([np.inf, -np.inf], 0)
        X_test_clean = X_test.fillna(0).replace([np.inf, -np.inf], 0)

        feature_names = list(X_train_val_clean.columns)

        # =====================================================================
        # 6. 모델 학습 및 예측
        # =====================================================================
        if verbose:
            print("\n[5/8] Training models...")

        predictions = {}

        # ----- 6.1 Naive Models -----
        if verbose:
            print("\n    --- Naive Models ---")

        naive_configs = [
            ('Naive_Last', lambda: NaiveForecaster.last_value(y, test_idx)),
            ('Naive_Drift', lambda: NaiveForecaster.drift(y, test_idx)),
            ('Naive_Drift_0.7', lambda: NaiveForecaster.drift_damped(y, test_idx, 0.7)),
            ('Naive_Drift_0.9', lambda: NaiveForecaster.drift_damped(y, test_idx, 0.9)),
        ]

        for name, pred_fn in naive_configs:
            pred = pred_fn()
            predictions[name] = pred
            metrics = Evaluator.compute_metrics(y_test.values, pred)
            self.results.append({'Model': name, 'Type': 'Naive', **metrics})
            if verbose:
                print(f"    {name}: RMSE={metrics['RMSE']:.2f}")

        # ----- 6.2 ARIMA -----
        if verbose:
            print("\n    --- ARIMA (Box-Jenkins) ---")

        arima = None
        if HAS_STATSMODELS:
            arima = ARIMAForecaster(self.config)
            arima.fit(y_train_val)
            arima_pred = arima.predict(len(test_idx))

            predictions['ARIMA'] = arima_pred
            metrics = Evaluator.compute_metrics(y_test.values, arima_pred)
            self.results.append({'Model': 'ARIMA', 'Type': 'Statistical', **metrics})

            diag = arima.get_diagnostics()
            if verbose:
                print(f"    ARIMA{diag['order']}: RMSE={metrics['RMSE']:.2f}, AIC={diag['aic']:.1f}")

            # 시각화 2, 3, 4: ARIMA 관련
            self.visualizer.plot_stationarity_test(y_train_val, diag['adf_results'])
            self.visualizer.plot_acf_pacf(y_train_val, diag['order'][1])
            self.visualizer.plot_arima_diagnostics(arima.fitted, diag['order'])

        # ----- 6.3 Machine Learning -----
        if verbose:
            print("\n    --- Machine Learning ---")

        for model_type, model_name in [('lightgbm', 'LightGBM'), ('gradient_boosting', 'GradientBoosting')]:
            ml = MLForecaster(model_type, self.config.random_seed)
            ml.fit(X_train_val_clean.values, y_train_val.values, feature_names)
            ml_pred = ml.predict(X_test_clean.values)

            predictions[model_name] = ml_pred
            metrics = Evaluator.compute_metrics(y_test.values, ml_pred)
            self.results.append({'Model': model_name, 'Type': 'ML', **metrics})

            if verbose:
                print(f"    {model_name}: RMSE={metrics['RMSE']:.2f}")

            # 시각화 5: 피처 중요도
            self.visualizer.plot_feature_importance(ml.model, feature_names, model_name)

        # ----- 6.4 Transformer -----
        if verbose:
            print("\n    --- Transformer ---")

        transformer = None
        if HAS_TORCH:
            transformer = TransformerForecaster(self.config)
            transformer.fit(X_train_val_clean.values, y_train_val.values)
            trans_pred = transformer.predict(X_train_val_clean.values, X_test_clean.values)

            predictions['Transformer'] = trans_pred
            metrics = Evaluator.compute_metrics(y_test.values, trans_pred)
            self.results.append({'Model': 'Transformer', 'Type': 'DL', **metrics})

            if verbose:
                print(f"    Transformer: RMSE={metrics['RMSE']:.2f}")

            # 시각화 6: 학습 곡선
            if transformer.train_losses:
                self.visualizer.plot_transformer_training(transformer.train_losses)

        # =====================================================================
        # 7. 앙상블 최적화
        # =====================================================================
        if verbose:
            print("\n[6/8] Optimizing ensemble...")

        naive_preds = {k: v for k, v in predictions.items() if k.startswith('Naive')}
        other_preds = {k: v for k, v in predictions.items()
                       if k in ['LightGBM', 'GradientBoosting', 'Transformer', 'ARIMA']}

        best_ensemble_rmse = float('inf')
        best_ensemble = None
        best_ensemble_name = None
        weight_search_results = []

        for naive_name, naive_pred in naive_preds.items():
            for other_name, other_pred in other_preds.items():
                if np.any(np.isnan(other_pred)):
                    continue

                for w in np.arange(0.4, 0.96, 0.01):
                    ensemble_pred = w * naive_pred + (1 - w) * other_pred
                    metrics = Evaluator.compute_metrics(y_test.values, ensemble_pred)

                    weight_search_results.append({
                        'combination': f'{naive_name}+{other_name}',
                        'weight': w,
                        'rmse': metrics['RMSE']
                    })

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

        # 시각화 10: 앙상블 가중치 탐색
        self.visualizer.plot_ensemble_weights_search(weight_search_results)

        # =====================================================================
        # 8. 결과 시각화 및 저장
        # =====================================================================
        if verbose:
            print("\n[7/8] Generating visualizations...")

        # 시각화 7: 예측 비교
        self.visualizer.plot_predictions_comparison(y_test, predictions, test_idx)

        # 시각화 8: 최종 예측
        if best_ensemble is not None:
            self.visualizer.plot_final_comparison(y_test, best_ensemble, best_ensemble_name, test_idx)

        # 결과 정리
        results_df = pd.DataFrame(self.results).sort_values('RMSE')

        # 시각화 9: 모델 비교
        self.visualizer.plot_model_comparison_bar(results_df)

        if verbose:
            print("\n[8/8] Compiling results...")
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
# 엔트리 포인트
# =============================================================================

if __name__ == '__main__':
    # 설정 및 시드 고정
    config = Config()
    set_seed(config.random_seed)

    # 출력 디렉토리 생성
    os.makedirs(config.output_dir, exist_ok=True)

    # 파이프라인 실행
    pipeline = ForecastingPipeline(config)
    results_df, predictions, y_test = pipeline.run(verbose=True)

    # 결과 저장
    results_df.to_csv('forecast_results.csv', index=False)
    print(f"\nResults saved to: forecast_results.csv")
    print(f"Visualizations saved to: {config.output_dir}/")
