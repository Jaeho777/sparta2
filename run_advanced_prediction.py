"""
Advanced Nickel Price Prediction - Main Runner
================================================
고급 피처 엔지니어링 + Optuna 최적화 + 딥러닝 모델 통합 실행
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import os

# 현재 디렉토리로 이동
os.chdir('/home/user/sparta2')

print("=" * 70)
print("Advanced Nickel Price Prediction Pipeline")
print("=" * 70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# 1. ML 파이프라인 실행
# =============================================================================

print("\n[PHASE 1] Machine Learning Pipeline with Advanced Features")
print("-" * 70)

from advanced_prediction import run_pipeline, load_data, filter_cols, create_advanced_features, CONFIG

# Optuna 사용 여부 확인
try:
    import optuna
    USE_OPTUNA = True
    print("Optuna available - using Bayesian optimization")
except ImportError:
    USE_OPTUNA = False
    print("Optuna not available - using default hyperparameters")

# ML 파이프라인 실행
ml_results = run_pipeline(use_optuna=USE_OPTUNA, verbose=True)

# =============================================================================
# 2. 딥러닝 파이프라인 실행
# =============================================================================

print("\n\n[PHASE 2] Deep Learning Pipeline")
print("-" * 70)

try:
    import torch
    PYTORCH_AVAILABLE = True
    print(f"PyTorch available - device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available - skipping deep learning models")

if PYTORCH_AVAILABLE:
    from advanced_dl_models import run_dl_pipeline
    from sklearn.preprocessing import RobustScaler

    # 데이터 준비
    df = load_data(CONFIG['data_file'])
    df = filter_cols(df, CONFIG['target_col'])
    df = create_advanced_features(df, CONFIG['target_col'], CONFIG['lag_periods'], CONFIG['rolling_windows'])

    y = df[CONFIG['target_col']]
    X = df.drop(columns=[CONFIG['target_col']]).shift(1)
    X = X.iloc[1:]
    y = y.iloc[1:]

    val_start = pd.to_datetime(CONFIG['val_start'])
    test_start = pd.to_datetime(CONFIG['test_start'])

    train_mask = X.index < val_start
    val_mask = (X.index >= val_start) & (X.index < test_start)
    test_mask = X.index >= test_start

    X_train, y_train = X[train_mask].ffill().bfill().fillna(0), y[train_mask]
    X_val, y_val = X[val_mask].ffill().bfill().fillna(0), y[val_mask]
    X_test, y_test = X[test_mask].ffill().bfill().fillna(0), y[test_mask]

    # 딥러닝 파이프라인 실행
    dl_results = run_dl_pipeline(
        X_train.values, y_train, X_val.values, y_val, X_test.values, y_test,
        seq_len=24, batch_size=32, epochs=100, verbose=True
    )

# =============================================================================
# 3. 결과 통합
# =============================================================================

print("\n\n[PHASE 3] Final Results Integration")
print("-" * 70)

# ML 결과
ml_test_df = ml_results['test_results'].copy()
ml_test_df['Type'] = 'ML'

# 딥러닝 결과 통합
if PYTORCH_AVAILABLE and len(dl_results['test_results']) > 0:
    dl_test_df = dl_results['test_results'].copy()
    dl_test_df['Type'] = 'DL'

    # RMSPE 추가 (없으면)
    if 'RMSPE' not in dl_test_df.columns:
        dl_test_df['RMSPE'] = np.nan

    # 결과 통합
    all_results = pd.concat([ml_test_df, dl_test_df], ignore_index=True)
else:
    all_results = ml_test_df

# 정렬
all_results = all_results.sort_values('RMSE')

# =============================================================================
# 4. 최종 결과 출력
# =============================================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS - All Models Ranked by RMSE")
print("=" * 70)
print(all_results.head(20).to_string(index=False))

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
best = all_results.iloc[0]
print(f"  Model: {best['Model']}")
print(f"  Type:  {best['Type']}")
print(f"  RMSE:  {best['RMSE']:.2f}")
print(f"  MAE:   {best['MAE']:.2f}")
print(f"  MAPE:  {best['MAPE']:.2f}%")

# 현재 베이스라인 대비 개선율
baseline_rmse = 406.80  # 기존 최고 성능 (Hybrid Naive*0.8 + GB*0.2)
improvement = (baseline_rmse - best['RMSE']) / baseline_rmse * 100

print(f"\n  Baseline RMSE: {baseline_rmse:.2f}")
if improvement > 0:
    print(f"  Improvement:   {improvement:.2f}% better")
else:
    print(f"  Difference:    {-improvement:.2f}% worse")

# =============================================================================
# 5. 결과 저장
# =============================================================================

print("\n\n[PHASE 4] Saving Results")
print("-" * 70)

# 전체 결과 저장
all_results.to_csv('advanced_all_results.csv', index=False)
print("  Saved: advanced_all_results.csv")

# ML 결과 상세
ml_test_df.to_csv('advanced_ml_results.csv', index=False)
print("  Saved: advanced_ml_results.csv")

# 검증 결과
ml_results['val_results'].to_csv('advanced_val_results.csv', index=False)
print("  Saved: advanced_val_results.csv")

# 최적 모델 정보
with open('advanced_best_model.txt', 'w') as f:
    f.write("Advanced Prediction Best Model\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: {best['Model']}\n")
    f.write(f"Type:  {best['Type']}\n")
    f.write(f"RMSE:  {best['RMSE']:.2f}\n")
    f.write(f"MAE:   {best['MAE']:.2f}\n")
    f.write(f"MAPE:  {best['MAPE']:.2f}%\n\n")
    f.write(f"Baseline RMSE: {baseline_rmse:.2f}\n")
    f.write(f"Improvement: {improvement:.2f}%\n")
print("  Saved: advanced_best_model.txt")

# 앙상블 가중치
if ml_results.get('ml_weights'):
    with open('advanced_ensemble_weights.txt', 'w') as f:
        f.write("ML Ensemble Weights\n")
        f.write("=" * 50 + "\n\n")
        for name, weight in ml_results['ml_weights'].items():
            if weight > 0.01:
                f.write(f"{name}: {weight:.4f}\n")
    print("  Saved: advanced_ensemble_weights.txt")

print("\n" + "=" * 70)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
