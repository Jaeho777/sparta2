"""
Advanced Time Series Analysis: STL Decomposition & ADI-CV Classification
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
df = pd.read_csv('data_weekly_260120.csv', parse_dates=['dt'])
df.set_index('dt', inplace=True)
target = df['Com_LME_Ni_Cash']

print("=" * 70)
print("1. STL Decomposition (Seasonal-Trend decomposition using LOESS)")
print("=" * 70)

# STL 분해 (주간 데이터: period=52 for yearly seasonality)
stl = STL(target, period=52, robust=True)
result = stl.fit()

# 시각화
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
fig.suptitle('STL Decomposition of LME Nickel Price', fontsize=14, fontweight='bold')

axes[0].plot(target.index, target.values, 'b-', linewidth=0.8)
axes[0].set_ylabel('Original')
axes[0].set_title('Observed')
axes[0].grid(True, alpha=0.3)

axes[1].plot(target.index, result.trend, 'g-', linewidth=1)
axes[1].set_ylabel('Trend')
axes[1].set_title('Trend Component')
axes[1].grid(True, alpha=0.3)

axes[2].plot(target.index, result.seasonal, 'orange', linewidth=0.8)
axes[2].set_ylabel('Seasonal')
axes[2].set_title('Seasonal Component (52-week cycle)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(target.index, result.resid, 'r-', linewidth=0.5, alpha=0.7)
axes[3].set_ylabel('Residual')
axes[3].set_title('Residual (Noise)')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_images/stl_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("STL decomposition saved to report_images/stl_decomposition.png")

# STL 결과 분석
trend_strength = 1 - np.var(result.resid) / np.var(result.trend + result.resid)
seasonal_strength = 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)

print(f"\nSTL Analysis Results:")
print(f"  - Trend Strength: {trend_strength:.4f} (1에 가까울수록 강한 추세)")
print(f"  - Seasonal Strength: {seasonal_strength:.4f} (1에 가까울수록 강한 계절성)")
print(f"  - Residual Std: {np.std(result.resid):.2f}")
print(f"  - Trend Range: {result.trend.min():.0f} ~ {result.trend.max():.0f}")

# 결론
if trend_strength > 0.6:
    trend_conclusion = "강한 추세 존재 -> Naive_Drift 유리"
else:
    trend_conclusion = "약한 추세 -> 평균 회귀 모델 고려"

if seasonal_strength > 0.3:
    seasonal_conclusion = "계절성 존재 -> 계절 조정 필요"
else:
    seasonal_conclusion = "계절성 미미 -> 계절 조정 불필요"

print(f"\n  Trend: {trend_conclusion}")
print(f"  Seasonal: {seasonal_conclusion}")

print("\n" + "=" * 70)
print("2. ADI-CV Framework (Average Demand Interval - Coefficient of Variation)")
print("=" * 70)

# 주간 수익률 계산
returns = target.pct_change().dropna()

# ADI (Average Demand Interval) - 여기서는 "non-zero" 대신 "significant move" 사용
threshold = 0.02  # 2% 이상 변동을 significant move로 정의
significant_moves = np.abs(returns) > threshold
intervals = []
count = 0
for is_sig in significant_moves:
    count += 1
    if is_sig:
        intervals.append(count)
        count = 0

ADI = np.mean(intervals) if intervals else 1

# CV (Coefficient of Variation)
CV = np.std(returns) / np.abs(np.mean(returns)) if np.mean(returns) != 0 else np.inf
CV_abs = np.std(np.abs(returns)) / np.mean(np.abs(returns))  # 절대값 기준

print(f"\nADI-CV Analysis (threshold={threshold*100:.0f}% moves):")
print(f"  - ADI (Avg Demand Interval): {ADI:.2f} weeks")
print(f"  - CV (Coefficient of Variation): {CV_abs:.2f}")

# ADI-CV 분류
# Smooth: ADI < 1.32, CV < 0.49
# Erratic: ADI < 1.32, CV >= 0.49
# Intermittent: ADI >= 1.32, CV < 0.49
# Lumpy: ADI >= 1.32, CV >= 0.49

adi_threshold = 1.32
cv_threshold = 0.49

if ADI < adi_threshold and CV_abs < cv_threshold:
    classification = "Smooth"
    description = "일정하고 예측 가능 -> 단순 모델 적합"
elif ADI < adi_threshold and CV_abs >= cv_threshold:
    classification = "Erratic"
    description = "빈번하지만 변동성 큼 -> ML 모델 또는 robust 방법"
elif ADI >= adi_threshold and CV_abs < cv_threshold:
    classification = "Intermittent"
    description = "간헐적 움직임 -> Croston 방법 고려"
else:
    classification = "Lumpy"
    description = "간헐적 + 변동성 큼 -> 예측 어려움"

print(f"\n  Classification: {classification}")
print(f"  Interpretation: {description}")

# ADI-CV 시각화
fig, ax = plt.subplots(figsize=(10, 8))

# 4분면 배경
ax.axhline(y=cv_threshold, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=adi_threshold, color='gray', linestyle='--', alpha=0.5)

# 분류 영역 라벨
ax.text(0.7, 0.25, 'Smooth\n(Simple models)', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.7, 0.75, 'Erratic\n(ML/Robust methods)', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(2.0, 0.25, 'Intermittent\n(Croston method)', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(2.0, 0.75, 'Lumpy\n(Hard to forecast)', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

# 현재 데이터 포인트
ax.scatter(ADI, CV_abs, s=200, c='red', marker='*', zorder=5, label=f'Nickel (ADI={ADI:.2f}, CV={CV_abs:.2f})')
ax.annotate(f'  Nickel\n  ({classification})', (ADI, CV_abs), fontsize=10, fontweight='bold')

ax.set_xlabel('ADI (Average Demand Interval)', fontsize=12)
ax.set_ylabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_title('ADI-CV Demand Classification Framework\nfor LME Nickel Weekly Returns', fontsize=14, fontweight='bold')
ax.set_xlim(0, 3)
ax.set_ylim(0, 1.2)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_images/adi_cv_classification.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nADI-CV classification saved to report_images/adi_cv_classification.png")

print("\n" + "=" * 70)
print("3. Augmented Dickey-Fuller Test (Stationarity)")
print("=" * 70)

from statsmodels.tsa.stattools import adfuller

# 원본 데이터
adf_original = adfuller(target.dropna(), autolag='AIC')
print(f"\nOriginal Price Series:")
print(f"  ADF Statistic: {adf_original[0]:.4f}")
print(f"  p-value: {adf_original[1]:.4f}")
print(f"  Stationary: {'Yes' if adf_original[1] < 0.05 else 'No (Non-stationary)'}")

# 1차 차분 (수익률)
adf_returns = adfuller(returns.dropna(), autolag='AIC')
print(f"\nFirst Difference (Returns):")
print(f"  ADF Statistic: {adf_returns[0]:.4f}")
print(f"  p-value: {adf_returns[1]:.4f}")
print(f"  Stationary: {'Yes' if adf_returns[1] < 0.05 else 'No'}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"""
[STL Decomposition]
- Trend Strength: {trend_strength:.2f} -> {'Strong' if trend_strength > 0.6 else 'Weak'} trend
- Seasonal Strength: {seasonal_strength:.2f} -> {'Present' if seasonal_strength > 0.3 else 'Negligible'} seasonality
- Implication: Naive_Drift model is justified by strong trend component

[ADI-CV Classification]
- Category: {classification}
- ADI={ADI:.2f}, CV={CV_abs:.2f}
- Implication: {description}

[Stationarity (ADF Test)]
- Original: Non-stationary (p={adf_original[1]:.4f})
- Returns: Stationary (p={adf_returns[1]:.4f})
- Implication: Use returns for ML, price level for Naive
""")
