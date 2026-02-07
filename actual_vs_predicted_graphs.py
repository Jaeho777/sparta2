# =====================================================
# Actual vs Predicted 시각화 코드
# sparta2_advanced.ipynb에 추가할 셀
# =====================================================

# Cell 1: 각 모델별 Actual vs Predicted 그래프
# -------------------------------------------------
"""
# Actual vs Predicted 시각화 (각 모델별)
import math

# 모델 개수
n_models = len(all_preds.columns)
n_cols = 3
n_rows = math.ceil(n_models / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for idx, model_name in enumerate(all_preds.columns):
    ax = axes[idx]
    
    # 실제값
    ax.plot(y_test.index, y_test.values, 'k-', label='Actual', linewidth=2)
    
    # 예측값
    ax.plot(y_test.index, all_preds[model_name].values, 'r--', label='Predicted', linewidth=1.5)
    
    # RMSE 계산
    try:
        rmse = np.sqrt(np.mean((y_test.values - all_preds[model_name].values)**2))
        ax.set_title(f'{model_name}\\nRMSE: {rmse:.2f}', fontsize=10)
    except:
        ax.set_title(model_name, fontsize=10)
    
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# 사용하지 않는 subplot 숨기기
for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Actual vs Predicted: 모든 모델 비교', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'all_models_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'저장: {OUTPUT_DIR}/all_models_actual_vs_predicted.png')
"""

# Cell 2: Top 5 모델 비교 (한 그래프에)
# -------------------------------------------------
"""
# 주요 모델 비교 (한 그래프에 Top 5 모델)
top5_models = metrics_df.head(5).index.tolist()

plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test.values, 'k-', label='Actual', linewidth=3, alpha=0.8)

colors = ['red', 'blue', 'green', 'orange', 'purple']
styles = ['--', '-.', ':', '--', '-.']

for i, model_name in enumerate(top5_models):
    if model_name in all_preds.columns:
        plt.plot(y_test.index, all_preds[model_name].values, 
                 color=colors[i], linestyle=styles[i], 
                 label=model_name, linewidth=1.5)

plt.title('Actual vs Predicted: Top 5 Models', fontsize=14, fontweight='bold')
plt.ylabel('Nickel Price (USD/tonne)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top5_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'저장: {OUTPUT_DIR}/top5_actual_vs_predicted.png')
"""
