#!/usr/bin/env python3
"""
Script to update the final results visualization with model composition details
"""
import json

NOTEBOOK_PATH = "/Users/jaeholee/Desktop/sparta_2/sparta2_advanced.ipynb"

# New visualization cell source with model composition mapping
new_vis_cell_source = [
    "# 모델 구성 매핑 (시각화용)\n",
    "# sparta2 기준선 추가\n",
    "sparta2_baseline = pd.DataFrame([{\n",
    "    'Experiment': 'sparta2: Naive×0.8 + GB×0.2 (baseline)',\n",
    "    'RMSE': SPARTA2_RMSE,\n",
    "    'Improvement': 0\n",
    "}])\n",
    "\n",
    "# 모델 구성 상세 매핑\n",
    "composition_mapping = {\n",
    "    'Hybrid_0.8': 'Hybrid: Naive×0.8 + LightGBM×0.2',\n",
    "    'Hybrid_0.7': 'Hybrid: Naive×0.7 + LightGBM×0.3',\n",
    "    'Hybrid_0.9': 'Hybrid: Naive×0.9 + LightGBM×0.1',\n",
    "    'GB_Basic': 'GradientBoosting (Basic)',\n",
    "    'LightGBM_Tuned': 'LightGBM (Tuned)',\n",
    "    'Stacking': 'Stacking Ensemble',\n",
    "    'ARIMA': 'ARIMA(1,1,0)',\n",
    "    'LSTM': 'LSTM Neural Network',\n",
    "    'Naive_Drift': 'Naive Drift',\n",
    "    'Naive_Last': 'Naive Last',\n",
    "    'Naive_Damped': 'Naive Damped (α=0.3)',\n",
    "    'Naive_Seasonal': 'Naive Seasonal',\n",
    "    'RidgeCV': 'Ridge Regression (CV)',\n",
    "    'Lasso': 'Lasso Regression',\n",
    "    'XGBoost_Tuned': 'XGBoost (Tuned)',\n",
    "    'RF_Basic': 'Random Forest',\n",
    "}\n",
    "\n",
    "# final_results_df에 구성 정보 추가\n",
    "viz_df = final_results_df.copy()\n",
    "viz_df['Composition'] = viz_df['Experiment'].map(\n",
    "    lambda x: composition_mapping.get(x, x)\n",
    ")\n",
    "\n",
    "# sparta2 기준선 추가 (비교용)\n",
    "viz_df = pd.concat([sparta2_baseline, viz_df], ignore_index=True)\n",
    "viz_df.loc[0, 'Composition'] = 'sparta2: Naive×0.8 + GB×0.2 (baseline)'\n",
    "\n",
    "# RMSE로 정렬\n",
    "viz_df = viz_df.sort_values('RMSE').reset_index(drop=True)\n",
    "\n",
    "# 전체 결과 시각화\n",
    "fig, ax = plt.subplots(figsize=(14, 8))\n",
    "\n",
    "# 색상: 기준선보다 좋으면 녹색, 나쁘면 빨강, 기준선은 파랑\n",
    "colors = []\n",
    "for i, row in viz_df.iterrows():\n",
    "    if 'baseline' in str(row['Composition']):\n",
    "        colors.append('blue')\n",
    "    elif row['RMSE'] < SPARTA2_RMSE:\n",
    "        colors.append('green')\n",
    "    else:\n",
    "        colors.append('red')\n",
    "\n",
    "bars = ax.barh(range(len(viz_df)), viz_df['RMSE'], color=colors, alpha=0.7)\n",
    "ax.axvline(SPARTA2_RMSE, color='blue', linestyle='--', linewidth=2, label=f'sparta2 Baseline ({SPARTA2_RMSE})')\n",
    "\n",
    "ax.set_yticks(range(len(viz_df)))\n",
    "ax.set_yticklabels(viz_df['Composition'])\n",
    "ax.invert_yaxis()\n",
    "ax.set_xlabel('RMSE (Lower is Better)')\n",
    "ax.set_title('All Experiments RMSE Comparison\\n(sparta2 baseline: Naive×0.8 + GradientBoosting×0.2, RMSE=406.80)', \n",
    "             fontweight='bold', fontsize=12)\n",
    "ax.legend(loc='lower right')\n",
    "ax.grid(True, alpha=0.3, axis='x')\n",
    "\n",
    "# RMSE 수치 표시\n",
    "for i, bar in enumerate(bars):\n",
    "    width = bar.get_width()\n",
    "    ax.text(width + 5, bar.get_y() + bar.get_height()/2, \n",
    "            f'{width:.1f}', ha='left', va='center', fontsize=9)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# 기준선 개선 통계 (baseline 제외)\n",
    "experiments_only = viz_df[~viz_df['Composition'].str.contains('baseline')]\n",
    "improved = len(experiments_only[experiments_only['RMSE'] < SPARTA2_RMSE])\n",
    "total = len(experiments_only)\n",
    "print(f'\\nBaseline improvement: {improved}/{total} experiments')\n",
    "if improved > 0:\n",
    "    best = experiments_only.iloc[0]\n",
    "    print(f'Best model: {best[\"Composition\"]} (RMSE: {best[\"RMSE\"]:.2f}, Δ: {SPARTA2_RMSE - best[\"RMSE\"]:+.2f})')"
]

def main():
    # Read notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the visualization cell
    target_cell_index = None
    for i, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if '전체 실험별 Test RMSE' in source and 'final_results_df' in source:
                target_cell_index = i
                break
    
    if target_cell_index is None:
        print("Could not find target visualization cell!")
        return False
    
    # Update the cell source
    notebook['cells'][target_cell_index]['source'] = new_vis_cell_source
    notebook['cells'][target_cell_index]['outputs'] = []  # Clear outputs
    
    # Write modified notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"Modified notebook saved: {NOTEBOOK_PATH}")
    print(f"  - Updated visualization cell at index {target_cell_index}")
    print("  - Added model composition mapping for clear labels")
    print("  - Included sparta2 baseline in comparison")
    return True

if __name__ == "__main__":
    main()
