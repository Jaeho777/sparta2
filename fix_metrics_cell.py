#!/usr/bin/env python3
"""
Script to fix the metrics table cell with proper formatting
"""
import json

NOTEBOOK_PATH = "/Users/jaeholee/Desktop/sparta_2/sparta2_advanced.ipynb"

# Proper cell source (each line ends with \n for JSON format)
new_metrics_cell_source = [
    "# 모델명 매핑 (구성 명확화)\n",
    "model_name_mapping = {\n",
    "    'Hybrid_0.8': 'Hybrid_Naive0.8_LGB0.2 (LightGBM)',\n",
    "    'Hybrid_0.7': 'Hybrid_Naive0.7_LGB0.3 (LightGBM)',\n",
    "    'Hybrid_0.9': 'Hybrid_Naive0.9_LGB0.1 (LightGBM)',\n",
    "    'GB_Basic': 'GB_Basic (GradientBoosting)',\n",
    "    'LightGBM_Tuned': 'LightGBM_Tuned',\n",
    "}\n",
    "\n",
    "# 성능 지표 계산\n",
    "results = []\n",
    "for col in all_preds.columns:\n",
    "    try:\n",
    "        pred = pd.Series(all_preds[col].values, index=y_test.index).dropna()\n",
    "        actual = y_test.reindex(pred.index).dropna()\n",
    "        common = pred.index.intersection(actual.index)\n",
    "        if len(common) == 0:\n",
    "            continue\n",
    "        pred = pred.loc[common]\n",
    "        actual = actual.loc[common]\n",
    "        \n",
    "        rmse = np.sqrt(np.mean((actual - pred)**2))\n",
    "        mae = np.mean(np.abs(actual - pred))\n",
    "        mape = np.mean(np.abs((actual - pred) / actual)) * 100\n",
    "        rmspe = np.sqrt(np.mean(((actual - pred) / actual)**2)) * 100\n",
    "        r2 = r2_score(actual, pred)\n",
    "        \n",
    "        # 모델명 매핑 적용\n",
    "        display_name = model_name_mapping.get(col, col)\n",
    "        results.append({'Model': display_name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'RMSPE': rmspe, 'R2': r2})\n",
    "    except Exception as e:\n",
    "        print(f'{col} 오류: {e}')\n",
    "\n",
    "metrics_df = pd.DataFrame(results).set_index('Model').sort_values('RMSE')\n",
    "\n",
    "# sparta2 기준선과 비교 (sparta2: Naive×0.8 + GradientBoosting×0.2)\n",
    "SPARTA2_RMSE = 406.80  # sparta2 기준선: Naive_Drift×0.8 + GradientBoosting×0.2\n",
    "metrics_df['vs_Baseline'] = SPARTA2_RMSE - metrics_df['RMSE']\n",
    "\n",
    "print('='*80)\n",
    "print('【모든 실험 최종 지표 비교】')\n",
    "print('='*80)\n",
    "print()\n",
    "print('📌 sparta2 기준선 (RMSE: 406.80)')\n",
    "print('   구성: Naive_Drift × 0.8 + GradientBoosting × 0.2')\n",
    "print()\n",
    "print('📌 sparta2_advanced 주요 변경사항:')\n",
    "print('   • ML 모델: GradientBoosting → LightGBM (GridSearchCV 튜닝)')\n",
    "print('   • 피처: +12개 추가 기술적 지표 및 래그 피처')\n",
    "print('='*80)\n",
    "print()\n",
    "\n",
    "display(metrics_df.style.format({\n",
    "    'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'MAPE': '{:.2f}%', \n",
    "    'RMSPE': '{:.2f}%', 'R2': '{:.4f}', 'vs_Baseline': '{:+.2f}'\n",
    "}).background_gradient(subset=['RMSE'], cmap='RdYlGn_r'))\n",
    "\n",
    "metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'all_experiments_comparison.csv'))\n",
    "print(f'\\n저장: {OUTPUT_DIR}/all_experiments_comparison.csv')"
]

def main():
    # Read notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the problematic cell (index 52)
    target_cell_index = 52
    
    # Update the cell source
    notebook['cells'][target_cell_index]['source'] = new_metrics_cell_source
    # Clear outputs to trigger re-run
    notebook['cells'][target_cell_index]['outputs'] = []
    
    # Write modified notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"Fixed notebook saved: {NOTEBOOK_PATH}")
    print(f"  - Fixed metrics calculation cell at index {target_cell_index}")
    return True

if __name__ == "__main__":
    main()
