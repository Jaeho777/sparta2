#!/usr/bin/env python3
"""
Script to add model clarification to sparta2_advanced.ipynb
- Clarifies difference between sparta2 baseline (GB) and advanced (LightGBM)
"""
import json
import os

NOTEBOOK_PATH = "/Users/jaeholee/Desktop/sparta_2/sparta2_advanced.ipynb"

# New markdown cell explaining the difference
clarification_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### ⚠️ 중요: 모델 명칭 정리\n",
        "\n",
        "| 구분 | 모델 구성 | RMSE |\n",
        "|------|----------|------|\n",
        "| **sparta2 기준선** | Naive_Drift × 0.8 + **GradientBoosting** × 0.2 | **406.80** |\n",
        "| **sparta2_advanced Hybrid** | Naive_Drift × 0.8 + **LightGBM_Tuned** × 0.2 | **398.00** |\n",
        "\n",
        "> 두 Hybrid 모델은 ML 부분이 다릅니다:\n",
        "> - sparta2: 기본 GradientBoosting\n",
        "> - sparta2_advanced: 튜닝된 LightGBM + 추가 피처 엔지니어링"
    ]
}

# Updated visualization code with clear baseline reference
updated_top5_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Top 5 Models Comparison (sparta2 기준선 포함)\n",
        "top5_models = metrics_df.head(5).index.tolist()\n",
        "\n",
        "plt.figure(figsize=(14, 7))\n",
        "\n",
        "# 실제값\n",
        "plt.plot(y_test.index, y_test.values, 'k-', label='Actual', linewidth=3, alpha=0.8)\n",
        "\n",
        "# Top 5 모델\n",
        "colors = ['red', 'blue', 'green', 'orange', 'purple']\n",
        "styles = ['--', '-.', ':', '--', '-.']\n",
        "\n",
        "for i, model_name in enumerate(top5_models):\n",
        "    if model_name in all_preds.columns:\n",
        "        # 모델명에 RMSE 추가\n",
        "        rmse = np.sqrt(np.mean((y_test.values - all_preds[model_name].values)**2))\n",
        "        label = f'{model_name} (RMSE: {rmse:.1f})'\n",
        "        plt.plot(y_test.index, all_preds[model_name].values, \n",
        "                 color=colors[i], linestyle=styles[i], \n",
        "                 label=label, linewidth=1.5)\n",
        "\n",
        "# sparta2 기준선 표시 (수평선)\n",
        "plt.axhline(y=y_test.mean(), color='gray', linestyle=':', alpha=0.5)\n",
        "\n",
        "plt.title('Actual vs Predicted: Top 5 Models\\n(sparta2 기준선 RMSE: 406.80 = Naive×0.8+GradientBoosting×0.2)', \n",
        "          fontsize=12, fontweight='bold')\n",
        "plt.ylabel('Nickel Price (USD/tonne)')\n",
        "plt.legend(loc='best', fontsize=9)\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.xticks(rotation=45)\n",
        "\n",
        "# 텍스트 박스로 기준선 설명\n",
        "textstr = 'sparta2 기준선:\\nHybrid(Naive×0.8+GB×0.2)\\nRMSE = 406.80'\n",
        "props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)\n",
        "plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,\n",
        "               verticalalignment='top', bbox=props)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(OUTPUT_DIR, 'top5_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(f'저장: {OUTPUT_DIR}/top5_actual_vs_predicted.png')"
    ]
}

# Updated metrics print with clear explanation
updated_metrics_intro_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 성능 지표 계산 + 명확한 기준선 설명 추가\n",
        "print('='*70)\n",
        "print('【기준선 정의】')\n",
        "print('  - sparta2 Hybrid: Naive_Drift × 0.8 + GradientBoosting × 0.2')\n",
        "print('  - sparta2 RMSE: 406.80')\n",
        "print()\n",
        "print('【sparta2_advanced 주요 변경사항】')\n",
        "print('  - ML 모델: GradientBoosting → LightGBM (튜닝됨)')\n",
        "print('  - 피처: 추가 기술적 지표 및 래그 피처')\n",
        "print('='*70)\n",
        "print()"
    ]
}

def main():
    # Read notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the cell with metrics comparison
    metrics_cell_index = None
    top5_cell_index = None
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if '모든 실험 최종 지표 비교' in source:
                metrics_cell_index = i
            if 'Top 5 Models Comparison' in source or 'top5_models' in source:
                top5_cell_index = i
    
    changes_made = []
    
    # 1. Add clarification markdown before metrics cell
    if metrics_cell_index is not None:
        notebook['cells'].insert(metrics_cell_index, clarification_cell)
        changes_made.append(f"Added clarification markdown at index {metrics_cell_index}")
        # Adjust indices after insertion
        if top5_cell_index is not None and top5_cell_index >= metrics_cell_index:
            top5_cell_index += 1
        metrics_cell_index += 1
    
    # 2. Add metrics intro cell right after clarification
    if metrics_cell_index is not None:
        notebook['cells'].insert(metrics_cell_index, updated_metrics_intro_cell)
        changes_made.append(f"Added metrics intro cell at index {metrics_cell_index}")
        if top5_cell_index is not None and top5_cell_index >= metrics_cell_index:
            top5_cell_index += 1
    
    # 3. Replace top5 visualization cell
    if top5_cell_index is not None:
        notebook['cells'][top5_cell_index] = updated_top5_cell
        changes_made.append(f"Updated top5 visualization cell at index {top5_cell_index}")
    
    if not changes_made:
        print("No target cells found!")
        return False
    
    # Write modified notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"Modified notebook saved: {NOTEBOOK_PATH}")
    for change in changes_made:
        print(f"  - {change}")
    return True

if __name__ == "__main__":
    main()
