#!/usr/bin/env python3
"""
Script to add Actual vs Predicted visualization cells to sparta2_advanced.ipynb
"""
import json
import os

NOTEBOOK_PATH = "/Users/jaeholee/Desktop/sparta_2/sparta2_advanced.ipynb"

# New cells to add
new_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 14.1 Actual vs Predicted 시각화\n",
        "각 모델별 실제값 대비 예측값을 비교합니다."
    ]
}

new_code_cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Actual vs Predicted 시각화 (각 모델별)\n",
        "import math\n",
        "\n",
        "n_models = len(all_preds.columns)\n",
        "n_cols = 3\n",
        "n_rows = math.ceil(n_models / n_cols)\n",
        "\n",
        "fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))\n",
        "axes = axes.flatten()\n",
        "\n",
        "for idx, model_name in enumerate(all_preds.columns):\n",
        "    ax = axes[idx]\n",
        "    ax.plot(y_test.index, y_test.values, 'k-', label='Actual', linewidth=2)\n",
        "    ax.plot(y_test.index, all_preds[model_name].values, 'r--', label='Predicted', linewidth=1.5)\n",
        "    try:\n",
        "        rmse = np.sqrt(np.mean((y_test.values - all_preds[model_name].values)**2))\n",
        "        ax.set_title(f'{model_name}\\nRMSE: {rmse:.2f}', fontsize=10)\n",
        "    except:\n",
        "        ax.set_title(model_name, fontsize=10)\n",
        "    ax.legend(loc='upper left', fontsize=8)\n",
        "    ax.grid(True, alpha=0.3)\n",
        "    ax.tick_params(axis='x', rotation=45)\n",
        "\n",
        "for idx in range(n_models, len(axes)):\n",
        "    axes[idx].set_visible(False)\n",
        "\n",
        "plt.suptitle('Actual vs Predicted: All Models', fontsize=14, fontweight='bold', y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(OUTPUT_DIR, 'all_models_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(f'저장: {OUTPUT_DIR}/all_models_actual_vs_predicted.png')"
    ]
}

new_code_cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Top 5 Models Comparison\n",
        "top5_models = metrics_df.head(5).index.tolist()\n",
        "\n",
        "plt.figure(figsize=(14, 6))\n",
        "plt.plot(y_test.index, y_test.values, 'k-', label='Actual', linewidth=3, alpha=0.8)\n",
        "\n",
        "colors = ['red', 'blue', 'green', 'orange', 'purple']\n",
        "styles = ['--', '-.', ':', '--', '-.']\n",
        "\n",
        "for i, model_name in enumerate(top5_models):\n",
        "    if model_name in all_preds.columns:\n",
        "        plt.plot(y_test.index, all_preds[model_name].values, \n",
        "                 color=colors[i], linestyle=styles[i], \n",
        "                 label=model_name, linewidth=1.5)\n",
        "\n",
        "plt.title('Actual vs Predicted: Top 5 Models', fontsize=14, fontweight='bold')\n",
        "plt.ylabel('Nickel Price (USD/tonne)')\n",
        "plt.legend(loc='best')\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.xticks(rotation=45)\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(OUTPUT_DIR, 'top5_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(f'저장: {OUTPUT_DIR}/top5_actual_vs_predicted.png')"
    ]
}

def main():
    # Read notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the cell with "all_experiments_comparison.csv"
    target_index = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'all_experiments_comparison.csv' in source and '모든 실험 최종 지표 비교' in source:
                target_index = i
                break
    
    if target_index is None:
        print("Target cell not found!")
        return False
    
    print(f"Found target cell at index {target_index}")
    
    # Insert new cells after the target cell
    notebook['cells'].insert(target_index + 1, new_markdown_cell)
    notebook['cells'].insert(target_index + 2, new_code_cell_1)
    notebook['cells'].insert(target_index + 3, new_code_cell_2)
    
    # Backup original
    backup_path = NOTEBOOK_PATH + '.backup'
    os.rename(NOTEBOOK_PATH, backup_path)
    print(f"Backup created: {backup_path}")
    
    # Write modified notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"Modified notebook saved: {NOTEBOOK_PATH}")
    print("Added 3 new cells (1 markdown + 2 code cells) for Actual vs Predicted visualization")
    return True

if __name__ == "__main__":
    main()
