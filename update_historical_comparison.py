#!/usr/bin/env python3
"""
Script to update model naming and add historical comparison in sparta2_advanced.ipynb
"""
import json
import os

NOTEBOOK_PATH = "/Users/jaeholee/Desktop/sparta_2/sparta2_advanced.ipynb"

# Cell to add: Historical Performance Comparison
history_comparison_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📊 역대 RMSE 성능 비교표\n",
        "\n",
        "| 버전 | 모델명 | 구성 | RMSE | 비고 |\n",
        "|------|--------|------|------|------|\n",
        "| **sparta2 (기준선)** | Hybrid_Naive0.8_GB0.2 | Naive_Drift × 0.8 + **GradientBoosting** × 0.2 | **406.80** | 기본 GradientBoosting 사용 |\n",
        "| **sparta2_advanced** | Hybrid_Naive0.8_LGB0.2 | Naive_Drift × 0.8 + **LightGBM_Tuned** × 0.2 | **398.00** | 튜닝된 LightGBM + 추가 피처 |\n",
        "| sparta2_advanced | GB_Basic | GradientBoosting 단독 | 1327.69 | ML 단독 사용 |\n",
        "| sparta2_advanced | LightGBM_Tuned | LightGBM 단독 | 940.11 | ML 단독 사용 |\n",
        "| sparta2_advanced | ARIMA | ARIMA(1,1,0) | 1211.88 | 전통 시계열 모델 |\n",
        "| sparta2_advanced | LSTM | 딥러닝 | 1039.28 | 신경망 모델 |\n",
        "| sparta2_advanced | Naive_Drift | 전주 + 변화량 | 480.67 | 단순 Naive 모델 |\n",
        "\n",
        "> ⚠️ **중요**: sparta2 기준선(406.80)은 **GradientBoosting** 기반, sparta2_advanced의 최고 성능(398.00)은 **LightGBM** 기반입니다."
    ]
}

# Updated code for metrics display with clearer model names
updated_metrics_display_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 모델명 가독성 향상 (주요 모델 구분)\n",
        "model_name_mapping = {\n",
        "    'Hybrid_0.8': 'Hybrid_Naive0.8_LGB0.2 (LightGBM)',\n",
        "    'Hybrid_0.7': 'Hybrid_Naive0.7_LGB0.3 (LightGBM)',\n",
        "    'Hybrid_0.9': 'Hybrid_Naive0.9_LGB0.1 (LightGBM)',\n",
        "    'GB_Basic': 'GB_Basic (GradientBoosting)',\n",
        "    'LightGBM_Tuned': 'LightGBM_Tuned (Tuned)',\n",
        "}\n",
        "\n",
        "# 결과 표시\n",
        "print('='*80)\n",
        "print('【최종 실험 결과 (sparta2 기준선 대비)】')\n",
        "print('='*80)\n",
        "print()\n",
        "print('📌 sparta2 기준선:')\n",
        "print('   모델: Hybrid (Naive_Drift × 0.8 + GradientBoosting × 0.2)')\n",
        "print(f'   RMSE: {SPARTA2_RMSE}')\n",
        "print()\n",
        "print('📌 sparta2_advanced 주요 변경사항:')\n",
        "print('   • ML 모델: GradientBoosting → LightGBM (GridSearchCV 튜닝)')\n",
        "print('   • 피처: +12개 추가 기술적 지표 및 래그 피처')\n",
        "print('   • 기법: ARIMA, LSTM, Stacking 추가 시도')\n",
        "print('='*80)\n",
        "print()"
    ]
}

def main():
    # Read notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find cell indices
    metrics_cell_index = None
    clarification_cell_index = None
    
    for i, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if '모든 실험 최종 지표 비교' in source and 'SPARTA2_RMSE' in source:
            metrics_cell_index = i
        if '중요: 모델 명칭 정리' in source:
            clarification_cell_index = i
    
    changes_made = []
    
    # 1. Replace the clarification cell with more comprehensive comparison
    if clarification_cell_index is not None:
        notebook['cells'][clarification_cell_index] = history_comparison_cell
        changes_made.append(f"Replaced clarification cell at index {clarification_cell_index} with historical comparison")
    
    # 2. Find and update the metrics intro cell (the one right after clarification)
    if clarification_cell_index is not None:
        intro_index = clarification_cell_index + 1
        if intro_index < len(notebook['cells']):
            cell = notebook['cells'][intro_index]
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if '기준선 정의' in source:
                    notebook['cells'][intro_index] = updated_metrics_display_cell
                    changes_made.append(f"Updated metrics intro cell at index {intro_index}")
    
    if not changes_made:
        print("No changes made!")
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
