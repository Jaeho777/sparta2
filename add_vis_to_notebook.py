
import json
import os

notebook_path = 'sparta2.ipynb'
output_path = 'sparta2_updated.ipynb'

# Source code for visualization cells
viz_func_code = """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_model_predictions(model_group_name, y_train, y_val, y_test, models_dict):
    # User requested ONLY Test set comparison
    splits = ['Test']
    y_trues = [y_test]
    
    n_models = len(models_dict)
    # Use 1 column, rows = n_models
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models), constrained_layout=True)
    
    if n_models == 1:
        axes = [axes] # Make it iterable

    for i, (model_name, preds) in enumerate(models_dict.items()):
        ax = axes[i]
        split = 'Test'
        y_true = y_trues[0]
        y_pred = preds.get(split)
        
        if y_pred is not None:
            common_idx = y_true.index.intersection(y_pred.index)
            if len(common_idx) > 0:
                ax.plot(common_idx, y_true.loc[common_idx], label='Actual', color='black', linewidth=1.5, alpha=0.7)
                ax.plot(common_idx, y_pred.loc[common_idx], label='Pred', color='red', linestyle='--', linewidth=2.0)
                rmse_val = np.sqrt(np.mean((y_true.loc[common_idx] - y_pred.loc[common_idx])**2))
                ax.set_title(f"{model_name} (Test Period)\\nRMSE: {rmse_val:.2f}", fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Highlight removed as per user request
            else:
                ax.text(0.5, 0.5, "No overlapping data", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "No prediction data", ha='center', va='center')
            
    plt.suptitle(f"{model_group_name} - Test Period Only", fontsize=16, y=1.02)
    plt.show()
"""

viz_exec_code = """
# 1. Prepare Data for Visualization
y_full = pd.concat([y_train, y_val, y_test]).sort_index()

# --- Naive Models ---
naive_last_full = y_full.shift(1)
naive_drift_full = y_full.shift(1) + (y_full.shift(1) - y_full.shift(2))
naive_damped_full = y_full.shift(1) + 0.7 * (y_full.shift(1) - y_full.shift(2))

naive_dict = {
    'Naive_Last': naive_last_full,
    'Naive_Drift': naive_drift_full,
    'Naive_Damped(0.7)': naive_damped_full
}

naive_viz_data = {}
for name, pred_full in naive_dict.items():
    naive_viz_data[name] = {
        'Train': pred_full.loc[y_train.index],
        'Val': pred_full.loc[y_val.index],
        'Test': pred_full.loc[y_test.index]
    }

# --- Hybrid Model (Naive 0.8 + GB 0.2) ---
# Find GradientBoosting model
gb_model = None
if 'base_models' in globals():
    for name, model in base_models.items():
        if 'GradientBoosting' in name:
            gb_model = model
            break

hybrid_viz_data = {}
if gb_model:
    # Generate GB preds
    gb_pred_train = pd.Series(gb_model.predict(X_train), index=y_train.index)
    gb_pred_val = pd.Series(gb_model.predict(X_val), index=y_val.index)
    gb_pred_test = pd.Series(gb_model.predict(X_test), index=y_test.index)
    
    w = 0.8
    hybrid_name = f'Hybrid(Naive{w}+GB{1-w:.1f})'
    hybrid_viz_data[hybrid_name] = {
        'Train': w * naive_viz_data['Naive_Drift']['Train'] + (1-w) * gb_pred_train,
        'Val': w * naive_viz_data['Naive_Drift']['Val'] + (1-w) * gb_pred_val,
        'Test': w * naive_viz_data['Naive_Drift']['Test'] + (1-w) * gb_pred_test
    }

# 3. Execute Plotting
print("Plotting Naive Models...")
plot_model_predictions("Naive Models", y_train, y_val, y_test, naive_viz_data)

if hybrid_viz_data:
    print("\\nPlotting Hybrid Models...")
    plot_model_predictions("Hybrid Models (Best)", y_train, y_val, y_test, hybrid_viz_data)
"""

# New cells to append
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 9. Integrated Visualization (All Models)\n",
            "사용자 요청에 따라 Naive, Stacking, Hybrid 모델의 Train/Val/Test 예측 결과를 시각화합니다."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": viz_func_code.strip().splitlines(True)
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": viz_exec_code.strip().splitlines(True)
    }
]

# Load, Append, Save
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    nb['cells'].extend(new_cells)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully created {output_path}")
    
except Exception as e:
    print(f"Error: {e}")
