import json
import os

# notebook_path = '/Users/jaeholee/Desktop/sparta_2/sparta2.ipynb'
notebook_path = '/Users/jaeholee/Desktop/sparta_2/sparta2.ipynb'

plot_code = """
# ------------------------------------------------------------
# Test Period Visualization (Actual vs Hybrid vs Pure ML vs Naive)
# ------------------------------------------------------------
plt.figure(figsize=(15, 7))
target_models_to_plot = ['Naive_Drift', 'Hybrid_Naive0.8_ML0.2', 'BASE_GradientBoosting']
colors = ['green', 'red', 'orange']
styles = ['--', '-', ':']

# Plot Actual
if 'y_test' in globals():
    plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=3, alpha=0.6)

# Plot Models
if 'preds' in globals():
    for name, color, style in zip(target_models_to_plot, colors, styles):
        if name in preds.columns:
            plt.plot(y_test.index, preds[name], label=name, color=color, linestyle=style, linewidth=2)

    plt.title('Test Period: Actual vs Predicted (Drift vs Hybrid vs ML)', fontsize=16)
    plt.ylabel('Nickel Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_period_comparison.png'), dpi=300, bbox_inches='tight')
    print('Saved test_period_comparison.png')
else:
    print('preds dataframe not found, skipping plot.')
"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell where metrics are saved (final_metrics_comparison.csv) and append plot code after it
inserted = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        if "final_metrics_comparison.csv" in "".join(cell['source']):
            # Check if already inserted to avoid duplicates
            if i+1 < len(nb['cells']) and "Test Period Visualization" in "".join(nb['cells'][i+1]['source']):
                print("Plotting cell already exists.")
                inserted = True
                break
            
            # Insert new cell after this one
            new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": plot_code.splitlines(keepends=True)
            }
            nb['cells'].insert(i+1, new_cell)
            print("Added Test Period Plotting cell.")
            inserted = True
            break

if not inserted:
    print("Could not find the target cell to insert plotting code.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Saved notebook with plotting code.")
