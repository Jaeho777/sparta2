import json
import os

nb_path = '/Users/jaeholee/Desktop/sparta_2/sparta2.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
modified_count = 0

# Target 1: compute_metrics_with_adj_r2
target_metrics_def = 'def compute_metrics_with_adj_r2(y_true, y_pred, n_features):'
new_metrics_code = r"""def compute_metrics_with_adj_r2(y_true, y_pred, n_features):
    from sklearn.metrics import r2_score
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    df = pd.concat([y_true, y_pred], axis=1).dropna()
    if len(df) == 0:
        return None
    yt = df.iloc[:, 0].values
    yp = df.iloc[:, 1].values
    n = len(yt)

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))

    eps = 1e-8
    denom = np.where(np.abs(yt) < eps, np.nan, np.abs(yt))
    ape = np.abs((yt - yp) / denom)
    spe = ((yt - yp) / denom) ** 2
    mape = float(np.nanmean(ape) * 100)
    rmspe = float(np.sqrt(np.nanmean(spe)) * 100)

    # Use sklearn r2_score
    if len(yt) > 1:
        r2 = r2_score(yt, yp)
    else:
        r2 = np.nan

    adj_r2 = np.nan
    if not np.isnan(r2) and n > (n_features + 1):
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return {
        'RMSE': rmse,
        'RMSPE': rmspe,
        'MAPE': mape,
        'MAE': mae,
        'Adj_R2': adj_r2,
        'N': n
    }
"""

# Target 2: Section 11 Report Generation
# We need to find the cell that contains "11. 최종 지표 비교 + 방향성 혼동행렬"
# And then inject the saving logic.
target_section_11 = '11. 최종 지표 비교 + 방향성 혼동행렬'

for cell in cells:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Patch metrics function
        if target_metrics_def in source_str:
            # We want to replace the whole function definition usually, but the cell might contain other things.
            # In the provided view, the cell starts with the prints and then defines the function. 
            # It seems the function definition is the main part of that cell or one of them.
            # Let's replace the manual R2 calculation with sklearn one in the existing source.
            
            # Identify the manual calculation block
            manual_r2_block = """    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = np.nan
    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)"""
            
            if manual_r2_block in source_str:
                new_r2_block = """    from sklearn.metrics import r2_score
    r2 = r2_score(yt, yp)"""
                
                # Careful with indentation. The original has 4 spaces indentation?
                # detailed check on indentation:
                # The manual block in view_file output:
                #     ss_res = np.sum((yt - yp) ** 2)
                # It has 4 spaces.
                
                source_str = source_str.replace(manual_r2_block, new_r2_block)
                cell['source'] = source_str.splitlines(keepends=True)
                modified_count += 1
                print("Patched metrics calculation.")

        # Patch Section 11 for saving images
        if target_section_11 in source_str:
            # Add OUTPUT_DIR creation if not present (it might be in CONFIG, but better safe)
            if "os.makedirs(OUTPUT_DIR, exist_ok=True)" not in source_str:
                # Insert at the beginning of the cell (after imports usually)
                # But imports are at top.
                # Let's define a marker.
                marker = "import os"
                if marker in source_str:
                    replacement = "import os\nos.makedirs(OUTPUT_DIR, exist_ok=True)\ntry:\n    os.chmod(OUTPUT_DIR, 0o755)\nexcept: pass\n"
                    source_str = source_str.replace(marker, replacement)
            
            # Patch Confusion Matrix plot
            # Look for disp.plot(...) ... plt.show()
            # Original:
            #         disp.plot(cmap='Blues')
            #         plt.title(f'Directional Confusion Matrix: {best_model}')
            #         plt.show()
            
            cm_plot_block = "plt.title(f'Directional Confusion Matrix: {best_model}')\n        plt.show()"
            if cm_plot_block in source_str:
                cm_save_block = "plt.title(f'Directional Confusion Matrix: {best_model}')\n        plt.savefig(os.path.join(OUTPUT_DIR, 'directional_confusion_matrix.png'))\n        plt.close()"
                source_str = source_str.replace(cm_plot_block, cm_save_block)
                print("Patched CM plot saving.")
            
            cell['source'] = source_str.splitlines(keepends=True)
            modified_count += 1

# Target 3: Test Period Visualization (might be in a separate cell or same)
# In the view_file output, it seemed to be in a separate cell starting with `execution_count: null`.
# Let's look for "Test Period Visualization (Actual vs Hybrid vs Pure ML vs Naive)"
target_viz_title = "# Test Period Visualization (Actual vs Hybrid vs Pure ML vs Naive)"

for cell in cells:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if target_viz_title in source_str:
            # Look for plt.show() at the end, or just append savefig if plt.show() is implicit/not present?
            # It usually ends with plt.show() in notebooks.
            # In the file view:
            # plt.plot(...)
            # ...
            # (End of view)
            
            # Let's assume there is a plt.show() or simply add the savefig call.
            # If we append it at the end of the cell, it should work.
            
            # Check if plt.show() is there
            if "plt.show()" in source_str:
                source_str = source_str.replace("plt.show()", "plt.savefig(os.path.join(OUTPUT_DIR, 'test_period_visualization.png'))\nplt.close()")
            else:
                # Just append it
                source_str += "\nplt.savefig(os.path.join(OUTPUT_DIR, 'test_period_visualization.png'))\nplt.close()"
            
            cell['source'] = source_str.splitlines(keepends=True)
            modified_count += 1
            print("Patched Test Period Visualization saving.")

if modified_count > 0:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1) # using indent=1 to minimize diff noise if possible, but standard is often 1 or 2
    print(f"Successfully modified {modified_count} cells.")
else:
    print("No cells matched for modification. Please check signatures.")
