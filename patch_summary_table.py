import json
import os

nb_path = '/Users/jaeholee/Desktop/sparta_2/sparta2.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
modified_count = 0

# Target: Section 11, after saving final_metrics_comparison.csv
target_marker = "final_metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'final_metrics_comparison.csv'))"

summary_table_code = r"""
# ------------------------------------------------------------
# [REQUESTED] Summary Table: Key Models Selection
# ------------------------------------------------------------
# 1. Baselines
baseline_mask = final_metrics_df.index.str.startswith('BASE_')

# 2. Naive Base (Naive_... but exclude complex ones)
# Simple Naive: Naive_Last, Naive_Drift, Naive_SMA
naive_base_mask = (final_metrics_df.index.str.startswith('Naive_')) & \
                  (~final_metrics_df.index.str.contains('Damped')) & \
                  (~final_metrics_df.index.str.contains('Residual')) & \
                  (~final_metrics_df.index.str.contains('ROR'))

# 3. Naive Follow-up (Damped, ML Residual/ROR stacking)
naive_followup_mask = (final_metrics_df.index.str.startswith('Naive_')) & \
                      (final_metrics_df.index.str.contains('Damped') | \
                       final_metrics_df.index.str.contains('Residual') | \
                       final_metrics_df.index.str.contains('ROR'))

# 4. Major Residual/ROR Models (RES_, ROR_) -> Pick Top 5 by RMSE
res_ror_all = final_metrics_df[final_metrics_df.index.str.startswith(('RES_', 'ROR_'))].sort_values('RMSE')
top_res_ror_indices = res_ror_all.head(5).index

# Combine
summary_indices = []
summary_indices.extend(final_metrics_df[baseline_mask].index.tolist())
summary_indices.extend(final_metrics_df[naive_base_mask].index.tolist())
summary_indices.extend(final_metrics_df[naive_followup_mask].index.tolist())
summary_indices.extend(top_res_ror_indices.tolist())

# Also include Hybrid if any (as they are important)
hybrid_mask = final_metrics_df.index.str.startswith('Hybrid_')
summary_indices.extend(final_metrics_df[hybrid_mask].index.tolist())

# Remove duplicates while preserving order
seen = set()
unique_indices = [x for x in summary_indices if not (x in seen or seen.add(x))]

summary_metrics_df = final_metrics_df.loc[unique_indices].sort_values('RMSE')

print('\n【주요 모델 요약 비교 (Baseline, Naive, Naive+, Top-5 RES/ROR)】')
display(summary_metrics_df[display_cols].style.format({
    'RMSE': '{:.4f}',
    'RMSPE': '{:.2f}',
    'MAPE': '{:.2f}',
    'MAE': '{:.4f}',
    'Adj_R2': '{:.4f}'
}))

summary_metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics_selected.csv'))
"""

for cell in cells:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if target_marker in source_str:
            # Append the summary table code after the marker
            if "summary_metrics_selected.csv" not in source_str:
                # insert code
                new_source = source_str.replace(target_marker, target_marker + "\n" + summary_table_code)
                cell['source'] = new_source.splitlines(keepends=True)
                modified_count += 1
                print("Added summary table code.")
            else:
                print("Summary table code already exists. Skipping.")

if modified_count > 0:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully patched {modified_count} cell.")
else:
    print("No matching cell found to patch.")
