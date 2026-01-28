import json

notebook_path = '/Users/jaeholee/Desktop/sparta_2/sparta2.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        modified = False
        for line in source:
            if "m = compute_metrics_with_adj_r2(y_test, preds[model_name], n_features)" in line:
                new_source.append("    current_n_features = 0 if (model_name.startswith('Naive') or model_name.startswith('Hybrid')) else n_features\n")
                new_source.append("    m = compute_metrics_with_adj_r2(y_test, preds[model_name], current_n_features)\n")
                modified = True
            else:
                new_source.append(line)
        
        if modified:
            cell['source'] = new_source
            print("Patched metrics calculation loop.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Saved patched notebook.")
