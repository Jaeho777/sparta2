import json
import os

nb_path = '/Users/jaeholee/Desktop/sparta_2/sparta2.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
modified_count = 0

# Target: The loop where current_n_features is defined
target_loop_line = "current_n_features = 0 if (model_name.startswith('Naive') or model_name.startswith('Hybrid')) else n_features"
new_loop_line = "    current_n_features = 0 # Force 0 to avoid NaN (Test set N=12 < P=20)"

for cell in cells:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if target_loop_line in source_str:
            # Replace the line
            new_source = source_str.replace(target_loop_line, new_loop_line)
            cell['source'] = new_source.splitlines(keepends=True)
            modified_count += 1
            print("Patched metrics loop.")

if modified_count > 0:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully patched {modified_count} cell.")
else:
    print("No matching cell found to patch.")
