
import json
import re

notebook_path = 'dl_lstm_transformer.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Loaded {notebook_path}")
    output_text = []
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if output['output_type'] == 'stream':
                        output_text.extend(output['text'])
                    elif output['output_type'] == 'execute_result':
                        if 'data' in output and 'text/plain' in output['data']:
                            output_text.extend(output['data']['text/plain'])
    
    full_output = "".join(output_text)
    
    # Simple regex to find result tables or metrics
    # Looking for patterns like "LSTM ... 123.45" or "RMSE: 123.45"
    print("\n--- Extracted Metric Candidates ---")
    
    # Look for tabular data rows (Model naming convention + numbers)
    # Regex: (Transformer|LSTM) followed by numbers
    patterns = [
        r"(LSTM|Transformer|Attention)\S*\s+\d+\.\d+",
        r"(BASE|RES|ROR)_\S+\s+\d+\.\d+"
    ]
    
    found = False
    for line in full_output.split('\n'):
        for pat in patterns:
            if re.search(pat, line):
                print(line.strip())
                found = True
                break
    
    if not found:
        print("No explicit metric tables found in outputs.")
        print("First 500 chars of output:")
        print(full_output[:500])

except Exception as e:
    print(f"Error: {e}")
