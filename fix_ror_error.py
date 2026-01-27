import nbformat as nbf

def fix_notebook(filepath):
    nb = nbf.read(filepath, as_version=4)
    found = 0
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'ROR_TARGET_MODE' in cell.source and "CONFIG['ror_target_mode']" not in cell.source:
                # Replace ROR_TARGET_MODE with CONFIG['ror_target_mode']
                # Be careful not to replace if it's already fixed or inside assignment if it was intended to be assigned (but here it seems it's used as value)
                # The usage is mode=ROR_TARGET_MODE, so replacement is safe.
                # However, let's just replace the exact string usage.
                
                # Check if it is being assigned to. If 'ROR_TARGET_MODE =' in source, we might need to be careful, 
                # but based on error it is used as a value.
                
                new_source = cell.source.replace("mode=ROR_TARGET_MODE", "mode=CONFIG['ror_target_mode']")
                if new_source != cell.source:
                    cell.source = new_source
                    found += 1
    
    if found > 0:
        nbf.write(nb, filepath)
        print(f"Successfully fixed {found} occurrences of ROR_TARGET_MODE in {filepath}")
    else:
        print("No occurrences of ROR_TARGET_MODE found to fix.")

if __name__ == "__main__":
    fix_notebook('sparta2.ipynb')
