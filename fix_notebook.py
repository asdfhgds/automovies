import json

# Read the notebook
with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

notebook = json.loads(content)

# Find the problematic cell - it's the one with the validation summary cell
# The error is at line 380 column 93, which is in the cells array
# Let's find the problematic cell and fix it

cells = notebook['cells']
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        # Check if this cell contains the problematic line
        source_text = ''.join(cell.get('source', []))
        if 'Total runtime:' in source and 'wall_clock_sec' in source:
            print(f'Found problematic cell at index {cell_index}')
            print(f'Cell source: {cell.get("source", [])}')
            
            # Check if the last element of source has a comma
            source_list = cell.get('source', [])
            if source_list and not source_list[-1].rstrip().endswith(','):
                # Add comma to the last element
                source_list[-1] = source_list[-1].rstrip() + ',\n'
                print(f'Fixed cell {cell_index} - added comma to last source line')
                
                # Save the fixed notebook
                with open('V4_Colab_Validation.ipynb', 'w', encoding='utf-8') as f:
                    json.dump(notebook, f, indent=2, ensure_ascii=False)
                print('Fixed and saved!')
                break
else:
    print('No problematic cell found')