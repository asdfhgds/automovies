import json

with open('GPU_VALIDATION.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells'][:10]):
    if cell['cell_type'] == 'markdown':
        print(f'Cell {i}: MD - {cell["source"][0][:150]}...')
    else:
        print(f'Cell {i}: CODE ({len(cell["source"])} chars)')
        if cell['source']:
            print('  ', cell['source'][0][:200])