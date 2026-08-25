import json

with open('GPU_VALIDATION.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells'][10:]):
    if cell['cell_type'] == 'markdown':
        print('Cell {}: MD - {}...'.format(i+10, cell['source'][0][:150]))
    else:
        print('Cell {}: CODE ({} chars)'.format(i+10, len(cell['source'])))
        if cell['source']:
            print('  ', cell['source'][0][:200])