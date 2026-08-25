with open('GPU_VALIDATION.ipynb', 'r', encoding='utf-8') as f:
    import json
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        print('Cell {}: MD - {}...'.format(i, cell['source'][0][:120]))
    else:
        print('Cell {}: CODE ({} chars)'.format(i, len(cell['source'])))
        if cell['source']:
            print('  ', cell['source'][0][:200])