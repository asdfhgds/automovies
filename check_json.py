import json

with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

try:
    json.loads(content)
    print('Valid JSON')
except json.JSONDecodeError as e:
    print(f'Error at line {e.lineno}, column {e.colno}: {e.msg}')
    with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(max(0, e.lineno-5), min(len(open('V4_Colab_Validation.ipynb', encoding='utf-8').readlines()), e.lineno+3)):
        print(f'{i+1:4d}: {open("V4_Colab_Validation.ipynb", encoding="utf-8").readlines()[i].rstrip()}')