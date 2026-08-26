import re

with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix missing commas between cells in the cells array
# Pattern: a cell ends with "        }\n" and next cell starts with "        {"
# Should be: "        },\n        {"

fixed = re.sub(
    r'(        \}\n)(\s+)(\{)',
    r'\1,\n\2\3',
    open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8').read()
)

with open('V4_Colab_Validation.ipynb', 'w', encoding='utf-8') as f:
    f.write(re.sub(
        r'(        \}\n)(\s+)(\{)',
        r'\1,\n\2\3',
        open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8').read()
    ))

print('Fixed!')

# Verify
import json
with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    json.load(f)
print('JSON is now valid!')