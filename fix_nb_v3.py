import re

with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix missing commas between cells in the notebook JSON
# Pattern: a cell ends with "        }\n" and next cell starts with "        {"
# Should be: "        },\n        {"

fixed = re.sub(
    r'(        \}\n)(\s+)(\{)',
    r'\1,\n\2\3',
    open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8').read()
)

# Also fix the specific line 380 issue - ensure comma after the print statement cell
# The issue is that a cell ending at line 380 is missing a comma before the next cell

with open('V4_Colab_Validation.ipynb', 'w', encoding='utf-8') as f:
    f.write(fixed)

print('Fixed!')

# Verify
import json
with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    json.load(f)
print('JSON is now valid!')