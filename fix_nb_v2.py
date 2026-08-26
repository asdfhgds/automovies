# Fix the missing comma in the notebook JSON
with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Line 380 (index 379) - need to add comma after the print statement
# Line 380 is: '        "    print(f\"Total runtime: {val.get(\'runtime\', {}).get(\'wall_clock_sec\', \'N/A\')}s\")\\n",\n'
# It should end with a comma

# Check line 380 (index 379)
print(f"Line 380: {repr(lines[379][:120])}")
print(f"Ends with comma: {lines[379].rstrip().endswith(',')}")

# Fix: add comma if missing
if not lines[379].rstrip().endswith(','):
    lines[379] = lines[379].rstrip() + ',\n'
    with open('V4_Colab_Validation.ipynb', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print('Fixed comma at line 380')
else:
    print('Already has comma')

# Also check if there's a missing comma after the last cell in the cells array
# The error might be that the cells array is missing a comma between cells

# Let's also check if there's a missing comma between cells in the cells array
import json
with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the cells array and check for missing commas between cells
# The error at line 380 column 93 suggests a missing comma after a cell

# Let's fix by ensuring the cells array has proper commas
import re

# Fix: ensure there's a comma between cells in the cells array
# The issue is likely that the last cell before the next cell is missing a comma

# Read the raw content
with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix: add comma after the cell that ends at line 380 if missing
# The pattern is: a cell ends with "        }\n" and should be followed by a comma if not the last cell

# Fix using regex: find },\n        }\n        } pattern and ensure comma
import re

# Fix: add comma after cell that ends before line 380
# The pattern is: a cell ends with "        }\n" and the next line starts with "        {\n"
# Should be: "        },\n        {\n"

# Fix using regex
fixed = re.sub(
    r'(        \}\n)(\s+)(\{)',
    r'\1,\n\2\3',
    content
)

# Also fix the specific line 380 issue - ensure comma after the print statement cell
# The cell ending at line 380 should have a comma if there's another cell after it

with open('V4_Colab_Validation.ipynb', 'w', encoding='utf-8') as f:
    f.write(fixed)

print('Fixed!')

# Verify
import json
with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    json.load(f)
print('JSON is now valid!')