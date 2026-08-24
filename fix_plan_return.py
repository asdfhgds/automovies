import re

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'r') as f:
    source = f.read()

# Find the plan_grounding method and fix the return structure
# The issue is that 'return result' is inside the if block at indent 12
# It should be at method level (indent 8) after all conditionals

# Pattern to find the problematic section
old_pattern = r'''            return result\r\n\r\n    def _legacy_prose_audit\('''

# Replacement with proper structure - return at method level
new_replacement = '''        return result

    def _legacy_prose_audit('''

# Use regex to find and replace
new_source = re.sub(old_pattern, new_replacement, source)

if new_source == source:
    print("Pattern not found, trying alternative...")
    # Try alternative pattern
    alt_pattern = r'return result\r\n\r\n    def _legacy_prose_audit\('
    new_source = re.sub(alt_pattern, '        return result\r\n\r\n    def _legacy_prose_audit(', source)
    if new_source == source:
        print("Pattern not found")
    else:
        with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'w', encoding='utf-8') as f:
            f.write(new_source)
        print('Fixed with alt pattern')
else:
    with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'w', encoding='utf-8') as f:
        f.write(new_source)
    print('Fixed with main pattern')

print('Done')