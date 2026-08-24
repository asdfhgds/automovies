import re

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Fix the plan_grounding method - move return result to method level
# Find the pattern where return result is inside the if block
old_pattern = r'''            return result\r\n\r\n    def _legacy_prose_audit\('''

new_replacement = '''        return result

    def _legacy_prose_audit('''

# Use regex with DOTALL flag to match across lines
new_source = re.sub(
    r'(\s{8})return result\r?\n\r?\n    def _legacy_prose_audit\(',
    r'        return result\r\n\r\n    def _legacy_prose_audit(',
    source,
    flags=re.MULTILINE
)

if new_source == source:
    print("Pattern not found, trying alternative...")
    # Try a more specific pattern
    new_source = re.sub(
        r'(\s{12})return result\r?\n\r?\n(\s{4})def _legacy_prose_audit\(',
        r'        return result\r\n\r\n    def _legacy_prose_audit(',
        source
    )
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