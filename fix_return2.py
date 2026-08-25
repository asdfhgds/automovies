import re

with open('src/director/editorial_director.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Fix the return pacing line - it should be at indent 8, followed by a blank line, then the next method
source = re.sub(
    r'(\s{8})return pacing\n \n(\s{4})def _build_audio_strategy',
    r'\1return pacing\n\n\2',
    source,
    flags=re.DOTALL
)

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\editorial_director.py', 'w', encoding='utf-8') as f:
    f.write(source)

print('Fixed')