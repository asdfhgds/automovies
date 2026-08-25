import os
os.chdir(r'C:\Users\hp\Documents\Default Project\automovies')

with open('src/director/editorial_director.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 434 (index 433) - should be empty line
lines[433] = '\n'

with open('src/director/editorial_director.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed')