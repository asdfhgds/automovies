with open('src/director/editorial_director.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 434 (index 433) - should be empty, not have a space
if len(lines[433]) > 0 and lines[433].strip() == '':
    lines[433] = '\n'

with open('src/director/editorial_director.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed')