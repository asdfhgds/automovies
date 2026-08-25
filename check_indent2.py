with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\editorial_director.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(430, 445):
    line = lines[i]
    indent = len(line) - len(line.lstrip())
    print('{0:4d}: indent={1} {1!r}'.format(i+1, indent, lines[i][:80]))