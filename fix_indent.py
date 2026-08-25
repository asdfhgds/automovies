import sys

with open('src/director/editorial_director.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the indentation issue around line 435
# The issue is that after the return pacing line, there's an empty line with wrong indent

# Find the line with "return pacing" and fix the following blank line
for i in range(len(lines)):
    if i > 0 and lines[i].strip() == 'return pacing':
        # Check the next non-empty line
        for j in range(i+1, min(i+5, len(lines))):
            if lines[j].strip() == '':
                lines[i+1] = '\n'
                break
            elif lines[i+1].strip() != '':
                break
        break

with open('src/director/editorial_director.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed')