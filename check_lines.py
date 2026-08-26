with open('V4_Colab_Validation.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Print lines around 380
for i in range(375, 390):
    line = lines[i]
    print(f'{i+1:4d}: {repr(lines[i][:120])}')