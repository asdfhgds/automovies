with open('src/director/editorial_director.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Fix the return pacing line - it should be indented by 8 spaces
source = source.replace(
    '            return pacing\n \n    def _build_audio_strategy',
    '        return pacing\n\n    def _build_audio_strategy'
)

with open('src/director/editorial_director.py', 'w', encoding='utf-8') as f:
    f.write(source)

print('Fixed')