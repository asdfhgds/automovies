import re

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Remove PLAN_EDITORIAL_TERMS definition from evidence.py
pattern = r'(#: Editorial/craft vocabulary allowed in plan ``editorial_direction`` prose\.\n#: These describe HOW to cut / score / frame the essay, never claims about\n#: on-screen content, so the plan auditor must not flag them as invented\.\n#: Also includes common neutral process/generic verbs and abstract staging\n#: nouns that appear in ANY editorial prose \(focusing, shifts, moments, \.\.\.\)\n#: regardless of the movie -- only concrete content nouns get audited\.\nPLAN_EDITORIAL_TERMS = frozenset\(\{.*?\))'

source = re.sub(
    r'#: Editorial/craft vocabulary allowed in plan ``editorial_direction`` prose\.\n#: These describe HOW to cut / score / frame the essay, never claims about\n#: on-screen content, so the plan auditor must not flag them as invented\.\n#: Also includes common neutral process/generic verbs and abstract staging\n#: nouns that appear in ANY editorial prose \(focusing, shifts, moments, \.\.\.\)\n#: regardless of the movie -- only concrete content nouns get audited\.\nPLAN_EDITORIAL_TERMS = frozenset\(\{.*?\}\)',
    '',
    source,
    flags=re.DOTALL
)

# Update imports in evidence.py
old_import = 'from director.concepts import concept_refs, render_ref'
new_import = '''from director.concepts import (
    concept_refs,
    render_ref,
    PLAN_EDITORIAL_TERMS,
    PLAN_TRANSITIONS,
    PLAN_PACING,
    PLAN_RHYTHM,
    PLAN_EMPHASIS,
    PLAN_REPETITION,
    PLAN_PURPOSE,
    PLAN_AUDIO_MOVIE,
    PLAN_AUDIO_NARRATION,
    PLAN_AUDIO_MUSIC,
)'''

source = re.sub(
    r'from director\.concepts import concept_refs, render_ref',
    '''from director.concepts import (
    concept_refs,
    render_ref,
    PLAN_EDITORIAL_TERMS,
    PLAN_TRANSITIONS,
    PLAN_PACING,
    PLAN_RHYTHM,
    PLAN_EMPHASIS,
    PLAN_REPETITION,
    PLAN_PURPOSE,
    PLAN_AUDIO_MOVIE,
    PLAN_AUDIO_NARRATION,
    PLAN_AUDIO_MUSIC,
)''',
    source
)

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'w', encoding='utf-8') as f:
    f.write(source)

print('Fixed evidence.py')