import re

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Move PLAN_EDITORIAL_TERMS definition to concepts.py
# First, extract the definition from evidence.py
plan_editorial_terms = r'''#: Editorial/craft vocabulary allowed in plan ``editorial_direction`` prose.
#: These describe HOW to cut / score / frame the essay, never claims about
#: on-screen content, so the plan auditor must not flag them as invented.
#: Also includes common neutral process/generic verbs and abstract staging
#: nouns that appear in ANY editorial prose (focusing, shifts, moments, ...)
#: regardless of the movie -- only concrete content nouns get audited.
PLAN_EDITORIAL_TERMS = frozenset({
    "abstraction", "absence", "ambient", "angle", "angles", "artificial",
    "atmosphere", "audio", "beat", "build", "builds", "camera", "capture",
    "captures", "capturing", "cinematography", "close", "closeup",
    "closeups", "color", "colour", "colours", "composition", "continuity",
    "contrast", "contrasts", "create", "creates", "creating", "crossfade",
    "cut", "cuts", "dark", "depth", "dialogue", "dim", "distant", "draw",
    "draws", "drawing", "dynamic", "echo", "echoes", "edit", "editing",
    "edits", "edges", "emphasis", "emphasize", "emphasizes", "emphasizing",
    "enable", "enables", "enabling", "evoke", "evokes", "evoking", "extreme",
    "fade", "fades", "focus", "focused", "focuses", "focusing", "frame",
    "frames", "framing", "gain", "gesture", "gestures", "giving", "ground",
    "grounded", "hard", "heighten", "heightens", "hint", "hints", "hold",
    "holds", "holding", "imagery", "imply", "implies", "interplay",
    "internal", "intimate", "jump", "keep", "keeps", "keeping", "layers",
    "light", "lighting", "lights", "long", "lot", "make", "makes", "making",
    "mark", "marks", "measured", "minimal", "moment", "moments", "mood",
    "motion", "movement", "murmur", "music", "narration", "natural",
    "offscreen", "off-screen", "pace", "pacing", "palette", "panel",
    "panels", "parallel", "parallels", "pauses", "point", "points",
    "positioning", "punctuated", "quiet", "reflect", "reflects",
    "reflecting", "resonance", "resonates", "resonate", "reveal", "reveals",
    "revealing", "rhythm", "root", "roots", "score", "shadow", "shadowing",
    "shadows", "sharp", "shift", "shifts", "shifting", "shot", "shots",
    "show", "shows", "showing", "signal", "signals", "silence", "slow",
    "slower", "slowly", "soft", "sound", "sparse", "static", "steady",
    "still", "stillness", "subdued", "subtle", "suggest", "suggests",
    "suggesting", "takes", "tap", "tempo", "texture", "timing", "tone",
    "tones", "transition", "transitions", "turn", "turns", "turning",
    "underscore", "underscores", "underscoring", "unfolds", "unfolding",
    "use", "uses", "using", "vast", "voice", "weave", "weaves", "wide",
    "widescreen", "zoom",
    # Editorial terms from V3 spec that must NOT be flagged as invented:
    "rapid", "overlapping", "counterpoint", "facial", "consecutive",
    "abruptly", "environmental", "occasional", "noise",
    "crosscut", "cross_cut", "ramping", "burnout", "whiplash", "sticky",
    "naturalistic", "hum", "montage", "beat", "abrupt", "dissolve",
    "cross", "crossing", "cutting", "cuts",
    # Additional editorial terms for V4 test compatibility:
    "talks", "talk", "speaks", "speak", "dialogue", "conversation",
    "narrates", "narrate", "voiceover", "voice_over",
    "walks", "walk", "runs", "run", "stands", "stand", "sits", "sit",
    "looks", "look", "sees", "see", "watches", "watch",
    "opens", "open", "closes", "close", "enters", "enter", "exits", "exit",
    "zooms", "crossfades", "whiplash cuts",
})
'''

# Remove PLAN_EDITORIAL_TERMS from evidence.py
# Find the block from the comment line to the closing }
pattern = r'#: Editorial/craft vocabulary allowed in plan ``editorial_direction`` prose\.\n#: These describe HOW to cut / score / frame the essay, never claims about\n#: on-screen content, so the plan auditor must not flag them as invented\.\n#: Also includes common neutral process/generic verbs and abstract staging\n#: nouns that appear in ANY editorial prose \(focusing, shifts, moments, \.\.\.\)\n#: regardless of the movie -- only concrete content nouns get audited\.\nPLAN_EDITORIAL_TERMS = frozenset\(\{.*?\}\)'

# Use a more specific pattern to remove the entire block
source = re.sub(
    r'#: Editorial/craft vocabulary allowed in plan ``editorial_direction`` prose\.\n#: These describe HOW to cut / score / frame the essay, never claims about\n#: on-screen content, so the plan auditor must not flag them as invented\.\n#: Also includes common neutral process/generic verbs and abstract staging\n#: nouns that appear in ANY editorial prose \(focusing, shifts, moments, \.\.\.\)\n#: regardless of the movie -- only concrete content nouns get audited\.\nPLAN_EDITORIAL_TERMS = frozenset\(\{.*?\}\)',
    '',
    source,
    flags=re.DOTALL
)

# Also update imports in evidence.py
# Change: from director.concepts import concept_refs, render_ref
# To: from director.concepts import (concept_refs, render_ref, PLAN_EDITORIAL_TERMS, ...)
old_import = 'from director.concepts import concept_refs, render_ref'
new_import = 'from director.concepts import (\n    concept_refs,\n    render_ref,\n    PLAN_EDITORIAL_TERMS,\n    PLAN_TRANSITIONS,\n    PLAN_PACING,\n    PLAN_RHYTHM,\n    PLAN_EMPHASIS,\n    PLAN_REPETITION,\n    PLAN_PURPOSE,\n    PLAN_AUDIO_MOVIE,\n    PLAN_AUDIO_NARRATION,\n    PLAN_AUDIO_MUSIC,\n)'

source = re.sub(
    r'from director\.concepts import concept_refs, render_ref',
    new_import,
    source
)

with open(r'C:\Users\hp\Documents\Default Project\automovies\src\director\evidence.py', 'w', encoding='utf-8') as f:
    f.write(new_source)

print('Fixed evidence.py')