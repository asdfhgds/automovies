import sys
sys.path.insert(0, r'C:\Users\hp\Documents\Default Project\automovies\src')
from director.scene_facts import SceneFacts
from director.grounded import MovieGroundedDirector
import json

facts = SceneFacts.from_movie_intelligence(movie_index={
    'movie': {'title': 'Test Western', 'duration_sec': 180.0},
    'scenes': [
        {'scene_id': 'scene-1', 'start_sec': 0.0, 'end_sec': 30.0, 'transcript': '',
         'story': {'characters': ['Barman'], 'location': 'saloon, dim light',
                   'actions': ['pouring', 'talking'], 'objects': ['revolver', 'glass'],
                   'visual_description': 'close-up of the barman hands',
                   'visual_events': ['a revolver is placed on the bar'],
                   'themes': ['tension', 'confrontation'], 'mood': 'tense',
                   'dialogue': [{'speaker': 'Barman', 'text': 'Keep your hands where I can see them.'}]}},
        {'scene_id': 'scene-2', 'start_sec': 30.0, 'end_sec': 60.0, 'transcript': '',
         'story': {'characters': ['Stranger'], 'location': 'outdoor, street at dusk',
                   'actions': ['riding'], 'objects': ['horse', 'dust'],
                   'visual_description': 'wide shot of a lone rider',
                   'themes': ['solitude'], 'mood': 'somber'}},
        {'scene_id': 'scene-3', 'start_sec': 60.0, 'end_sec': 90.0, 'transcript': '',
         'story': {'location': 'riverbank', 'actions': ['walking'],
                   'objects': ['water', 'counter with various items'],
                   'visual_description': 'a person walks through shallow water',
                   'themes': ['nature'], 'mood': 'serene'}},
    ],
})

class _LLM:
    def __init__(self):
        self.calls = []
    def __call__(self, prompt):
        self.calls.append(prompt)
        if 'finalizing the plan' in prompt or 'finalizing the STRUCTURED plan' in prompt:
            return json.dumps({
                "concept": {
                    "title": "A Different Movie",
                    "hook": "invented",
                    "thesis": "an invented thesis about a hospital clock"
                },
                "format": {"type": "short_video_essay", "duration_sec": 90},
                "editorial_direction": {
                    "pacing": "slow",
                    "visual_style": "close-up on the revolver while the barman talks",
                    "audio_style": "minimal",
                    "editing_style": "quiet cuts"
                }
            })
        return json.dumps({"concepts": [{
            "title": "Real One", "hook": "h",
            "thesis": "a specific grounded claim about the saloon",
            "why_interesting": "w",
            "evidence_refs": [
                {"kind": "scene", "scene_id": "scene-1"},
                {"kind": "object", "value": "revolver"}
            ],
            "visual_opportunity": "close-up", "format": "f"
        }]})

llm = _LLM()
director = MovieGroundedDirector(llm)
facts = SceneFacts.from_movie_intelligence(movie_index={
    'movie': {'title': 'Test Western', 'duration_sec': 180.0},
    'scenes': [
        {'scene_id': 'scene-1', 'start_sec': 0.0, 'end_sec': 30.0, 'transcript': '',
         'story': {'characters': ['Barman'], 'location': 'saloon, dim light',
                   'actions': ['pouring', 'talking'], 'objects': ['revolver', 'glass'],
                   'visual_description': 'close-up of the barman hands',
                   'visual_events': ['a revolver is placed on the bar'],
                   'themes': ['tension', 'confrontation'], 'mood': 'tense',
                   'dialogue': [{'speaker': 'Barman', 'text': 'Keep your hands where I can see them.'}]}},
        {'scene_id': 'scene-2', 'start_sec': 30.0, 'end_sec': 60.0, 'transcript': '',
         'story': {'characters': ['Stranger'], 'location': 'outdoor, street at dusk',
                   'actions': ['riding'], 'objects': ['horse', 'dust'],
                   'visual_description': 'wide shot of a lone rider',
                   'themes': ['solitude'], 'mood': 'somber'}},
        {'scene_id': 'scene-3', 'start_sec': 60.0, 'end_sec': 90.0, 'transcript': '',
         'story': {'location': 'riverbank', 'actions': ['walking'],
                   'objects': ['water', 'counter with various items'],
                   'visual_description': 'a person walks through shallow water',
                   'themes': ['nature'], 'mood': 'serene'}},
    ],
})

res = director.develop({'title': 'T', 'duration_sec': 90}, facts, num_concepts=1, min_coverage=0.4)
print('Result:', res.get('plan'))
print('Plan rejection:', res.get('plan_rejection'))