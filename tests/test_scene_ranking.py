import json
from pathlib import Path

from scene_selection.ranker import rank_scenes


def test_rank_scenes_basic(tmp_path: Path):
    proj = tmp_path / 'proj'
    scenes_dir = proj / 'scenes'
    scenes_dir.mkdir(parents=True)
    scenes = [
        {"scene_id": "scene-1", "transcript": "This scene discusses fate and chance and destiny"},
        {"scene_id": "scene-2", "transcript": "A quiet walk in the park with no mention of fate"}
    ]
    (scenes_dir / 'scene_index.json').write_text(json.dumps(scenes))

    rankings = rank_scenes(proj, "fate and destiny in this scene", top_k=2)
    assert isinstance(rankings, list)
    assert rankings[0]['scene_id'] == 'scene-1'
    assert rankings[0]['score'] > rankings[1]['score']
    # scene_ranking.json should exist
    assert (scenes_dir / 'scene_ranking.json').exists()
