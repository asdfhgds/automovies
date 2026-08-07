import json
from pathlib import Path
import tempfile

from director.planner import plan_director
from scene_selection.ranker import rank_scenes


def test_director_produces_thesis_and_ranker_consumes(tmp_path: Path):
    # prepare simple project structure with scenes
    proj = tmp_path / 'proj'
    scenes_dir = proj / 'scenes'
    scenes_dir.mkdir(parents=True)

    scenes = [
        {"scene_id": "scene_001", "start_sec": 0.0, "end_sec": 5.0, "transcript": "Hello world this is a test"},
        {"scene_id": "scene_002", "start_sec": 5.0, "end_sec": 20.0, "transcript": "Dramatic reveal and emotional conflict rise here repeated repeated"}
    ]
    (scenes_dir / 'scene_index.json').write_text(json.dumps(scenes))

    # run director
    out = plan_director(proj, title='Unit Test')
    assert out.exists()
    data = json.loads(out.read_text(encoding='utf-8'))
    assert 'thesis' in data and isinstance(data['thesis'], str)

    # pass thesis to ranker and ensure ranking file created
    rankings = rank_scenes(proj, data['thesis'], top_k=2)
    assert isinstance(rankings, list)
    ranking_file = proj / 'scenes' / 'scene_ranking.json'
    assert ranking_file.exists()
    rf = json.loads(ranking_file.read_text(encoding='utf-8'))
    assert len(rf) > 0
    # top ranking should be scene_002 because it has more words and repeated tokens
    assert rf[0]['scene_id'] == 'scene_002'
