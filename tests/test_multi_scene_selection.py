import json
from pathlib import Path

import pytest

from scene_selection.selector import select_scenes, select_best_scene


def _make_project(tmp_path: Path, scenes, ranking):
    proj = tmp_path / 'proj'
    scenes_dir = proj / 'scenes'
    scenes_dir.mkdir(parents=True)
    (scenes_dir / 'scene_index.json').write_text(json.dumps(scenes), encoding='utf-8')
    (scenes_dir / 'scene_ranking.json').write_text(json.dumps(ranking), encoding='utf-8')
    return proj


def test_select_scenes_writes_multi_and_single_files(tmp_path):
    scenes = [
        {"scene_id": "s1", "start_sec": 0.0, "end_sec": 10.0, "transcript": "alpha"},
        {"scene_id": "s2", "start_sec": 10.0, "end_sec": 20.0, "transcript": "beta"},
        {"scene_id": "s3", "start_sec": 20.0, "end_sec": 30.0, "transcript": "gamma"},
    ]
    ranking = [
        {"scene_id": "s2", "score": 0.9, "reason": "best"},
        {"scene_id": "s1", "score": 0.8, "reason": "second"},
        {"scene_id": "s3", "score": 0.7, "reason": "third"},
    ]
    proj = _make_project(tmp_path, scenes, ranking)

    entries = select_scenes(proj, top_n=2)

    assert len(entries) == 2
    assert [e["scene_id"] for e in entries] == ["s2", "s1"]
    assert [e["order"] for e in entries] == [0, 1]
    assert entries[0]["start_sec"] == 10.0
    assert entries[0]["end_sec"] == 20.0

    multi = json.loads((proj / 'scenes' / 'selected_scenes.json').read_text(encoding='utf-8'))
    single = json.loads((proj / 'scenes' / 'selected_scene.json').read_text(encoding='utf-8'))
    assert multi == entries
    assert single["scene_id"] == "s2"


def test_select_scenes_skips_overlapping_scenes(tmp_path):
    scenes = [
        {"scene_id": "s1", "start_sec": 0.0, "end_sec": 10.0},
        {"scene_id": "s2", "start_sec": 5.0, "end_sec": 15.0},  # overlaps s1
        {"scene_id": "s3", "start_sec": 15.0, "end_sec": 25.0},
    ]
    ranking = [
        {"scene_id": "s2", "score": 1.0, "reason": "top"},
        {"scene_id": "s1", "score": 0.5, "reason": "a"},
        {"scene_id": "s3", "score": 0.5, "reason": "b"},
    ]
    proj = _make_project(tmp_path, scenes, ranking)

    entries = select_scenes(proj, top_n=3)

    # s1 overlaps s2, so it must be dropped; s3 is non-overlapping
    assert [e["scene_id"] for e in entries] == ["s2", "s3"]


def test_select_scenes_filters_invalid_timestamps(tmp_path):
    scenes = [
        {"scene_id": "s1", "start_sec": 0.0, "end_sec": 1.0},
        {"scene_id": "bad", "start_sec": 5.0, "end_sec": 5.0},   # zero length
        {"scene_id": "none", "start_sec": None, "end_sec": None},
    ]
    ranking = [
        {"scene_id": "bad", "score": 1.0, "reason": "x"},
        {"scene_id": "none", "score": 0.9, "reason": "y"},
        {"scene_id": "s1", "score": 0.4, "reason": "z"},
    ]
    proj = _make_project(tmp_path, scenes, ranking)

    entries = select_scenes(proj, top_n=3)
    assert [e["scene_id"] for e in entries] == ["s1"]


def test_select_best_scene_backward_compat(tmp_path):
    scenes = [
        {"scene_id": "s1", "start_sec": 0.0, "end_sec": 5.0},
        {"scene_id": "s2", "start_sec": 5.0, "end_sec": 10.0},
    ]
    ranking = [
        {"scene_id": "s1", "score": 0.9, "reason": "top"},
        {"scene_id": "s2", "score": 0.2, "reason": "low"},
    ]
    proj = _make_project(tmp_path, scenes, ranking)

    path = select_best_scene(proj)
    assert path.name == "selected_scene.json"
    single = json.loads(path.read_text(encoding='utf-8'))
    assert single["scene_id"] == "s1"


def test_select_scenes_missing_ranking_raises(tmp_path):
    proj = tmp_path / 'proj'
    (proj / 'scenes').mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        select_scenes(proj, top_n=3)
