import json
import subprocess
from pathlib import Path
import pytest

from scene_selection.ranker import rank_scenes
from scene_selection.selector import select_best_scene, select_scenes
from editor.clip_extractor import extract_clip, probe_duration


def has_ffmpeg():
    from shutil import which
    return which('ffmpeg') is not None and which('ffprobe') is not None


def make_test_video(path: Path, duration: float = 2.0):
    # Create a tiny test video using ffmpeg testsrc and sine audio
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', f'testsrc=size=320x240:rate=25:duration={duration}',
        '-f', 'lavfi', '-i', f'sine=frequency=1000:duration={duration}',
        '-c:v', 'libx264', '-c:a', 'aac', str(path)
    ]
    subprocess.run(cmd, check=True)


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not available")
def test_selection_and_extraction_tmp(tmp_path: Path):
    proj = tmp_path / 'proj'
    proj.mkdir()
    # write project_meta with source video
    data_dir = proj
    video = data_dir / 'input.mp4'
    make_test_video(video, duration=2.0)

    meta = {"project_id": "test1", "title": "Test", "source_path": str(video)}
    (proj / 'project_meta.json').write_text(json.dumps(meta))

    scenes_dir = proj / 'scenes'
    scenes_dir.mkdir()
    # create two scenes covering parts of the video
    scenes = [
        {"scene_id": "scene-1", "start_sec": 0.0, "end_sec": 1.0, "transcript": "fate destiny"},
        {"scene_id": "scene-2", "start_sec": 1.0, "end_sec": 2.0, "transcript": "park and birds"}
    ]
    (scenes_dir / 'scene_index.json').write_text(json.dumps(scenes))

    # create a director plan with a thesis
    plan = {"project_id": "test1", "thesis": "fate destiny"}
    (proj / 'director_plan.json').write_text(json.dumps(plan))

    # run ranking
    rank_scenes(proj, plan['thesis'], top_k=2)
    ranking = json.loads((scenes_dir / 'scene_ranking.json').read_text())
    assert len(ranking) >= 1

    # selection
    sel_path = select_best_scene(proj)
    sel = json.loads(sel_path.read_text())
    assert sel['scene_id'] == 'scene-1'

    # extraction
    out_dir = proj / 'assets' / 'scenes'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sel['scene_id']}.mp4"
    extract_clip(str(video), sel['start_sec'], sel['end_sec'], str(out_file))

    assert out_file.exists()
    dur = probe_duration(str(out_file))
    assert 0.9 <= dur <= 1.5


def test_multi_scene_selection_preserves_legacy_artifact(tmp_path: Path):
    proj = tmp_path / "proj"
    scenes_dir = proj / "scenes"
    scenes_dir.mkdir(parents=True)
    scenes = [
        {"scene_id": "scene-1", "start_sec": 0, "end_sec": 1, "scene_type": "dialogue"},
        {"scene_id": "scene-2", "start_sec": 1, "end_sec": 2, "scene_type": "revelation"},
        {"scene_id": "scene-3", "start_sec": 2, "end_sec": 3, "scene_type": "dialogue"},
    ]
    rankings = [
        {"scene_id": "scene-1", "score": 0.9, "reason": "best"},
        {"scene_id": "scene-2", "score": 0.8, "reason": "second"},
        {"scene_id": "scene-3", "score": 0.7, "reason": "third"},
    ]
    (scenes_dir / "scene_index.json").write_text(json.dumps(scenes))
    (scenes_dir / "scene_ranking.json").write_text(json.dumps(rankings))

    path = select_scenes(
        proj,
        max_scenes=2,
        scene_requirements=[{"purpose": "reveal", "preferred_scene_types": ["revelation"]}],
    )
    selected = json.loads(path.read_text())
    assert [item["scene_id"] for item in selected] == ["scene-2", "scene-1"]
    assert selected[0]["evidence_requirement"]["purpose"] == "reveal"
    assert json.loads((scenes_dir / "selected_scene.json").read_text())["scene_id"] == "scene-2"
