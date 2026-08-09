import json
import os
import shutil
from pathlib import Path
import pytest
import time

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get('STUDIO_RUN_REAL_TESTS') != '1',
        reason='Runs the real transcription/scene-detection pipeline. Enable with STUDIO_RUN_REAL_TESTS=1',
    ),
]


def has_whisperx():
    try:
        import whisperx  # type: ignore
        return True
    except Exception:
        return False


def has_pyscenedetect():
    try:
        import scenedetect  # type: ignore
        return True
    except Exception:
        try:
            import pyscenedetect  # type: ignore
            return True
        except Exception:
            return False


def has_ffmpeg():
    return shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None


def test_end_to_end_pipeline(tmp_path: Path):
    """End-to-end integration test that runs the real pipeline end-to-end.

    Requirements (skip if missing):
      - whisperx installed
      - pyscenedetect installed
      - ffmpeg and ffprobe on PATH
      - a small test fixture at tests/fixtures/test_speech.mp4

    This test is marked integration and should be executed with: pytest -m integration
    """
    if not has_whisperx():
        pytest.skip('whisperx not installed')
    if not has_pyscenedetect():
        pytest.skip('pyscenedetect not installed')
    if not has_ffmpeg():
        pytest.skip('ffmpeg/ffprobe not available')

    fixture = Path('tests') / 'fixtures' / 'test_speech.mp4'
    if not fixture.exists():
        pytest.skip('test fixture not present at tests/fixtures/test_speech.mp4; generate locally with tests/fixtures/generate_test_fixture.py')

    # init project
    from main import init_project
    class A: pass
    args = A()
    args.title = 'Integration Test'
    args.source = str(fixture.resolve())
    project_id = init_project(args)

    # run pipeline
    from app.orchestrator import start_pipeline
    start_pipeline(project_id)

    proj = Path('data') / project_id
    transcripts = proj / 'transcripts' / 'transcript.json'
    scenes_index = proj / 'scenes' / 'scene_index.json'
    ranking = proj / 'scenes' / 'scene_ranking.json'
    selected = proj / 'scenes' / 'selected_scene.json'

    # wait briefly for ffmpeg extraction if pipeline is async; otherwise immediate
    time.sleep(1)

    assert transcripts.exists(), 'transcript.json not produced'
    data = json.loads(transcripts.read_text(encoding='utf-8'))
    assert 'segments' in data and len(data['segments']) > 0

    assert scenes_index.exists(), 'scene_index.json not produced'
    scenes = json.loads(scenes_index.read_text(encoding='utf-8'))
    assert isinstance(scenes, list) and len(scenes) > 0
    for s in scenes:
        assert s.get('start_sec') is not None
        assert s.get('end_sec') is not None
        assert s.get('end_sec') > s.get('start_sec')

    assert ranking.exists(), 'scene_ranking.json not produced'
    rf = json.loads(ranking.read_text(encoding='utf-8'))
    assert isinstance(rf, list) and len(rf) > 0

    assert selected.exists(), 'selected_scene.json not produced'
    sf = json.loads(selected.read_text(encoding='utf-8'))
    assert sf.get('scene_id') in [s.get('scene_id') for s in scenes]

    # check extracted clip exists
    assets_dir = proj / 'assets' / 'scenes'
    assert assets_dir.exists()
    clips = list(assets_dir.glob('*.mp4'))
    assert len(clips) > 0

    # probe the first clip to verify media properties
    clip = clips[0]
    import subprocess
    res = subprocess.run([shutil.which('ffprobe'), '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(clip)], capture_output=True, text=True)
    assert res.returncode == 0
    dur = float(res.stdout.strip())
    assert dur > 0
