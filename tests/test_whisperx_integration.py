import pytest
import shutil
from pathlib import Path
import subprocess
import json


def has_whisperx():
    try:
        import whisperx  # type: ignore
        return True
    except Exception:
        return False


def has_ffmpeg():
    return shutil.which('ffmpeg') is not None


def test_whisperx_integration(tmp_path: Path):
    """Optional integration test for WhisperX.

    This test is marked to run only when explicitly requested (pytest -m integration).
    It requires:
      - whisperx installed
      - ffmpeg available
      - a small speech test fixture at tests/fixtures/test_speech.mp4

    The test will run transcription and assert transcript.json segments exist.
    """
    pytest.importorskip('whisperx')
    if not has_ffmpeg():
        pytest.skip('ffmpeg not available')

    fixture = Path('tests') / 'fixtures' / 'test_speech.mp4'
    if not fixture.exists():
        pytest.skip('test fixture not present at tests/fixtures/test_speech.mp4; generate locally with tests/fixtures/generate_test_fixture.py')

    # create project dir
    proj = tmp_path / 'proj'
    proj.mkdir()
    # copy fixture to project
    target = proj / 'input.mp4'
    shutil.copyfile(fixture, target)
    # write project meta
    meta = {"project_id": "integration1", "title": "WhisperX Integration", "source_path": str(target)}
    (proj / 'project_meta.json').write_text(json.dumps(meta))

    # run transcription via adapter
    from transcription.adapter import transcribe
    out = transcribe(proj, str(target))
    assert out.exists()
    data = json.loads(out.read_text(encoding='utf-8'))
    assert 'segments' in data
    assert isinstance(data['segments'], list)
