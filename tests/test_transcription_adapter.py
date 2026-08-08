import json
from pathlib import Path

from transcription import adapter


def test_transcription_adapter_fallback(tmp_path: Path):
    proj = tmp_path / 'proj'
    proj.mkdir()
    # create dummy source (empty file) and project_meta
    video = proj / 'input.mp4'
    video.write_text('')
    meta = {"project_id": "t1", "title": "Test", "source_path": str(video)}
    (proj / 'project_meta.json').write_text(json.dumps(meta))

    # call adapter.transcribe; because system likely lacks whisperx/whisper, it should fall back to stub via adapter logic
    out = adapter.transcribe(proj, str(video))
    assert out.exists()
    data = json.loads(out.read_text(encoding='utf-8'))
    assert 'segments' in data
    # transcript.txt also exists
    assert (proj / 'transcripts' / 'transcript.txt').exists()
