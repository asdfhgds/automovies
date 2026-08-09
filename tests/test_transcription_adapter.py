import json
from pathlib import Path

from transcription import adapter


def test_transcription_adapter_fallback(tmp_path: Path, monkeypatch):
    proj = tmp_path / 'proj'
    proj.mkdir()
    # create dummy source (empty file) and project_meta
    video = proj / 'input.mp4'
    video.write_text('')
    meta = {"project_id": "t1", "title": "Test", "source_path": str(video)}
    (proj / 'project_meta.json').write_text(json.dumps(meta))

    # Force the real backend to fail so the adapter deterministically falls
    # back to the stub (no model loading, no network, fast unit test).
    def _boom(*args, **kwargs):
        raise RuntimeError('whisperx backend unavailable')

    monkeypatch.setattr('transcription.whisperx_adapter.transcribe', _boom)

    # call adapter.transcribe; it should fall back to the stub via adapter logic
    out = adapter.transcribe(proj, str(video))
    assert out.exists()
    data = json.loads(out.read_text(encoding='utf-8'))
    assert 'segments' in data
    # transcript.txt also exists
    assert (proj / 'transcripts' / 'transcript.txt').exists()
