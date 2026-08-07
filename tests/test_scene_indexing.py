import json
from pathlib import Path
import pytest

from scene_indexing import adapter


def test_adapter_falls_back_to_stub(tmp_path: Path):
    project_dir = tmp_path / 'proj'
    project_dir.mkdir()
    # create minimal transcripts to satisfy stub
    transcripts = project_dir / 'transcripts'
    transcripts.mkdir()
    (transcripts / 'transcript.json').write_text(json.dumps({"full_text": "Hello world", "words": [{"text": "Hello", "start": 0.1, "end": 0.5}, {"text": "world", "start": 0.6, "end": 1.0}]}))

    # call adapter with no scenedetect installed (test environment may or may not have it)
    out = adapter.build_scene_cards(project_dir)
    assert out.exists()
    data = json.loads(out.read_text(encoding='utf-8'))
    assert isinstance(data, list)
    assert len(data) >= 1


@pytest.mark.skipif(True, reason="Integration test with PySceneDetect and a real video — enable when package+fixture present")
def test_pyscenedetect_integration(tmp_path: Path):
    # Placeholder integration test: requires scenedetect and a small test video fixture
    pass
