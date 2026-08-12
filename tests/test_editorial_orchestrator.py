"""Editorial orchestrator wiring: EDITORIAL_MODE=true produces the expected
artifacts (director_plan.json, editorial_plan.json, script.json, timeline/
editorial_timeline.json, renders/final_render.mp4, provider_manifest.json,
qc_report.json). Runs the REAL start_pipeline on a local testsrc movie with
stub transcription/scene/LLM providers and mock TTS, then real FFmpeg.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from tests.editorial_fixtures import SEGMENTS, SCENES


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_source(path: Path, duration: float = 30.0):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=duration=%.1f:size=320x180:rate=15" % duration,
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, text=True,
    )


def _seed_fixture(project_dir: Path):
    """Seed scene_index + transcript so the stub stages have real content."""
    project_dir = Path(project_dir)
    (project_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (project_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (project_dir / "scenes" / "scene_index.json").write_text(
        json.dumps(SCENES), encoding="utf-8")
    (project_dir / "transcripts" / "transcript.json").write_text(
        json.dumps({"segments": SEGMENTS}), encoding="utf-8")


def _noop_transcribe(project_dir: Path, source_path=None):
    """Preserve the seeded transcript instead of running real WhisperX on CPU."""
    return project_dir / "transcripts" / "transcript.json"


def _noop_scene_cards(project_dir: Path, source_path=None):
    """Preserve the seeded scene index; write scene_cards.json for QC."""
    project_dir = Path(project_dir)
    (project_dir / "scenes").mkdir(parents=True, exist_ok=True)
    import json as _j
    idx = (project_dir / "scenes" / "scene_index.json")
    if idx.exists():
        scenes = _j.loads(idx.read_text(encoding="utf-8"))
        (project_dir / "scenes" / "scene_cards.json").write_text(
            _j.dumps(scenes), encoding="utf-8")
    return project_dir / "scenes" / "scene_cards.json"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg required")
def test_orchestrator_editorial_mode_produces_expected_artifacts(tmp_path, monkeypatch):
    import app.orchestrator as orch

    # Providers: stub transcription/scenes, deterministic director, mock TTS.
    monkeypatch.setenv("EDITORIAL_MODE", "true")
    monkeypatch.setenv("STUDIO_PROFILE", "local")
    monkeypatch.setenv("CREATIVE_DIRECTOR_ENABLED", "false")
    monkeypatch.setenv("TTS_PROVIDER", "mock")
    monkeypatch.setenv("SCRIPT_PROVIDER", "mock")
    monkeypatch.setenv("REQUIRE_REAL_LLM", "false")
    monkeypatch.setenv("REQUIRE_REAL_TTS", "false")
    monkeypatch.setenv("BURN_SUBTITLES", "true")

    # Redirect the orchestrator's data dir to the temp tree.
    monkeypatch.setattr(orch, "ROOT", tmp_path)

    # Keep the heavy real-model stages (WhisperX/PySceneDetect) out of this
    # wiring test: patch them to preserve the seeded fixture instead.
    monkeypatch.setattr("transcription.adapter.transcribe", _noop_transcribe)
    monkeypatch.setattr("scene_indexing.adapter.build_scene_cards", _noop_scene_cards)

    data_dir = tmp_path / "data"
    source = tmp_path / "movie.mp4"
    _make_source(source, duration=30.0)
    project_dir = data_dir / "proj-1"
    project_dir.mkdir(parents=True)
    (project_dir / "project_meta.json").write_text(
        json.dumps({"project_id": "proj-1", "title": "Editorial Test",
                    "source_path": str(source)}), encoding="utf-8")
    _seed_fixture(project_dir)

    orch.start_pipeline("proj-1")

    essentials = [
        "director_plan.json",
        "editorial_plan.json",
        "script.json",
        "movie_index.json",
        "timeline/editorial_timeline.json",
        "renders/final_render.mp4",
        "provider_manifest.json",
        "reports/qc_report.json",
    ]
    for rel in essentials:
        assert (project_dir / rel).exists(), f"missing artifact: {rel}"

    # Editorial script shape.
    script = json.loads((project_dir / "script.json").read_text(encoding="utf-8"))
    assert script.get("editorial") is True
    assert isinstance(script.get("sections"), list) and script["sections"]
    assert all(s.get("narrative_evidence") for s in script["sections"])

    # Editorial timeline with extracted excerpts.
    tl = json.loads((project_dir / "timeline" / "editorial_timeline.json").read_text(encoding="utf-8"))
    assert tl["mode"] == "editorial"
    excerpts = [
        Path(c["content_path"])
        for seg in tl["segments"] for c in seg["video"]
    ]
    assert excerpts, "editorial timeline must reference excerpt clips"
    assert all(p.exists() and p.stat().st_size > 0 for p in excerpts)

    # Render is a playable MP4.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(project_dir / "renders" / "final_render.mp4")],
        check=True, capture_output=True, text=True,
    )
    assert float(probe.stdout.strip()) > 0

    # Manifest + QC reflect the editorial path.
    manifest = json.loads((project_dir / "provider_manifest.json").read_text(encoding="utf-8"))
    assert manifest["editorial_mode"] is True
    assert manifest["editorial_plan_built"] is True
    assert manifest["editorial_timeline_built"] is True

    qc = json.loads((project_dir / "reports" / "qc_report.json").read_text(encoding="utf-8"))
    assert qc["checks"]["editorial_timeline"] is True
    assert qc["checks"]["editorial_excerpts"] is True