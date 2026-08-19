"""End-to-end editorial render: real excerpt clips + narration -> edited MP4.

Mirrors test_timeline_rendering but through the editorial path (excerpts,
SPEED/CROP/HOLD directives, transitions, caption SRT, editorial render job).
Requires local ffmpeg/ffprobe.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from tests.editorial_fixtures import seed_project


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_source(path: Path, duration: float = 12.0):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=%.1f:size=320x180:rate=15" % duration,
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, text=True,
    )


def _make_voice(path: Path, duration: float = 8.0):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=%.1f" % duration,
         "-ar", "44100", "-c:a", "pcm_s16le", str(path)],
        check=True, capture_output=True, text=True,
    )


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg required")
def test_editorial_pipeline_produces_valid_render(tmp_path):
    from movie_understanding.analyzer import MovieAnalyzer
    from editorial.director import create_editorial_plan
    from editorial.script import build_editorial_script
    from editorial.timeline import EditorialTimelineBuilder
    from editorial.render import assemble_editorial

    source = tmp_path / "movie.mp4"
    # The fixture scenes span 0..24s, so the source must contain them or the
    # multi-scene gate (>=3 distinct excerpts) legitimately fails.
    _make_source(source, duration=30.0)
    seed_project(tmp_path)
    _make_voice(tmp_path / "audio" / "voice.wav", duration=8.0)
    (tmp_path / "project_meta.json").write_text(
        json.dumps({"project_id": "t", "title": "Coinflip",
                    "source_path": str(source)}), encoding="utf-8")
    (tmp_path / "movie_index.json").write_text(
        json.dumps(MovieAnalyzer().analyze(tmp_path)), encoding="utf-8")

    plan = create_editorial_plan(tmp_path, target_sec=60)
    movie_index = json.loads((tmp_path / "movie_index.json").read_text(encoding="utf-8"))
    script = build_editorial_script(tmp_path, plan, movie_index)

    timeline = EditorialTimelineBuilder(source_path=str(source)).build(tmp_path, plan, script)
    assert timeline["mode"] == "editorial"
    excerpts = [
        Path(c["content_path"])
        for seg in timeline["segments"] for c in seg["video"]
    ]
    assert excerpts, "excerpt clips should be extracted from the source movie"
    assert all(p.exists() and p.stat().st_size > 0 for p in excerpts)

    output = assemble_editorial(tmp_path)
    assert output.exists() and output.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(output)],
        check=True, capture_output=True, text=True,
    )
    assert float(probe.stdout.strip()) > 0

    job = json.loads((tmp_path / "renders" / "render_job.json").read_text(encoding="utf-8"))
    assert job["status"] == "done"
    assert job["mode"] == "editorial"
    assert job["audio_mix"]["no_clipping"] is True

    srt = tmp_path / "renders" / "subtitles.srt"
    assert srt.exists(), "short-caption SRT should be written"
    srt_text = srt.read_text(encoding="utf-8")
    # cinematic captions: short uppercase lines with SRT timestamps
    assert "-->" in srt_text and srt_text.strip()