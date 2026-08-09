import json
import shutil
import subprocess
from pathlib import Path

import pytest

from audio.tts_adapter import synthesize_voice
from editor.ffmpeg_editor import assemble, build_timeline
from script.writer import generate_script


def _make_video(path: Path, duration: float = 1.5):
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=24:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is required for rendering tests",
)
def test_pipeline_artifacts_produce_valid_render(tmp_path: Path):
    project = tmp_path / "project"
    (project / "scenes").mkdir(parents=True)
    (project / "assets" / "scenes").mkdir(parents=True)
    (project / "audio").mkdir()

    source = project / "assets" / "scenes" / "scene-1.mp4"
    _make_video(source)
    (project / "project_meta.json").write_text(
        json.dumps({"project_id": "render-test", "title": "Render Test"}),
        encoding="utf-8",
    )
    (project / "director_plan.json").write_text(
        json.dumps({
            "project_id": "render-test",
            "thesis": "contrast gives the scene its emotional force",
            "tone": "analytical",
            "structure": [
                {"id": "intro", "goal": "Hook", "target_seconds": 2},
                {"id": "closing", "goal": "Close", "target_seconds": 2},
            ],
        }),
        encoding="utf-8",
    )
    (project / "scenes" / "scene_index.json").write_text(
        json.dumps([{
            "scene_id": "scene-1",
            "start_sec": 0,
            "end_sec": 1.5,
            "summary": "A character faces a difficult choice",
            "transcript": "A difficult choice",
        }]),
        encoding="utf-8",
    )
    (project / "scenes" / "selected_scene.json").write_text(
        json.dumps({"scene_id": "scene-1", "start_sec": 0, "end_sec": 1.5}),
        encoding="utf-8",
    )

    generate_script(project)
    synthesize_voice(project)
    timeline, timeline_path = build_timeline(project)
    assert timeline_path.exists()
    assert timeline.get_track("video") is not None
    assert timeline.get_track("voice") is not None

    output = assemble(project)
    assert output.exists()
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(probe.stdout.strip()) > 0
    render_job = json.loads((project / "renders" / "render_job.json").read_text())
    assert render_job["status"] == "done"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is required for rendering tests",
)
def test_timeline_uses_all_selected_scene_clips(tmp_path: Path):
    project = tmp_path / "project"
    clips_dir = project / "assets" / "scenes"
    clips_dir.mkdir(parents=True)
    (project / "scenes").mkdir()
    (project / "audio").mkdir()
    _make_video(clips_dir / "scene-1.mp4", duration=0.5)
    _make_video(clips_dir / "scene-2.mp4", duration=0.5)
    # A valid WAV is required only so build_timeline can probe it.
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=mono", "-t", "0.2", str(project / "audio" / "voice.wav"),
    ], check=True)
    (project / "scenes" / "selected_scenes.json").write_text(json.dumps([
        {"scene_id": "scene-1"}, {"scene_id": "scene-2"},
    ]), encoding="utf-8")

    timeline, _ = build_timeline(project)
    video_track = timeline.get_track("video")
    assert [item.content_path.name for item in video_track.items] == ["scene-1.mp4", "scene-2.mp4"]
    assert video_track.items[1].start_sec >= video_track.items[0].end_sec
    output = assemble(project)
    assert output.exists()
    render_job = json.loads((project / "renders" / "render_job.json").read_text())
    assert len(render_job["timeline"]) == 2
