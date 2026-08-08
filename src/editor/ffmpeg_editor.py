"""Timeline-backed FFmpeg renderer for the local MVP profile."""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from editing.timeline import TimelineBuilder, TrackType


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(0.1, float(result.stdout.strip()))


def _selected_clip(project_dir: Path) -> Optional[Path]:
    selected_path = project_dir / "scenes" / "selected_scene.json"
    if not selected_path.exists():
        return None
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    scene_id = selected.get("scene_id")
    if not scene_id:
        return None
    clip = project_dir / "assets" / "scenes" / f"{scene_id}.mp4"
    return clip if clip.exists() else None


def build_timeline(project_dir: Path):
    """Build and persist a timeline from the selected scene and voiceover."""
    project_dir = Path(project_dir)
    clip = _selected_clip(project_dir)
    voice = project_dir / "audio" / "voice.wav"
    if clip is None:
        raise FileNotFoundError("Selected scene clip not found")
    if not voice.exists():
        raise FileNotFoundError("Voiceover audio not found")

    clip_duration = _probe_duration(clip)
    voice_duration = _probe_duration(voice)
    total_duration = max(clip_duration, voice_duration)
    builder = TimelineBuilder(total_duration)
    builder.add_video_clip(clip, 0.0, total_duration)
    builder.add_voiceover(voice, 0.0, voice_duration)

    script_path = project_dir / "script.json"
    if script_path.exists():
        script = json.loads(script_path.read_text(encoding="utf-8"))
        cursor = 0.0
        for section in script.get("sections", []):
            duration = max(0.1, float(section.get("estimated_seconds", 1)))
            duration = min(duration, total_duration - cursor)
            if duration <= 0:
                break
            builder.add_subtitle(section.get("text", ""), cursor, duration)
            cursor += duration

    timeline = builder.build()
    timeline_dir = project_dir / "timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = timeline_dir / "timeline.json"
    timeline_path.write_text(
        json.dumps(timeline.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return timeline, timeline_path


def _render_job(project_dir: Path, timeline, output_path: Path) -> Dict[str, Any]:
    video_track = timeline.get_track(TrackType.VIDEO)
    voice_track = timeline.get_track(TrackType.VOICE)
    clip = video_track.items[0].content_path
    voice = voice_track.items[0].content_path
    return {
        "project_id": project_dir.name,
        "timeline": [
            {
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "source_type": "scene_clip",
                "source_path": str(item.content_path),
            }
            for item in video_track.items
        ],
        "audio_mix": {
            "voice_path": str(voice),
            "voice_gain_db": 0,
            "music_gain_db": 0,
        },
        "export": {
            "format": "mp4",
            "resolution": "1280x720",
            "fps": 30,
            "output_path": str(output_path),
        },
    }


def assemble(project_dir: Path):
    """Render a valid MP4 from the selected clip and generated voiceover."""
    project_dir = Path(project_dir)
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required for assembly")

    timeline, timeline_path = build_timeline(project_dir)
    errors = timeline.validate()
    if errors:
        raise ValueError("Invalid timeline: " + "; ".join(errors))

    renders_dir = project_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    out_file = renders_dir / "final_render.mp4"
    video_track = timeline.get_track(TrackType.VIDEO)
    voice_track = timeline.get_track(TrackType.VOICE)
    clip = video_track.items[0].content_path
    voice = voice_track.items[0].content_path
    duration = timeline.total_duration_sec

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-stream_loop", "-1", "-i", str(clip),
        "-i", str(voice),
        "-filter_complex",
        "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v]",
        "-map", "[v]", "-map", "1:a:0",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100",
        str(out_file),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg assembly failed: {exc.stderr.strip()}") from exc
    if not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not produce a render")

    job = _render_job(project_dir, timeline, out_file)
    job["timeline_path"] = str(timeline_path)
    job["status"] = "done"
    (renders_dir / "render_job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Assembled render -> {out_file}")
    return out_file
