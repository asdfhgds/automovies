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


def _selected_clips(project_dir: Path):
    """Return selected clips in narrative order, with legacy fallback."""
    multi_path = project_dir / "scenes" / "selected_scenes.json"
    legacy_path = project_dir / "scenes" / "selected_scene.json"
    if multi_path.exists():
        selections = json.loads(multi_path.read_text(encoding="utf-8"))
    elif legacy_path.exists():
        selections = [json.loads(legacy_path.read_text(encoding="utf-8"))]
    else:
        return []
    clips = []
    for selection in selections:
        scene_id = selection.get("scene_id")
        clip = project_dir / "assets" / "scenes" / f"{scene_id}.mp4"
        if scene_id and clip.exists():
            clips.append(clip)
    return clips


def build_timeline(project_dir: Path):
    """Build and persist a timeline from the selected scene and voiceover."""
    project_dir = Path(project_dir)
    clips = _selected_clips(project_dir)
    voice = project_dir / "audio" / "voice.wav"
    if not clips:
        raise FileNotFoundError("Selected scene clips not found")
    if not voice.exists():
        raise FileNotFoundError("Voiceover audio not found")

    clip_durations = [_probe_duration(clip) for clip in clips]
    voice_duration = _probe_duration(voice)
    total_duration = max(sum(clip_durations), voice_duration)
    builder = TimelineBuilder(total_duration)
    cursor = 0.0
    for clip, clip_duration in zip(clips, clip_durations):
        builder.add_video_clip(clip, cursor, clip_duration)
        cursor += clip_duration
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
    voice = voice_track.items[0].content_path
    duration = timeline.total_duration_sec

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for item in video_track.items:
        command.extend(["-i", str(item.content_path)])
    voice_input_index = len(video_track.items)
    command.extend(["-i", str(voice)])
    filters = []
    video_labels = []
    for index, _ in enumerate(video_track.items):
        label = f"v{index}"
        filters.append(
            f"[{index}:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[{label}]"
        )
        video_labels.append(f"[{label}]")
    if len(video_labels) == 1:
        filters.append(f"{video_labels[0]}null[v]")
    else:
        filters.append(f"{''.join(video_labels)}concat=n={len(video_labels)}:v=1:a=0[v]")
    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[v]", "-map", f"{voice_input_index}:a:0",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100",
        str(out_file),
    ])
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
