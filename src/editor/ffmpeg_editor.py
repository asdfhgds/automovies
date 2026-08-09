"""Timeline-backed FFmpeg renderer for the local MVP profile.

Supports a multi-scene cut: every selected scene clip is concatenated in
selection order, scaled to the export resolution, and mixed with the generated
voiceover. Subtitles from the script are recorded in the timeline (not burned).
"""
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


def _selected_clips(project_dir: Path) -> list:
    """Return [(scene_entry, clip_path), ...] for the selected cut.

    Prefers selected_scenes.json (multi-scene) and falls back to the legacy
    selected_scene.json single-scene file.
    """
    scenes_dir = project_dir / "scenes"
    scenes = []
    multi = scenes_dir / "selected_scenes.json"
    single = scenes_dir / "selected_scene.json"
    if multi.exists():
        data = json.loads(multi.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        scenes = data if isinstance(data, list) else []
    elif single.exists():
        scenes = [json.loads(single.read_text(encoding="utf-8"))]

    clips = []
    for scene in scenes:
        scene_id = scene.get("scene_id")
        if not scene_id:
            continue
        clip = project_dir / "assets" / "scenes" / f"{scene_id}.mp4"
        if clip.exists():
            clips.append((scene, clip))
    return clips


def build_timeline(project_dir: Path):
    """Build and persist a timeline from the selected clips and voiceover."""
    project_dir = Path(project_dir)
    clips = _selected_clips(project_dir)
    voice = project_dir / "audio" / "voice.wav"
    if not clips:
        raise FileNotFoundError("No selected scene clips found")
    if not voice.exists():
        raise FileNotFoundError("Voiceover audio not found")

    clip_durations = [_probe_duration(clip) for _, clip in clips]
    voice_duration = _probe_duration(voice)
    video_total = sum(clip_durations)
    total_duration = max(video_total, voice_duration)

    builder = TimelineBuilder(total_duration)
    cursor = 0.0
    for (_, clip), duration in zip(clips, clip_durations):
        builder.add_video_clip(clip, cursor, duration)
        cursor += duration
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
    voice = voice_track.items[0].content_path if voice_track and voice_track.items else None
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
    """Render a valid MP4 from the selected clips and generated voiceover."""
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
    if not video_track or not video_track.items:
        raise ValueError("Timeline has no video clips")
    if not voice_track or not voice_track.items:
        raise ValueError("Timeline has no voiceover")
    clips = [item.content_path for item in video_track.items]
    voice = voice_track.items[0].content_path
    duration = timeline.total_duration_sec

    inputs = []
    for clip in clips:
        inputs += ["-i", str(clip)]
    inputs += ["-i", str(voice)]
    voice_index = len(clips)

    chains = []
    for i in range(len(clips)):
        chains.append(
            f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
        )
    concat_in = "".join(f"[v{i}]" for i in range(len(clips)))
    chains.append(f"{concat_in}concat=n={len(clips)}:v=1:a=0[v]")
    filter_complex = ";".join(chains)

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", f"{voice_index}:a:0",
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
