"""Timeline-backed FFmpeg renderer for the MVP profile.

Supports a multi-scene cut: every selected scene clip is concatenated in
selection order with short dissolve transitions between cuts (and a fade
in/out at the edges), scaled to the export resolution, and mixed with the
generated voiceover. The audio pipeline now provides:

- real narration mixed over the selected movie clips (film audio preserved),
- basic movie-audio ducking (sidechain compressor keyed on the narration),
- music ducking when an ``assets/music/*`` file is present,
- loudness normalization (EBU R128) and a final true-peak limiter,
- burned subtitles from the script (SRT rendered via the ``subtitles`` filter)
  when the local ffmpeg has libass.
"""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from editing.timeline import TimelineBuilder, TrackType

EXPORT_WIDTH = 1280
EXPORT_HEIGHT = 720
EXPORT_FPS = 30
AUDIO_SR = 44100


# --------------------------------------------------------------------------
# ffprobe helpers
# --------------------------------------------------------------------------

def _probe(path: Path) -> Dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-show_entries", "stream=index,codec_type",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    duration = 0.0
    try:
        duration = float(data.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    streams = data.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return {"duration_sec": max(0.0, duration), "video": has_video, "audio": has_audio}


def _probe_duration(path: Path) -> float:
    return max(0.1, _probe(path).get("duration_sec", 0.0))


def _probe_has_audio(path: Path) -> bool:
    return _probe(path).get("audio", False)


def _find_music(project_dir: Path) -> Optional[Path]:
    music_dir = project_dir / "assets" / "music"
    if not music_dir.exists():
        return None
    for ext in ("*.mp3", "*.wav", "*.m4a", "*.aac", "*.ogg", "*.flac"):
        for f in sorted(music_dir.glob(ext)):
            return f
    return None


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


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------

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
        # Distribute subtitle windows proportionally to the REAL narration
        # duration (voice.wav) instead of the rough estimated_seconds, so each
        # line appears roughly when it is actually spoken.
        sections = script.get("sections", [])
        total_est = sum(max(0.1, float(s.get("estimated_seconds", 1))) for s in sections)
        if total_est <= 0:
            total_est = 1.0
        cursor = 0.0
        for section in sections:
            frac = max(0.1, float(section.get("estimated_seconds", 1))) / total_est
            duration = frac * voice_duration
            duration = min(duration, max(0.0, total_duration - cursor))
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


def _build_srt(timeline, srt_path: Path) -> Path:
    """Write an SRT file from the timeline's subtitle (TEXT) track."""
    text_track = timeline.get_track(TrackType.TEXT)
    if not text_track or not text_track.items:
        raise ValueError("Timeline has no subtitle track")

    def _ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, mss = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{mss:03d}"

    lines = []
    for i, item in enumerate(text_track.items, start=1):
        start = max(0.0, item.start_sec)
        end = start + max(0.1, item.duration_sec)
        text = (item.content_text or "").strip()
        if not text:
            continue
        lines.append(f"{i}\n{_ts(start)} --> {_ts(end)}\n{text}\n")
    if not lines:
        raise ValueError("Subtitle track contains no usable text")
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def _escape_srt_path(path: Path) -> str:
    s = str(path)
    s = s.replace("\\", "/")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    return s


def _burn_subtitles_supported() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
        )
        return "subtitles" in out.stdout or "subtitles" in out.stderr
    except Exception:
        return False


# --------------------------------------------------------------------------
# Render command builder (unit-testable without running ffmpeg)
# --------------------------------------------------------------------------

def build_render_command(
    clip_paths: list,
    voice_path: Path,
    output_path: Path,
    srt_path: Optional[Path] = None,
    music_path: Optional[Path] = None,
    width: int = EXPORT_WIDTH,
    height: int = EXPORT_HEIGHT,
    fps: int = EXPORT_FPS,
    audio_sr: int = AUDIO_SR,
    normalize: bool = True,
    crossfade: bool = True,
) -> Dict[str, Any]:
    """Build the ffmpeg command for the full audio-mixed render.

    Returns a dict with the prepared ``command`` list plus metadata so callers
    and tests can inspect the audio pipeline. Set ``normalize=False`` to skip
    loudnorm+limiter (used as a fallback when the normalized graph fails) and
    ``crossfade=False`` to concatenate clips with hard cuts instead of dissolve
    transitions (fallback for ffmpeg builds without the ``xfade`` filter).
    """
    clips = [Path(c) for c in clip_paths]
    voice = Path(voice_path)
    output = Path(output_path)
    if not clips:
        raise ValueError("No scene clips to render")
    if not voice.exists():
        raise FileNotFoundError(f"Voiceover audio not found: {voice}")

    voice_duration = _probe_duration(voice)
    clip_durations = [_probe_duration(c) for c in clips]

    # "The edit": dissolve transitions between consecutive clips so the cut
    # feels intentional instead of clip-hopping. Each transition overlaps two
    # clips by xfade_duration, so the assembled video is (N-1) * xfade_duration
    # shorter than the raw sum. Requires ffmpeg >= 4.3 (`xfade` filter).
    n_clips = len(clips)
    xfade_duration = 0.0
    crossfades_used = crossfade and n_clips >= 2
    if crossfades_used:
        xfade_duration = min(0.6, min(clip_durations) / 2.0)
        video_total = sum(clip_durations) - (n_clips - 1) * xfade_duration
    else:
        video_total = sum(clip_durations)
    total_duration = max(video_total, voice_duration)
    pad = max(0.0, total_duration - video_total)

    # Inputs
    inputs = []
    input_count = 0
    for clip in clips:
        inputs += ["-i", str(clip)]
        input_count += 1
    inputs += ["-i", str(voice)]
    voice_index = input_count
    input_count += 1
    music_index = None
    if music_path is not None:
        inputs += ["-i", str(music_path)]
        music_index = input_count
        input_count += 1

    audio_clip_indices = [i for i, c in enumerate(clips) if _probe_has_audio(c)]
    silence_index = None
    if not audio_clip_indices:
        inputs += [
            "-f", "lavfi",
            "-t", f"{video_total:.3f}",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={audio_sr}",
        ]
        silence_index = input_count
        input_count += 1

    chains = []

    # --- video chain ---
    for i in range(n_clips):
        chains.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
        )

    if crossfades_used:
        # Dissolve between consecutive clips (video_total already accounts for
        # the overlap shortening the cut).
        prev = "[v0]"
        offset = clip_durations[0] - xfade_duration
        for i in range(1, n_clips):
            out = f"[xf{i}]"
            chains.append(
                f"{prev}[v{i}]xfade=transition=dissolve:"
                f"duration={xfade_duration:.3f}:offset={offset:.3f}{out}"
            )
            prev = out
            offset += clip_durations[i] - xfade_duration
        video_label = prev
    else:
        # Hard cuts (single clip, or crossfade disabled as a fallback).
        concat_in = "".join(f"[v{i}]" for i in range(n_clips))
        chains.append(f"{concat_in}concat=n={n_clips}:v=1:a=0[vc]")
        video_label = "[vc]"

    # Edge fades: open from black, close to black at the end of the cut.
    chains.append(
        f"{video_label}fade=t=in:st=0:d=0.4,"
        f"fade=t=out:st={max(0.0, video_total - 0.8):.3f}:d=0.8[vfc]"
    )
    video_label = "[vfc]"

    if pad > 0.001:
        chains.append(f"[vfc]tpad=stop_mode=clone:stop_duration={pad:.3f}[vt]")
        video_label = "[vt]"

    subtitle_label = video_label
    if srt_path is not None and Path(srt_path).exists():
        escaped = _escape_srt_path(srt_path)
        chains.append(
            f"{video_label}subtitles=filename='{escaped}'"
            f":force_style='FontSize=18,MarginV=24'[vsub]"
        )
        subtitle_label = "[vsub]"

    # --- film audio chain (movie dialogue/sound under narration) ---
    if audio_clip_indices:
        for i in audio_clip_indices:
            chains.append(
                f"[{i}:a]aresample={audio_sr},aformat=channel_layouts=stereo,"
                f"volume=0.5[af{i}]"
            )
        if len(audio_clip_indices) == 1:
            i = audio_clip_indices[0]
            chains.append(f"[af{i}]anull[film]")
        else:
            concat_a = "".join(f"[af{i}]" for i in audio_clip_indices)
            chains.append(f"{concat_a}concat=n={len(audio_clip_indices)}:v=0:a=1[film]")
    else:
        chains.append(f"[{silence_index}:a]anull[film]")

    # --- voice chain (fan out once per consumer: a filtergraph output pad can
    # only be consumed once -- newer ffmpeg rejects reuse with "Invalid stream
    # specifier"). Narration gets a small gain so it clearly leads the mix. ---
    voice_consumers = 2 + (1 if music_path is not None else 0)
    voice_pads = "".join(f"[voice{i}]" for i in range(voice_consumers))
    chains.append(
        f"[{voice_index}:a:0]aresample={audio_sr},aformat=channel_layouts=stereo,"
        f"volume=1.25,asplit={voice_consumers}{voice_pads}"
    )

    # Duck film audio hard under narration so the voiceover leads the mix.
    chains.append(
        f"[film][voice0]sidechaincompress=threshold=0.01:ratio=12:attack=15:"
        f"release=300:makeup=1[filmD]"
    )

    mix_inputs = f"[filmD][voice{voice_consumers - 1}]"
    num_mix = 2
    music_used = False
    if music_path is not None:
        chains.append(
            f"[{music_index}:a:0]aresample={audio_sr},aformat=channel_layouts=stereo,"
            f"volume=0.25[mus]"
        )
        chains.append(
            f"[mus][voice1]sidechaincompress=threshold=0.01:ratio=12:attack=20:"
            f"release=400[musD]"
        )
        mix_inputs += "[musD]"
        num_mix = 3
        music_used = True

    # --- mix + normalize + limiter (no clipping) ---
    chains.append(
        f"{mix_inputs}amix=inputs={num_mix}:duration=longest:normalize=0[mix]"
    )
    if normalize:
        chains.append(
            f"[mix]loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary,"
            f"alimiter=limit=0.95[aout]"
        )
        normalization = "loudnorm=-16LUFS+alimiter-0.95"
    else:
        chains.append("[mix]anull[aout]")
        normalization = None

    filter_complex = ";".join(chains)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", subtitle_label, "-map", "[aout]",
        "-t", f"{total_duration:.3f}",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(audio_sr),
        "-movflags", "+faststart",
        str(output),
    ]
    return {
        "command": command,
        "filter_complex": filter_complex,
        "inputs": inputs,
        "total_duration_sec": total_duration,
        "video_total_sec": video_total,
        "pad_sec": pad,
        "crossfades_used": crossfades_used,
        "xfade_duration_sec": xfade_duration,
        "subtitle_burned": subtitle_label == "[vsub]",
        "music_used": music_used,
        "ducking": True,
        "normalization": normalization,
        "output_path": str(output),
    }


# --------------------------------------------------------------------------
# Render job metadata
# --------------------------------------------------------------------------

def _render_job(project_dir: Path, timeline, output_path: Path, render_info: Dict[str, Any]) -> Dict[str, Any]:
    video_track = timeline.get_track(TrackType.VIDEO)
    voice_track = timeline.get_track(TrackType.VOICE)
    voice = voice_track.items[0].content_path if voice_track and voice_track.items else None
    text_track = timeline.get_track(TrackType.TEXT)
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
        "crossfades_used": render_info.get("crossfades_used", False),
        "xfade_duration_sec": render_info.get("xfade_duration_sec", 0.0),
        "subtitles": [
            {
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "text": item.content_text,
            }
            for item in text_track.items
        ] if text_track else [],
        "audio_mix": {
            "voice_path": str(voice),
            "voice_gain_db": 2,
            "film_ducking": "sidechaincompress threshold=0.01 ratio=12 makeup=1",
            "music_path": render_info.get("music_path"),
            "music_gain_db": None,
            "music_ducking": render_info.get("music_used", False),
            "normalization": render_info.get("normalization"),
            "no_clipping": True,
        },
        "export": {
            "format": "mp4",
            "resolution": f"{EXPORT_WIDTH}x{EXPORT_HEIGHT}",
            "fps": EXPORT_FPS,
            "audio_sample_rate": AUDIO_SR,
            "output_path": str(output_path),
        },
    }


# --------------------------------------------------------------------------
# Assembly entry point
# --------------------------------------------------------------------------

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

    music_path = _find_music(project_dir)
    srt_path = None
    burn = __import__("os").getenv("BURN_SUBTITLES", "true").lower() != "false"
    if burn and _burn_subtitles_supported():
        srt_path = renders_dir / "subtitles.srt"
        try:
            _build_srt(timeline, srt_path)
        except Exception as e:
            print(f"Subtitle build failed, continuing without subtitles: {e}")
            srt_path = None
    else:
        print("Subtitles disabled (BURN_SUBTITLES=false or ffmpeg lacks libass)")

    render_info = build_render_command(
        clip_paths=clips,
        voice_path=voice,
        output_path=out_file,
        srt_path=srt_path,
        music_path=music_path,
    )
    render_info["music_path"] = str(music_path) if music_path else None
    render_info["subtitle_path"] = str(srt_path) if srt_path else None

    # Try progressively simpler graphs. The most common Colab failure is the
    # `subtitles` filter (libass/fontconfig); loudnorm can also fail on short or
    # silent inputs. Only if every attempt fails do we raise.
    attempts = [(render_info["command"], "full (subtitles + loudnorm)")]
    attempts.append(
        (
            build_render_command(
                clip_paths=clips, voice_path=voice, output_path=out_file,
                srt_path=None, music_path=music_path,
            )["command"],
            "without subtitles",
        )
    )
    attempts.append(
        (
            build_render_command(
                clip_paths=clips, voice_path=voice, output_path=out_file,
                srt_path=None, music_path=music_path, normalize=False,
            )["command"],
            "without subtitles + without loudnorm",
        )
    )
    attempts.append(
        (
            build_render_command(
                clip_paths=clips, voice_path=voice, output_path=out_file,
                srt_path=None, music_path=music_path, normalize=False,
                crossfade=False,
            )["command"],
            "without subtitles + without loudnorm + without crossfades",
        )
    )

    first_err = None
    done = False
    for cmd, label in attempts:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            done = True
            break
        except subprocess.CalledProcessError as exc:
            if first_err is None:
                first_err = (exc.stderr or "").strip()
            print(f"   render attempt '{label}' failed: {(exc.stderr or '').strip()[-1500:]}")
    if not done:
        raise RuntimeError(
            f"ffmpeg assembly failed on all attempts.\n"
            f"Primary command stderr:\n{first_err}"
        )
    if not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not produce a render")

    job = _render_job(project_dir, timeline, out_file, render_info)
    job["timeline_path"] = str(timeline_path)
    job["status"] = "done"
    (renders_dir / "render_job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Assembled render -> {out_file}")
    return out_file
