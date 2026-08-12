"""Editorial renderer.

Consumes ``timeline/editorial_timeline.json`` and builds the ffmpeg command for
an *edited* short: per-excerpt SPEED / CROP / HOLD directives, CUT / CROSSFADE
/ FADE transitions between segments, edge FADEs, and the same narration-
dominant audio mix as the plain renderer (shared ``_audio_mix_chains`` helper).
"""
from pathlib import Path
from typing import Dict, List, Optional

from editor.ffmpeg_editor import (
    AUDIO_SR,
    EXPORT_FPS,
    EXPORT_HEIGHT,
    EXPORT_WIDTH,
    _audio_mix_chains,
    _escape_srt_path,
    _probe_has_audio,
)
from editorial.timeline import CROSSFADE_SEC, CUT_SEC, FADE_SEC

_TRANSITION_DURATION = {
    "cut": CUT_SEC,
    "crossfade": CROSSFADE_SEC,
    "fade": FADE_SEC,
}
_XFADE_TRANSITION = {
    "cut": "fadeblack",      # ~1 frame black dip == a hard cut in the xfade chain
    "crossfade": "dissolve",
    "fade": "fadeblack",
}

_EDGE_FADE_IN = 0.4
_EDGE_FADE_OUT = 0.8


def build_editorial_render_command(
    editorial_timeline: dict,
    voice_path: Path,
    output_path: Path,
    srt_path: Optional[Path] = None,
    music_path: Optional[Path] = None,
    width: int = EXPORT_WIDTH,
    height: int = EXPORT_HEIGHT,
    fps: int = EXPORT_FPS,
    audio_sr: int = AUDIO_SR,
    normalize: bool = True,
) -> Dict:
    """Build the ffmpeg command for the editorial cut (unit-testable)."""
    voice_path = Path(voice_path)
    output_path = Path(output_path)
    segments = editorial_timeline.get("segments", [])
    flat = _flatten(segments)
    if not flat:
        raise ValueError("editorial timeline has no video clips to render")

    narration_total = float(editorial_timeline.get("narration_total_sec", 0.0))
    music_path_p = Path(music_path) if music_path else None

    # ---- inputs ----
    inputs = []
    input_count = 0
    clip_input_index: Dict[str, int] = {}
    transformed_durations: List[float] = []
    audio_clip_indices: List[int] = []

    for i, item in enumerate(flat):
        clip = item["clip"]
        path = Path(clip["content_path"])
        inputs += ["-i", str(path)]
        clip_input_index[path.name] = i
        input_count += 1
        transformed_durations.append(float(clip.get("duration_sec", 0.5)))
        if not clip.get("mute_film_audio", False) and _probe_has_audio(path):
            audio_clip_indices.append(i)

    inputs += ["-i", str(voice_path)]
    voice_index = input_count
    input_count += 1
    music_index = None
    if music_path_p is not None:
        inputs += ["-i", str(music_path_p)]
        music_index = input_count
        input_count += 1

    boundaries = _boundaries(flat)
    video_total = _xchain_total(transformed_durations, boundaries)
    total_duration = max(video_total, narration_total)
    pad = max(0.0, total_duration - video_total)

    chains: List[str] = []

    # ---- video chain: per-clip transforms ----
    for i, item in enumerate(flat):
        clip = item["clip"]
        speed = float(clip.get("speed", 1.0))
        crop_zoom = float(clip.get("crop_zoom", 1.0))
        hold = float(clip.get("hold_sec", 0.0))
        vf = [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
        ]
        if crop_zoom > 1.001:
            z = crop_zoom
            vf.append(f"crop=w=iw/{z}:h=ih/{z},scale={width}:{height},setsar=1")
        if abs(speed - 1.0) > 0.001:
            vf.append(f"setpts=PTS/{speed}")
        # Normalize every chain to a common frame rate and timebase. xfade
        # hard-fails when its inputs carry different timebases (e.g. clips cut
        # from a 15fps source vs a 30fps one re-encode to 1/15360 vs 1/30), so
        # fps= is applied to ALL clips, not just speed-adjusted ones.
        vf.append(f"fps={fps}")
        if hold > 0.001:
            vf.append(f"tpad=stop_mode=clone:stop_duration={hold:.3f}")
        chains.append(f"[{i}:v]{','.join(vf)}[e{i}]")

    video_label = _xchain_video(
        chains, len(flat), [f"[e{i}]" for i in range(len(flat))],
        transformed_durations, boundaries,
    )

    # Edge fades: open from black, close to black.
    chains.append(
        f"{video_label}fade=t=in:st=0:d={_EDGE_FADE_IN},"
        f"fade=t=out:st={max(0.0, video_total - _EDGE_FADE_OUT):.3f}:"
        f"d={_EDGE_FADE_OUT}[vfc]"
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

    # ---- audio: shared narration-dominant mix ----
    silence_index = None
    if not audio_clip_indices:
        inputs += [
            "-f", "lavfi", "-t", f"{video_total:.3f}",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={audio_sr}",
        ]
        silence_index = input_count
        input_count += 1
    music_used, normalization = _audio_mix_chains(
        chains, audio_clip_indices, silence_index, voice_index, audio_sr,
        music_path_p, music_index, normalize,
    )

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
        str(output_path),
    ]
    return {
        "command": command,
        "filter_complex": filter_complex,
        "inputs": inputs,
        "total_duration_sec": round(total_duration, 3),
        "video_total_sec": round(video_total, 3),
        "pad_sec": round(pad, 3),
        "narration_total_sec": round(narration_total, 3),
        "mode": "editorial",
        "subtitle_burned": subtitle_label == "[vsub]",
        "music_used": music_used,
        "normalization": normalization,
        "output_path": str(output_path),
    }


# --------------------------------------------------------------------------
# Assembly (runs ffmpeg with graceful degradation, like the plain renderer)
# --------------------------------------------------------------------------

def build_editorial_srt(script: dict, output_path: Path) -> bool:
    """Write an SRT of short cinema captions from script.json. Returns False if
    there are no captions to burn."""
    from editorial.subtitles import _fmt_ts

    lines = [] if False else []
    count = 0
    for section in script.get("sections", []):
        for cap in section.get("subtitle_captions", []):
            count += 1
            s = float(cap.get("start_sec", 0.0))
            e = float(cap.get("end_sec", s + 1.0))
            lines.append(f"{count}\n{_fmt_ts(s)} --> {_fmt_ts(e)}\n{cap.get('text', '').upper()}\n")
    if count == 0:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def assemble_editorial(project_dir: Path) -> Path:
    """Render the editorial cut -> renders/final_render.mp4 (+ render_job.json)."""
    import json
    import shutil
    import subprocess

    from editor.ffmpeg_editor import (
        _burn_subtitles_supported,
        _find_music,
        _render_job,
    )
    from movie_understanding import movie_memory

    project_dir = Path(project_dir)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for assembly")

    timeline = movie_memory.load_json(project_dir, "timeline/editorial_timeline.json")
    if not timeline:
        raise ValueError("editorial timeline missing (run the editorial planning stage first)")
    voice = project_dir / "audio" / "voice.wav"
    if not voice.exists():
        raise FileNotFoundError("Voiceover audio not found — run TTS before assembling")

    # Reconcile narration pacing with the REAL synthesized voice: the plan only
    # estimates narration length, but the final mix must span the actual speech
    # so the narration is never clipped mid-sentence.
    from editor.ffmpeg_editor import _probe_duration
    from movie_understanding import movie_memory as _mm

    real_voice_sec = _probe_duration(voice)
    if real_voice_sec > float(timeline.get("narration_total_sec", 0.0)):
        timeline["narration_total_sec"] = round(real_voice_sec, 3)
        timeline["total_duration_sec"] = round(real_voice_sec, 3)
        _mm.save_json(project_dir, "timeline/editorial_timeline.json", timeline)
        print(f"Editorial: narration total reconciled to real voice ({real_voice_sec:.1f}s)")

    renders_dir = project_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    out_file = renders_dir / "final_render.mp4"
    script = movie_memory.load_json(project_dir, "script.json", {})

    music_path = _find_music(project_dir)
    srt_path = None
    burn = __import__("os").getenv("BURN_SUBTITLES", "true").lower() != "false"
    if burn and _burn_subtitles_supported():
        srt_path = renders_dir / "subtitles.srt"
        if not build_editorial_srt(script, srt_path):
            print("Editorial SRT empty; continuing without subtitles")
            srt_path = None
    else:
        print("Subtitles disabled (BURN_SUBTITLES=false or ffmpeg lacks libass)")

    render_info = build_editorial_render_command(
        timeline, voice, out_file, srt_path=srt_path, music_path=music_path,
    )
    render_info["music_path"] = str(music_path) if music_path else None
    render_info["subtitle_path"] = str(srt_path) if srt_path else None

    attempts = [(render_info["command"], "editorial full (subtitles + loudnorm)")]
    attempts.append((
        build_editorial_render_command(
            timeline, voice, out_file, srt_path=None, music_path=music_path,
        )["command"],
        "editorial without subtitles",
    ))
    attempts.append((
        build_editorial_render_command(
            timeline, voice, out_file, srt_path=None, music_path=music_path,
            normalize=False,
        )["command"],
        "editorial without subtitles + without loudnorm",
    ))
    cuts_only = _only_cuts(timeline)
    attempts.append((
        build_editorial_render_command(
            cuts_only, voice, out_file, srt_path=None, music_path=music_path,
            normalize=False,
        )["command"],
        "editorial cut-only fallback",
    ))

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
            print(f"   render attempt '{label}' failed: {(exc.stderr or '')[-1500:]}")
    if not done:
        raise RuntimeError(
            f"ffmpeg editorial assembly failed on all attempts.\n{first_err}"
        )
    if not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not produce a render")

    job = {
        "project_id": project_dir.name,
        "status": "done",
        "mode": "editorial",
        "segments": [
            {"seg_id": s["seg_id"], "video": s.get("video", [])}
            for s in timeline.get("segments", [])
        ],
        "subtitles": render_info.get("subtitle_burned"),
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
            "output_path": str(out_file),
        },
        "timeline_path": str(project_dir / "timeline" / "editorial_timeline.json"),
    }
    (renders_dir / "render_job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Assembled editorial render -> {out_file}")
    return out_file


def _only_cuts(timeline: dict) -> dict:
    import copy

    t = copy.deepcopy(timeline)
    for seg in t.get("segments", []):
        seg["transition_to_next"] = "cut"
    return t


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _flatten(segments: List[dict]) -> List[dict]:
    out = []
    for seg_index, seg in enumerate(segments):
        for clip in seg.get("video", []):
            out.append({"seg_index": seg_index, "seg": seg, "clip": clip})
    return out


def _boundaries(flat: List[dict]) -> List[dict]:
    """One junction per consecutive pair: ``{style, duration}``.

    ``style`` comes from the segment that the boundary leaves (its
    ``transition_to_next``); junctions *within* a segment hard cut.
    """
    out = []
    for i in range(len(flat) - 1):
        if flat[i + 1]["seg_index"] != flat[i]["seg_index"]:
            style = flat[i]["seg"].get("transition_to_next", "crossfade")
        else:
            style = "cut"
        out.append({
            "style": style,
            "duration": _TRANSITION_DURATION.get(style, CROSSFADE_SEC),
        })
    return out


def _xchain_total(durations: List[float], boundaries: List[dict]) -> float:
    total = durations[0] if durations else 0.0
    for d, b in zip(durations[1:], boundaries):
        total += d - b["duration"]
    return max(0.1, round(total, 3))


def _xchain_video(chains: list, n_clips: int, labels: List[str],
                  durations: List[float], boundaries: List[dict]) -> str:
    """Link clips with xfade; transition chosen per junction. Returns label."""
    if n_clips == 1:
        return labels[0]
    prev = labels[0]
    offset = durations[0] - boundaries[0]["duration"]
    label = prev
    for i in range(1, n_clips):
        out = f"[xa{i}]"
        b = boundaries[i - 1]
        chains.append(
            f"{prev}{labels[i]}xfade=transition={_XFADE_TRANSITION[b['style']]}:"
            f"duration={b['duration']:.3f}:offset={offset:.3f}{out}"
        )
        prev = out
        label = out
        offset += durations[i] - b["duration"]
    return label