"""Unit tests for the audio-mix render command (ducking/normalization/subtitles).

Requires ffmpeg + ffprobe (already required by the project) but does NOT run a
full render, so it stays fast.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from editing.timeline import TimelineBuilder, TrackType
from editor.ffmpeg_editor import (
    _build_srt,
    build_render_command,
    _probe_has_audio,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is required for render command tests",
)


def _make_clip(path: Path, duration: float = 1.0, with_audio: bool = True):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=24:duration={duration}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def test_probe_has_audio_detects_tracks(tmp_path):
    with_audio = tmp_path / "a.mp4"
    silent = tmp_path / "s.mp4"
    _make_clip(with_audio, with_audio=True)
    _make_clip(silent, with_audio=False)
    assert _probe_has_audio(with_audio) is True
    assert _probe_has_audio(silent) is False


def test_build_render_command_includes_ducking_and_limiter(tmp_path):
    c1 = tmp_path / "c1.mp4"
    c2 = tmp_path / "c2.mp4"
    voice = tmp_path / "voice.wav"
    _make_clip(c1)
    _make_clip(c2)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration=2",
         "-ar", "44100", "-c:a", "pcm_s16le", str(voice)],
        check=True, capture_output=True, text=True,
    )
    info = build_render_command(
        clip_paths=[c1, c2],
        voice_path=voice,
        output_path=tmp_path / "out.mp4",
        srt_path=None,
        music_path=None,
    )
    fc = info["filter_complex"]
    assert "sidechaincompress" in fc          # film ducking keyed on narration
    assert "loudnorm=I=-16" in fc             # loudness normalization
    assert "alimiter=limit=0.95" in fc        # true-peak limiter (no clipping)
    assert "amix=inputs=2" in fc              # film + voice
    assert info["ducking"] is True
    assert info["subtitle_burned"] is False
    # video label + audio label are mapped
    assert "-map" in info["command"]
    assert info["total_duration_sec"] > 0


def test_build_render_command_includes_music_ducking(tmp_path):
    c1 = tmp_path / "c1.mp4"
    voice = tmp_path / "voice.wav"
    music = tmp_path / "music.wav"
    _make_clip(c1)
    for out, freq, dur in ((voice, 300, 2.0), (music, 150, 2.0)):
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={dur}",
             "-ar", "44100", "-c:a", "pcm_s16le", str(out)],
            check=True, capture_output=True, text=True,
        )
    info = build_render_command(
        clip_paths=[c1],
        voice_path=voice,
        output_path=tmp_path / "out.mp4",
        srt_path=None,
        music_path=music,
    )
    fc = info["filter_complex"]
    assert "amix=inputs=3" in fc
    assert "[mus][voice]sidechaincompress" in fc
    assert info["music_used"] is True


def test_build_render_command_handles_silent_clips(tmp_path):
    c1 = tmp_path / "c1.mp4"
    voice = tmp_path / "voice.wav"
    _make_clip(c1, with_audio=False)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=1.5",
         "-ar", "44100", "-c:a", "pcm_s16le", str(voice)],
        check=True, capture_output=True, text=True,
    )
    info = build_render_command(
        clip_paths=[c1], voice_path=voice, output_path=tmp_path / "out.mp4"
    )
    # silent fallback input for film audio (anullsrc as an ffmpeg input)
    assert any(arg.startswith("anullsrc=") for arg in info["command"])
    assert "anull[film]" in info["filter_complex"]


def test_build_srt_writes_valid_timestamps(tmp_path):
    builder = TimelineBuilder(10.0)
    builder.add_subtitle("First line", 0.0, 2.0)
    builder.add_subtitle("Second line", 2.5, 3.0)
    timeline = builder.build()
    srt = _build_srt(timeline, tmp_path / "subs.srt")
    text = srt.read_text(encoding="utf-8")
    assert "First line" in text
    assert "00:00:00,000 --> 00:00:02,000" in text
    assert "00:00:02,500 --> 00:00:05,500" in text
    assert text.count("-->") == 2
