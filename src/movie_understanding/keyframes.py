"""Keyframe extraction for vision-based scene enrichment.

Extracts one representative JPEG frame per scene (or per N samples across the
scene) using FFmpeg so a Qwen3-VL-style encoder can describe what is happening
on screen. Pure FFmpeg: no OpenCV/PyAV dependency. Returns the output paths on
success and raises on failure so callers can degrade to heuristic enrichment.
"""
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def ffmpeg_available() -> bool:
    return _ffmpeg_available()


def _probe_duration(source_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(source_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()}")
    try:
        return float(res.stdout.strip())
    except ValueError:
        raise RuntimeError(f"ffprobe returned non-numeric duration: {res.stdout!r}")


def extract_scene_keyframes(
    source_path: str,
    scene_start_sec: float,
    scene_end_sec: float,
    output_dir,
    scene_id: str = "scene",
    max_frames: int = 1,
) -> List[Path]:
    """Extract up to ``max_frames`` JPEG frames spaced across a scene window.

    Frames are named ``{scene_id}_k{i:02d}.jpg`` and written to ``output_dir``.
    If the window is too short to place ``max_frames`` distinct samples, fewer
    frames are produced. Raises RuntimeError when ffmpeg is unavailable or
    extraction fails; returns the created frame paths on success.
    """
    src = Path(source_path)
    out_dir = Path(output_dir)

    try:
        start = float(scene_start_sec)
        end = float(scene_end_sec)
    except (TypeError, ValueError):
        raise ValueError("Invalid scene start/end times")

    if end <= start:
        raise ValueError("Scene end must be greater than scene start")

    if not src.exists():
        raise FileNotFoundError(f"Source video not found: {src}")

    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg or ffprobe not found on PATH")

    duration = end - start
    max_frames = max(1, int(max_frames))
    n = min(max_frames, max(1, int(duration)))
    if duration < 0.5:
        n = 1

    out_dir.mkdir(parents=True, exist_ok=True)
    frames: List[Path] = []
    for i in range(n):
        # Distribute samples across the window (not just the first frame).
        if n == 1:
            t = start + duration * 0.35  # slightly past the start: less likely to be a disolve edge
        else:
            t = start + duration * ((i + 0.5) / n)
        out_path = out_dir / f"{scene_id}_k{i + 1:02d}.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{t:.3f}",
            "-i", str(src),
            "-frames:v", "1",
            "-q:v", "3",
            str(out_path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg timed out extracting frame for {scene_id}")
        if res.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed extracting frame for {scene_id}: "
                f"{res.stderr.decode('utf-8', errors='ignore')}"
            )
        if not out_path.exists():
            raise RuntimeError(f"ffmpeg did not produce keyframe {out_path}")
        frames.append(out_path)
    return frames


def extract_all_scene_keyframes(
    source_path: str,
    scenes: List[dict],
    output_dir,
    max_frames_per_scene: int = 1,
) -> dict:
    """Extract keyframes for every scene and return ``{scene_id: [paths]}``.

    ``scenes`` entries need ``scene_id`` / ``start_sec`` / ``end_sec``. A failed
    scene is skipped (returns ``""`` path list) so a partial vision pass never
    breaks the rest of the pipeline.
    """
    result: dict = {}
    for scene in scenes:
        sid = scene.get("scene_id")
        if not sid:
            continue
        try:
            frames = extract_scene_keyframes(
                source_path,
                scene.get("start_sec", 0.0),
                scene.get("end_sec", 0.0),
                output_dir,
                scene_id=sid,
                max_frames=max_frames_per_scene,
            )
            result[sid] = [str(p) for p in frames]
        except Exception as e:
            result[sid] = []
    return result


def snapshot_frame(
    source_path: str,
    time_sec: float,
    output_path,
) -> Path:
    """Extract a single frame at a precise timestamp (used by tests and tools)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg or ffprobe not found on PATH")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{float(time_sec):.3f}",
        "-i", str(source_path),
        "-frames:v", "1",
        "-q:v", "3",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting frame: "
            f"{res.stderr.decode('utf-8', errors='ignore')}"
        )
    if not out.exists():
        raise RuntimeError(f"ffmpeg did not produce frame {out}")
    return out