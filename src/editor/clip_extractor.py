"""FFmpeg-based clip extractor.

Provides extract_clip(source_path, start_sec, end_sec, output_path)
Returns output_path on success, raises RuntimeError on failure.
"""
from pathlib import Path
import shutil
import subprocess


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None


def extract_clip(source_path: str, start_sec: float, end_sec: float, output_path: str, reencode: bool = True):
    src = Path(source_path)
    out = Path(output_path)

    if not src.exists():
        raise FileNotFoundError(f"Source video not found: {src}")
    if start_sec is None or end_sec is None:
        raise ValueError("Start and end times must be provided")
    try:
        start = float(start_sec)
        end = float(end_sec)
    except Exception:
        raise ValueError("Invalid start or end time")
    if end <= start:
        raise ValueError("End time must be greater than start time")

    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg or ffprobe not found on PATH")

    out.parent.mkdir(parents=True, exist_ok=True)

    # Use -ss after -i for accurate cutting (slower) and re-encode for safety
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src),
        '-ss', str(start), '-to', str(end),
        '-c:v', 'libx264', '-c:a', 'aac',
        str(out)
    ]

    try:
        res = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode('utf-8', errors='ignore')}")

    if not out.exists():
        raise RuntimeError("ffmpeg did not produce output file")

    return out


def probe_duration(path: str) -> float:
    """Return duration in seconds using ffprobe."""
    if not _ffmpeg_available():
        raise RuntimeError("ffprobe not available")
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]
    res = subprocess.run(cmd, check=True, capture_output=True)
    out = res.stdout.decode('utf-8').strip()
    try:
        return float(out)
    except Exception:
        return 0.0
