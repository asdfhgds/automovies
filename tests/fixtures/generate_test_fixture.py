"""Generate a small test video fixture with optional speech.

Usage:
    python generate_test_fixture.py output.mp4 "Text to speak"

The script will try these methods in order:
- Use pyttsx3 (offline) to synthesize speech to WAV, then combine with ffmpeg testsrc to produce an MP4.
- If pyttsx3 is unavailable, produce a silent MP4 with a short test pattern and note that speech is missing.

This script is intended for local test setup and is not run automatically in CI. It avoids committing large binaries to the repo.
"""
import sys
from pathlib import Path
import subprocess
import shutil


def has_ffmpeg():
    return shutil.which('ffmpeg') is not None


def synthesize_with_pyttsx3(text: str, wav_out: Path):
    try:
        import pyttsx3
    except Exception:
        return False
    engine = pyttsx3.init()
    engine.save_to_file(text, str(wav_out))
    engine.runAndWait()
    return True


def make_video_with_audio(wav_path: Path, out_mp4: Path, duration: float = 3.0):
    # create a simple testsrc video and mux with provided wav
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', f"testsrc=size=320x240:rate=25:duration={duration}",
        '-i', str(wav_path),
        '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(out_mp4)
    ]
    subprocess.run(cmd, check=True)


def make_silent_video(out_mp4: Path, duration: float = 3.0):
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', f"testsrc=size=320x240:rate=25:duration={duration}",
        '-c:v', 'libx264', '-an', str(out_mp4)
    ]
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 2:
        print('Usage: python generate_test_fixture.py output.mp4 "Optional text to speak"')
        return 1
    out = Path(sys.argv[1])
    text = sys.argv[2] if len(sys.argv) > 2 else 'Hello, this is a test.'

    out.parent.mkdir(parents=True, exist_ok=True)

    if not has_ffmpeg():
        print('ffmpeg not available on PATH. Cannot produce test video.')
        return 1

    # try pyttsx3
    wav_tmp = out.with_suffix('.wav')
    ok = synthesize_with_pyttsx3(text, wav_tmp)
    if ok and wav_tmp.exists():
        try:
            make_video_with_audio(wav_tmp, out)
            print(f'Created test video with speech -> {out}')
            wav_tmp.unlink()
            return 0
        except Exception as e:
            print('Failed to mux audio with ffmpeg:', e)
            if wav_tmp.exists():
                wav_tmp.unlink()
    # fallback: silent video
    try:
        make_silent_video(out)
        print(f'Created silent test video -> {out} (no speech)')
        return 0
    except Exception as e:
        print('Failed to create silent video:', e)
        return 1

if __name__ == '__main__':
    sys.exit(main())
