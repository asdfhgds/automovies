"""TTS adapter for the local profile.

The local profile uses the deterministic mock TTS provider.  Unlike the old
stub, it writes a valid WAV whose duration is derived from the generated
script, so FFmpeg can consume it during assembly.
"""
import json
from pathlib import Path
from generation.mock import MockTTSProvider


def synthesize_voice(project_dir: Path, script_path: str = None):
    project_dir = Path(project_dir)
    script_file = Path(script_path) if script_path else project_dir / "script.json"
    if not script_file.exists():
        raise FileNotFoundError(f"Script not found: {script_file}")
    script = json.loads(script_file.read_text(encoding="utf-8"))
    text = script.get("voiceover_text", "").strip()
    if not text:
        raise ValueError("Script contains no voiceover_text")

    out_dir = Path(project_dir) / 'audio'
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_path = out_dir / 'voice.wav'
    result = MockTTSProvider().synthesize(text, output_path=voice_path)
    meta = {
        "voice_path": str(voice_path),
        "voice_model": "mock",
        "duration_sec": result["duration_sec"],
        "sample_rate": result["sample_rate"],
        "text": text,
    }
    meta_path = out_dir / 'tts_meta.json'
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Synthesized voice -> {voice_path}")
    return voice_path
