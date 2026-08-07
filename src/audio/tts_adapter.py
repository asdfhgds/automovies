"""TTS adapter stub: write a small placeholder WAV file path and metadata."""
import json
from pathlib import Path

def synthesize_voice(project_dir: Path, script_path: str = None):
    out_dir = Path(project_dir) / 'audio'
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_path = out_dir / 'voice.wav'
    # create small placeholder file
    with voice_path.open('wb') as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVE")
    meta = {"voice_path": str(voice_path), "voice_model": "tts-stub"}
    meta_path = out_dir / 'tts_meta.json'
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Synthesized voice -> {voice_path}")
    return voice_path
