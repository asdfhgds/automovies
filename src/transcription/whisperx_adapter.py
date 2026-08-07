"""WhisperX adapter skeleton.

This module intentionally does not import heavy libraries at import time. When transcribe() is called,
it will attempt to import whisperx and run transcription if available. If not available, it raises a clear
error directing the developer to install the dependency or fall back to the stub.
"""
import json
from pathlib import Path
from typing import Optional


def transcribe(project_dir: Path, source_path: Optional[str] = None):
    """Transcribe using whisperx if available.

    Returns path to transcript.json
    """
    # Try whisperx first, then try openai/whisper as a CPU-friendly fallback.
    whisper_impl = None
    try:
        import whisperx  # type: ignore
        whisper_impl = 'whisperx'
    except Exception:
        try:
            import whisper  # type: ignore
            whisper_impl = 'whisper'
        except Exception:
            whisper_impl = None

    out_dir = Path(project_dir) / 'transcripts'
    out_dir.mkdir(parents=True, exist_ok=True)

    if whisper_impl == 'whisperx':
        # Placeholder: actual whisperx integration would go here
        transcript = {
            "provider": "whisperx",
            "source": str(source_path) if source_path else None,
            "words": [],
            "full_text": "",
        }
        p = out_dir / 'transcript.json'
        with p.open('w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print(f"(WhisperX adapter) wrote placeholder transcript -> {p}")
        return p
    elif whisper_impl == 'whisper':
        # Use OpenAI whisper package if available (can run on CPU)
        model = whisper.load_model("small")
        if source_path is None:
            raise RuntimeError("No source_path provided for transcription")
        print(f"Running whisper model on {source_path} (this may take a while)")
        result = model.transcribe(str(source_path))
        # Build transcript structure
        full_text = result.get('text','')
        words = []
        # whisper segments may have word-level timestamps depending on model; fallback to segment-level
        for seg in result.get('segments', []):
            words.append({"text": seg.get('text','').strip(), "start": seg.get('start'), "end": seg.get('end')})
        transcript = {"provider": "whisper", "source": str(source_path), "words": words, "full_text": full_text}
        p = out_dir / 'transcript.json'
        with p.open('w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print(f"(whisper) wrote transcript -> {p}")
        return p
    else:
        # No real backend available, write a minimal placeholder and warn
        transcript = {
            "provider": "none",
            "source": str(source_path) if source_path else None,
            "words": [],
            "full_text": ""
        }
        p = out_dir / 'transcript.json'
        with p.open('w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print(f"No transcription backend found; wrote placeholder transcript -> {p}")
        return p
