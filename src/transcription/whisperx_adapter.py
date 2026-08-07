"""WhisperX adapter skeleton.

This module intentionally does not import heavy libraries at import time. When transcribe() is called,
it will attempt to import whisperx and run transcription if available. If not available, it falls back
to the stub.

For now, if whisperx is available but we need a real transcription, we use openai/whisper as a fallback.
"""
import json
from pathlib import Path
from typing import Optional


def transcribe(project_dir: Path, source_path: Optional[str] = None):
    """Transcribe using whisperx if available, else use openai/whisper as fallback.

    Returns path to transcript.json (normalized format with 'segments' key)
    
    Also creates a human-readable transcript.txt file.
    """
    out_dir = Path(project_dir) / 'transcripts'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try whisper (openai/whisper) for real transcription on CPU or GPU
    try:
        import whisper  # type: ignore
        print(f"Running openai/whisper model on {source_path}")
        model = whisper.load_model("small")
        if source_path is None:
            raise RuntimeError("No source_path provided for transcription")
        result = model.transcribe(str(source_path))
        
        # Build normalized transcript structure
        segments = []
        full_text_lines = []
        for i, seg in enumerate(result.get('segments', [])):
            segment_obj = {
                "id": f"seg_{i:03d}",
                "start_sec": seg.get('start'),
                "end_sec": seg.get('end'),
                "text": seg.get('text','').strip(),
                "speaker": None,
                "words": []
            }
            segments.append(segment_obj)
            full_text_lines.append(segment_obj['text'])
            
        transcript = {
            "provider": "whisper",
            "source": str(source_path),
            "language": result.get('language'),
            "segments": segments
        }
        p = out_dir / 'transcript.json'
        with p.open('w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        
        # Also write human-readable text version
        txt_path = out_dir / 'transcript.txt'
        with txt_path.open('w', encoding='utf-8') as f:
            f.write(' '.join(full_text_lines))
        
        print(f"(whisper) wrote transcript -> {p}")
        return p
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        # If no real backend works, return a placeholder with proper structure
        transcript = {
            "provider": "none",
            "source": str(source_path) if source_path else None,
            "language": None,
            "segments": []
        }
        p = out_dir / 'transcript.json'
        with p.open('w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        
        # Also write empty text version
        txt_path = out_dir / 'transcript.txt'
        txt_path.write_text('')
        
        print(f"No transcription backend succeeded; wrote placeholder -> {p}")
        return p
