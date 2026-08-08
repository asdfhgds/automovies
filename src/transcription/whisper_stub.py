"""Simple transcription stub that writes a sample transcript.json with timestamps."""
import json
from pathlib import Path

def transcribe(project_dir: Path, source_path: str = None):
    """Create a normalized transcript.json in project_dir/transcripts for testing.

    Produces the normalized 'segments' schema so the rest of the pipeline can consume it.
    """
    out_dir = Path(project_dir) / 'transcripts'
    out_dir.mkdir(parents=True, exist_ok=True)
    words = [
        {"word": "It", "start_sec": 1.2, "end_sec": 1.6},
        {"word": "was", "start_sec": 1.6, "end_sec": 1.9},
        {"word": "the", "start_sec": 1.9, "end_sec": 2.0},
        {"word": "best", "start_sec": 2.0, "end_sec": 2.4},
        {"word": "of", "start_sec": 2.4, "end_sec": 2.6},
        {"word": "times", "start_sec": 2.6, "end_sec": 3.0},
        {"word": "It", "start_sec": 3.1, "end_sec": 3.3},
        {"word": "was", "start_sec": 3.3, "end_sec": 3.6},
        {"word": "the", "start_sec": 3.6, "end_sec": 3.7},
        {"word": "worst", "start_sec": 3.7, "end_sec": 4.3},
        {"word": "of", "start_sec": 4.3, "end_sec": 4.5},
        {"word": "times", "start_sec": 4.5, "end_sec": 5.0}
    ]
    segments = [
        {
            "id": "seg_001",
            "start_sec": 1.2,
            "end_sec": 5.0,
            "text": "It was the best of times. It was the worst of times.",
            "speaker": None,
            "words": words
        }
    ]
    transcript = {
        "source": str(source_path) if source_path else None,
        "language": "en",
        "segments": segments
    }
    p = out_dir / 'transcript.json'
    with p.open('w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    # also write text
    with (out_dir / 'transcript.txt').open('w', encoding='utf-8') as f:
        f.write(segments[0]['text'])
    print(f"Wrote transcript -> {p}")
    return p
