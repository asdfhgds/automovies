"""Simple scene card builder stub that reads transcript and creates scene_cards.json."""
import json
from pathlib import Path


def build_scene_cards(project_dir: Path):
    transcripts_dir = Path(project_dir) / 'transcripts'
    infile = transcripts_dir / 'transcript.json'
    if not infile.exists():
        raise FileNotFoundError(f"Transcript not found at {infile}")
    with infile.open('r', encoding='utf-8') as f:
        tr = json.load(f)

    scenes = [
        {
            "scene_id": "scene-1",
            "title_id": "example-title",
            "start_sec": 0.0,
            "end_sec": 6.0,
            "transcript": tr.get('full_text',''),
            "summary": "A short sample scene",
            "shot_count": 3,
            "key_frames": [],
            "speaker_labels": [],
            "keywords": ["times", "contrast"]
        }
    ]
    out_dir = Path(project_dir) / 'scenes'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'scene_cards.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    print(f"Wrote scene cards -> {out_path}")
    return out_path
