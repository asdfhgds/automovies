"""Character index.

Deterministic character candidate extraction from transcript text (repeated
capitalized proper nouns). No diarization/NER yet — single-mention characters
and pronoun-only roles are honest gaps. Scene membership + first appearance
come from scene dialogue alignment.
"""
from typing import Dict, List

from movie_understanding import text_utils


def build_character_index(scenes: List[dict], transcript_segments: List[dict]) -> List[dict]:
    """Return characters::

        [{"name", "mentions", "scene_ids", "first_appearance_sec"}]
    """
    mention_times: Dict[str, List[float]] = {}
    scene_ids: Dict[str, List[str]] = {}

    for scene in scenes:
        scene_id = scene.get("scene_id")
        start = float(scene.get("start_sec", 0.0))
        end = float(scene.get("end_sec", 0.0))
        for seg in transcript_segments or []:
            try:
                s = float(seg.get("start_sec", 0.0))
                e = float(seg.get("end_sec", 0.0))
            except (TypeError, ValueError):
                continue
            if s <= end and e >= start:
                for name in text_utils.candidate_names((seg.get("text") or ""), min_mentions=1):
                    mention_times.setdefault(name, []).append(s)
                    if name not in scene_ids.setdefault(scene_id, []):
                        scene_ids[scene_id].append(name)

    characters = []
    for name, times in mention_times.items():
        if len(times) < 2:
            continue
        characters.append({
            "name": name,
            "mentions": len(times),
            "first_appearance_sec": round(min(times), 3),
            "scene_ids": [
                sid for sid, names in scene_ids.items() if name in names
            ],
        })
    characters.sort(key=lambda c: -c["mentions"])
    return characters
