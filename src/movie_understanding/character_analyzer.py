"""Character index.

Characters are *only* what diarization/speaker labels give us: a person we can
point at in the transcript and say "this is who is speaking". We deliberately do
NOT invent characters by scanning for capitalized words in the transcript text —
"Ser" / "Com" / sentence-initial words fit that pattern and are noise, not
people.

Consequences we accept and record honestly:

- a speaker label that appears exactly once still counts as a character
  (the person spoke; that is evidence enough to exist),
- transcripts without diarization yield an empty character index,
  ``scene.story.characters`` falls back to the speakers on that scene's
  dialogue and is empty when there are none,
- name+role resolution is out of scope: speaker labels are literal, not
  aliased to roles by an LLM/NER pass.

Scene membership + first appearance come from scene dialogue alignment.
"""
from typing import Dict, List, Optional

_UNKNOWN_SPEAKERS = {"", "unknown", "none", "null", "?", "speaker"}


def _speaker_label(seg: dict) -> Optional[str]:
    name = seg.get("speaker")
    if not name:
        return None
    name = str(name).strip()
    if name.lower() in _UNKNOWN_SPEAKERS:
        return None
    return name


def build_character_index(scenes: List[dict], transcript_segments: List[dict]) -> List[dict]:
    """Return characters::

        [{"name", "mentions", "scene_ids", "first_appearance_sec"}]

    ``mentions`` counts the speaker-labeled utterances; ``scene_ids`` lists the
    scenes containing one of their utterances; ``first_appearance_sec`` is the
    exact earliest utterance start (unrounded).
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
                speaker = _speaker_label(seg)
                if speaker is None:
                    continue
                mention_times.setdefault(speaker, []).append(s)
                if speaker not in scene_ids.setdefault(scene_id, []):
                    scene_ids[scene_id].append(speaker)

    characters = []
    for name, times in mention_times.items():
        first = min(times)
        characters.append({
            "name": name,
            "mentions": len(times),
            "first_appearance_sec": first,
            "scene_ids": [
                sid for sid, names in scene_ids.items() if name in names
            ],
        })
    characters.sort(key=lambda c: -c["mentions"])
    return characters


def scene_characters_from_speakers(scene: dict, transcript_segments: List[dict]
                                   ) -> List[str]:
    """Ordered, deduplicated speaker labels appearing on this scene's dialogue."""
    start = float(scene.get("start_sec", 0.0))
    end = float(scene.get("end_sec", 0.0))
    seen: List[str] = []
    for seg in transcript_segments or []:
        try:
            s = float(seg.get("start_sec", 0.0))
            e = float(seg.get("end_sec", 0.0))
        except (TypeError, ValueError):
            continue
        if s > end or e < start:
            continue
        speaker = _speaker_label(seg)
        if speaker is not None and speaker not in seen:
            seen.append(speaker)
    return seen