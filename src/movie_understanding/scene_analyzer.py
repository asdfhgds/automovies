"""Scene enrichment.

Turns a bare scene (``scene_id / start_sec / end_sec / transcript``) into a
story-bearing unit the editorial director can reason about:

``story = {summary, topics, dialogue, characters, location, actions,
           visual_description, emotional_tone, themes, provenance}``

Everything is deterministic-first. Fields that genuinely require vision or an
LLM (location, actions, visual description, real themes) are left ``None`` and
recorded as ``available: false`` in ``provenance`` rather than faked.
"""
from typing import Dict, List, Optional

from movie_understanding import text_utils

# Small lexicon used for a defensible, coarse emotional tone heuristic.
_TONE_LEXICON = {
    "tension": {"fear", "afraid", "danger", "chase", "gun", "knife", "scream",
                "dark", "shadow", "run", "threat", "break", "fight", "blood"},
    "joy": {"smile", "laugh", "happy", "joy", "love", "dance", "celebrate",
            "fun", "warm", "good", "gift", "friends", "family", "kiss"},
    "sadness": {"tears", "cry", "cried", "alone", "lost", "goodbye", "death",
                "dead", "gone", "hurt", "pain", "miss", "sorry", "empty"},
    "anger": {"angry", "rage", "shout", "yell", "hate", "slam", "throw",
              "furious", "argue", "curse", "betray", "enemy"},
    "mystery": {"secret", "hidden", "strange", "unknown", "silent", "mystery",
                "whisper", "mask", "memory", "forgot", "dream", "shadow"},
    "calm": {"quiet", "still", "peace", "calm", "slow", "soft", "gentle",
             "morning", "night", "stars", "water", "silent", "breathe"},
}


def _detect_emotional_tone(text: str) -> str:
    scores: Dict[str, int] = {}
    tokens = set(text_utils.tokenize(text))
    for tone, words in _TONE_LEXICON.items():
        scores[tone] = len(tokens & words)
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else "neutral"


def _scene_dialogue(scene: dict, transcript_segments: List[dict]) -> List[dict]:
    """Transcript segments overlapping the scene window (interval overlap)."""
    start = float(scene.get("start_sec", 0.0))
    end = float(scene.get("end_sec", 0.0))
    out = []
    for seg in transcript_segments or []:
        try:
            s = float(seg.get("start_sec", 0.0))
            e = float(seg.get("end_sec", 0.0))
        except (TypeError, ValueError):
            continue
        if s <= end and e >= start:
            out.append({
                "speaker": seg.get("speaker") or None,
                "text": (seg.get("text") or "").strip(),
                "start_sec": s,
                "end_sec": e,
            })
    return out


class SceneEnricher:
    """Provider interface for scene enrichment."""

    name = "base"

    def enrich(self, scene: dict, transcript_segments: List[dict]) -> dict:
        raise NotImplementedError


class HeuristicSceneEnricher(SceneEnricher):
    """Deterministic enrichment: summary, topics, dialogue, tone, characters.

    ``location`` / ``actions`` / ``visual_description`` / ``themes`` are
    vision/LLM-only and reported as unavailable rather than invented.
    """

    name = "heuristic"

    def enrich(self, scene: dict, transcript_segments: List[dict]) -> dict:
        scene_id = scene.get("scene_id", "scene-0")
        text = (scene.get("transcript") or scene.get("summary") or "").strip()
        dialogue = _scene_dialogue(scene, transcript_segments)
        dialogue_text = " ".join(d["text"] for d in dialogue if d["text"])
        corpus = f"{text} {dialogue_text}".strip()

        sentences = text_utils.sentencize(text)
        summary = sentences[0] if sentences else None
        if summary is not None and len(summary) > 120:
            summary = summary[:117] + "..."
        topics = text_utils.top_keywords(corpus, k=5)
        characters = text_utils.candidate_names(corpus, min_mentions=2)
        tone = _detect_emotional_tone(corpus)

        return {
            "scene_id": scene_id,
            "start_sec": float(scene.get("start_sec", 0.0)),
            "end_sec": float(scene.get("end_sec", 0.0)),
            "duration_sec": float(scene.get("duration", 0.0)) or (
                float(scene.get("end_sec", 0.0)) - float(scene.get("start_sec", 0.0))
            ),
            "transcript": text,
            "story": {
                "summary": summary,
                "topics": topics,
                "dialogue": dialogue,
                "characters": characters,
                "location": None,
                "actions": None,
                "visual_description": None,
                "emotional_tone": tone,
                "themes": None,
                "provenance": {
                    "summary": "transcript",
                    "topics": "transcript_frequency",
                    "dialogue": "transcript_alignment",
                    "characters": "transcript_capitalized_names",
                    "location": "unavailable (vision/LLM)",
                    "actions": "unavailable (vision/LLM)",
                    "visual_description": "unavailable (vision/LLM)",
                    "emotional_tone": "transcript_lexicon",
                    "themes": "unavailable (LLM)",
                },
            },
        }


def enrich_scene(scene: dict, transcript_segments: List[dict],
                 enricher: Optional[SceneEnricher] = None) -> dict:
    """Enrich one scene entry. Defaults to the heuristic enricher."""
    enricher = enricher or HeuristicSceneEnricher()
    return enricher.enrich(scene, transcript_segments)
