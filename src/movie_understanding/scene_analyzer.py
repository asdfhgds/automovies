"""Scene enrichment.

Turns a bare scene (``scene_id / start_sec / end_sec / transcript``) into a
story-bearing unit the editorial director can reason about.

Repair: the enrichment output is split into two *epistemically different*
half-cards so downstream consumers can tell WHERE a fact came from:

- ``analysis.transcript`` — what the transcript/diarization alone supports:
  summary, topics, dialogue alignment, ``characters`` (from diarization
  speaker labels ONLY — capitalized transcript words are never treated as
  characters), emotional tone (lexicon).
- ``analysis.visual`` — what vision (a Qwen3-VL keyframe read) alone supports:
  location, actions, objects, visual description, visual events, emotional
  cues, themes, mood, cinematography, confidence. When vision is unavailable
  every field stays ``None`` and is flagged ``unavailable (vision/LLM)`` in its
  provenance rather than faked.

``story`` remains the merged view (transcript + visual) for backward
compatibility with the editorial director and retrieval layer, with the same
per-field ``provenance`` as before.
"""
from typing import Dict, List, Optional

from movie_understanding import character_analyzer, text_utils

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

TRANSCRIPT_FIELDS = ("summary", "topics", "dialogue", "characters", "emotional_tone")
VISUAL_FIELDS = ("location", "actions", "objects", "visual_description",
                 "visual_events", "emotional_cues", "themes", "mood",
                 "cinematography", "confidence")


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


def _transcript_analysis(scene: dict, transcript_segments: List[dict]) -> dict:
    """Deterministic, transcript-derived half of the scene analysis."""
    text = (scene.get("transcript") or scene.get("summary") or "").strip()
    dialogue = _scene_dialogue(scene, transcript_segments)
    dialogue_text = " ".join(d["text"] for d in dialogue if d["text"])
    corpus = f"{text} {dialogue_text}".strip()

    sentences = text_utils.sentencize(text)
    summary = sentences[0] if sentences else None
    if summary is not None and len(summary) > 120:
        summary = summary[:117] + "..."
    topics = text_utils.top_keywords(corpus, k=5)
    characters = character_analyzer.scene_characters_from_speakers(scene, transcript_segments)
    tone = _detect_emotional_tone(corpus)

    return {
        "summary": summary,
        "topics": topics,
        "dialogue": dialogue,
        "characters": characters,
        "emotional_tone": tone,
        "provenance": {
            "summary": "transcript",
            "topics": "transcript_frequency",
            "dialogue": "transcript_alignment",
            "characters": "diarization_speaker_labels",
            "emotional_tone": "transcript_lexicon",
        },
    }


_VISUAL_UNAVAILABLE = "unavailable (vision/LLM)"


def _visual_analysis_unavailable() -> dict:
    """Vision half with every field honestly unset + provenance reason."""
    return {
        field: None for field in VISUAL_FIELDS
    } | {
        "provenance": {field: _VISUAL_UNAVAILABLE for field in VISUAL_FIELDS},
    }


def merge_story(analysis: dict) -> dict:
    """Merge ``analysis.transcript`` + ``analysis.visual`` into the story card."""
    transcript = (analysis or {}).get("transcript", {}) or {}
    visual = (analysis or {}).get("visual", {}) or {}
    story = {field: transcript.get(field) for field in TRANSCRIPT_FIELDS}
    story.update({field: visual.get(field, None) for field in VISUAL_FIELDS})
    provenance = {}
    provenance.update(transcript.get("provenance", {}) or {})
    for field in TRANSCRIPT_FIELDS:
        provenance.setdefault(field, _VISUAL_UNAVAILABLE)
    provenance.update(visual.get("provenance", {}) or {})
    for field in VISUAL_FIELDS:
        provenance.setdefault(field, _VISUAL_UNAVAILABLE)
    story["provenance"] = provenance
    return story


def _scene_shell(scene: dict) -> dict:
    return {
        "scene_id": scene.get("scene_id", "scene-0"),
        "start_sec": float(scene.get("start_sec", 0.0)),
        "end_sec": float(scene.get("end_sec", 0.0)),
        "duration_sec": float(scene.get("duration", 0.0)) or (
            float(scene.get("end_sec", 0.0)) - float(scene.get("start_sec", 0.0))
        ),
        "transcript": (scene.get("transcript") or "").strip(),
        "shot_ids": scene.get("shot_ids") or [],
        "shot_count": scene.get("shot_count") or (
            len(scene.get("shots") or []) if scene.get("shots") else None
        ),
    }


class SceneEnricher:
    """Provider interface for scene enrichment."""

    name = "base"

    def enrich(self, scene: dict, transcript_segments: List[dict]) -> dict:
        raise NotImplementedError


class HeuristicSceneEnricher(SceneEnricher):
    """Deterministic enrichment: transcript analysis + honest None visual half.

    ``analysis.transcript`` (summary/topics/dialogue/characters/emotional_tone)
    is derived from the transcript; ``analysis.visual`` is entirely unset and
    flagged ``unavailable (vision/LLM)`` — nothing is invented.
    """

    name = "heuristic"

    def enrich(self, scene: dict, transcript_segments: List[dict]) -> dict:
        transcript = _transcript_analysis(scene, transcript_segments)
        visual = _visual_analysis_unavailable()
        return {
            **_scene_shell(scene),
            "analysis": {
                "transcript": transcript,
                "visual": visual,
            },
            "story": merge_story({"transcript": transcript, "visual": visual}),
        }


def enrich_scene(scene: dict, transcript_segments: List[dict],
                 enricher: Optional[SceneEnricher] = None) -> dict:
    """Enrich one scene entry. Defaults to the heuristic enricher."""
    enricher = enricher or HeuristicSceneEnricher()
    return enricher.enrich(scene, transcript_segments)