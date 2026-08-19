"""Editorial script builder.

Converts an EditorialPlan into the canonical ``script.json`` *plus* the new
editorial fields: per-section narration text, per-section delivery (pace /
energy / pauses used by per-section TTS), short caption chunks with word
timings, and the narration->evidence mapping. The ordinary fields
(``voiceover_text`` / ``sections`` / legacy ``narration_properties``) are kept
so existing consumers and the QC schema still pass.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from editorial.plan import EditorialPlan, EditorialSegment
from editorial.subtitles import (
    MAX_LINE_WORDS,
    caption_word_timings,
    merge_with_real_word_timestamps,
    split_into_captions,
)
from movie_understanding import movie_memory, text_utils

WORDS_PER_SEC = 2.4


def _estimate_seconds(text: str, pace: float = 1.0) -> float:
    words = len(text.split())
    return max(1.0, round((words / max(0.1, WORDS_PER_SEC * pace)), 2))


def _segment_delivery(delivery) -> dict:
    return {
        "tone": delivery.tone,
        "emotion": delivery.emotion,
        "energy": delivery.energy,
        "pace": delivery.pace,
        "dramatic_intensity": delivery.dramatic_intensity,
        "pause_before": delivery.pause_before,
        "pause_after": delivery.pause_after,
    }


def _global_narration_properties(plan: EditorialPlan) -> dict:
    deliv = plan.segments[0].narration.delivery if plan.segments else None
    return {
        "tone": deliv.tone if deliv else "analytical",
        "emotion": deliv.emotion if deliv else "neutral",
        "pace": deliv.pace if deliv else 1.0,
        "energy": deliv.energy if deliv else 0.5,
        "dramatic_intensity": deliv.dramatic_intensity if deliv else 0.5,
    }


def build_editorial_script(project_dir: Path, plan: EditorialPlan,
                           movie_index: dict,
                           transcript_word_map: Dict[str, List[dict]] | None = None,
                           progress: Dict[str, dict] | None = None) -> dict:
    """Write ``script.json`` (editorial variant). Returns the script dict.

    ``transcript_word_map``: optional scene_id -> word timestamps (unused when
    real narration word timings come from per-section TTS alignment).
    ``progress``: optional narration timing map from the audio stage —
    ``{section_id: {"start_sec", "duration_sec"}}`` — used to anchor captions
    and the narration->evidence mapping to real spoken positions.
    """
    project_dir = Path(project_dir)
    progress = progress or {}

    hook_text = (plan.hook.get("text") or "").strip()
    sections: List[dict] = []
    captions_by_section: Dict[str, List[dict]] = {}

    all_texts = []
    if hook_text:
        all_texts.append(hook_text)
    for seg in plan.segments:
        all_texts.append(seg.narration.text)

    # Deduplicate the hook: the plan's hook.text is also carried by the first
    # hook segment, and the grounded path already emits a dedicated "hook"
    # section. Speaking it twice is a bug.
    if len(all_texts) >= 2 and all_texts[0] == all_texts[1]:
        all_texts.pop(0)

    voiceover_text = " ".join(all_texts)

    for idx, seg in enumerate(plan.segments):
        delivery = seg.narration.delivery
        estimate = _estimate_seconds(seg.narration.text, delivery.pace)
        timings = progress.get(seg.id)
        if timings is None:
            timings = _default_section_timing(estimate, idx, sections)
        captions = _captions_for(seg, timings, transcript_word_map)
        captions_by_section[seg.id] = captions
        sections.append({
            "section_id": seg.id,
            "text": seg.narration.text,
            "estimated_seconds": timings.get("duration_sec", estimate),
            "narration_start_sec": timings.get("start_sec", 0.0),
            "narration_duration_sec": timings.get("duration_sec", estimate),
            "purpose": seg.purpose,
            "scene_ids": [e.scene_id for e in seg.evidence],
            "narrative_evidence": [
                {
                    "scene_id": e.scene_id,
                    "start_sec": e.start_sec,
                    "end_sec": e.end_sec,
                    "reason": e.reason,
                }
                for e in seg.evidence
            ],
            "subtitle_captions": captions,
            "delivery": _segment_delivery(delivery),
        })

    script = {
        "project_id": str(Path(project_dir).name),
        "voiceover_text": voiceover_text,
        "hook": plan.hook,
        "sections": sections,
        "cta": "",
        "style_notes": "editorial short-film essay, evidence-driven",
        "scene_ids": _all_scene_ids(sections),
        "narration_properties": _global_narration_properties(plan),
        "editorial": True,
        "thesis": plan.thesis,
    }
    movie_memory.save_json(project_dir, "script.json", script)
    return script


def _default_section_timing(duration_sec: float, idx: int,
                            prior_sections: List[dict]) -> dict:
    # cumulative placeholder: starts where the previous section ended.
    cursor = 0.0
    if prior_sections:
        cursor = float(prior_sections[-1].get("narration_start_sec", 0.0)) + float(
            prior_sections[-1].get("narration_duration_sec", 0.0))
    return {"start_sec": round(cursor, 3), "duration_sec": duration_sec}


def _captions_for(seg: EditorialSegment, timings: dict,
                  transcript_word_map: Optional[Dict[str, List[dict]]]) -> List[dict]:
    text = seg.narration.text
    start = float(timings.get("start_sec", 0.0))
    duration = float(timings.get("duration_sec", 1.0))
    return caption_word_timings(split_into_captions(text, max_words=MAX_LINE_WORDS),
                                start, duration)


def _all_scene_ids(sections: List[dict]) -> List[str]:
    seen: List[str] = []
    for s in sections:
        for sid in s.get("scene_ids", []):
            if sid not in seen:
                seen.append(sid)
    return seen


def align_captions_to_voice(project_dir: Path) -> None:
    """Re-anchor segment captions + start times after real per-section TTS.

    Reads ``audio/segment_timings.json`` (written by the audio stage) and
    rewrites ``script.json`` so captions and evidence windows line up with the
    actual spoken audio.
    """
    project_dir = Path(project_dir)
    timings = movie_memory.load_json(project_dir, "audio/segment_timings.json", {})
    if not timings:
        return
    script = movie_memory.load_json(project_dir, "script.json", {})
    if not script.get("sections"):
        return
    for section in script["sections"]:
        t = timings.get(section["section_id"])
        if not t:
            continue
        start = float(t["start_sec"])
        duration = float(t["duration_sec"])
        section["narration_start_sec"] = start
        section["narration_duration_sec"] = duration
        section["subtitle_captions"] = _remake_captions(section, start, duration)
    movie_memory.save_json(project_dir, "script.json", script)


def _remake_captions(section: dict, start: float, duration: float) -> List[dict]:
    from editorial.subtitles import caption_word_timings, split_into_captions

    return caption_word_timings(
        split_into_captions(section.get("text", ""), max_words=MAX_LINE_WORDS),
        start, duration)