"""Editorial timeline.

Converts an EditorialPlan + aligned script into an *edited* sequence: short
excerpt clips extracted from the source movie, each with editing directives
(SPEED / CROP / HOLD / MUTE), grouped into segments with transitions, aligned
to narration windows, plus a compatible ``timeline.json`` structure so existing
tooling (QC, renderer conventions) still understands it.

Output artifact: ``timeline/editorial_timeline.json``
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from editorial.plan import EditorialPlan
from movie_understanding import movie_memory

DEFAULT_TRANSITION = "crossfade"
CROSSFADE_SEC = 0.6
FADE_SEC = 0.5
CUT_SEC = 0.04          # imperceptible dissolve = hard cut in the xfade chain

MAX_EXCERPTS_PER_SEGMENT = 6

ExcerptFactory = Callable[[str, float, float, str, bool], Path]


def default_excerpt_factory(source: str, start: float, end: float,
                            output_path: str, include_audio: bool) -> Path:
    from editor.clip_extractor import extract_clip

    return extract_clip(source, start, end, output_path, reencode=True)


class EditorialTimelineBuilder:
    """Builds the editorial timeline JSON + extracts excerpt clips.

    Allocates enough distinct short excerpt windows to cover every narration
    window (up to ``max_excerpt_sec`` each). If a segment cannot be visually
    covered, the timeline records the deficit so the renderer's coverage
    validator fails the pipeline instead of silently stretching/padding.
    """

    def __init__(self, source_path: Optional[str] = None,
                 excerpt_factory: Optional[ExcerptFactory] = None,
                 max_excerpt_sec: float = 6.0,
                 min_excerpt_sec: float = 1.2,
                 lead_sec: float = 0.4):
        self.source_path = source_path
        self.excerpt_factory = excerpt_factory or default_excerpt_factory
        self.max_excerpt_sec = max_excerpt_sec
        self.min_excerpt_sec = min_excerpt_sec
        self.lead_sec = lead_sec

    def build(self, project_dir: Path, plan: EditorialPlan,
              script: dict) -> dict:
        project_dir = Path(project_dir)
        source = self.source_path or _movie_source(project_dir)
        source_duration = None
        if source and Path(source).exists():
            from editor.clip_extractor import probe_duration

            source_duration = probe_duration(source) or None
        sections = {s["section_id"]: s for s in script.get("sections", [])}
        excerpts_dir = project_dir / "assets" / "excerpts"
        excerpts_dir.mkdir(parents=True, exist_ok=True)

        segments_out = []
        video_items = []
        text_items = []
        total_coverage = 0.0
        for seg in plan.segments:
            section = sections.get(seg.id, {})
            narration_start = float(section.get("narration_start_sec", 0.0))
            narration_dur = float(section.get("narration_duration_sec", 1.0))
            budget = max(0.0, narration_dur)

            # Distinct evidence windows to draw from (deduplicated).
            windows = _dedupe_windows(seg.evidence)

            # Allocate windows until the narration budget is visually covered.
            # Each window is at most ``max_excerpt_sec`` long (short excerpts,
            # never the whole scene). Extra windows keep the edit moving.
            chosen = []
            covered = 0.0
            for evidence in windows:
                if len(chosen) >= MAX_EXCERPTS_PER_SEGMENT:
                    break
                start = float(evidence.start_sec)
                want = min(self.max_excerpt_sec, max(
                    self.min_excerpt_sec, budget - covered))
                end = min(float(evidence.end_sec), start + want)
                if source_duration is not None:
                    end = min(end, source_duration)
                if end - start < self.min_excerpt_sec:
                    continue
                if source_duration is not None and start >= source_duration:
                    continue
                chosen.append({
                    "evidence": evidence, "start": start, "end": end,
                })
                covered += end - start
                if covered >= budget - 1e-6:
                    break

            seg_clips = []
            for idx, c in enumerate(chosen):
                evidence = c["evidence"]
                start, end = c["start"], c["end"]
                out_path = excerpts_dir / f"{seg.id}-{idx}.mp4"
                extracted = False
                if source and Path(source).exists():
                    self.excerpt_factory(
                        source, start, end, str(out_path),
                        not seg.editing.mute_film_audio,
                    )
                    extracted = True
                dur = _transformed_duration(end - start,
                                            seg.editing.speed, seg.editing.hold_sec)
                total_coverage += dur
                seg_clips.append({
                    "excerpt_index": idx,
                    "source_scene": evidence.scene_id,
                    "source_start_sec": round(start, 3),
                    "source_end_sec": round(end, 3),
                    "content_path": str(out_path),
                    "extracted": extracted,
                    "duration_sec": round(dur, 3),
                    "speed": seg.editing.speed,
                    "crop_zoom": seg.editing.crop_zoom,
                    "hold_sec": seg.editing.hold_sec,
                    "mute_film_audio": seg.editing.mute_film_audio,
                })
                video_items.append({
                    "type": "video_clip",
                    "start_sec": round(narration_start, 3),
                    "duration_sec": round(dur, 3),
                    "content_path": str(out_path),
                    "content_text": None,
                    "metadata": {
                        "seg_id": seg.id,
                        "editing": seg.editing.to_json(),
                    },
                })

            if not seg_clips:
                continue

            uncovered = max(0.0, budget - covered)
            segments_out.append({
                "seg_id": seg.id,
                "purpose": seg.purpose,
                "transition_to_next": seg.editing.transition,
                "narration": {
                    "start_sec": round(narration_start, 3),
                    "end_sec": round(narration_start + narration_dur, 3),
                    "duration_sec": round(narration_dur, 3),
                },
                "video": seg_clips,
                "visual_coverage_sec": round(covered, 3),
                "narration_uncovered_sec": round(uncovered, 3),
                "audio": {
                    "duck_level": seg.editing.duck_level,
                    "mute_film": seg.editing.mute_film_audio,
                },
            })

        # Subtitles from caption chunks (already anchored to narration windows).
        for section in sections.values():
            for cap in section.get("subtitle_captions", []):
                text_items.append({
                    "type": "subtitle",
                    "start_sec": cap["start_sec"],
                    "duration_sec": round(max(0.1, cap["end_sec"] - cap["start_sec"]), 3),
                    "content_path": None,
                    "content_text": cap.get("text", ""),
                    "metadata": {"words": cap.get("words", [])},
                })

        narration_total = _narration_total(plan, sections)
        timeline = {
            "mode": "editorial",
            "total_duration_sec": round(narration_total, 3),
            "narration_total_sec": round(narration_total, 3),
            "source_path": source,
            "segments": segments_out,
            "movie_audio": _movie_audio_defaults(),
            "tracks": {
                "video": {
                    "type": "video",
                    "items": video_items,
                    "volume": 1.0,
                    "mute": False,
                },
                "voice": {
                    "type": "voice",
                    "items": [{
                        "type": "audio_clip",
                        "start_sec": 0.0,
                        "duration_sec": round(narration_total, 3),
                        "content_path": str(project_dir / "audio" / "voice.wav"),
                        "content_text": None,
                        "metadata": {},
                    }],
                    "volume": 1.0,
                    "mute": False,
                },
                "text": {
                    "type": "text",
                    "items": text_items,
                    "volume": 1.0,
                    "mute": False,
                },
            },
            "metadata": {
                "builder": "editorial",
                "thesis": plan.thesis,
            },
        }
        movie_memory.save_json(project_dir, "timeline/editorial_timeline.json", timeline)
        return timeline


def _transformed_duration(raw_duration: float, speed: float, hold_sec: float) -> float:
    speed = max(0.25, float(speed or 1.0))
    return round(max(0.1, raw_duration / speed + float(hold_sec or 0.0)), 3)


def _movie_source(project_dir: Path) -> Optional[str]:
    meta = movie_memory.load_json(project_dir, "project_meta.json", {})
    return meta.get("source_path")


def _evidence_attr(e, name: str, default):
    """Read an attribute from an :class:`EditorialEvidence` dataclass or a dict."""
    if isinstance(e, dict):
        return e.get(name, default)
    return getattr(e, name, default)


def _dedupe_windows(evidence: List) -> List:
    """Deduplicate evidence windows by (scene,start,end) so the same excerpt is
    never used twice within one segment without an editorial reason."""
    seen = set()
    out = []
    for e in evidence:
        key = (str(_evidence_attr(e, "scene_id", "")),
               round(float(_evidence_attr(e, "start_sec", 0)), 3),
               round(float(_evidence_attr(e, "end_sec", 0)), 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _narration_total(plan: EditorialPlan, sections: Dict[str, dict]) -> float:
    total = 0.0
    for seg in plan.segments:
        section = sections.get(seg.id, {})
        total += float(section.get("narration_duration_sec",
                       seg.narration.text.count(" ") / 2.0 or 1.0))
        total += float(seg.narration.delivery.pause_after or 0.0)
    return max(1.0, round(total, 3))


def _movie_audio_defaults() -> dict:
    """Explicit movie-audio contract: preserve the movie's own soundtrack by
    default and duck it under the narration. Never silently strip it."""
    import os
    return {
        "enabled": os.getenv("MOVIE_AUDIO_ENABLED", "true").lower() != "false",
        "gain_db": float(os.getenv("MOVIE_AUDIO_GAIN_DB", "-6.0")),
        "duck_under_narration": os.getenv("MOVIE_AUDIO_DUCK", "true").lower() != "false",
    }


def editorial_timeline_excerpts(timeline: dict) -> List[str]:
    return [
        clip["content_path"]
        for seg in timeline.get("segments", [])
        for clip in seg.get("video", [])
    ]
