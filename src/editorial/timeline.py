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

ExcerptFactory = Callable[[str, float, float, str, bool], Path]


def default_excerpt_factory(source: str, start: float, end: float,
                            output_path: str, include_audio: bool) -> Path:
    from editor.clip_extractor import extract_clip

    return extract_clip(source, start, end, output_path, reencode=True)


class EditorialTimelineBuilder:
    """Builds the editorial timeline JSON + extracts excerpt clips."""

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
        for seg in plan.segments:
            section = sections.get(seg.id, {})
            narration_start = float(section.get("narration_start_sec", 0.0))
            narration_dur = float(section.get("narration_duration_sec", 1.0))
            n_ev = max(1, len(seg.evidence))
            per = max(self.min_excerpt_sec, min(self.max_excerpt_sec, narration_dur / n_ev))

            seg_clips = []
            for idx, evidence in enumerate(seg.evidence[:2]):
                start = float(evidence.start_sec)
                end = min(float(evidence.end_sec), start + per)
                if source_duration is not None:
                    end = min(end, source_duration)
                if end - start < self.min_excerpt_sec:
                    continue
                if source_duration is not None and start >= source_duration:
                    continue
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


def _narration_total(plan: EditorialPlan, sections: Dict[str, dict]) -> float:
    total = 0.0
    for seg in plan.segments:
        section = sections.get(seg.id, {})
        total += float(section.get("narration_duration_sec",
                       seg.narration.text.count(" ") / 2.0 or 1.0))
        total += float(seg.narration.delivery.pause_after or 0.0)
    return max(1.0, round(total, 3))


def editorial_timeline_excerpts(timeline: dict) -> List[str]:
    return [
        clip["content_path"]
        for seg in timeline.get("segments", [])
        for clip in seg.get("video", [])
    ]
