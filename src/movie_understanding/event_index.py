"""Event index.

Groups transcript dialogue into discrete *events* (a beat of screen action
anchored to dialogue). An event is a maximal run of transcript segments inside
one scene with gaps below ``gap_threshold_sec``. Without vision, events are
dialogue-visible; silent action beats are not (yet) discoverable.
"""
from typing import Dict, List, Optional


def build_event_index(scenes: List[dict], transcript_segments: List[dict],
                      gap_threshold_sec: float = 3.0) -> List[dict]:
    """Return events::

        [{"event_id", "scene_id", "start_sec", "end_sec",
          "dialogue_ids", "summary_text", "keywords"}]
    """
    # Map each transcript segment to its scene.
    scene_ranges = []
    for scene in scenes:
        scene_ranges.append({
            "scene_id": scene.get("scene_id"),
            "start": float(scene.get("start_sec", 0.0)),
            "end": float(scene.get("end_sec", 0.0)),
        })

    def _scene_for(start: float, end: float) -> Optional[str]:
        for r in scene_ranges:
            if start <= r["end"] and end >= r["start"]:
                return r["scene_id"]
        return None

    # Assign each segment to a scene, preserving order.
    assigned = []
    for seg in transcript_segments or []:
        try:
            s = float(seg.get("start_sec", 0.0))
            e = float(seg.get("end_sec", 0.0))
        except (TypeError, ValueError):
            continue
        scene_id = _scene_for(s, e)
        if scene_id is None:
            continue
        assigned.append({"scene_id": scene_id, "seg": seg, "start": s, "end": e})

    # Cluster into events.
    events = []
    for scene_id, group in _groupby_scene(assigned):
        current = None
        for item in group:
            if current is None or item["start"] - current["end"] > gap_threshold_sec:
                if current is not None:
                    events.append(_finish_event(current))
                current = {
                    "scene_id": scene_id,
                    "start": item["start"],
                    "end": item["end"],
                    "segs": [item["seg"]],
                }
            else:
                current["end"] = max(current["end"], item["end"])
                current["segs"].append(item["seg"])
        if current is not None:
            events.append(_finish_event(current))

    for i, ev in enumerate(events):
        ev["event_id"] = f"event_{i:03d}"
    return events


def _groupby_scene(assigned: List[dict]) -> List[tuple]:
    """Group consecutive segments by scene, preserving order (no reordering)."""
    out: List[tuple] = []
    for item in assigned:
        if out and out[-1][0] == item["scene_id"]:
            out[-1][1].append(item)
        else:
            out.append((item["scene_id"], [item]))
    return out


def _finish_event(current: dict) -> dict:
    from movie_understanding import text_utils

    text = " ".join((s.get("text") or "").strip() for s in current["segs"] if s)
    return {
        "scene_id": current["scene_id"],
        "start_sec": round(current["start"], 3),
        "end_sec": round(current["end"], 3),
        "dialogue_ids": [s.get("id") or "seg" for s in current["segs"]],
        "summary_text": text[:220] if text else None,
        "keywords": text_utils.top_keywords(text, k=5),
    }
