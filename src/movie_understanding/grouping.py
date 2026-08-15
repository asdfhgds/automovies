"""Deterministic shot -> narrative scene grouping.

PySceneDetect emits raw *shots* (a cut is a shot boundary), which are far too
granular for editorial reasoning: a 120s trailer can contain 30+ shots.  This
module groups contiguous shots into *narrative scenes* — the unit the
MovieAnalyzer enriches, retrieves against and hands to the editorial layer.

The grouping is deliberately deterministic and dependency-free:

- shots are walked in temporal order and greedily accumulated into the current
  narrative scene;
- a scene closes the moment its span (from the first member shot's start to the
  current shot's end) would exceed ``max_scene_sec``, or its shot count reaches
  ``max_shots``;
- a single shot longer than ``max_scene_sec`` becomes its own scene rather than
  being split (we never fabricate boundaries inside a shot);
- ``gap_threshold_sec`` (default 0.0) splits on any inter-shot temporal gap
  larger than the value — a discontinuity means a different narrative unit.

ID scheme (keeps the two collections unambiguous and downstream-compatible):
raw shots become ``shot-N`` (original PySceneDetect id kept as
``source_scene_id``); narrative scenes keep the ``scene-N`` namespace the
editorial layer already consumes.

Exact temporal coordinates are preserved end to end: boundaries are the raw
min/max of the member shots' floats, never rounded.
"""
from typing import Dict, List, Optional

DEFAULT_GAP_THRESHOLD_SEC = 0.0


def shots_from_scene_index(scene_index: List[dict]) -> List[dict]:
    """Normalize raw scene_index entries into shot records.

    Each shot gets an unambiguous ``shot-N`` id; the original PySceneDetect
    ``scene_id`` is preserved as ``source_scene_id`` for traceability. Exact
    ``start_sec`` / ``end_sec`` / ``transcript`` are kept verbatim.
    """
    shots: List[dict] = []
    for entry in scene_index or []:
        if not isinstance(entry, dict):
            continue
        try:
            start = float(entry.get("start_sec"))
            end = float(entry.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        shots.append({
            "shot_id": f"shot-{len(shots) + 1}",
            "source_scene_id": entry.get("scene_id") or entry.get("shot_id"),
            "start_sec": start,
            "end_sec": end,
            "transcript": (entry.get("transcript") or "").strip(),
        })
    return shots


def group_shots_into_narrative_scenes(
    shots: List[dict],
    max_scene_sec: float = 30.0,
    max_shots: int = 12,
    gap_threshold_sec: Optional[float] = DEFAULT_GAP_THRESHOLD_SEC,
) -> List[dict]:
    """Group ordered shots into deterministic narrative scenes.

    Narrative scenes keep the ``scene-N`` namespace (editorial contract); raw
    shots are ``shot-N``.  Returns::

        [{"scene_id": "scene-1", "start_sec", "end_sec", "duration_sec",
          "shot_count", "shot_ids": [...], "shots": [...]}]
    """
    ordered = sorted(shots, key=lambda s: (float(s.get("start_sec", 0.0)),
                                           float(s.get("end_sec", 0.0))))
    scenes: List[dict] = []
    current: Optional[dict] = None
    next_index = 1

    def _flush() -> None:
        nonlocal current
        if not current:
            return
        members = current["shots"]
        start = min(float(s["start_sec"]) for s in members)
        end = max(float(s["end_sec"]) for s in members)
        current["start_sec"] = start
        current["end_sec"] = end
        current["duration_sec"] = end - start
        current["shot_count"] = len(members)
        current["shot_ids"] = [s["shot_id"] for s in members]
        current["transcript"] = " ".join(
            s["transcript"] for s in members if s.get("transcript")
        ).strip()
        for shot in members:
            shot.setdefault("duration_sec",
                            float(shot["end_sec"]) - float(shot["start_sec"]))
        scenes.append(current)
        current = None

    for shot in ordered:
        if current is None:
            current = {"scene_id": f"scene-{next_index}", "shots": [shot]}
            next_index += 1
            continue

        span = float(shot["end_sec"]) - float(current["shots"][0]["start_sec"])
        exceeds_duration = span > max(float(max_scene_sec), 0.0)
        exceeds_count = len(current["shots"]) >= max(1, int(max_shots))

        gap_split = False
        if gap_threshold_sec is not None:
            last_end = max(float(s["end_sec"]) for s in current["shots"])
            gap_split = (float(shot["start_sec"]) - last_end) > float(gap_threshold_sec)

        if exceeds_duration or exceeds_count or gap_split:
            _flush()
            current = {"scene_id": f"scene-{next_index}", "shots": [shot]}
            next_index += 1
            continue

        current["shots"].append(shot)

    if current is not None:
        _flush()
    return scenes


def scene_span_of_shots(shot_ids: List[str], shots: List[dict]) -> Dict[str, float]:
    """Exact ``{start, end}`` span of referenced shots, or empty dict."""
    by_id = {str(s.get("shot_id")): s for s in shots}
    span = [by_id[sid] for sid in shot_ids if sid in by_id]
    if not span:
        return {}
    return {
        "start_sec": min(float(s["start_sec"]) for s in span),
        "end_sec": max(float(s["end_sec"]) for s in span),
    }


def describe_grouping(
    max_scene_sec: float = 30.0,
    max_shots: int = 12,
    gap_threshold_sec: Optional[float] = DEFAULT_GAP_THRESHOLD_SEC,
) -> dict:
    """Return a provenance record for the grouping that produced the scenes."""
    return {
        "method": "deterministic_greedy",
        "max_scene_sec": float(max_scene_sec),
        "max_shots": int(max_shots),
        "gap_threshold_sec": gap_threshold_sec,
        "deterministic": True,
        "note": "contiguous PySceneDetect shots greedily grouped into "
                "narrative scenes; >gap_threshold_sec of inter-shot silence "
                "starts a new scene; boundaries are exact, never rounded",
    }