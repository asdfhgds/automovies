"""Normalized grounding contract: the director decision as data.

The milestone pipeline treats the grounded director's output as the *source of
truth* for everything downstream. ``build_grounding_contract`` folds the
director result (selected concept + plan + evidence analysis) into one stable,
typed schema — the contract the grounded script generator consumes. It is
deterministic (no LLM call) and reuses the exact keys the director already
produces, so no duplicate formats appear.
"""
from pathlib import Path
from typing import Any, Dict, List

from .scene_facts import SceneFacts
from movie_understanding import movie_memory

# Keys every well-formed contract carries. Downstream stages (script generator,
# editorial planner) can rely on this set without re-deriving it.
CONTRACT_KEYS = (
    "concept",
    "evidence_refs",
    "evidence_requirements",
    "supporting_scenes",
    "visual_motifs",
    "character_focus",
    "format",
    "editorial_intent",
)


def build_grounding_contract(
    director_result: Dict[str, Any],
    scene_facts: SceneFacts,
    movie_index: Dict[str, Any],
) -> Dict[str, Any]:
    """Fold a ``MovieGroundedDirector.develop()`` result into the contract.

    Everything is read from data the director already produced:

    - ``concept``            <- plan.concept + selected_concept.why_interesting
    - ``evidence_refs``       <- selected_concept.evidence_refs (structured refs)
    - ``evidence_requirements`` <- selected_concept.required_evidence
    - ``supporting_scenes``  <- plan.evidence_strategy.scene_ids (with times)
    - ``visual_motifs``      <- plan.evidence_strategy.visual_motifs
    - ``character_focus``    <- plan.evidence_strategy.character_focus
    - ``format``             <- plan.format
    - ``editorial_intent``   <- plan.editorial_direction

    Times on supporting scenes are resolved from the scene facts so downstream
    stages can build excerpt windows without a second lookup.
    """
    selected = director_result.get("selected_concept") or {}
    plan = director_result.get("plan") or {}
    concept = dict(plan.get("concept") or {})
    concept.setdefault("title", selected.get("title", ""))
    concept.setdefault("hook", selected.get("hook", ""))
    concept.setdefault("thesis", selected.get("thesis", ""))
    concept.setdefault("why_interesting", selected.get("why_interesting", ""))

    strategy = plan.get("evidence_strategy") or {}
    scene_ids = strategy.get("scene_ids") or []
    supporting_scenes = []
    for sid in scene_ids:
        fact = scene_facts.by_id(sid)
        supporting_scenes.append({
            "scene_id": sid,
            "start_sec": float(fact.start_sec) if fact and fact.start_sec is not None else None,
            "end_sec": float(fact.end_sec) if fact and fact.end_sec is not None else None,
        })

    fmt = dict(plan.get("format") or {})
    if "duration_sec" not in fmt:
        fmt["duration_sec"] = int(movie_index.get("movie", {}).get("duration_sec") or 90)

    editorial = dict(plan.get("editorial_direction") or {})
    return {
        "concept": concept,
        "evidence_refs": [
            {
                "kind": r.get("kind", "text"),
                "scene_id": r.get("scene_id"),
                "value": r.get("value"),
            }
            for r in (selected.get("evidence_refs") or [])
        ],
        "evidence_requirements": list(selected.get("required_evidence") or []),
        "supporting_scenes": supporting_scenes,
        "visual_motifs": list(strategy.get("visual_motifs") or []),
        "character_focus": list(strategy.get("character_focus") or []),
        "format": {
            "type": fmt.get("type", "short_video_essay"),
            "duration_sec": int(fmt.get("duration_sec") or 90),
        },
        "editorial_intent": {
            "pacing": editorial.get("pacing", ""),
            "tone": editorial.get("tone", ""),
            "visual_style": editorial.get("visual_style", ""),
            "audio_style": editorial.get("audio_style", ""),
        },
    }


def contract_is_valid(contract: Dict[str, Any]) -> List[str]:
    """Return a list of missing/empty contract fields (empty list == valid)."""
    errors = []
    for key in CONTRACT_KEYS:
        if key not in contract:
            errors.append(f"grounding contract missing: {key}")
    if not contract.get("concept", {}).get("thesis"):
        errors.append("contract concept.thesis is empty")
    if not contract.get("supporting_scenes"):
        errors.append("contract supporting_scenes is empty")
    return errors


def save_grounding_contract(project_dir: Path, contract: Dict[str, Any]) -> Path:
    project_dir = Path(project_dir)
    path = movie_memory.save_json(project_dir, "grounding_contract.json", contract)
    return path


def load_grounding_contract(project_dir: Path) -> Dict[str, Any]:
    return movie_memory.load_json(project_dir, "grounding_contract.json", {})
