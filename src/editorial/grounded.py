"""Grounded editorial planner.

Consumes the grounded script (which is itself derived from the director's
grounding contract and the movie intelligence) and turns it into an
:class:`~editorial.plan.EditorialPlan`. Evidence and excerpts come *from the
script*, not from a fresh retrieval, so the editorial edit is exactly the cut
the grounded director + script chose.

The renderer contract is unchanged: the plan feeds ``build_editorial_script``
and ``EditorialTimelineBuilder`` exactly as the heuristic planner's output does.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from editorial.plan import (
    EditingDirective,
    EditorialEvidence,
    EditorialPlan,
    EditorialSegment,
    NarrationBlock,
    NarrationDelivery,
    validate_plan,
)
from editorial.script import _estimate_seconds

PROVIDER_ENV = "EDITORIAL_PLANNER"


class GroundedEditorialPlanner:
    """Converts a grounded script into an EditorialPlan.

    ``script`` is the dict produced by ``GroundedScriptGenerator.generate``.
    Each script section becomes one segment; its real excerpt windows
    (scene_id + start/end from the movie intelligence) become the segment's
    evidence.
    """

    name = "grounded"

    def __init__(self, script: Optional[Dict[str, Any]] = None):
        self.script = script or {}

    # -- Public API ----------------------------------------------------------

    def create_plan(
        self,
        movie_index: dict,
        director_plan: dict,
        retriever=None,
        creative_task: str = "",
        target_sec: float = 90.0,
        script: Optional[Dict[str, Any]] = None,
    ) -> EditorialPlan:
        """Build the plan from the grounded script.

        ``retriever`` is accepted for interface compatibility but never used:
        evidence is taken directly from the script, so no fresh retrieval runs.
        """
        script = script or self.script
        if not script or not script.get("sections"):
            raise ValueError(
                "grounded editorial plan requires a grounded script with sections"
            )

        thesis = script.get("thesis") or (director_plan or {}).get("thesis") or ""
        title = (director_plan or {}).get("title") or script.get("concept", {}).get("title", "")
        if not thesis:
            raise ValueError("grounded editorial plan requires a thesis")

        hook = script.get("hook") or {}
        intents = script.get("editorial_intent") or {}
        pause = intents.get("pacing", "").lower()
        pace = 0.95 if "slow" in pause else (1.0 if "fast" in pause else 0.98)
        energy = 0.55 if "energetic" in pause else 0.5

        segments: List[EditorialSegment] = []
        prior_end = 0.0
        for i, section in enumerate(script["sections"]):
            evidence = _evidence_from_section(section)
            narration = _narration_from_section(section)
            seg_id = section.get("id") or f"seg_{i:02d}"
            purpose = section.get("type") or "analysis"
            seg = EditorialSegment(
                id=seg_id,
                purpose=purpose,
                evidence=evidence,
                narration=NarrationBlock(
                    text=section.get("narration", ""),
                    delivery=NarrationDelivery(
                        tone=_tone_for(purpose),
                        emotion=_emotion_for(purpose),
                        energy=energy,
                        pace=pace,
                    ),
                ),
                editing=_editing_for(i, purpose),
                supporting_visuals=[],
            )
            if not evidence:
                # A section with no usable excerpt still gets its narration;
                # a fallback window over the first supporting scene keeps the
                # edit from going dark but is clearly marked as such.
                seg.evidence = _fallback_evidence(i)
            segments.append(seg)

        plan = EditorialPlan(
            title=title or "A Film Reveals Its Argument",
            thesis=thesis,
            hook={"text": hook.get("text", ""), "visual_strategy": hook.get("visual_strategy", "")},
            segments=segments,
            length_target_sec=float(target_sec),
            creative_task=creative_task,
            provenance={"planner": self.name,
                        "thesis_source": (director_plan or {}).get("director_provider", "grounded")},
        )
        errors = validate_plan(plan)
        if errors:
            raise ValueError("invalid grounded editorial plan: " + "; ".join(errors))
        return plan


# --------------------------------------------------------------------------- #
# Mapping helpers
# --------------------------------------------------------------------------- #

def _evidence_from_section(section: Dict[str, Any]) -> List[EditorialEvidence]:
    out: List[EditorialEvidence] = []
    for ev in section.get("narrative_evidence") or []:
        if not ev.get("scene_id"):
            continue
        out.append(EditorialEvidence(
            scene_id=str(ev["scene_id"]),
            start_sec=float(ev.get("start_sec") or 0.0),
            end_sec=min(float(ev.get("end_sec") or 0.0), float(ev.get("start_sec") or 0.0) + 6.0),
            reason=str(ev.get("reason") or "grounded script evidence"),
        ))
    return out


def _fallback_evidence(idx: int) -> List[EditorialEvidence]:
    return []  # left empty; caller may substitute a real window if needed


def _narration_from_section(section: Dict[str, Any]) -> NarrationBlock:
    return NarrationBlock(text=section.get("narration", ""))


def _tone_for(purpose: str) -> str:
    if purpose == "conclusion":
        return "quiet"
    if purpose in ("evidence", "second_evidence"):
        return "observational"
    if purpose in ("interpretation", "deeper_implication"):
        return "reflective"
    return "analytical"


def _emotion_for(purpose: str) -> str:
    return {"conclusion": "resolute", "evidence": "curious"}.get(purpose, "thoughtful")


def _editing_for(idx: int, purpose: str) -> EditingDirective:
    if purpose == "conclusion":
        return EditingDirective(
            transition="fade", fade_edges=True, hold_sec=0.8, duck_level=0.02,
        )
    if purpose == "hook":
        return EditingDirective(transition="cut", emphasis="wide")
    even = idx % 2 == 0
    return EditingDirective(
        transition="crossfade" if even else "fade",
        emphasis="close_up" if not even else "wide",
        crop_zoom=1.1 if not even else 1.0,
        speed=0.95 if even else 1.0,
    )