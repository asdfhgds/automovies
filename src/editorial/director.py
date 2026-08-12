"""Editorial director.

Turns the (already LLM-generated) thesis + hook + ranking into an
:class:`~editorial.plan.EditorialPlan`: a structured decision about how the
argument becomes an edit. The planning logic is deterministic and transparent;
the *thesis itself* comes from the (real, in GPU mode) Creative Director, so a
real movie run is still genuinely LLM-driven at the conceptual level.

The provider pattern means a future LLM editorial planner can replace the
heuristic one without changing callers.
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from editorial.plan import (
    EditingDirective,
    EditorialEvidence,
    EditorialPlan,
    EditorialSegment,
    NarrationBlock,
    NarrationDelivery,
    validate_plan,
)
from editorial.retrieval import EvidenceRetriever, blend_director_requirements
from movie_understanding import movie_memory, text_utils

STRICT_ENV = "REQUIRE_REAL_LLM"
PROVIDER_ENV = "EDITORIAL_PLANNER"


class EditorialPlanner(ABC):
    name = "base"

    @abstractmethod
    def create_plan(self, movie_index: dict, director_plan: dict,
                    retriever: EvidenceRetriever,
                    creative_task: str, target_sec: float) -> EditorialPlan:
        raise NotImplementedError


class HeuristicEditorialPlanner(EditorialPlanner):
    """Deterministic composition around the (real) thesis, evidence-driven."""

    name = "heuristic"

    # ------------------------------------------------------------------
    # Narration building blocks (original, thesis-grounded)
    # ------------------------------------------------------------------

    def _hook(self, director_plan: dict) -> str:
        hook = (director_plan.get("hook") or "").strip()
        if hook:
            return hook
        thesis = (director_plan.get("thesis") or "").strip()
        return f"What if the movie was hiding its real argument in plain sight? {thesis}"

    def _claim(self, thesis: str, topic: str) -> str:
        thesis = thesis.rstrip(".!?")
        topic = (topic or "this detail").strip()
        return (
            f"This is where the film makes its point about {topic}: "
            f"what looks like a throwaway moment is actually the thesis — "
            f"{thesis} — taking shape."
        )

    def _observation(self, segment: EditorialSegment, scene_summary: str) -> Optional[str]:
        if not segment.evidence:
            return None
        ev = segment.evidence[0]
        snippet = None
        if scene_summary:
            snippet = scene_summary[:120]
        if snippet:
            return f"Watch {ev.scene_id} at {ev.start_sec:.1f}s: {snippet}"
        return f"The film shows us this in {ev.scene_id}."

    def _close(self, thesis: str) -> str:
        thesis = thesis.rstrip(".!?")
        return (
            f"Strip away the plot and {thesis} is what remains — "
            f"an argument the movie makes with pictures, not words."
        )

    # ------------------------------------------------------------------

    def create_plan(self, movie_index: dict, director_plan: dict,
                    retriever: EvidenceRetriever,
                    creative_task: str, target_sec: float) -> EditorialPlan:
        thesis = (director_plan.get("thesis") or "").strip()
        if not thesis:
            raise ValueError("director_plan.thesis is required to build an editorial plan")

        scenes = movie_index.get("scenes", [])
        if not scenes:
            raise ValueError("movie_index has no scenes to draw evidence from")

        structure = self._structure(target_sec, len(scenes))
        query = blend_director_requirements(f"{creative_task}. {thesis}", thesis)
        used = set()
        segments: List[EditorialSegment] = []
        cursor = 0.0

        # 1) Hook segment: claim + first piece of evidence.
        hook_seg = self._make_segment(
            idx=0,
            purpose="hook and thesis",
            claim=self._hook(director_plan),
            query=query,
            retriever=retriever,
            exclude=used,
            transition="cut",
            emphasis="wide",
        )
        used.update(e.scene_id for e in hook_seg.evidence)
        segments.append(hook_seg)

        # 2) Evidence segments
        remaining = structure["evidence_count"]
        for i in range(remaining):
            topic = _next_topic(movie_index, used)
            purpose = self._purpose_for(i, remaining, topic)
            claim = self._claim(thesis, topic)
            seg = self._make_segment(
                idx=i + 1,
                purpose=purpose,
                claim=claim,
                query=blend_director_requirements(purpose, thesis),
                retriever=retriever,
                exclude=used,
                transition="crossfade" if i % 2 == 0 else "fade",
                emphasis="close_up" if i % 2 == 1 else "wide",
                crop_zoom=1.15 if i % 2 == 1 else 1.0,
                speed=0.9 if i % 2 == 0 else 1.0,
            )
            for e in seg.evidence:
                used.add(e.scene_id)
            segments.append(seg)

        # 3) Closing segment: derive the observation from the first piece of
        #    evidence we already used, then land the argument.
        closing = EditorialSegment(
            id="seg_close",
            purpose="conclusion",
            evidence=segments[0].evidence[:1],
            narration=self._close_block(self._close(thesis), pause_before=0.4),
            editing=EditingDirective(
                transition="fade",
                fade_edges=True,
                hold_sec=0.8,
                duck_level=0.02,
            ),
        )
        segments.append(closing)

        plan = EditorialPlan(
            title=(director_plan.get("title") or "A Film Reveals Its Argument"),
            thesis=thesis,
            hook={
                "text": self._hook(director_plan),
                "visual_strategy": "black-and-white hold, slow push in",
            },
            segments=segments,
            length_target_sec=target_sec,
            creative_task=creative_task,
            provenance={"planner": self.name, "thesis_source": director_plan.get("director_provider", "unknown")},
        )
        errors = validate_plan(plan)
        if errors:
            raise ValueError("invalid editorial plan: " + "; ".join(errors))
        return plan

    # ------------------------------------------------------------------

    def _structure(self, target_sec: float, n_scenes: int) -> dict:
        # One scene anchors the hook, one re-anchors the close (the close
        # reuses the hook's evidence, so it does not consume a new scene).
        desired = 3 if target_sec >= 90 else 2
        evidence_count = max(1, min(desired, n_scenes - 1))
        if n_scenes < 2:
            raise ValueError("need at least 2 scenes to build an editorial plan")
        return {"hook": 1, "evidence_count": evidence_count, "close": 1}

    def _purpose_for(self, i: int, total: int, topic: str) -> str:
        purposes = [
            "establish the central contradiction the thesis exposes",
            "turn the thesis into a question the visuals answer",
            "make the argument concrete with dialogue evidence",
            "resolve the question the film itself posed",
        ]
        base = purposes[min(i, len(purposes) - 1)]
        return f"{base} ({topic})"

    def _make_segment(self, idx: int, purpose: str, claim: str, query: str,
                      retriever: EvidenceRetriever, exclude: set,
                      transition: str, emphasis: str,
                      crop_zoom: float = 1.0, speed: float = 1.0,
                      evidence_k: int = 1) -> EditorialSegment:
        evidence = retriever.retrieve(query, k=evidence_k, exclude=exclude)
        if not evidence:
            raise RuntimeError(f"no evidence found for segment {idx} (query: {query})")
        return EditorialSegment(
            id=f"seg_{idx:02d}",
            purpose=purpose,
            evidence=evidence,
            narration=self._claim_block(claim),
            editing=EditingDirective(
                shot_order=[e.scene_id for e in evidence],
                transition=transition,
                emphasis=emphasis,
                crop_zoom=crop_zoom,
                speed=speed,
            ),
        )

    @staticmethod
    def _claim_block(text: str) -> NarrationBlock:
        return NarrationBlock(
            text=text,
            delivery=NarrationDelivery(tone="analytical", emotion="thoughtful",
                                       energy=0.55, pace=0.95),
        )

    @staticmethod
    def _close_block(text: str, pause_before: float = 0.0) -> NarrationBlock:
        return NarrationBlock(
            text=text,
            delivery=NarrationDelivery(tone="quiet", emotion="resolute",
                                       energy=0.4, pace=0.85,
                                       dramatic_intensity=0.7,
                                       pause_before=pause_before),
        )


class QwenEditorialPlanner(EditorialPlanner):
    """Real-LLM editorial planner (stub pending a Qwen-backed implementation).

    Presence over a heuristic default is gated on the strict GPU env so a
    production run can require a genuine LLM composition rather than silently
    degrading to heuristics. The thesis is already real; this planner composes
    segments/narration. It is not yet wired — create_plan raises so nobody
    mistakes the stub for a real LLM.
    """

    name = "qwen"

    def __init__(self):
        if not os.getenv("REQUIRE_REAL_LLM", "").lower() == "true":
            raise RuntimeError("QwenEditorialPlanner requires REQUIRE_REAL_LLM=true")

    def create_plan(self, movie_index: dict, director_plan: dict,
                    retriever: EvidenceRetriever,
                    creative_task: str, target_sec: float) -> EditorialPlan:
        raise NotImplementedError(
            "Qwen editorial planning is not wired yet; the heuristic planner "
            "is the production path for this milestone."
        )


def editorial_planner_from_env() -> EditorialPlanner:
    """Provider factory: ``EDITORIAL_PLANNER=heuristic|qwen`` (default heuristic)."""
    provider = os.getenv(PROVIDER_ENV, "").strip().lower() or "heuristic"
    if provider == "qwen":
        return QwenEditorialPlanner()
    return HeuristicEditorialPlanner()


def create_editorial_plan(project_dir: Path, creative_task: str = "",
                          target_sec: float = 90.0,
                          planner: Optional[EditorialPlanner] = None) -> EditorialPlan:
    """Pipeline entry point: reads movie_index + director plan, writes
    ``editorial_plan.json`` at the project root."""
    project_dir = Path(project_dir)
    movie_index = movie_memory.load_movie_index(project_dir)
    director_plan = movie_memory.load_json(project_dir, "director_plan.json", {})
    retriever = EvidenceRetriever.from_project_dicts(movie_index)
    planner = planner or editorial_planner_from_env()

    plan = planner.create_plan(
        movie_index, director_plan, retriever,
        creative_task or director_plan.get("creative_task", ""),
        float(target_sec),
    )
    movie_memory.save_json(project_dir, "editorial_plan.json", plan.to_dict())
    return plan


def _next_topic(movie_index: dict, used: set) -> str:
    for scene in movie_index.get("scenes", []):
        if scene.get("scene_id") in used:
            continue
        topics = scene.get("story", {}).get("topics", [])
        if topics:
            return topics[0]
    return "the film's central idea"