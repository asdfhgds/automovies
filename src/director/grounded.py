"""Movie-grounded Creative Director.

This is the milestone director: it turns an existing Movie Intelligence
representation in to multiple creative concepts, checks that each concept is
supported by evidence the movie ACTUALLY contains, rejects unsupported ideas
(rather than trusting its own inventions), selects the strongest grounded
concept, and emits a scene-aware production plan.

Pipeline (stops at the plan — NOT wired to the script stage yet)::

    Movie Intelligence
      |  SceneFacts (normalize existing scene_index_v2 / movie_index / scenes)
      v
    DirectorContextBuilder (compact, token-limited, fact-grounded context)
      |
      v
    llm -> 5 diverse concepts (milestone schema, with required_evidence)
      |
      v
    EvidenceAnalyzer (ground each concept to real scenes; coverage level)
      |-- concepts below threshold -> REJECTED (regenerate substitutes)
      |
      v
    ConceptCritic (feasibility dimensions) + evidence coverage -> select
      |
      v
    plan (concept + evidence_strategy + format + editorial_direction)
      |
      v
    CreativeMemory (store selected concept)  +  reports/director_reasoning.md

The ``llm`` is a callable ``str -> str`` (raw model output). Real runs pass the
Qwen provider's ``generate_text``; unit tests pass a mock. This keeps real-Qwen
tests gated and everything else hermetic.
"""
import logging
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

from director.concepts import (
    build_generation_prompt,
    build_rejection_prompt,
    build_plan_prompt,
    parse_concepts,
    parse_plan,
    is_generic_thesis,
    compute_diversity_metric,
)
from director.context_builder import DirectorContextBuilder
from director.critic import ConceptCritic
from director.evidence import EvidenceAnalyzer
from director.memory import CreativeMemory
from director.report import build_report, write_report
from director.scene_facts import SceneFacts

logger = logging.getLogger(__name__)

DEFAULT_NUM_CONCEPTS = 5
DEFAULT_MIN_COVERAGE = 0.4
# Bounded regeneration: one initial batch, then at most a single corrective
# retry. If retries still cannot produce a grounded concept, the run FAILS by
# leaving selected_concept/plan None (never forced through).
DEFAULT_MAX_REJECT_ROUNDS = 1


class MovieGroundedDirector:
    """Orchestrates evidence-grounded concept development for a movie."""

    def __init__(
        self,
        llm: Callable[[str], str],
        memory_dir: Optional[Path] = None,
        context_tokens: int = 4096,
        reserve_for_output: int = 2048,
    ):
        self.llm = llm
        self.memory = CreativeMemory(memory_dir)
        self.critic = ConceptCritic()
        self.context_builder = DirectorContextBuilder(
            max_tokens=context_tokens, reserve_for_output=reserve_for_output
        )
        # Honest run bookkeeping for validation reports (llm call counts,
        # regeneration rounds) — never fabricated, only incremented here.
        self.stats = {
            "llm_calls": 0,
            "regeneration_rounds": 0,
            "substitutes_generated": 0,
        }

    # -- Public API ---------------------------------------------------------

    def develop(
        self,
        movie_metadata: Dict[str, Any],
        scale_facts: SceneFacts,
        num_concepts: int = DEFAULT_NUM_CONCEPTS,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        user_topic: Optional[str] = None,
        duration_sec: int = 90,
    ) -> Dict[str, Any]:
        """Run the full grounded director flow and return the full result."""
        analyzer = EvidenceAnalyzer(scale_facts)
        memory_summary = self.memory.get_concepts_summary(limit=3)

        context, ctx_meta = self.context_builder.build_concept_generation_context(
            movie_metadata, scale_facts, creative_memory=memory_summary,
            user_topic=user_topic,
        )

        # 1. Generate an initial batch of grounded concepts.
        concepts = self._generate(context, num_concepts, reject_previous=None)

        # 2. Evidence-gate: reject unsupported/generic concepts.
        concepts, rejected = self._evidence_gate(
            context, concepts, analyzer, min_coverage,
            max_rounds=DEFAULT_MAX_REJECT_ROUNDS
        )

        # 3. Critique every surviving concept; select the strongest.
        for concept in concepts:
            concept["critique"] = self.critic.critique(concept)
        selected, selected_index = self._select(concepts, analyzer)
        if selected is not None:
            selected["_evidence"] = analyzer.concept_evidence(selected)

        # 4. Build the scene-aware final plan.
        plan = None
        if selected is not None:
            plan = self._build_plan(movie_metadata, selected, scale_facts,
                                    analyzer, duration_sec)

        # 5. Persist selected concept in creative memory.
        if selected is not None:
            self._store_in_memory(selected, movie_metadata)

        diversity = compute_diversity_metric(concepts)
        result = {
            "movie": movie_metadata.get("title", "Unknown"),
            "context_meta": ctx_meta,
            "generated_concepts": concepts,
            "rejected_concepts": rejected,
            "selected_concept": selected,
            "selected_concept_index": selected_index,
            "plan": plan,
            "diversity_metric": diversity,
            "llm_stats": dict(self.stats),
            "_scene_facts": scale_facts,
        }
        return result

    def write_reasoning_report(self, project_dir: Path, result: Dict[str, Any]) -> Path:
        """Write ``reports/director_reasoning.md`` for the last run."""
        analyzer = EvidenceAnalyzer(result.get("_scene_facts") or SceneFacts([]))
        text = build_report(
            movie_title=result.get("movie", "Unknown"),
            concepts=result.get("generated_concepts", []),
            rejected=result.get("rejected_concepts", []),
            selected=result.get("selected_concept"),
            selected_index=result.get("selected_concept_index"),
            analyzer=analyzer,
            plan=result.get("plan"),
            diversity_metric=result.get("diversity_metric", 0.0),
        )
        return write_report(Path(project_dir), text)

    def write_report(self, project_dir: Path, result: Dict[str, Any]) -> Path:
        """Public alias of :meth:`write_reasoning_report`."""
        return self.write_reasoning_report(project_dir, result)

    # -- Internals -----------------------------------------------------------

    def _generate(
        self, context: str, num_concepts: int,
        reject_previous: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if reject_previous:
            prompt = build_rejection_prompt(context, reject_previous, num_concepts)
        else:
            prompt = build_generation_prompt(context, num_concepts)
        raw = self.llm(prompt)
        self.stats["llm_calls"] += 1
        concepts = parse_concepts(raw)
        if not concepts:
            logger.warning("Model returned no parseable concepts; retrying once.")
            raw = self.llm(build_generation_prompt(context, num_concepts))
            self.stats["llm_calls"] += 1
            concepts = parse_concepts(raw)
        if reject_previous:
            self.stats["regeneration_rounds"] += 1
            self.stats["substitutes_generated"] += len(concepts)
        return concepts

    def _evidence_gate(
        self,
        context: str,
        concepts: List[Dict[str, Any]],
        analyzer: EvidenceAnalyzer,
        min_coverage: float,
        max_rounds: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Keep concepts with real evidence; reject + regenerate the rest.

        ``rejected`` accumulates every concept that was ever rejected across the
        regneration rounds (so the reasoning report can show them), while only
        concepts that remain admissible survive. If regeneration can't find a
        sufficient replacement within ``max_rounds``, the last attempt is kept
        in ``rejected`` rather than forced into the candidates.
        """
        admissible: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        def _admissible(c):
            if is_generic_thesis(c.get("thesis", "")):
                return False
            return analyzer.is_sufficient(c, min_coverage=min_coverage)

        # First pass: classify.
        pending: List[Dict[str, Any]] = []
        for c in concepts:
            (admissible if _admissible(c) else pending).append(c)
        rejected.extend(pending)

        # Regenerate substitutes for pending ones, up to max_rounds times.
        rounds = 0
        while pending and rounds < max_rounds:
            substitutes = self._generate(context, len(pending),
                                         reject_previous=pending)
            pending = []
            for sub in substitutes:
                (admissible if _admissible(sub) else pending).append(sub)
            rejected.extend(pending)
            rounds += 1

        return admissible, rejected

    def _select(
        self, concepts: List[Dict[str, Any]], analyzer: EvidenceAnalyzer
    ) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
        if not concepts:
            return None, None
        best = None
        best_score = -1.0
        best_idx = 0
        for i, concept in enumerate(concepts):
            critique = concept.get("critique") or {}
            coverage = analyzer.concept_evidence(concept)["coverage"]
            cov_boost = {"HIGH": 0.15, "MED": 0.05, "LOW": 0.0}.get(coverage, 0.0)
            score = float(critique.get("overall", 0.0)) + cov_boost
            if score > best_score:
                best_score = score
                best = concept
                best_idx = i
        return best, best_idx

    def _build_plan(
        self,
        movie_metadata: Dict[str, Any],
        selected: Dict[str, Any],
        scene_facts: SceneFacts,
        analyzer: EvidenceAnalyzer,
        duration_sec: int,
    ) -> Dict[str, Any]:
        evidence_strategy = analyzer.build_evidence_strategy(selected)
        plan_ctx = self.context_builder.build_plan_context(
            selected, scene_facts, evidence_strategy.get("scene_ids", [])
        )
        prompt = build_plan_prompt(plan_ctx, duration_sec=duration_sec)
        plan = parse_plan(self.llm(prompt)) or {}
        self.stats["llm_calls"] += 1
        plan.setdefault("concept", {})
        plan.setdefault("format", {"type": "short_video_essay",
                                   "duration_sec": duration_sec})
        plan.setdefault("editorial_direction", {})
        # The evidence_strategy is deterministic, not model-invented.
        plan["evidence_strategy"] = evidence_strategy
        # Alias for any existing downstream consumer (no duplicate schema).
        plan["concept"].setdefault("thesis",
                                   selected.get("thesis") or plan["concept"].get("thesis", ""))
        return plan

    def _store_in_memory(self, selected: Dict[str, Any], movie_metadata: Dict[str, Any]) -> None:
        self.memory.add_concept(
            title=selected.get("title", "Untitled"),
            thesis=selected.get("thesis", ""),
            hook=selected.get("hook", ""),
            why_interesting=selected.get("why_interesting", ""),
            tone="analytical",
            structure=[],
            visual_strategy=selected.get("visual_opportunity", ""),
            duration_sec=90,
            movie_title=movie_metadata.get("title", "Unknown"),
            themes=self._extract_themes(selected),
        )

    @staticmethod
    def _extract_themes(concept: Dict[str, Any]) -> List[str]:
        text = (concept.get("thesis", "") + " " + concept.get("why_interesting", "")).lower()
        known = {
            "identity", "morality", "power", "love", "death", "freedom", "control",
            "choice", "fate", "justice", "betrayal", "redemption", "loss", "hope",
            "fear", "truth", "illusion", "ambition", "sacrifice", "loyalty",
            "solitude", "nature", "violence", "silence", "confrontation",
        }
        found = [kw for kw in known if kw in text]
        return found[:5]
