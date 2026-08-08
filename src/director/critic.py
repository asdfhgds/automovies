"""Critic: evaluates creative concepts."""
import json
from typing import Dict, Any, List


class ConceptCritic:
    """Evaluates creative concepts on multiple dimensions."""

    @staticmethod
    def critique(
        concept: Dict[str, Any], scene_index: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Critique a concept.
        
        Returns a dict with scores (0.0-1.0) and overall assessment.
        """
        # Extract fields
        thesis = concept.get("thesis", "")
        why_interesting = concept.get("why_interesting", "")
        visual_strategy = concept.get("visual_strategy", "")
        supporting_scene_types = concept.get("supporting_scene_types", [])

        # Score dimensions
        originality = ConceptCritic._score_originality(concept)
        thesis_strength = ConceptCritic._score_thesis_strength(thesis)
        evidence_strength = ConceptCritic._score_evidence_strength(
            scene_index, supporting_scene_types
        )
        visual_potential = ConceptCritic._score_visual_potential(visual_strategy)
        audience_curiosity = ConceptCritic._score_audience_curiosity(
            thesis, why_interesting
        )
        feasibility = ConceptCritic._score_feasibility(supporting_scene_types)

        # Calculate overall
        scores = {
            "originality": originality,
            "thesis_strength": thesis_strength,
            "evidence_strength": evidence_strength,
            "visual_potential": visual_potential,
            "audience_curiosity": audience_curiosity,
            "feasibility": feasibility,
        }
        overall = sum(scores.values()) / len(scores)

        # Generate critique
        critique_text = ConceptCritic._generate_critique(
            concept, scores, overall
        )

        return {
            "originality": originality,
            "thesis_strength": thesis_strength,
            "evidence_strength": evidence_strength,
            "visual_potential": visual_potential,
            "audience_curiosity": audience_curiosity,
            "feasibility": feasibility,
            "overall": overall,
            "critique": critique_text,
        }

    @staticmethod
    def _score_originality(concept: Dict[str, Any]) -> float:
        """Score originality (0-1)."""
        # Check if thesis is generic or vague
        thesis = concept.get("thesis", "").lower()
        generic_phrases = [
            "focused analysis",
            "key moment",
            "important scene",
            "character development",
            "plot summary",
        ]

        for phrase in generic_phrases:
            if phrase in thesis:
                return 0.3  # Generic

        # Check length and specificity
        if len(thesis.split()) < 5:
            return 0.4
        if "specific" in thesis or "unique" in thesis or "particular" in thesis:
            return 0.8

        return 0.6

    @staticmethod
    def _score_thesis_strength(thesis: str) -> float:
        """Score how strong/defensible the thesis is (0-1)."""
        # Length indicates depth
        words = len(thesis.split())
        if words < 5:
            return 0.2
        if words < 15:
            return 0.5
        if words < 30:
            return 0.8
        return 0.9

    @staticmethod
    def _score_evidence_strength(
        scene_index: List[Dict[str, Any]], scene_types: List[str]
    ) -> float:
        """Score evidence availability (0-1)."""
        if not scene_index:
            return 0.5  # Can't evaluate without scene data

        if not scene_types:
            return 0.4  # No specific scenes requested

        # Check how many requested scene types exist
        available_count = 0
        for scene in scene_index:
            # Rough heuristic: check if scene has transcript
            if scene.get("transcript"):
                available_count += 1

        if not available_count:
            return 0.3

        # Proportional to availability
        return min(1.0, (available_count / len(scene_types)) * 0.9 + 0.1)

    @staticmethod
    def _score_visual_potential(visual_strategy: str) -> float:
        """Score visual potential (0-1)."""
        if not visual_strategy:
            return 0.3

        words = len(visual_strategy.split())
        if words < 10:
            return 0.4
        if words < 30:
            return 0.7
        return 0.9

    @staticmethod
    def _score_audience_curiosity(thesis: str, why_interesting: str) -> float:
        """Score how curious audience will be (0-1)."""
        curiosity_markers = [
            "why",
            "how",
            "what if",
            "secret",
            "hidden",
            "reveal",
            "discover",
            "proves",
            "contradiction",
            "paradox",
        ]

        text = (thesis + " " + why_interesting).lower()
        matches = sum(1 for marker in curiosity_markers if marker in text)

        # 0-2 matches: low curiosity, 3-5: medium, 6+: high
        if matches < 2:
            return 0.3
        if matches < 5:
            return 0.6
        return 0.9

    @staticmethod
    def _score_feasibility(scene_types: List[str]) -> float:
        """Score how feasible with available scenes (0-1)."""
        # More scene types = higher feasibility requirement
        if not scene_types:
            return 0.7

        # Assume most movie projects have 20-50 scenes
        # 5 requested types is highly feasible
        # 10+ is challenging
        if len(scene_types) <= 3:
            return 0.9
        if len(scene_types) <= 6:
            return 0.7
        if len(scene_types) <= 10:
            return 0.5
        return 0.3

    @staticmethod
    def _generate_critique(
        concept: Dict[str, Any], scores: Dict[str, float], overall: float
    ) -> str:
        """Generate a text critique based on scores."""
        lines = []

        # Overall assessment
        if overall >= 0.8:
            lines.append("Excellent concept with strong potential.")
        elif overall >= 0.6:
            lines.append("Good concept with solid foundations.")
        elif overall >= 0.4:
            lines.append("Acceptable concept; some refinement needed.")
        else:
            lines.append("Weak concept; consider revision.")

        # Detailed feedback
        strengths = []
        weaknesses = []

        for dimension, score in scores.items():
            if score >= 0.7:
                strengths.append(dimension)
            elif score <= 0.4:
                weaknesses.append(dimension)

        if strengths:
            lines.append(f"Strengths: {', '.join(strengths)}")
        if weaknesses:
            lines.append(f"Areas for improvement: {', '.join(weaknesses)}")

        return " ".join(lines)
