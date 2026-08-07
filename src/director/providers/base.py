"""Base class for LLM providers."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class LLMProvider(ABC):
    """Base interface for LLM providers."""

    @abstractmethod
    def generate_concepts(
        self,
        movie_metadata: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
        creative_memory: str,
        user_topic: str = None,
        num_concepts: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple creative concepts.

        Returns list of dicts with:
        - title
        - hook
        - thesis
        - why_interesting
        - supporting_scene_types
        - tone
        - visual_strategy
        - estimated_duration_sec
        """
        pass

    @abstractmethod
    def refine_concept(
        self, concept: Dict[str, Any], feedback: str
    ) -> Dict[str, Any]:
        """
        Refine a concept based on feedback.

        Returns refined concept dict.
        """
        pass

    @abstractmethod
    def generate_production_plan(
        self,
        concept: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a production plan for a concept.

        Returns dict with:
        - structure (list of sections with durations)
        - scene_requirements (evidence needs)
        - visual_strategy
        - music_strategy
        - editing_strategy
        """
        pass
