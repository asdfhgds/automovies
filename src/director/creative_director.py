"""Creative director: generates creative video concepts using LLM."""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from .memory import CreativeMemory
from .critic import ConceptCritic
from .providers.base import LLMProvider
from .providers.mock_llm import MockLLMProvider


class CreativeDirector:
    """Orchestrates creative concept generation and production planning."""

    def __init__(self, provider: Optional[LLMProvider] = None, memory_dir: Optional[Path] = None):
        """
        Initialize director.

        Args:
            provider: LLM provider (defaults to MockLLMProvider for testing)
            memory_dir: Where to store creative memory
        """
        self.provider = provider or MockLLMProvider()
        self.memory = CreativeMemory(memory_dir)
        self.critic = ConceptCritic()

    def develop_production_plan(
        self,
        movie_metadata: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
        user_topic: Optional[str] = None,
        num_concepts: int = 3,
    ) -> Dict[str, Any]:
        """
        Full creative development process.

        1. Generate multiple concepts
        2. Critique each concept
        3. Select strongest
        4. Generate production plan
        5. Store in memory

        Returns final production plan dict.
        """
        # Get creative memory summary
        memory_summary = self.memory.get_concepts_summary(limit=3)

        # Step 1: Generate concepts
        concepts = self.provider.generate_concepts(
            movie_metadata=movie_metadata,
            scene_index=scene_index,
            transcript=transcript,
            creative_memory=memory_summary,
            user_topic=user_topic,
            num_concepts=num_concepts,
        )

        if not concepts:
            raise RuntimeError("LLM provider failed to generate concepts")

        # Step 2: Critique each concept
        critiques = []
        for concept in concepts:
            critique = self.critic.critique(concept, scene_index)
            critiques.append(critique)

        # Attach critiques to concepts
        for concept, critique in zip(concepts, critiques):
            concept["critique"] = critique

        # Step 3: Select strongest concept
        selected_idx = self._select_best_concept(concepts, critiques)
        selected_concept = concepts[selected_idx]

        # Step 4: Generate production plan
        production_plan = self.provider.generate_production_plan(
            concept=selected_concept,
            scene_index=scene_index,
            transcript=transcript,
        )

        # Step 5: Store in memory
        self._store_in_memory(selected_concept, movie_metadata)

        # Build final output
        result = {
            "generated_concepts": concepts,
            "critiques": critiques,
            "selected_concept_index": selected_idx,
            "selected_concept": selected_concept,
            "production_plan": production_plan,
        }

        return result

    def _select_best_concept(
        self, concepts: List[Dict[str, Any]], critiques: List[Dict[str, Any]]
    ) -> int:
        """Select best concept by overall score."""
        scores = [c.get("overall", 0.0) for c in critiques]
        if not scores:
            return 0
        return scores.index(max(scores))

    def _store_in_memory(
        self, concept: Dict[str, Any], movie_metadata: Dict[str, Any]
    ) -> None:
        """Store selected concept in creative memory."""
        self.memory.add_concept(
            title=concept.get("title", "Untitled"),
            thesis=concept.get("thesis", ""),
            hook=concept.get("hook", ""),
            why_interesting=concept.get("why_interesting", ""),
            tone=concept.get("tone", ""),
            structure=concept.get("structure", []),
            visual_strategy=concept.get("visual_strategy", ""),
            duration_sec=concept.get("estimated_duration_sec", 60),
            movie_title=movie_metadata.get("title", "Unknown"),
            themes=self._extract_themes(concept),
        )

    @staticmethod
    def _extract_themes(concept: Dict[str, Any]) -> List[str]:
        """Extract themes from concept."""
        # Simple extraction: look for common theme keywords
        text = (
            concept.get("thesis", "") + " " + concept.get("why_interesting", "")
        ).lower()

        themes = []
        theme_keywords = [
            "identity",
            "morality",
            "power",
            "love",
            "death",
            "freedom",
            "control",
            "choice",
            "fate",
            "justice",
            "betrayal",
            "redemption",
            "loss",
            "hope",
            "fear",
            "truth",
            "illusion",
            "ambition",
            "sacrifice",
            "loyalty",
        ]

        for keyword in theme_keywords:
            if keyword in text:
                themes.append(keyword)

        return themes[:5]  # Return top 5
