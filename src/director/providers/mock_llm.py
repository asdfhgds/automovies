"""Mock LLM provider for testing (no API calls)."""
import json
from typing import Dict, Any, List
from .base import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock provider that returns deterministic creative concepts for testing."""

    def generate_concepts(
        self,
        movie_metadata: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
        creative_memory: str,
        user_topic: str = None,
        num_concepts: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate mock concepts."""
        movie_title = movie_metadata.get("title", "Unknown Movie")

        # Generate diverse concepts based on input
        concepts = []

        # Concept 1: Thematic analysis
        concepts.append({
            "title": f"The Hidden Philosophy in {movie_title}",
            "hook": f"What if {movie_title} is actually exploring a deeper philosophical truth?",
            "thesis": f"{movie_title} uses its narrative structure to explore themes of fate, choice, and moral ambiguity through the tension between determinism and free will.",
            "why_interesting": "Examines how visual language and dialogue reinforce philosophical ideas without stating them explicitly.",
            "supporting_scene_types": ["dialogue", "decision", "consequence", "revelation"],
            "tone": "analytical_philosophical",
            "visual_strategy": "Track how cinematography emphasizes character agency or constraint. Use close-ups during moral choices, wide shots during consequences.",
            "estimated_duration_sec": 90,
        })

        # Concept 2: Character study
        if len(scene_index) > 0:
            concepts.append({
                "title": f"The Architect Within: Character Psychology in {movie_title}",
                "hook": "Every character builds worlds in their mind before acting. Let's decode the architecture of their choices.",
                "thesis": f"{movie_title} reveals how characters construct meaning from chaos, using symbolic objects and recurring patterns to impose order on their circumstances.",
                "why_interesting": "Explores the psychological defense mechanisms characters employ, making abstract psychology visually concrete.",
                "supporting_scene_types": ["character_alone", "symbolic_moment", "conflict", "resolution"],
                "tone": "psychological_intimate",
                "visual_strategy": "Use object tracking and spatial composition to show emotional states. Repeat visual motifs when characters revisit their internal conflicts.",
                "estimated_duration_sec": 75,
            })

        # Concept 3: Meta narrative
        concepts.append({
            "title": f"Watching vs. Living: The Viewer's Paradox in {movie_title}",
            "hook": "Is the film asking us to judge, or to feel the weight of impossible choices ourselves?",
            "thesis": f"{movie_title} positions the viewer as both witness and participant, blurring moral judgment through perspective and withholding information that would allow certainty.",
            "why_interesting": "Examines the film's grammar of storytelling—what it shows, hides, and forces us to infer—as the engine of moral complexity.",
            "supporting_scene_types": ["ambiguous_moment", "point_of_view_shift", "information_reveal", "final_judgment"],
            "tone": "metanarrative_interrogative",
            "visual_strategy": "Highlight editing choices and camera positioning that create identification with different characters. Show how perspective shift changes moral weight.",
            "estimated_duration_sec": 85,
        })

        return concepts[:num_concepts]

    def refine_concept(
        self, concept: Dict[str, Any], feedback: str
    ) -> Dict[str, Any]:
        """Refine a concept based on feedback (mock: just returns with updated field)."""
        concept["refined_from_feedback"] = feedback
        return concept

    def generate_production_plan(
        self,
        concept: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a production plan (mock)."""
        duration_sec = concept.get("estimated_duration_sec", 90)

        # Rough structure based on duration
        structure = [
            {"section": "hook", "duration_sec": int(duration_sec * 0.1)},
            {"section": "setup", "duration_sec": int(duration_sec * 0.15)},
            {"section": "analysis", "duration_sec": int(duration_sec * 0.55)},
            {"section": "conclusion", "duration_sec": int(duration_sec * 0.2)},
        ]

        return {
            "concept": {
                "title": concept.get("title", "Untitled"),
                "hook": concept.get("hook", ""),
                "thesis": concept.get("thesis", ""),
                "why_interesting": concept.get("why_interesting", ""),
            },
            "format": {
                "type": "video_essay",
                "duration_sec": duration_sec,
                "aspect_ratio": "16:9",
            },
            "tone": concept.get("tone", "analytical"),
            "structure": structure,
            "scene_requirements": [
                {"purpose": "demonstrate_thesis", "preferred_scene_types": concept.get("supporting_scene_types", [])}
            ],
            "visual_strategy": concept.get("visual_strategy", ""),
            "music_strategy": "Underscore analysis sections with contemplative instrumental; build during moments of revelation.",
            "editing_strategy": "Cut on ideas, not on dialogue. Use juxtaposition to highlight contrasts. Let important moments breathe.",
        }
