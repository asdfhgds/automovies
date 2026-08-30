"""Mock LLM provider for testing (no API calls)."""
import json
from typing import Dict, Any, List, Optional
from .base import LLMProvider


class GroundedMockLLM:
    """A ``Callable[[str], str]`` mock for the grounded director's raw ``llm``.

    The grounded director (``MovieGroundedDirector``) expects an ``llm`` of the
    form ``str -> str`` that answers its concept-generation and plan prompts
    with milestone-schema JSON. This callable does that deterministically and
    keeps required_evidence grounded in the *real* scene facts (via
    ``_mock_evidence``), so concepts can pass the evidence gate — exactly like
    the real runs, but with no model download.
    """

    def __init__(self, scene_index: Optional[List[Dict[str, Any]]] = None,
                 concepts: Optional[List[Dict[str, Any]]] = None):
        self.scene_index = scene_index or []
        self._concepts = concepts or []
        self.calls: List[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "finalizing the plan" in prompt:
            return json.dumps({
                "concept": {
                    "title": self._concepts[0]["title"] if self._concepts else "Grounded Concept",
                    "hook": self._concepts[0]["hook"] if self._concepts else "",
                    "thesis": self._concepts[0]["thesis"] if self._concepts else "",
                },
                "format": {"type": "short_video_essay", "duration_sec": 90},
                "editorial_plan": {
                    "visual": {
                        "scene_id": "scene-1",
                        "start_sec": 1.2,
                        "end_sec": 3.8,
                        "source_fact_refs": []
                    },
                    "editing": {
                        "transition": "cut",
                        "pacing": "gradual",
                        "rhythm": "steady",
                        "emphasis": "character",
                        "repetition": "none",
                        "purpose": "contrast"
                    },
                    "audio": {
                        "movie_audio": "retain",
                        "narration": "dominant",
                        "music": "low"
                    }
                },
            })
        if self._concepts:
            return json.dumps({"concepts": self._concepts})
        # Build grounded concepts from the real scene index (nothing invented).
        evidence = _mock_evidence(self.scene_index)
        concepts = [
            {
                "title": "The Hidden Argument in Plain Sight",
                "hook": "What looks like a small detail is actually the film's argument.",
                "thesis": "The film arranges its visible details so that the smallest "
                          "on-screen moment carries the weight of the whole idea.",
                "why_interesting": "The movie makes its point through what it chooses to show, "
                                   "not through what it says.",
                "required_evidence": evidence,
                "visual_opportunity": "Slow push-ins on the evidence objects during the reveals.",
                "format": "short_video_essay",
                "diversity_angle": "symbolism",
            },
            {
                "title": "Decoding the Frame",
                "hook": "Every frame is a decision. Let's decode what this one chose.",
                "thesis": "The film's framing and placement of characters quietly "
                          "organizes how we should feel about the action.",
                "why_interesting": "The composition itself is the argument — the camera is the writer.",
                "required_evidence": evidence,
                "visual_opportunity": "Match-cuts between the composition and the action it frames.",
                "format": "short_video_essay",
                "diversity_angle": "cinematography",
            },
        ]
        return json.dumps({"concepts": concepts})


def _mock_evidence(scene_index: List[Dict[str, Any]]) -> List[str]:
    """Derive grounded required_evidence from the actual scene index.

    The milestone director rejects any concept whose ``required_evidence`` is
    empty or un-matchable (nothing invented). The mock reads real scene facts
    (objects/locations/dialogue) so its concepts can pass the evidence gate.
    """
    claims: List[str] = []

    def _add(text: str):
        text = (text or "").strip()
        if text and text not in claims:
            claims.append(text)

    for scene in scene_index or []:
        story = scene.get("story") or {}
        for obj in (story.get("objects") or [])[:1]:
            _add(f"the {obj} is visible on screen")
        for loc in ([story.get("location")] if story.get("location") else [])[:1]:
            _add(f"the scene takes place in {loc}")
        for line in (story.get("dialogue") or [])[:1]:
            if isinstance(line, dict):
                _add((line.get("text") or "").strip())
            elif line:
                _add(str(line))
        for act in (story.get("actions") or [])[:1]:
            _add(f"a character is {act}")

    if not claims:
        claims = ["on-screen dialogue between characters"]
    return claims[:5]


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
            "required_evidence": _mock_evidence(scene_index),
            "visual_opportunity": "Track how cinematography emphasizes character agency or constraint. Use close-ups during moral choices, wide shots during consequences.",
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
                "required_evidence": _mock_evidence(scene_index),
                "visual_opportunity": "Use object tracking and spatial composition to show emotional states. Repeat visual motifs when characters revisit their internal conflicts.",
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
            "required_evidence": _mock_evidence(scene_index),
            "visual_opportunity": "Highlight editing choices and camera positioning that create identification with different characters. Show how perspective shift changes moral weight.",
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
