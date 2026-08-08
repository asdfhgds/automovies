"""Prompt construction for production planning."""
from typing import Dict, Any, List


def build_production_plan_prompt(
    concept: Dict[str, Any],
    scene_index: List[Dict[str, Any]],
) -> str:
    """
    Build a prompt for generating a detailed production plan.
    
    Covers:
    - Structure (distinct sections with durations)
    - Scene requirements for evidence
    - Visual strategy
    - Music strategy
    - Editing strategy
    """
    duration = concept.get("estimated_duration_sec", 90)

    # Scene types available
    scene_types = set()
    for scene in scene_index:
        types = scene.get("supporting_scene_types", [])
        if isinstance(types, list):
            scene_types.update(types)

    scene_types_str = ", ".join(sorted(scene_types)[:10]) if scene_types else "unknown"

    prompt = f"""You are a film director planning production for a video essay.

## Concept
Title: {concept.get('title', 'Untitled')}
Thesis: {concept.get('thesis', '')}
Tone: {concept.get('tone', 'analytical')}

## Constraints
- Target duration: {duration} seconds
- Available scenes: {scene_types_str}

## Task
Create a detailed production plan with:
1. Structure (distinct sections with durations that sum to {duration}s)
2. Scene requirements for evidence
3. Visual strategy (cinematography, editing, effects)
4. Music strategy (mood, instrumentation)
5. Editing strategy (pacing, cuts, transitions)

Return ONLY valid JSON (no markdown, no fencing):
{{
  "structure": [
    {{"section": "hook", "duration_sec": 10}},
    {{"section": "setup", "duration_sec": 20}},
    {{"section": "analysis", "duration_sec": 50}},
    {{"section": "conclusion", "duration_sec": 10}}
  ],
  "scene_requirements": [
    {{"purpose": "demonstrate_thesis", "preferred_scene_types": ["type1", "type2"]}}
  ],
  "visual_strategy": "Description of visual approach",
  "music_strategy": "Description of music approach",
  "editing_strategy": "Description of editing approach"
}}

Plan now:"""

    return prompt
