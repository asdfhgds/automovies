"""Prompt construction for concept critique."""
from typing import Dict, Any


def build_critique_prompt(concept: Dict[str, Any]) -> str:
    """
    Build a prompt for evaluating a creative concept.
    
    Scores across 6 dimensions:
    - originality: How unique is this interpretation?
    - thesis_strength: Is the thesis clear and compelling?
    - evidence_strength: Could this be supported by scenes?
    - visual_potential: Can this be visually compelling?
    - audience_curiosity: Would audiences find this interesting?
    - feasibility: Can this realistically be produced?
    """
    prompt = f"""You are a film critic evaluating a creative concept.

## Concept to Critique
Title: {concept.get('title', 'Untitled')}
Hook: {concept.get('hook', '')}
Thesis: {concept.get('thesis', '')}
Why Interesting: {concept.get('why_interesting', '')}

## Evaluation Criteria
Score each 0.0-1.0:
1. originality: How unique is this interpretation?
2. thesis_strength: Is the thesis clear, specific, and compelling?
3. evidence_strength: Could this concept be supported by scenes?
4. visual_potential: Can this be made visually compelling?
5. audience_curiosity: Would audiences find this interesting?
6. feasibility: Can this realistically be produced?

Return ONLY valid JSON (no markdown, no fencing):
{{
  "scores": {{
    "originality": 0.7,
    "thesis_strength": 0.8,
    "evidence_strength": 0.6,
    "visual_potential": 0.8,
    "audience_curiosity": 0.7,
    "feasibility": 0.8
  }},
  "overall": 0.75,
  "critique": "Brief summary of strengths and areas for improvement"
}}

Evaluate now:"""

    return prompt
