"""Prompt construction for concept generation."""
from typing import Dict, Any, List, Optional


def build_concept_generation_prompt(
    movie_metadata: Dict[str, Any],
    scene_index: List[Dict[str, Any]],
    transcript: Dict[str, Any],
    creative_memory: str,
    user_topic: Optional[str] = None,
    num_concepts: int = 3,
) -> str:
    """
    Build a detailed prompt for concept generation.
    
    Focuses on:
    - Specific claims, not generic platitudes
    - Evidence from actual scenes
    - Visual storytelling opportunities
    - Diversity across multiple dimensions
    """
    movie_title = movie_metadata.get("title", "Unknown")
    movie_duration = movie_metadata.get("duration_sec", 0)

    # Build scene summary (limit to avoid token explosion)
    scene_summary = ""
    if scene_index:
        # Include first 5 scenes with timestamps
        for scene in scene_index[:5]:
            scene_id = scene.get("scene_id", "unknown")
            start = scene.get("start_sec", 0)
            end = scene.get("end_sec", 0)
            transcript_text = scene.get("transcript", "")[:200]
            scene_summary += f"\n- {scene_id}: {start:.1f}s-{end:.1f}s | {transcript_text}"

    # Build memory context
    memory_context = ""
    if creative_memory and creative_memory.strip():
        memory_context = f"\n\n## Previous Concepts (for avoiding repetition):\n{creative_memory}"

    # Build transcript summary (limit)
    transcript_text = ""
    if transcript and "segments" in transcript:
        segments = transcript["segments"][:10]  # First 10 segments
        for seg in segments:
            text = seg.get("text", "")[:100]
            transcript_text += f"\n- {text}"

    prompt = f"""You are a creative director analyzing a film to generate engaging video essay concepts.

## Movie Details
- Title: {movie_title}
- Duration: {movie_duration:.1f} seconds

## Scene Breakdown{scene_summary}

## Transcript (first segments){transcript_text}{memory_context}

## Task
Generate {num_concepts} distinct creative concepts for this film. Each concept must:
1. Have a SPECIFIC claim (not generic platitudes)
2. Reference actual scenes or dialogue from the film
3. Be visually compelling and suitable for a video essay
4. Have a clear hook and thesis
5. Be diverse in approach (psychology, philosophy, symbolism, narrative structure, etc.)

Do NOT generate generic concepts like:
- "This movie teaches us about life"
- "Characters show emotions"
- "Good versus evil"

Instead, generate specific interpretations with evidence.

{f"User Focus: {user_topic}" if user_topic else ""}

Return ONLY valid JSON (no markdown, no fencing) with this structure:
{{
  "concepts": [
    {{
      "title": "Specific concept title",
      "hook": "Engaging question or statement that draws viewers in",
      "thesis": "The core argument about the film (specific, evidence-based)",
      "why_interesting": "Why this interpretation reveals something important",
      "supporting_scene_types": ["type1", "type2"],
      "tone": "analytical_philosophical|psychological_intimate|metanarrative_interrogative|other",
      "visual_strategy": "How to visualize this concept in editing, effects, etc.",
      "estimated_duration_sec": 85
    }}
  ]
}}

Generate now:"""

    return prompt
