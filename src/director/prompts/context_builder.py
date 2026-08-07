"""Context builder for intelligent context limiting."""
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds limited context for LLM to avoid exceeding token limits."""

    def __init__(self, max_tokens: int = 4096, reserve_for_output: int = 2048):
        """
        Initialize context builder.
        
        Args:
            max_tokens: Total model context window
            reserve_for_output: Tokens reserved for model output
        """
        self.max_tokens = max_tokens
        self.reserve_for_output = reserve_for_output
        self.available_for_context = max_tokens - reserve_for_output

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars)."""
        return len(text) // 4

    def build_concept_generation_context(
        self,
        movie_metadata: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
        transcript: Dict[str, Any],
        creative_memory: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build limited context for concept generation.
        
        Prioritizes:
        1. Movie metadata (always included, small)
        2. Scene summaries (first N scenes that fit)
        3. Transcript excerpts (samples to avoid explosion)
        4. Creative memory (last concepts to avoid repetition)
        
        Args:
            movie_metadata: Movie info
            scene_index: All detected scenes
            transcript: Full transcript
            creative_memory: Previous concepts as formatted string
            
        Returns:
            (context_str, metadata_about_what_was_included)
        """
        context_parts = []
        metadata = {
            "movie_metadata_included": True,
            "num_scenes_included": 0,
            "transcript_samples": 0,
            "memory_included": False,
            "truncated": False,
        }

        current_tokens = 0

        # 1. Movie metadata (always include, it's small)
        movie_title = movie_metadata.get("title", "Unknown")
        movie_duration = movie_metadata.get("duration_sec", 0)
        metadata_str = f"Movie: {movie_title} ({movie_duration:.1f}s)"
        tokens = self._estimate_tokens(metadata_str)
        context_parts.append(metadata_str)
        current_tokens += tokens

        # 2. Scene index (include what fits)
        if scene_index:
            scene_str = "Scenes:\n"
            for scene in scene_index:
                scene_id = scene.get("scene_id", "unknown")
                start = scene.get("start_sec", 0)
                end = scene.get("end_sec", 0)
                transcript = scene.get("transcript", "")[:150]  # Limit per scene
                line = f"  {scene_id}: {start:.1f}s-{end:.1f}s | {transcript}\n"

                tokens = self._estimate_tokens(line)
                if current_tokens + tokens > self.available_for_context * 0.4:  # Use 40% for scenes
                    metadata["truncated"] = True
                    break

                scene_str += line
                current_tokens += tokens
                metadata["num_scenes_included"] += 1

            context_parts.append(scene_str)

        # 3. Transcript samples (high-level summary)
        if transcript and "segments" in transcript:
            segments = transcript["segments"]
            transcript_str = "Dialogue samples:\n"

            # Sample: first, middle, last
            sample_indices = [0, len(segments) // 2, len(segments) - 1]
            sample_indices = sorted(set(i for i in sample_indices if 0 <= i < len(segments)))

            for idx in sample_indices:
                seg = segments[idx]
                text = seg.get("text", "")[:100]
                start = seg.get("start_sec", 0)
                line = f"  @{start:.1f}s: {text}\n"

                tokens = self._estimate_tokens(line)
                if current_tokens + tokens > self.available_for_context * 0.5:  # Use 50% for context
                    metadata["truncated"] = True
                    break

                transcript_str += line
                current_tokens += tokens
                metadata["transcript_samples"] += 1

            if metadata["transcript_samples"] > 0:
                context_parts.append(transcript_str)

        # 4. Creative memory (previous concepts)
        if creative_memory and creative_memory.strip():
            memory_tokens = self._estimate_tokens(creative_memory)
            if current_tokens + memory_tokens <= self.available_for_context:
                context_parts.append(f"Previous concepts:\n{creative_memory}")
                metadata["memory_included"] = True
                current_tokens += memory_tokens
            else:
                logger.warning("Creative memory too large, skipping")
                metadata["truncated"] = True

        context = "\n\n".join(context_parts)
        logger.info(f"Built context ({self._estimate_tokens(context)} estimated tokens): {metadata}")

        return context, metadata

    def build_critique_context(self, concept: Dict[str, Any]) -> str:
        """Build context for critique (typically small, no limiting needed)."""
        return f"""Concept: {concept.get('title', '')}
Hook: {concept.get('hook', '')}
Thesis: {concept.get('thesis', '')}
Why Interesting: {concept.get('why_interesting', '')}"""

    def build_production_plan_context(
        self,
        concept: Dict[str, Any],
        scene_index: List[Dict[str, Any]],
    ) -> str:
        """Build context for production planning."""
        duration = concept.get("estimated_duration_sec", 90)

        # Collect available scene types
        scene_types = set()
        for scene in scene_index:
            types = scene.get("supporting_scene_types", [])
            if isinstance(types, list):
                scene_types.update(types)

        scene_types_str = ", ".join(sorted(scene_types)[:15])

        return f"""Concept: {concept.get('title', '')}
Thesis: {concept.get('thesis', '')}
Tone: {concept.get('tone', '')}
Target Duration: {duration}s
Available Scenes: {scene_types_str}"""
