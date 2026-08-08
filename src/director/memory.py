"""Creative memory: persists previous concepts to avoid repetition."""
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class CreativeMemory:
    """Stores and retrieves previous creative concepts."""

    def __init__(self, memory_dir: Path = None):
        if memory_dir is None:
            memory_dir = Path("data/memory")
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.concepts_file = self.memory_dir / "concepts.jsonl"

    def add_concept(
        self,
        title: str,
        thesis: str,
        tone: str,
        structure: List[Dict[str, Any]],
        visual_strategy: str,
        duration_sec: int,
        movie_title: str,
        themes: List[str],
        hook: str = None,
        why_interesting: str = None,
    ):
        """Add a concept to memory."""
        concept = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "thesis": thesis,
            "hook": hook,
            "why_interesting": why_interesting,
            "tone": tone,
            "structure": structure,
            "visual_strategy": visual_strategy,
            "duration_sec": duration_sec,
            "movie_title": movie_title,
            "themes": themes,
        }
        with self.concepts_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(concept, ensure_ascii=False) + "\n")

    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """Retrieve all stored concepts."""
        if not self.concepts_file.exists():
            return []
        concepts = []
        with self.concepts_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        concepts.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return concepts

    def get_concepts_summary(self, limit: int = 5) -> str:
        """Get a summary of recent concepts to inform the director."""
        concepts = self.get_all_concepts()
        if not concepts:
            return "No previous concepts in memory."

        recent = concepts[-limit:]
        summary_lines = ["## Previous Creative Concepts:"]
        for i, c in enumerate(recent, 1):
            summary_lines.append(
                f"\n{i}. {c.get('title', 'Untitled')} (movie: {c.get('movie_title', 'Unknown')})"
            )
            summary_lines.append(f"   Thesis: {c.get('thesis', 'N/A')}")
            summary_lines.append(f"   Tone: {c.get('tone', 'N/A')}")
            summary_lines.append(f"   Themes: {', '.join(c.get('themes', []))}")

        return "\n".join(summary_lines)

    def clear_memory(self):
        """Clear all stored concepts."""
        if self.concepts_file.exists():
            self.concepts_file.unlink()
