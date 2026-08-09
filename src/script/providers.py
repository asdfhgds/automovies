"""LLM-backed script providers. Kept separate from creative direction."""
from typing import Any, Dict, List, Optional

from generation.base import ScriptProvider


class QwenScriptProvider(ScriptProvider):
    """Create grounded narration with Qwen; the caller supplies only selected evidence."""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def generate_script(self, thesis: str, selected_scenes: List[Dict[str, Any]],
                        movie_context: Optional[Dict[str, Any]] = None,
                        target_duration: int = 60, tone: str = "analytical",
                        structure: str = "classic") -> Dict[str, Any]:
        scene_lines = []
        valid_ids = []
        for scene in selected_scenes:
            scene_id = scene.get("scene_id")
            if not scene_id:
                continue
            valid_ids.append(scene_id)
            scene_lines.append(f"{scene_id} ({scene.get('start_sec', '?')}-{scene.get('end_sec', '?')}s): "
                               f"{scene.get('transcript') or scene.get('summary') or ''}")
        evidence = "\n".join(scene_lines)
        prompt = f"""Write a {target_duration}-second {tone} cinematic video essay.
Thesis: {thesis}
Available evidence (do not invent scenes or dialogue):
{evidence}
Return ONLY JSON: {{"sections":[{{"id":"hook","type":"hook","narration":"...","scene_ids":["..."]}}]}}.
Use claim -> evidence -> interpretation -> implication. Include hook, analysis, and conclusion.
Every non-hook section must cite one or more IDs from: {valid_ids}. Keep narration near {max(40, target_duration * 2.5):.0f} words."""
        result = self.llm_provider.generate_json(prompt, ["sections"])
        sections = result["sections"]
        if not isinstance(sections, list) or not sections:
            raise ValueError("Script has no sections")
        valid = set(valid_ids)
        for section in sections:
            if not isinstance(section, dict) or not section.get("narration"):
                raise ValueError("Script section is malformed")
            ids = section.get("scene_ids", [])
            if any(scene_id not in valid for scene_id in ids):
                raise ValueError("Script referenced a scene outside selected evidence")
            if section.get("type") != "hook" and not ids:
                raise ValueError("Analytical sections must cite selected scenes")
        voiceover = " ".join(section["narration"].strip() for section in sections)
        return {"sections": sections, "voiceover_text": voiceover,
                "word_count": len(voiceover.split()), "estimated_duration_sec": round(len(voiceover.split()) / 2.5, 1),
                "provider": "qwen", "tone": tone}
