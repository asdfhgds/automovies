"""Director planner stub: generate a simple director_plan.json from project metadata."""
import json
import uuid
from pathlib import Path

def plan_director(project_dir: Path, title: str = None):
    out_dir = Path(project_dir)
    plan = {
        "project_id": str(uuid.uuid4()),
        "content_type": "scene_analysis",
        "topic": title or "Example Title",
        "thesis": "Contrast between optimism and despair in the chosen scene.",
        "hook": "Why this moment flips the whole movie",
        "tone": "analytical",
        "structure": [
            {"id": "intro", "goal": "Hook and thesis", "target_seconds": 20},
            {"id": "scene_discussion", "goal": "Explain the scene", "target_seconds": 60},
            {"id": "closing", "goal": "Wrap and CTA", "target_seconds": 10}
        ],
        "visual_strategy": ["use_scene_clip", "generated_illustration"],
        "music_mood": "subtle_tension",
        "length_target_sec": 90,
        "novelty_constraints": {"avoid_recent_hook_types": True, "min_style_distance": 0.2}
    }
    out_path = out_dir / 'director_plan.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"Wrote director plan -> {out_path}")
    return out_path
