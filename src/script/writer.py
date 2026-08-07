"""Simple script generator stub that reads director_plan and scene_cards and writes script.json."""
import json
from pathlib import Path

def generate_script(project_dir: Path):
    project_dir = Path(project_dir)
    plan_p = project_dir / 'director_plan.json'
    scenes_p = project_dir / 'scenes' / 'scene_cards.json'
    out_dir = project_dir
    if not plan_p.exists() or not scenes_p.exists():
        raise FileNotFoundError('director_plan.json or scene_cards.json missing')
    with plan_p.open('r', encoding='utf-8') as f:
        plan = json.load(f)
    with scenes_p.open('r', encoding='utf-8') as f:
        scenes = json.load(f)

    voiceover = f"In this video, we explore: {plan.get('thesis')}\nSummary of scene: {scenes[0].get('summary')}"
    sections = [
        {"section_id": s['id'], "text": f"{s['goal']} - expands on the thesis.", "estimated_seconds": s['target_seconds']} for s in plan.get('structure',[])
    ]
    script = {"project_id": plan.get('project_id'), "voiceover_text": voiceover, "sections": sections, "cta": "Subscribe for more", "style_notes": "concise analytical tone"}
    out_path = out_dir / 'script.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(script, f, ensure_ascii=False, indent=2)
    print(f"Wrote script -> {out_path}")
    return out_path
