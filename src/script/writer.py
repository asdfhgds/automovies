"""Generate the first usable narration script for a project.

The MVP intentionally keeps script generation deterministic, but it now consumes
the same scene-index formats used by the rest of the pipeline.  This makes the
output useful to the TTS and editing stages instead of being a disconnected
placeholder.
"""
import json
from pathlib import Path


def _load_scenes(project_dir: Path):
    """Load the preferred scene index, falling back to legacy scene cards."""
    for path in (
        project_dir / "scenes" / "scene_index.json",
        project_dir / "scenes" / "scene_cards.json",
    ):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("scenes", [])
        if isinstance(data, list):
            return data
    return []


def generate_script(project_dir: Path):
    project_dir = Path(project_dir)
    plan_p = project_dir / 'director_plan.json'
    out_dir = project_dir
    if not plan_p.exists():
        raise FileNotFoundError('director_plan.json missing')
    with plan_p.open('r', encoding='utf-8') as f:
        plan = json.load(f)
    scenes = _load_scenes(project_dir)
    if not scenes:
        raise FileNotFoundError('scene_index.json or scene_cards.json missing')

    first_scene = scenes[0]
    thesis = plan.get("thesis") or "the meaning hidden inside a pivotal scene"
    summary = first_scene.get("summary") or first_scene.get("transcript") or "a pivotal moment"
    structure = plan.get("structure") or [
        {"id": "intro", "goal": "Hook and thesis", "target_seconds": 20},
        {"id": "scene_discussion", "goal": "Explain the scene", "target_seconds": 60},
        {"id": "closing", "goal": "Wrap and CTA", "target_seconds": 10},
    ]

    sections = []
    for section in structure:
        section_id = section.get("id", f"section_{len(sections) + 1}")
        goal = section.get("goal", "Develop the analysis")
        duration = max(1, int(section.get("target_seconds", 15)))
        if section_id in ("intro", "hook"):
            text = f"At first glance, this moment seems simple. But {thesis}"
        elif section_id in ("closing", "conclusion"):
            text = f"That is why this scene matters: {thesis}."
        else:
            text = f"{goal}. In this scene, {summary}. This gives us a way to see how {thesis}"
        sections.append({
            "section_id": section_id,
            "text": text,
            "estimated_seconds": duration,
            "scene_ids": [first_scene.get("scene_id", "scene-1")],
        })

    voiceover = " ".join(section["text"] for section in sections)
    project_id = plan.get("project_id")
    meta_path = project_dir / "project_meta.json"
    if meta_path.exists():
        project_id = json.loads(meta_path.read_text(encoding="utf-8")).get("project_id", project_id)
    script = {
        "project_id": project_id or project_dir.name,
        "voiceover_text": voiceover,
        "sections": sections,
        "cta": "Subscribe for more",
        "style_notes": f"{plan.get('tone', 'analytical')} commentary with a concise, cinematic pace",
    }
    out_path = out_dir / 'script.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(script, f, ensure_ascii=False, indent=2)
    print(f"Wrote script -> {out_path}")
    return out_path
