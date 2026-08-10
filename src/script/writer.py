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


def _load_selected_scenes(project_dir: Path):
    """Load the scenes chosen for the cut (multi-scene first, then legacy single)."""
    for path in (
        project_dir / "scenes" / "selected_scenes.json",
        project_dir / "scenes" / "selected_scene.json",
    ):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
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

    by_id = {s.get("scene_id"): s for s in scenes if s.get("scene_id")}
    selected = _load_selected_scenes(project_dir)
    selected = [s for s in selected if s.get("scene_id") in by_id]
    if not selected:
        selected = [scenes[0]]

    thesis = plan.get("thesis") or "the meaning hidden inside a pivotal scene"
    structure = plan.get("structure") or [
        {"id": "intro", "goal": "Hook and thesis", "target_seconds": 20},
        {"id": "scene_discussion", "goal": "Explain the scene", "target_seconds": 60},
        {"id": "closing", "goal": "Wrap and CTA", "target_seconds": 10},
    ]

    def _scene_text(scene):
        return scene.get("summary") or scene.get("transcript") or "a pivotal moment"

    scene_cursor = 0
    sections = []
    for section in structure:
        section_id = section.get("id", f"section_{len(sections) + 1}")
        goal = section.get("goal", "Develop the analysis")
        duration = max(1, int(section.get("target_seconds", 15)))
        if section_id in ("intro", "hook"):
            text = f"At first glance, this moment seems simple. But {thesis}"
            scene_ids = [selected[0]["scene_id"]]
        elif section_id in ("closing", "conclusion"):
            text = f"That is why this scene matters: {thesis}."
            scene_ids = [selected[-1]["scene_id"]]
        else:
            scene = selected[scene_cursor % len(selected)]
            scene_cursor += 1
            summary = _scene_text(by_id[scene["scene_id"]])
            text = f"{goal}. In this scene, {summary}. This gives us a way to see how {thesis}"
            scene_ids = [scene["scene_id"]]
        sections.append({
            "section_id": section_id,
            "text": text,
            "estimated_seconds": duration,
            "scene_ids": scene_ids,
        })

    voiceover = " ".join(section["text"] for section in sections)
    project_id = plan.get("project_id")
    meta_path = project_dir / "project_meta.json"
    if meta_path.exists():
        project_id = json.loads(meta_path.read_text(encoding="utf-8")).get("project_id", project_id)

    from script.narration import narration_properties_from_env

    narration_props = narration_properties_from_env(plan)
    script = {
        "project_id": project_id or project_dir.name,
        "voiceover_text": voiceover,
        "sections": sections,
        "cta": "Subscribe for more",
        "style_notes": f"{plan.get('tone', 'analytical')} commentary with a concise, cinematic pace",
        "scene_ids": [s["scene_id"] for s in selected],
        "narration_properties": narration_props,
    }
    out_path = out_dir / 'script.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(script, f, ensure_ascii=False, indent=2)
    print(f"Wrote script -> {out_path}")
    return out_path
